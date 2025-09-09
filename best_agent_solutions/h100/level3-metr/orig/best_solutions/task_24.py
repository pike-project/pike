import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------
# 1. Custom CUDA Kernels and C++ Wrappers
# ----------------------------------------------------
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// --- Helper for CUDA error checking ---
#define CUDA_CHECK(err) \\
    if (err != cudaSuccess) { \\
        TORCH_CHECK(false, "CUDA kernel failed: ", cudaGetErrorString(err)); \\
    }

// --- Kernel 1 (from previous): Fused Bias Add + ReLU (In-place) ---
__global__ void add_bias_relu_inplace_kernel(float* x, const float* bias, long total_elements, int C, int HW) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int c = (idx / HW) % C;
    x[idx] = fmaxf(0.f, x[idx] + bias[c]);
}

// --- Kernel 2 (from previous): Fused Add Bias (In-place) ---
__global__ void add_bias_inplace_kernel(float* data, const float* bias, long total_elements, int C, int HW) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int c = (idx / HW) % C;
    data[idx] = data[idx] + bias[c];
}

// --- Kernel 3 (New): Global Average Pooling ---
// Each block computes the average for one channel (over H and W)
__global__ void global_avg_pool_kernel(const float* input, float* output, int N, int C, int H, int W) {
    extern __shared__ float sdata[];
    int c = blockIdx.x;
    int n = blockIdx.y;
    
    if (c >= C || n >= N) return;

    const float* channel_in = input + n * C * H * W + c * H * W;
    int tid = threadIdx.x;
    int HW = H * W;

    float sum = 0.0f;
    for (int i = tid; i < HW; i += blockDim.x) {
        sum += channel_in[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[n * C + c] = sdata[0] / (float)HW;
    }
}

// --- Kernel 4 (New): Tiled Matmul + ReLU ---
#define TILE_DIM 16
__global__ void matmul_relu_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // Computes C = relu(A @ B), where A(M,K), B(K,N), C(M,N)
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < num_tiles; ++t) {
        int a_idx = row * K + (t * TILE_DIM + threadIdx.x);
        if (row < M && (t * TILE_DIM + threadIdx.x) < K) {
            sA[threadIdx.y][threadIdx.x] = A[a_idx];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_idx = (t * TILE_DIM + threadIdx.y) * N + col;
        if (col < N && (t * TILE_DIM + threadIdx.y) < K) {
            sB[threadIdx.y][threadIdx.x] = B[b_idx];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = fmaxf(0.f, acc); // Apply ReLU
    }
}

// --- Kernel 5 (New): Tiled Matmul + Sigmoid ---
__global__ void matmul_sigmoid_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // Computes C = sigmoid(A @ B)
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float acc = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;
    
    for (int t = 0; t < num_tiles; ++t) {
        int a_idx = row * K + (t * TILE_DIM + threadIdx.x);
        if (row < M && (t * TILE_DIM + threadIdx.x) < K) {
            sA[threadIdx.y][threadIdx.x] = A[a_idx];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_idx = (t * TILE_DIM + threadIdx.y) * N + col;
        if (col < N && (t * TILE_DIM + threadIdx.y) < K) {
            sB[threadIdx.y][threadIdx.x] = B[b_idx];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_DIM; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = 1.0f / (1.0f + expf(-acc)); // Apply Sigmoid
    }
}


// --- C++ Wrappers for PyTorch ---

void add_bias_relu_inplace_cuda(torch::Tensor x, torch::Tensor bias) {
    const long total_elements = x.numel();
    if (total_elements == 0) return;
    TORCH_CHECK(x.dim() == 4, "Input tensor 'x' must be 4-dimensional");
    const int C = x.size(1);
    const int HW = x.size(2) * x.size(3);
    const int block_size = 256;
    const dim3 num_blocks((total_elements + block_size - 1) / block_size);
    add_bias_relu_inplace_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), bias.data_ptr<float>(), total_elements, C, HW);
    CUDA_CHECK(cudaGetLastError());
}

void add_bias_inplace_cuda(torch::Tensor data, torch::Tensor bias) {
    const long total_elements = data.numel();
    if (total_elements == 0) return;
    TORCH_CHECK(data.dim() == 4, "Input tensor 'data' must be 4-dimensional");
    const int C = data.size(1);
    const int HW = data.size(2) * data.size(3);
    const int block_size = 256;
    const dim3 num_blocks((total_elements + block_size - 1) / block_size);
    add_bias_inplace_kernel<<<num_blocks, block_size>>>(data.data_ptr<float>(), bias.data_ptr<float>(), total_elements, C, HW);
    CUDA_CHECK(cudaGetLastError());
}

torch::Tensor global_avg_pool_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    
    auto output = torch::empty({N, C, 1, 1}, input.options());
    
    dim3 grid_dim(C, N);
    const int block_size = 256; // Can be tuned
    dim3 block_dim(block_size);
    size_t smem_size = block_size * sizeof(float);
    
    global_avg_pool_kernel<<<grid_dim, block_dim, smem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W);
    CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor matmul_relu_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions mismatch for matmul");
    
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    auto C = torch::empty({M, N}, a.options());

    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    matmul_relu_kernel<<<grid_dim, block_dim>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    CUDA_CHECK(cudaGetLastError());
    return C;
}

torch::Tensor matmul_sigmoid_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions mismatch for matmul");
    
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);

    auto C = torch::empty({M, N}, a.options());

    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    matmul_sigmoid_kernel<<<grid_dim, block_dim>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    CUDA_CHECK(cudaGetLastError());
    return C;
}
"""

fused_ops_cpp_source = """
void add_bias_relu_inplace_cuda(torch::Tensor x, torch::Tensor bias);
void add_bias_inplace_cuda(torch::Tensor data, torch::Tensor bias);
torch::Tensor global_avg_pool_cuda(torch::Tensor input);
torch::Tensor matmul_relu_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor matmul_sigmoid_cuda(torch::Tensor a, torch::Tensor b);
"""

# Compile the CUDA kernels
custom_ops = load_inline(
    name="custom_fused_ops_efficientnet_v2",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=[
        "add_bias_relu_inplace_cuda",
        "add_bias_inplace_cuda",
        "global_avg_pool_cuda",
        "matmul_relu_cuda",
        "matmul_sigmoid_cuda",
    ],
    verbose=False,
)

# ----------------------------------------------------
# 2. Custom PyTorch Modules for Fusion
# ----------------------------------------------------

def _fuse_conv_bn_params(conv, bn):
    """Helper function to fuse Conv and BN parameters for inference."""
    conv.eval()
    bn.eval()
    w_conv = conv.weight.clone()
    b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=w_conv.device)
    running_mean, running_var, gamma, beta, eps = bn.running_mean, bn.running_var, bn.weight, bn.bias, bn.eps
    std = (running_var + eps).sqrt()
    scale_factor = gamma / std
    fused_conv_w = w_conv * scale_factor.reshape([-1] + [1] * (w_conv.dim() - 1))
    fused_conv_b = (b_conv - running_mean) * scale_factor + beta
    return fused_conv_w, fused_conv_b

class FusedConvBN(nn.Module):
    def __init__(self, conv, bn):
        super().__init__()
        self.stride, self.padding, self.dilation, self.groups = conv.stride, conv.padding, conv.dilation, conv.groups
        fused_w, fused_b = _fuse_conv_bn_params(conv, bn)
        self.register_buffer('fused_weight', fused_w)
        self.register_buffer('fused_bias', fused_b)

    def forward(self, x):
        x = F.conv2d(x, self.fused_weight, None, self.stride, self.padding, self.dilation, self.groups)
        custom_ops.add_bias_inplace_cuda(x, self.fused_bias)
        return x

class FusedConvBNReLU(nn.Module):
    def __init__(self, conv, bn):
        super().__init__()
        self.stride, self.padding, self.dilation, self.groups = conv.stride, conv.padding, conv.dilation, conv.groups
        fused_w, fused_b = _fuse_conv_bn_params(conv, bn)
        self.register_buffer('fused_weight', fused_w)
        self.register_buffer('fused_bias', fused_b)

    def forward(self, x):
        x = F.conv2d(x, self.fused_weight, None, self.stride, self.padding, self.dilation, self.groups)
        custom_ops.add_bias_relu_inplace_cuda(x, self.fused_bias)
        return x

class FusedSEBlock(nn.Module):
    """
    A fully fused Squeeze-and-Excitation block using custom CUDA kernels.
    Replaces AdaptiveAvgPool -> Conv -> ReLU -> Conv -> Sigmoid.
    """
    def __init__(self, conv_squeeze, conv_excite):
        super().__init__()
        # 1x1 Conv weights are (C_out, C_in, 1, 1). We need (C_in, C_out) for matmul.
        # So we squeeze and transpose the weight matrices.
        w1 = conv_squeeze.weight.detach().squeeze().T
        w2 = conv_excite.weight.detach().squeeze().T
        self.register_buffer('w1', w1)
        self.register_buffer('w2', w2)
    
    def forward(self, x):
        n, c, _, _ = x.shape
        
        # 1. Custom Global Average Pool
        pooled = custom_ops.global_avg_pool_cuda(x)
        
        # 2. Reshape for Matmul
        pooled_reshaped = pooled.view(n, c)
        
        # 3. Fused Matmul + ReLU
        intermediate = custom_ops.matmul_relu_cuda(pooled_reshaped, self.w1)
        
        # 4. Fused Matmul + Sigmoid
        out_reshaped = custom_ops.matmul_sigmoid_cuda(intermediate, self.w2)
        
        return out_reshaped.view(n, c, 1, 1)

class MBConvOptimized(nn.Module):
    """
    An optimized MBConv block using fused Conv-BN-ReLU and a fully fused SE Block.
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        expanded_channels = in_channels * expand_ratio

        # Expansion phase
        if expand_ratio != 1:
            self.expand = FusedConvBNReLU(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels)
            )
        else:
            self.expand = nn.Identity()
            
        # Depthwise convolution phase
        self.dw_conv = FusedConvBNReLU(
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels)
        )
        
        # Squeeze-and-Excitation phase (Fully Fused)
        self.se = FusedSEBlock(
            nn.Conv2d(expanded_channels, expanded_channels // 4, 1, bias=False),
            nn.Conv2d(expanded_channels // 4, expanded_channels, 1, bias=False)
        )
        
        # Projection phase
        self.project = FusedConvBN(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Replicate the original model's sequential (and flawed) data flow
        out = self.expand(x)
        out = self.dw_conv(out)
        out = self.se(out) 
        out = self.project(out)
        return out

# ----------------------------------------------------
# 3. Final Optimized Model
# ----------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        # Fused initial Conv+BN+ReLU stem
        self.fused_conv1 = FusedConvBNReLU(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )
        
        # Optimized MBConv blocks with fully fused SE
        self.mbconv1 = MBConvOptimized(32, 96, 1, 3)
        self.mbconv2 = MBConvOptimized(96, 144, 2, 6)
        self.mbconv3 = MBConvOptimized(144, 192, 2, 6)
        self.mbconv4 = MBConvOptimized(192, 288, 2, 6)
        self.mbconv5 = MBConvOptimized(288, 384, 1, 6)
        
        # Fused final Conv+BN+ReLU head
        self.fused_conv_final = FusedConvBNReLU(
            nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1408)
        )
        
        # Standard final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1408, num_classes)
    
    def forward(self, x):
        x = self.fused_conv1(x)
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.fused_conv_final(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x