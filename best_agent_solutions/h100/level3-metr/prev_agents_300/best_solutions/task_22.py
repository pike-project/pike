import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel for fused implicit GEMM depthwise convolution, BatchNorm, and ReLU6
fused_depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

// TILE_SIZE_M and TILE_SIZE_N define the output tile size computed by a thread block.
#define TILE_SIZE_M 16
#define TILE_SIZE_N 16

template <int KERNEL_SIZE, int STRIDE>
__global__ void fused_depthwise_conv_bn_relu6_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* output,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    float bn_eps,
    int N, int C, int H, int W,
    int H_out, int W_out)
{
    constexpr int PADDING = (KERNEL_SIZE - 1) / 2;

    // Shared memory for input and weight tiles
    constexpr int SHARED_TILE_H = (TILE_SIZE_M - 1) * STRIDE + KERNEL_SIZE;
    constexpr int SHARED_TILE_W = (TILE_SIZE_N - 1) * STRIDE + KERNEL_SIZE;

    __shared__ float input_tile[SHARED_TILE_H][SHARED_TILE_W];
    __shared__ float weight_tile[KERNEL_SIZE][KERNEL_SIZE];

    // Thread and block indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int thread_id_in_block = ty * TILE_SIZE_N + tx;

    // Global output indices
    const int out_x = blockIdx.x * TILE_SIZE_N + tx;
    const int out_y = blockIdx.y * TILE_SIZE_M + ty;
    
    // Batch and channel index from blockIdx.z
    const int c = blockIdx.z % C;
    const int n = blockIdx.z / C;

    // --- 1. Load weight tile into shared memory ---
    if (thread_id_in_block < KERNEL_SIZE * KERNEL_SIZE) {
        int ky = thread_id_in_block / KERNEL_SIZE;
        int kx = thread_id_in_block % KERNEL_SIZE;
        weight_tile[ky][kx] = weight[c * KERNEL_SIZE * KERNEL_SIZE + ky * KERNEL_SIZE + kx];
    }

    // --- 2. Load input tile into shared memory ---
    const int in_x_origin = blockIdx.x * TILE_SIZE_N * STRIDE - PADDING;
    const int in_y_origin = blockIdx.y * TILE_SIZE_M * STRIDE - PADDING;
    const int block_size = TILE_SIZE_M * TILE_SIZE_N;

    for (int i = thread_id_in_block; i < SHARED_TILE_H * SHARED_TILE_W; i += block_size) {
        int smem_y = i / SHARED_TILE_W;
        int smem_x = i % SHARED_TILE_W;
        int gmem_y = in_y_origin + smem_y;
        int gmem_x = in_x_origin + smem_x;
        const int input_idx = n * C * H * W + c * H * W + gmem_y * W + gmem_x;

        if (gmem_y >= 0 && gmem_y < H && gmem_x >= 0 && gmem_x < W) {
            input_tile[smem_y][smem_x] = input[input_idx];
        } else {
            input_tile[smem_y][smem_x] = 0.0f;
        }
    }

    __syncthreads();

    // --- 3. Perform convolution and fused operations ---
    if (out_x < W_out && out_y < H_out) {
        float acc = 0.0f;
        const int smem_start_y = ty * STRIDE;
        const int smem_start_x = tx * STRIDE;

        #pragma unroll
        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                acc += input_tile[smem_start_y + ky][smem_start_x + kx] * weight_tile[ky][kx];
            }
        }
        
        // --- 4. Apply fused BatchNorm and ReLU6 ---
        float mean = bn_mean[c];
        float var = bn_var[c];
        float gamma = bn_weight[c];
        float beta = bn_bias[c];
        
        float inv_std = rsqrtf(var + bn_eps);
        acc = (acc - mean) * inv_std * gamma + beta;
        
        acc = fminf(fmaxf(acc, 0.0f), 6.0f);

        const int output_idx = n * C * H_out * W_out + c * H_out * W_out + out_y * W_out + out_x;
        output[output_idx] = acc;
    }
}

torch::Tensor fused_depthwise_conv_bn_relu6_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    double bn_eps,
    int stride)
{
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight tensor must be contiguous");
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    
    const int K = weight.size(2);
    const int P = (K - 1) / 2;

    const int H_out = (H + 2 * P - K) / stride + 1;
    const int W_out = (W + 2 * P - K) / stride + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    dim3 block_dim(TILE_SIZE_N, TILE_SIZE_M, 1);
    dim3 grid_dim(
        (W_out + TILE_SIZE_N - 1) / TILE_SIZE_N,
        (H_out + TILE_SIZE_M - 1) / TILE_SIZE_M,
        N * C
    );
    
    auto stream = at::cuda::getCurrentCUDAStream();

    if (K == 3 && stride == 1) {
        fused_depthwise_conv_bn_relu6_kernel<3, 1><<<grid_dim, block_dim, 0, stream>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
            bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
            bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
            (float)bn_eps, N, C, H, W, H_out, W_out);
    } else if (K == 3 && stride == 2) {
        fused_depthwise_conv_bn_relu6_kernel<3, 2><<<grid_dim, block_dim, 0, stream>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
            bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
            bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
            (float)bn_eps, N, C, H, W, H_out, W_out);
    } else if (K == 5 && stride == 1) {
        fused_depthwise_conv_bn_relu6_kernel<5, 1><<<grid_dim, block_dim, 0, stream>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
            bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
            bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
            (float)bn_eps, N, C, H, W, H_out, W_out);
    } else if (K == 5 && stride == 2) {
        fused_depthwise_conv_bn_relu6_kernel<5, 2><<<grid_dim, block_dim, 0, stream>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
            bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
            bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
            (float)bn_eps, N, C, H, W, H_out, W_out);
    } else {
        TORCH_CHECK(false, "Unsupported kernel size or stride for fused depthwise conv. Got K=", K, ", S=", stride);
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}

"""

fused_depthwise_conv_cpp_source = """
torch::Tensor fused_depthwise_conv_bn_relu6_cuda(
    torch::Tensor input, torch::Tensor weight,
    torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var,
    double bn_eps, int stride);
"""

# JIT compile the custom CUDA kernel
custom_conv_op = load_inline(
    name="fused_depthwise_conv",
    cpp_sources=fused_depthwise_conv_cpp_source,
    cuda_sources=fused_depthwise_conv_source,
    functions=["fused_depthwise_conv_bn_relu6_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class FusedDepthwiseConvBNReLU6(nn.Module):
    """
    Custom nn.Module for a fused (DepthwiseConv2d + BatchNorm2d + ReLU6) operation.
    It holds the parameters for both convolution and batch normalization.
    This module assumes it is used for inference (i.e., model.eval() mode).
    """
    def __init__(self, channels, kernel_size, stride):
        super().__init__()
        self.stride = stride
        # Depthwise conv weight has shape (out_channels, in_channels/groups, k, k)
        # For depthwise, groups=in_channels=out_channels, so in_channels/groups = 1
        self.conv_weight = nn.Parameter(torch.Tensor(channels, 1, kernel_size, kernel_size))
        self.bn = nn.BatchNorm2d(channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))

    def forward(self, x):
        return custom_conv_op.fused_depthwise_conv_bn_relu6_cuda(
            x, self.conv_weight,
            self.bn.weight, self.bn.bias,
            self.bn.running_mean, self.bn.running_var,
            self.bn.eps, self.stride
        )


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation using the custom fused kernel for the depthwise stage.
        """
        super(MBConv, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        # Replace the original depthwise_conv with the custom fused one
        # The original was:
        # self.depthwise_conv = nn.Sequential(
        #     nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.ReLU6(inplace=True)
        # )
        self.fused_depthwise_conv = FusedDepthwiseConvBNReLU6(hidden_dim, kernel_size, stride)

        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        """
        Forward pass of the MBConv block.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        x = self.fused_depthwise_conv(x)
        
        x = self.project_conv(x)
        
        if self.use_residual:
            x += identity
        
        return x


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB0 architecture implementation in PyTorch.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(Model, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            # MBConv1 (32, 16, 1, 1)
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the EfficientNetB0 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]