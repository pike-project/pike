# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# --- Fused CUDA Kernels ---
# This version synthesizes the best features from all previous attempts.
# 1. bn_relu: The high-performing fused BatchNorm+ReLU kernel from the top solution,
#    with its robust float4 vectorization path.
# 2. bn_relu_pool: A new, more advanced version that includes BOTH a scalar and a
#    vectorized implementation. A smart C++ dispatcher chooses the fastest path based
#    on tensor shape and memory alignment, making it faster and more general.
# 3. Micro-optimizations like '#pragma unroll' are included for warp reductions.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm> // For std::min

// --- Kernel 1: Fused BatchNorm2d + ReLU (Scalar + Vectorized) ---

// Scalar version (fallback)
__global__ void batch_norm_relu_kernel_scalar(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int total_elements,
    int C,
    int spatial_dim)
{
    // Grid-stride loop for robustness
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_elements;
         i += blockDim.x * gridDim.x) {

        int c = (i / spatial_dim) % C;
        // Pre-calculate scale and shift for FMA
        float scale = weight[c] * rsqrtf(running_var[c] + eps);
        float shift = bias[c] - running_mean[c] * scale;
        // Fused operation
        output[i] = fmaxf(0.0f, input[i] * scale + shift);
    }
}

// Vectorized (float4) version
__global__ void batch_norm_relu_kernel_vec4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int total_vec4_elements,
    int C,
    int spatial_dim)
{
    // Grid-stride loop over float4 elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_vec4_elements;
         i += blockDim.x * gridDim.x) {
        
        float4 in_val = input[i];
        
        const int base_idx = i * 4;
        const int c = (base_idx / spatial_dim) % C;

        // Pre-calculate scale and shift (once for 4 elements)
        const float scale = weight[c] * rsqrtf(running_var[c] + eps);
        const float shift = bias[c] - running_mean[c] * scale;

        // Apply operations element-wise on the vector
        float4 out_val;
        out_val.x = fmaxf(0.0f, in_val.x * scale + shift);
        out_val.y = fmaxf(0.0f, in_val.y * scale + shift);
        out_val.z = fmaxf(0.0f, in_val.z * scale + shift);
        out_val.w = fmaxf(0.0f, in_val.w * scale + shift);
        
        output[i] = out_val;
    }
}

// C++ dispatcher for BN+ReLU
torch::Tensor batch_norm_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps) {

    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int total_elements = input.numel();
    const int spatial_dim = H * W;

    auto output = torch::empty_like(input);
    if (total_elements == 0) return output;

    const int block_size = 256;
    const float f_eps = static_cast<float>(eps);

    if ( (spatial_dim % 4 == 0) && ( ( (intptr_t)input.data_ptr() % 16) == 0 ) && ( ( (intptr_t)output.data_ptr() % 16) == 0 ) ) {
        const int total_vec4_elements = total_elements / 4;
        const int num_blocks = std::min((total_vec4_elements + block_size - 1) / block_size, 4096);
        batch_norm_relu_kernel_vec4<<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            weight.data_ptr<float>(), bias.data_ptr<float>(), 
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            f_eps, total_vec4_elements, C, spatial_dim);
    } else {
        const int num_blocks = std::min((total_elements + block_size - 1) / block_size, 4096);
        batch_norm_relu_kernel_scalar<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), weight.data_ptr<float>(),
            bias.data_ptr<float>(), running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            f_eps, total_elements, C, spatial_dim);
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// --- Kernel 2: Fused BN+ReLU+Pool with Warp Shuffle Reduction (Scalar + Vectorized) ---

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
  #pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

template <typename T>
__device__ __forceinline__ T blockReduceSum(T val, T* shared) {
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);

  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : (T)0.0;
  if (wid == 0) {
    val = warpReduceSum(val);
  }
  return val;
}

// Vectorized (float4) version of the pool kernel
__global__ void fused_bn_relu_pool_kernel_vec4(
    const float4* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W) {

    const int n = blockIdx.y;
    const int c = blockIdx.x;
    
    extern __shared__ float shared_mem[];

    const float scale = weight[c] * rsqrtf(running_var[c] + eps);
    const float shift = bias[c] - running_mean[c] * scale;

    const int plane_size = H * W;
    const int plane_size_vec4 = plane_size / 4;
    const float4* plane_ptr = input + (n * C + c) * plane_size_vec4;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < plane_size_vec4; i += blockDim.x) {
        float4 val4 = plane_ptr[i];
        local_sum += fmaxf(0.0f, val4.x * scale + shift);
        local_sum += fmaxf(0.0f, val4.y * scale + shift);
        local_sum += fmaxf(0.0f, val4.z * scale + shift);
        local_sum += fmaxf(0.0f, val4.w * scale + shift);
    }

    float total_sum = blockReduceSum(local_sum, shared_mem);

    if (threadIdx.x == 0) {
        output[n * C + c] = total_sum / plane_size;
    }
}

// Scalar version of the pool kernel (fallback)
__global__ void fused_bn_relu_pool_kernel_scalar(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W) {

    const int n = blockIdx.y;
    const int c = blockIdx.x;
    
    extern __shared__ float shared_mem[];

    const float scale = weight[c] * rsqrtf(running_var[c] + eps);
    const float shift = bias[c] - running_mean[c] * scale;

    const int plane_size = H * W;
    const float* plane_ptr = input + (n * C + c) * plane_size;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < plane_size; i += blockDim.x) {
        local_sum += fmaxf(0.0f, plane_ptr[i] * scale + shift);
    }

    float total_sum = blockReduceSum(local_sum, shared_mem);

    if (threadIdx.x == 0) {
        output[n * C + c] = total_sum / plane_size;
    }
}

// C++ Dispatcher for BN+ReLU+Pool
torch::Tensor bn_relu_adaptive_avg_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps) {

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const int spatial_dim = H * W;

    auto output = torch::empty({N, C}, input.options());
    if (input.numel() == 0) return output;

    const int block_size = 512;
    dim3 grid_dim(C, N);
    
    const int threads_per_plane = std::min(block_size, spatial_dim);
    dim3 block_dim(threads_per_plane);
    size_t smem_size = (threads_per_plane / 32 + 1) * sizeof(float);

    const float f_eps = static_cast<float>(eps);

    const bool can_vectorize = (spatial_dim % 4 == 0) &&
                               (reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0);

    if (can_vectorize) {
        fused_bn_relu_pool_kernel_vec4<<<grid_dim, block_dim, smem_size>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            output.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            f_eps, N, C, H, W);
    } else {
        fused_bn_relu_pool_kernel_scalar<<<grid_dim, block_dim, smem_size>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), weight.data_ptr<float>(),
            bias.data_ptr<float>(), running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            f_eps, N, C, H, W);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor batch_norm_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps);

torch::Tensor bn_relu_adaptive_avg_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps);
"""

# JIT compile the fused kernels
fused_ops = load_inline(
    name="fused_densenet_ops_ultimate_v2",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["batch_norm_relu_cuda", "bn_relu_adaptive_avg_pool_cuda"],
    verbose=True,
)


# --- PyTorch Modules using Fused Kernels ---

class FusedBatchNormReLU(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(FusedBatchNormReLU, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps)

    def forward(self, x):
        # Fallback to standard PyTorch ops during training
        if self.training:
            return F.relu(self.bn(x), inplace=True)
        else:
            return fused_ops.batch_norm_relu_cuda(
                x, self.bn.weight, self.bn.bias,
                self.bn.running_mean, self.bn.running_var, self.bn.eps)

class FusedBatchNormReLUAvgPool(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(FusedBatchNormReLUAvgPool, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps)

    def forward(self, x):
        if self.training:
            x = F.relu(self.bn(x), inplace=True)
            return F.adaptive_avg_pool2d(x, (1, 1))
        else:
            return fused_ops.bn_relu_adaptive_avg_pool_cuda(
                x, self.bn.weight, self.bn.bias,
                self.bn.running_mean, self.bn.running_var, self.bn.eps)

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            FusedBatchNormReLU(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-allocation strategy to avoid torch.cat bottleneck (inference-only optimization)
        batch_size, num_initial_features, H, W = x.shape
        final_num_features = num_initial_features + self.num_layers * self.growth_rate

        all_features = torch.empty(batch_size, final_num_features, H, W, dtype=x.dtype, device=x.device,
                                   memory_format=torch.contiguous_format)
        all_features[:, :num_initial_features, :, :] = x
        
        current_offset = num_initial_features
        for layer in self.layers:
            # Input is a zero-copy slice of accumulated features
            input_tensor = all_features.narrow(1, 0, current_offset)
            
            new_feature = layer(input_tensor)
            
            # Write new feature into its slice in the output tensor
            all_features[:, current_offset:current_offset + self.growth_rate, :, :] = new_feature
            
            current_offset += self.growth_rate
            
        return all_features

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            FusedBatchNormReLU(num_input_features),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class Model(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            FusedBatchNormReLU(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = 64
        block_layers = [6, 12, 24, 16]
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features //= 2

        self.final_op = FusedBatchNormReLUAvgPool(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transition_layers):
                x = self.transition_layers[i](x)
        
        x = self.final_op(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

batch_size = 10
num_classes = 10
height, width = 224, 224

def get_inputs():
    # Ensure contiguous memory for safe vectorized access in CUDA
    return [torch.randn(batch_size, 3, height, width).cuda().contiguous()]

def get_init_inputs():
    return [32, num_classes]

# EVOLVE-BLOCK-END