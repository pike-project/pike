# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source code for the fused operation: BatchNorm -> ReLU -> AvgPool2x2
# This version builds on the top-performing float32 implementation with a tuned launch configuration.
#
# Optimizations:
# 1. Algorithmic Reordering: (BN -> ReLU -> Pool) -> Conv, reducing Conv workload by 4x.
# 2. Operator Fusion: BN, ReLU, and AvgPool are combined into a single memory-bound kernel.
# 3. Vectorization: Uses float4 to maximize memory bandwidth, with each thread processing a 1x4 output tile.
# 4. Warp-Aligned Thread Block: A 32x8 thread block is used instead of 16x16. This aligns an entire
#    32-thread warp along the horizontal dimension, which can improve memory access coalescence.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 8

__global__ void bn_relu_avgpool_vectorized_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    const float bn_eps,
    float* __restrict__ out,
    const int N, const int C, const int H, const int W)
{
    // Shared memory for fused BN parameters, computed once per block.
    __shared__ float s_scale;
    __shared__ float s_shift;

    // Each thread computes a float4 (4 horizontal output pixels).
    // With THREADS_PER_BLOCK_X = 32, a full warp processes a contiguous 32*4 = 128-pixel wide segment.
    const int w_out_start = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int n_c_idx = blockIdx.z;

    const int H_out = H / 2;
    const int W_out = W / 2;

    // Boundary check for the entire vector of 4 pixels.
    if (h_out >= H_out || w_out_start >= W_out) return;

    // Decompose batch and channel index from grid's z-dimension.
    const int n = n_c_idx / C;
    const int c = n_c_idx % C;

    // Thread 0 of each block computes the fused BN parameters from the raw stats.
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float inv_std = rsqrtf(bn_var[c] + bn_eps);
        s_scale = bn_weight[c] * inv_std;
        s_shift = bn_bias[c] - bn_mean[c] * s_scale;
    }
    __syncthreads(); // Ensure all threads in the block see the computed parameters.

    // A 1x4 output requires a 2x8 input patch.
    const int h_in_start = h_out * 2;
    const int w_in_start = w_out_start * 2;
    
    const long long base_in_idx = (long long)(n * C + c) * H * W;
    const float* x_ptr = x + base_in_idx;
    
    // Create float4 pointers for vectorized loads.
    const float4* x_row0_ptr = (const float4*)(x_ptr + (long long)h_in_start * W + w_in_start);
    const float4* x_row1_ptr = (const float4*)(x_ptr + (long long)(h_in_start + 1) * W + w_in_start);

    // Load the 2x8 patch using four float4 loads.
    float4 in00 = x_row0_ptr[0];
    float4 in01 = x_row0_ptr[1];
    float4 in10 = x_row1_ptr[0];
    float4 in11 = x_row1_ptr[1];

    // Apply fused BN and ReLU to all 16 values in registers.
    #pragma unroll
    for (int i=0; i<4; ++i) {
      ((float*)&in00)[i] = fmaxf(0.0f, ((float*)&in00)[i] * s_scale + s_shift);
      ((float*)&in01)[i] = fmaxf(0.0f, ((float*)&in01)[i] * s_scale + s_shift);
      ((float*)&in10)[i] = fmaxf(0.0f, ((float*)&in10)[i] * s_scale + s_shift);
      ((float*)&in11)[i] = fmaxf(0.0f, ((float*)&in11)[i] * s_scale + s_shift);
    }

    // Perform the 2x2 average pooling for each of the 4 output pixels.
    float4 out_val;
    out_val.x = (in00.x + in00.y + in10.x + in10.y) * 0.25f;
    out_val.y = (in00.z + in00.w + in10.z + in10.w) * 0.25f;
    out_val.z = (in01.x + in01.y + in11.x + in11.y) * 0.25f;
    out_val.w = (in01.z + in01.w + in11.z + in11.w) * 0.25f;

    // Calculate output index and perform a single vectorized store.
    const long long base_out_idx = (long long)(n * C + c) * H_out * W_out;
    float4* out_ptr = (float4*)(out + base_out_idx + (long long)h_out * W_out + w_out_start);
    *out_ptr = out_val;
}

torch::Tensor bn_relu_avgpool_vectorized_cuda(
    torch::Tensor x,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_running_mean,
    torch::Tensor bn_running_var,
    double bn_eps)
{
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous NCHW");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Only float32 is supported");

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    TORCH_CHECK(W % 8 == 0, "Input width must be divisible by 8 for vectorization");

    const int H_out = H / 2;
    const int W_out = W / 2;

    auto out = torch::empty({N, C, H_out, W_out}, x.options());

    dim3 threads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
    dim3 blocks(
        (W_out + (threads.x * 4) - 1) / (threads.x * 4),
        (H_out + threads.y - 1) / threads.y,
        N * C);
    
    bn_relu_avgpool_vectorized_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        bn_running_mean.data_ptr<float>(),
        bn_running_var.data_ptr<float>(),
        (float)bn_eps,
        out.data_ptr<float>(),
        N, C, H, W
    );
    
    // Check for errors during kernel launch for easier debugging.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return out;
}
"""

cpp_source = "torch::Tensor bn_relu_avgpool_vectorized_cuda(torch::Tensor x, torch::Tensor bn_weight, torch::Tensor bn_bias, torch::Tensor bn_running_mean, torch::Tensor bn_running_var, double bn_eps);"

# JIT compile the CUDA kernel. Use a unique name to avoid caching issues.
fused_op = load_inline(
    name="fused_bn_relu_avgpool_32x8",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["bn_relu_avgpool_vectorized_cuda"],
    verbose=True,
)

class Model(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(Model, self).__init__()
        # Layers with parameters are kept for state_dict compatibility
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        
    def forward(self, x):
        # The custom kernel fuses BatchNorm -> ReLU -> AvgPool.
        # This reordered approach runs the Conv layer on a 4x smaller feature map.
        # The model must be in .eval() mode to use the running_mean/var stats.
        intermediate = fused_op.bn_relu_avgpool_vectorized_cuda(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps
        )
        # Apply the standard, highly-optimized 1x1 convolution afterwards
        return self.conv(intermediate)

batch_size = 10
num_input_features = 32
num_output_features = 64
height, width = 224, 224

def get_inputs():
    # Use float32, as it proved fastest in previous attempts.
    # Input must be NCHW contiguous for vectorized loads to work correctly.
    return [torch.randn(batch_size, num_input_features, height, width).cuda().contiguous()]

def get_init_inputs():
    return [num_input_features, num_output_features]

# EVOLVE-BLOCK-END