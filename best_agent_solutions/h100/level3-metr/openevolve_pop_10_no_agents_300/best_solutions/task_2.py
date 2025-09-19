# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for fused (Linear + ReLU) and (Linear) operations.
# This version builds upon the top-performing solution by adding a minor but impactful optimization.
# 1. C++ Templating: A single kernel `fused_gemv_template_kernel` generates both the ReLU and
#    non-ReLU variants, avoiding code duplication.
# 2. float4 Vectorization: Retains the highly effective `float4` memory access to maximize bandwidth.
# 3. Loop Unrolling (Compute): Keeps the 2x loop unrolling for the main computation to increase
#    instruction-level parallelism.
# 4. Warp-Level Reductions: Uses the fast `__shfl_down_sync` warp shuffle instructions for reduction,
#    minimizing shared memory traffic and synchronization overhead.
# 5. Loop Unrolling (Reduction): This is the key improvement. By adding `#pragma unroll` to the
#    reduction loops, we explicitly tell the compiler to unroll these small, fixed-iteration loops.
#    This eliminates loop overhead (branching and counter updates) for the reduction phase,
#    potentially leading to a small but consistent performance gain.

fused_mlp_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Template-based kernel for Fused GEMV + Bias, with optional ReLU.
// Optimized with float4 vectorization, compute loop unrolling, and unrolled warp-level reductions.
template <bool ApplyReLU>
__global__ void fused_gemv_template_kernel(const float* __restrict__ x,
                                           const float* __restrict__ w,
                                           const float* __restrict__ b,
                                           float* __restrict__ y,
                                           int N, int K) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int block_size = blockDim.x;
    const int warps_per_block = block_size / 32;

    // Shared memory to store the partial sum from each warp. Size is warps_per_block.
    extern __shared__ float sdata[];

    if (row < N) {
        float partial_sum = 0.0f;
        
        // Cast pointers to float4 for vectorized memory access.
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* w4_row = reinterpret_cast<const float4*>(w + row * K);
        const int K_vec = K / 4;

        // Main compute loop with 2x unrolling.
        int k = tid;
        for (; k + block_size < K_vec; k += 2 * block_size) {
            const float4 x_val1 = x4[k];
            const float4 w_val1 = w4_row[k];
            partial_sum += x_val1.x * w_val1.x + x_val1.y * w_val1.y + x_val1.z * w_val1.z + x_val1.w * w_val1.w;
            
            const float4 x_val2 = x4[k + block_size];
            const float4 w_val2 = w4_row[k + block_size];
            partial_sum += x_val2.x * w_val2.x + x_val2.y * w_val2.y + x_val2.z * w_val2.z + x_val2.w * w_val2.w;
        }
        // Handle remaining elements.
        if (k < K_vec) {
            const float4 x_val = x4[k];
            const float4 w_val = w4_row[k];
            partial_sum += x_val.x * w_val.x + x_val.y * w_val.y + x_val.z * w_val.z + x_val.w * w_val.w;
        }

        // --- Reduction Phase ---

        // Stage 1: Intra-warp reduction using shuffle instructions.
        // Unrolling this loop removes branch overhead for a faster reduction.
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }

        // Stage 2: Inter-warp reduction.
        // Lane 0 of each warp writes its partial sum to shared memory.
        if (lane_id == 0) {
            sdata[warp_id] = partial_sum;
        }

        // Synchronize to ensure all warp sums are in shared memory.
        __syncthreads();

        // The first warp performs the final reduction.
        if (warp_id == 0) {
            // Load warp sums into registers of the first warp's threads.
            float warp_sum = (lane_id < warps_per_block) ? sdata[lane_id] : 0.0f;
            
            // Final reduction within the first warp.
            // Unrolling this loop removes branch overhead for a faster reduction.
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
            }

            // Thread 0 adds bias, optionally applies ReLU, and writes the final result.
            if (lane_id == 0) {
                float total_sum = warp_sum + b[row];
                if (ApplyReLU) {
                    y[row] = fmaxf(0.0f, total_sum);
                } else {
                    y[row] = total_sum;
                }
            }
        }
    }
}

// Common C++ launcher function that handles tensor checks and calls the correct kernel template.
torch::Tensor gemv_launcher(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, bool apply_relu) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda(), "All tensors must be on a CUDA device");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "Input x must be of shape [1, K]");

    const auto x_cont = x.contiguous();
    const auto weight_cont = weight.contiguous();
    const auto bias_cont = bias.contiguous();

    const int K = x_cont.size(1);
    const int N = weight_cont.size(0);

    TORCH_CHECK(K % 4 == 0, "Input features K must be divisible by 4 for float4 kernel");

    auto y = torch::empty({1, N}, x_cont.options());

    const int block_size = 256;
    const int num_blocks = N;
    const int warps_per_block = block_size / 32;
    const int shared_mem_size = warps_per_block * sizeof(float);

    auto x_vec = x_cont.squeeze(0);
    auto y_vec = y.squeeze(0);

    if (apply_relu) {
        fused_gemv_template_kernel<true><<<num_blocks, block_size, shared_mem_size>>>(
            x_vec.data_ptr<float>(), weight_cont.data_ptr<float>(), bias_cont.data_ptr<float>(),
            y_vec.data_ptr<float>(), N, K);
    } else {
        fused_gemv_template_kernel<false><<<num_blocks, block_size, shared_mem_size>>>(
            x_vec.data_ptr<float>(), weight_cont.data_ptr<float>(), bias_cont.data_ptr<float>(),
            y_vec.data_ptr<float>(), N, K);
    }
    
    C10_CUDA_CHECK(cudaGetLastError());
    return y;
}

// Python-callable wrapper for Linear + ReLU.
torch::Tensor fused_linear_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    return gemv_launcher(x, weight, bias, true);
}

// Python-callable wrapper for Linear only.
torch::Tensor fused_linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    return gemv_launcher(x, weight, bias, false);
}
"""

fused_mlp_kernels_cpp_source = """
torch::Tensor fused_linear_relu_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
torch::Tensor fused_linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
"""

# JIT compile the CUDA kernels. Using a unique name to prevent caching issues.
fused_mlp_kernels = load_inline(
    name="fused_mlp_kernels_warp_v7_unrolled",
    cpp_sources=fused_mlp_kernels_cpp_source,
    cuda_sources=fused_mlp_kernels_source,
    functions=["fused_linear_relu_cuda", "fused_linear_cuda"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(Model, self).__init__()
        
        self.layers = nn.ModuleList()
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            self.layers.append(nn.Linear(current_input_size, hidden_size))
            current_input_size = hidden_size
        
        self.layers.append(nn.Linear(current_input_size, output_size))
    
    def forward(self, x):
        num_hidden_layers = len(self.layers) - 1
        
        # Apply the custom fused Linear + ReLU kernel for all hidden layers.
        for i in range(num_hidden_layers):
            layer = self.layers[i]
            x = fused_mlp_kernels.fused_linear_relu_cuda(x, layer.weight, layer.bias)
            
        # Apply the custom fused Linear kernel (no ReLU) for the final layer.
        last_layer = self.layers[-1]
        x = fused_mlp_kernels.fused_linear_cuda(x, last_layer.weight, last_layer.bias)
        
        return x

# Test code
batch_size = 1
input_size = 1000
hidden_layer_sizes = [2000, 2000]
output_size = 10

def get_inputs():
    # The custom CUDA kernel requires input tensors to be on the GPU.
    return [torch.randn(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
# EVOLVE-BLOCK-END