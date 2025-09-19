# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This solution enhances the top-performing program by introducing vectorized memory access.
# The core GEMV (matrix-vector) operation is often memory-bound. By loading data in
# larger chunks (using float4), we reduce the number of memory transactions and
# better utilize GPU memory bandwidth. This is a potent optimization given that all
# relevant dimensions (1000, 400, 800) are divisible by 4.
# We retain the highly effective two-stage warp-shuffle reduction from the best prior solution.

fused_gemv_vectorized_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Templated kernel for matrix-vector product, optimized with float4 vectorization and WLP reduction.
// The template parameter `WithReLU` controls whether a ReLU activation is applied.
template <bool WithReLU>
__global__ void matvec_kernel_vec4(
    const float* __restrict__ X, // input vector, shape (K)
    const float* __restrict__ W, // weight matrix, shape (N, K)
    const float* __restrict__ B, // bias vector, shape (N)
    float* __restrict__ Y,       // output vector, shape (N)
    int N, int K) {
    
    // Each CUDA block computes one element of the output vector Y.
    const int n = blockIdx.x;
    if (n >= N) return;

    const int tid = threadIdx.x;
    
    // --- Vectorized Dot Product Computation ---
    // Cast pointers to float4 to load 4 floats at a time.
    const float4* X_vec = (const float4*)X;
    const float4* w_row_vec = (const float4*)(W + n * K);
    const int K_vec = K / 4;

    float partial_sum = 0.0f;
    // Each thread computes a partial sum by striding over the vectorized input.
    for (int k = tid; k < K_vec; k += blockDim.x) {
        float4 x_val = X_vec[k];
        float4 w_val = w_row_vec[k];
        // Manually unroll the dot product of the two float4 vectors.
        partial_sum += x_val.x * w_val.x;
        partial_sum += x_val.y * w_val.y;
        partial_sum += x_val.z * w_val.z;
        partial_sum += x_val.w * w_val.w;
    }

    // --- STAGE 1: Intra-Warp Reduction ---
    // Reduce partial sums within each warp using fast shuffle-down instructions.
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }
    
    // --- STAGE 2: Inter-Warp Reduction ---
    extern __shared__ float sdata[];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // The first thread in each warp writes its sum to shared memory.
    if (lane_id == 0) {
        sdata[warp_id] = partial_sum;
    }
    __syncthreads();

    // The first warp reduces the final sums from shared memory.
    if (warp_id == 0) {
        partial_sum = (lane_id < blockDim.x / 32) ? sdata[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }
    }

    // Thread 0 of the block holds the final sum. It adds the bias, applies ReLU
    // if enabled, and writes the final result.
    if (tid == 0) {
        float final_sum = partial_sum + B[n];
        if (WithReLU) {
            Y[n] = fmaxf(0.0f, final_sum);
        } else {
            Y[n] = final_sum;
        }
    }
}

// Helper to launch the templated kernel.
template<bool WithReLU>
void launch_matvec_kernel(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor y) {
    const int N = w.size(0);
    const int K = w.size(1);

    // Heuristic for block size, same as the top-performing solution.
    int block_size = 256;
    if (K > 1024) block_size = 1024;
    else if (K > 512) block_size = 512;
    
    const int num_blocks = N;
    const int shared_mem_size = (block_size / 32) * sizeof(float);

    matvec_kernel_vec4<WithReLU><<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), N, K);
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
}

// C++ wrapper for fused_linear_relu (WithReLU = true)
torch::Tensor fused_linear_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "Input x must be of shape [1, K]");
    TORCH_CHECK(x.size(1) == w.size(1), "Inner dimensions of x and w must match");
    TORCH_CHECK(w.size(0) == b.size(0), "Dimensions of w and b must match");
    TORCH_CHECK(w.size(1) % 4 == 0, "Inner dimension K must be a multiple of 4 for vectorized kernel");

    auto y = torch::empty({1, w.size(0)}, x.options());
    launch_matvec_kernel<true>(x.contiguous(), w.contiguous(), b.contiguous(), y);
    return y;
}

// C++ wrapper for linear (WithReLU = false)
torch::Tensor fused_linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "Input x must be of shape [1, K]");
    TORCH_CHECK(x.size(1) == w.size(1), "Inner dimensions of x and w must match");
    TORCH_CHECK(w.size(0) == b.size(0), "Dimensions of w and b must match");
    TORCH_CHECK(w.size(1) % 4 == 0, "Inner dimension K must be a multiple of 4 for vectorized kernel");

    auto y = torch::empty({1, w.size(0)}, x.options());
    launch_matvec_kernel<false>(x.contiguous(), w.contiguous(), b.contiguous(), y);
    return y;
}
"""

fused_gemv_vectorized_cpp_source = """
torch::Tensor fused_linear_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
torch::Tensor fused_linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
"""

# JIT compile the inline CUDA code. A unique name prevents caching issues.
fused_op = load_inline(
    name="fused_gemv_op_vec4_v2",
    cpp_sources=fused_gemv_vectorized_cpp_source,
    cuda_sources=fused_gemv_vectorized_source,
    functions=["fused_linear_relu_cuda", "fused_linear_cuda"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(Model, self).__init__()
        
        # Store weights and biases in ParameterLists for direct access by the custom kernel.
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        all_sizes = [input_size] + layer_sizes + [output_size]
        
        # Manually create parameters, initializing them identically to nn.Linear
        # to ensure numerical correctness with the baseline model.
        for i in range(len(all_sizes) - 1):
            temp_layer = nn.Linear(all_sizes[i], all_sizes[i+1])
            self.weights.append(nn.Parameter(temp_layer.weight.clone()))
            self.biases.append(nn.Parameter(temp_layer.bias.clone()))
            
    def forward(self, x):
        # Apply fused (linear + relu) for all hidden layers
        num_hidden_layers = len(self.weights) - 1
        for i in range(num_hidden_layers):
            x = fused_op.fused_linear_relu_cuda(x, self.weights[i], self.biases[i])
        
        # Apply the final fused linear layer (without ReLU)
        x = fused_op.fused_linear_cuda(x, self.weights[-1], self.biases[-1])
        
        return x

# Test code
batch_size = 1
input_size = 1000
layer_sizes = [400, 800]
output_size = 500

def get_inputs():
    # The custom kernel requires inputs to be on the GPU.
    return [torch.randn(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]
# EVOLVE-BLOCK-END