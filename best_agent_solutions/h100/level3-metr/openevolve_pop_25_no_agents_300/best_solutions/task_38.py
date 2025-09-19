# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# This custom CUDA kernel fuses slicing and linear layer operations using a
# vectorized, batched GEMV (General Matrix-Vector) strategy.
#
# Rationale:
# The operation is equivalent to `fc(lstm_out[:, -1, :])`, which is a batched
# matrix-vector multiplication. The key insight is to combine the best features
# of the top-performing prior solutions.
#
# 1.  **"One Row per Block" Strategy:** Each CUDA block is assigned to one item
#     in the batch. This launches 10 blocks, providing a good level of parallelism
#     to occupy the GPU, a major improvement over the single-block approach of the
#     top solution.
# 2.  **Maximal Data Reuse:** Each block loads its corresponding input vector
#     (the slice `lstm_out[b, -1, :]`) into shared memory *once*. This shared vector
#     is then reused for all 10 dot products against the rows of the weight matrix.
#     This is a significant advantage over "one element per block" strategies, which
#     suffer from redundant global memory reads of the input vector.
# 3.  **Vectorized `float4` Memory Access:** To maximize memory bandwidth (the main
#     bottleneck), all memory operations on the 512-element vectors are done using
#     `float4`. This includes the initial load into shared memory and the reads from
#     the weight matrix. This effectively quadruples the memory throughput.
# 4.  **Efficient Reduction:** A two-stage reduction using fast warp-level shuffle
#     intrinsics (`__shfl_down_sync`) followed by a shared memory reduction in the
#     first warp ensures the dot products are computed efficiently.
# 5.  **Optimized Configuration:** A block size of 128 is chosen, as it perfectly
#     matches the vectorization scheme (128 threads * 4 floats/thread = 512 elements).
fused_gemv_vectorized_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Block size is 128 threads. Since K=512, each thread handles a float4.
#define BLOCK_SIZE 128
// Inner dimension (hidden_size * 2), must be 512 for this kernel.
#define K_DIM 512

// Efficient warp-level reduction using shuffle instructions. Result in lane 0.
__inline__ __device__ float warpReduceSum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void fused_gemv_vectorized_kernel(
    const float* __restrict__ lstm_out, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ out, 
    const int B, 
    const int S, 
    const int N) {
    
    // Each block computes one row of the output matrix: out[blockIdx.x, :]
    const int row = blockIdx.x;
    
    // Shared memory for the input vector and for warp reduction partials.
    extern __shared__ float s_mem[];
    float* s_input_vec = s_mem; // Size K_DIM
    const int num_warps = BLOCK_SIZE / 32;
    float* s_partials = &s_mem[K_DIM]; // Size num_warps

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // 1. Collaboratively load the input slice into shared memory using vectorized loads.
    const float4* input_vec_global_f4 = reinterpret_cast<const float4*>(
        lstm_out + (long)row * S * K_DIM + (long)(S - 1) * K_DIM);
    if (tid < K_DIM / 4) { // tid < 128
        reinterpret_cast<float4*>(s_input_vec)[tid] = input_vec_global_f4[tid];
    }
    __syncthreads();

    // Cast shared memory pointer for vectorized access in the loop.
    const float4* s_input_vec_f4 = reinterpret_cast<const float4*>(s_input_vec);

    // 2. Loop over output columns. The entire block computes one dot product at a time.
    for (int col = 0; col < N; ++col) {
        const float4* weight_vec_global_f4 = reinterpret_cast<const float4*>(weight + (long)col * K_DIM);
        
        float partial_sum = 0.0f;
        if (tid < K_DIM / 4) {
            const float4 input_vals = s_input_vec_f4[tid];
            const float4 weight_vals = weight_vec_global_f4[tid];
            partial_sum = input_vals.x * weight_vals.x + 
                          input_vals.y * weight_vals.y + 
                          input_vals.z * weight_vals.z + 
                          input_vals.w * weight_vals.w;
        }

        // --- Stage 1: Warp-level reduction ---
        partial_sum = warpReduceSum(partial_sum);
        
        // --- Stage 2: Block-level reduction ---
        if (lane_id == 0) {
            s_partials[warp_id] = partial_sum;
        }
        __syncthreads();

        float final_sum = 0.0f;
        if (warp_id == 0) {
            final_sum = (lane_id < num_warps) ? s_partials[lane_id] : 0.0f;
            final_sum = warpReduceSum(final_sum);
        }
        
        if (tid == 0) {
            out[(long)row * N + col] = final_sum + bias[col];
        }
    }
}

torch::Tensor fused_gemv_vectorized_cuda(
    torch::Tensor lstm_out, 
    torch::Tensor weight, 
    torch::Tensor bias) {
    
    lstm_out = lstm_out.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    const int B = lstm_out.size(0);
    const int S = lstm_out.size(1);
    const int K = lstm_out.size(2);
    const int N = weight.size(0);

    TORCH_CHECK(K == K_DIM, "This kernel is specialized for K == K_DIM (512)");
    TORCH_CHECK(K % 4 == 0, "K must be a multiple of 4 for vectorized loads");

    auto out = torch::empty({B, N}, lstm_out.options());

    const dim3 grid_size(B);
    const dim3 block_size(BLOCK_SIZE);
    const size_t shared_mem_size = (K_DIM + (BLOCK_SIZE / 32)) * sizeof(float);

    fused_gemv_vectorized_kernel<<<grid_size, block_size, shared_mem_size>>>(
        lstm_out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        B, S, N
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

fused_gemv_vectorized_cpp_source = (
    "torch::Tensor fused_gemv_vectorized_cuda(torch::Tensor lstm_out, torch::Tensor weight, torch::Tensor bias);"
)

# JIT compile the CUDA kernel. A unique name is used to avoid caching issues.
fused_op = load_inline(
    name="fused_op_gemv_vectorized_v1",
    cpp_sources=fused_gemv_vectorized_cpp_source,
    cuda_sources=fused_gemv_vectorized_source,
    functions=["fused_gemv_vectorized_cuda"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        # The PyTorch LSTM is highly optimized with cuDNN and is retained.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        # The nn.Linear layer provides a convenient way to handle weights and biases.
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x, h0, c0):
        # Forward propagate LSTM.
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Replace `self.fc(lstm_out[:, -1, :])` with our optimized, fused CUDA kernel.
        out = fused_op.fused_gemv_vectorized_cuda(lstm_out, self.fc.weight, self.fc.bias)
        
        return out

# Model parameters and helper functions for generating inputs
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    # Generate random input tensors directly on the CUDA device.
    x = torch.randn(batch_size, sequence_length, input_size, device='cuda', dtype=torch.float32)
    h0 = torch.randn(num_layers * 2, batch_size, hidden_size, device='cuda', dtype=torch.float32)
    c0 = torch.randn(num_layers * 2, batch_size, hidden_size, device='cuda', dtype=torch.float32)
    return [x, h0, c0]

def get_init_inputs():
    # Return parameters required for model initialization.
    return [input_size, hidden_size, num_layers, output_size, dropout]
# EVOLVE-BLOCK-END