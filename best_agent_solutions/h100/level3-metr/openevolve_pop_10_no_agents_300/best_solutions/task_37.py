# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused kernel for Slicing + Batched GEMV (x @ W.T + bias)
# This kernel replaces `self.fc(out[:, -1, :])`.
#
# Optimization Strategy:
# This kernel is a hybrid of the best features from prior attempts:
# 1. Fusion: Combines three operations (slicing, matmul, bias add) into one kernel
#    to eliminate intermediate tensors and kernel launch overhead.
# 2. Block-per-Batch Strategy: Each thread block is responsible for one item in the batch.
#    This is highly efficient for this problem's dimensions.
# 3. Shared Memory Caching: The input vector `out[b, -1, :]` is loaded into shared
#    memory once per block, drastically reducing global memory reads.
# 4. float4 Vectorization: It leverages `float4` for both loading data into shared
#    memory and for the dot-product computation. This maximizes memory bandwidth and
#    reduces the number of instructions.
# 5. Warp-Shuffle Reduction: A fast warp-shuffle instruction (`shfl_down`) is used
#    for the final reduction of the dot product, which is much faster than using
#    shared memory for reduction within a single warp.
#
fused_gemv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h> // Required for warp-level primitives

namespace cg = cooperative_groups;

__global__ void fused_gemv_warp_float4_kernel(
    const float* __restrict__ input,      // Shape: (B, S, H)
    const float* __restrict__ weight,     // Shape: (O, H)
    const float* __restrict__ bias,       // Shape: (O)
    float* __restrict__ output,           // Shape: (B, O)
    int B, int S, int H, int O) {

    // One block is launched per batch item.
    const int batch_idx = blockIdx.x;
    
    // Use a single warp per block for computation.
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    int lane_id = warp.thread_rank();

    // Shared memory to hold the input vector for the current batch item.
    // Size is H/4 float4 elements.
    extern __shared__ float4 shared_input_f4[];

    // Pointer to the start of the last hidden state for the current batch item.
    const float* input_vec = input + (long)batch_idx * S * H + (long)(S - 1) * H;
    const float4* input_vec_f4 = (const float4*)input_vec;

    // Collaboratively load the input vector into shared memory using float4.
    // Each thread loads (H / 4) / warpSize elements.
    for (int i = lane_id; i < H / 4; i += warp.size()) {
        shared_input_f4[i] = input_vec_f4[i];
    }
    // Synchronize the block to ensure shared memory is fully populated before use.
    __syncthreads();

    // The warp iterates through the output features, computing one dot product at a time.
    for (int o = 0; o < O; ++o) {
        const float4* weight_row_f4 = (const float4*)(weight + (long)o * H);
        float thread_sum = 0.0f;

        // Each thread in the warp computes a partial sum of the dot product using float4.
        #pragma unroll
        for (int k = lane_id; k < H / 4; k += warp.size()) {
            float4 in_val = shared_input_f4[k];
            float4 wt_val = weight_row_f4[k];
            thread_sum += in_val.x * wt_val.x + in_val.y * wt_val.y + in_val.z * wt_val.z + in_val.w * wt_val.w;
        }

        // Reduce the sum across the warp using shuffle-down instructions.
        for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
            thread_sum += warp.shfl_down(thread_sum, offset);
        }
        
        // Lane 0 of the warp holds the final sum and writes the result.
        if (lane_id == 0) {
            output[batch_idx * O + o] = thread_sum + bias[o];
        }
    }
}

torch::Tensor fused_gemv_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    const int B = input.size(0);
    const int S = input.size(1);
    const int H = input.size(2);
    const int O = weight.size(0);

    TORCH_CHECK(H % 4 == 0, "hidden_size must be a multiple of 4 for float4 vectorization");

    auto output = torch::empty({B, O}, input.options());

    // Launch configuration:
    // Grid: One block per batch item.
    // Block: One warp (32 threads) per block.
    const int block_size = 32;
    const int num_blocks = B;
    
    // Shared memory size needed is (H/4) * sizeof(float4) = H * sizeof(float).
    const int shared_mem_size = (H / 4) * sizeof(float4);

    fused_gemv_warp_float4_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, S, H, O
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

fused_gemv_cpp_source = (
    "torch::Tensor fused_gemv_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

# JIT compile the custom CUDA kernel. Use a unique name to avoid caching issues.
fused_gemv_op = load_inline(
    name="fused_gemv_float4_v3",
    cpp_sources=fused_gemv_cpp_source,
    cuda_sources=fused_gemv_source,
    functions=["fused_gemv_cuda"],
    verbose=False,
    # Add C++17 standard flag which is required for cooperative_groups.
    extra_cuda_cflags=["-std=c++17"] 
)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.fused_op = fused_gemv_op

    def forward(self, x, h0, c0):
        # Forward propagate LSTM, which is the most expensive operation and is left to cuDNN.
        out, state = self.lstm(x, (h0, c0))
        
        # Replace `self.fc(out[:, -1, :])` with the highly optimized custom CUDA kernel.
        # This operation is performed to match the original model's workload.
        _ = self.fused_op.fused_gemv_cuda(out, self.fc.weight, self.fc.bias)
        
        # Return the final cell state, as in the original model.
        return state[1]

# Test code parameters from the problem description
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    # Ensure input tensors are on the correct device and contiguous for the CUDA kernel.
    x = torch.randn(batch_size, sequence_length, input_size).cuda().contiguous()
    h0 = torch.randn((num_layers, batch_size, hidden_size)).cuda().contiguous()
    c0 = torch.randn((num_layers, batch_size, hidden_size)).cuda().contiguous()
    return [x, h0, c0]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]

# EVOLVE-BLOCK-END