# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# This custom CUDA kernel fuses two operations into one for better performance:
# 1. Slicing the LSTM output to get the last time step: `out[:, -1, :]`
# 2. Applying the fully connected (linear) layer: `fc(...)`
#
# Key Optimizations:
# - Operator Fusion: Reduces kernel launch overhead and avoids writing an
#   intermediate tensor (the slice) to global memory.
# - Stride-Awareness: The kernel accepts tensor strides as arguments, allowing it
#   to operate directly on the potentially non-contiguous output of the LSTM layer
#   without requiring an expensive `.contiguous()` memory copy.
# - Parallel Reduction: Implements an efficient dot product using a hierarchical
#   reduction strategy. It uses ultra-fast warp-level shuffle instructions for
#   intra-warp reduction and shared memory for inter-warp reduction. This fully
#   parallelizes the most compute-intensive part of the operation.
fused_slice_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// In-warp reduction using shuffle instructions. This is highly efficient as it avoids
// shared memory and synchronizes implicitly within a warp.
__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__global__ void slice_linear_stride_aware_kernel(
    const float* __restrict__ out_lstm,
    const float* __restrict__ fc_weight,
    const float* __restrict__ fc_bias,
    float* __restrict__ final_out,
    const int B, const int S, const int H2, const int O,
    const long stride_b, // Stride for batch dimension
    const long stride_s  // Stride for sequence dimension
) {

    // Each block computes one output element: final_out[b, o]
    const int b = blockIdx.y;
    const int o = blockIdx.x;

    if (b >= B || o >= O) return;

    // Calculate a pointer to the start of the relevant input row using strides.
    // This performs the `out_lstm[b, S-1, :]` slice without a memory copy.
    const float* a_row = out_lstm + b * stride_b + (S - 1) * stride_s;
    const float* w_row = fc_weight + o * H2;

    // Each thread in the block computes a partial sum of the dot product.
    float my_sum = 0.0f;
    for (int k = threadIdx.x; k < H2; k += blockDim.x) {
        // The innermost dimension (k) is assumed to be contiguous (stride=1).
        my_sum += a_row[k] * w_row[k];
    }

    // --- Efficient Hierarchical Reduction within the Block ---
    // Stage 1: Reduce sums within each warp.
    float warp_sum = warpReduceSum(my_sum);

    // Shared memory for inter-warp reduction.
    extern __shared__ float sdata[];

    // Stage 2: The first thread of each warp writes its result to shared memory.
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) {
        sdata[warp_id] = warp_sum;
    }
    __syncthreads();

    // Stage 3: The first warp reduces the partial sums from all warps.
    if (warp_id == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float final_sum = (lane_id < num_warps) ? sdata[lane_id] : 0.0f;

        // Final reduction across the first warp's threads.
        final_sum = warpReduceSum(final_sum);

        // The very first thread (lane_id == 0) writes the final result with bias.
        if (lane_id == 0) {
            final_out[b * O + o] = final_sum + fc_bias[o];
        }
    }
}

torch::Tensor fused_slice_linear_cuda(
    torch::Tensor out_lstm,
    torch::Tensor fc_weight,
    torch::Tensor fc_bias) {

    TORCH_CHECK(out_lstm.is_cuda(), "out_lstm must be a CUDA tensor");
    TORCH_CHECK(fc_weight.is_cuda(), "fc_weight must be a CUDA tensor");
    TORCH_CHECK(fc_bias.is_cuda(), "fc_bias must be a CUDA tensor");
    TORCH_CHECK(out_lstm.stride(2) == 1, "The innermost dimension of out_lstm must have stride 1");

    // We don't require out_lstm to be contiguous because the kernel is stride-aware.
    // Weight and bias are small and accessed repeatedly, so contiguity is good practice.
    fc_weight = fc_weight.contiguous();
    fc_bias = fc_bias.contiguous();

    // Get dimensions
    const auto B = out_lstm.size(0);
    const auto S = out_lstm.size(1);
    const auto H2 = out_lstm.size(2);
    const auto O = fc_weight.size(0);

    // Get strides for non-contiguous access
    const long stride_b = out_lstm.stride(0);
    const long stride_s = out_lstm.stride(1);

    auto final_out = torch::empty({B, O}, out_lstm.options());

    // --- Kernel Launch Configuration ---
    // Each block calculates one element of the output matrix.
    dim3 grid_size(O, B);

    // Use a block size that matches the inner dimension of the matmul (H2) for
    // efficient reduction. In this problem H2 = hidden_size * 2 = 512.
    int block_dim_x = 512;
    dim3 block_size(block_dim_x);

    // Calculate shared memory size needed for inter-warp reduction.
    // The `warpSize` variable is a device-only variable and cannot be used in host code.
    // We use its known constant value (32) for host-side calculations.
    const int host_warp_size = 32;
    int num_warps = (block_dim_x + host_warp_size - 1) / host_warp_size;
    int shared_mem_size = num_warps * sizeof(float);

    slice_linear_stride_aware_kernel<<<grid_size, block_size, shared_mem_size>>>(
        out_lstm.data_ptr<float>(),
        fc_weight.data_ptr<float>(),
        fc_bias.data_ptr<float>(),
        final_out.data_ptr<float>(),
        B, S, H2, O,
        stride_b, stride_s
    );
    
    // Check for any CUDA errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA Error in fused_slice_linear: ") + cudaGetErrorString(err));
    }

    return final_out;
}
"""

fused_slice_linear_cpp_source = """
torch::Tensor fused_slice_linear_cuda(torch::Tensor out_lstm, torch::Tensor fc_weight, torch::Tensor fc_bias);
"""

# JIT compile the CUDA kernel
fused_slice_linear = load_inline(
    name="fused_slice_linear_stride_aware",
    cpp_sources=fused_slice_linear_cpp_source,
    cuda_sources=fused_slice_linear_source,
    functions=["fused_slice_linear_cuda"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        # Replace nn.Linear with manually created parameters. This is cleaner as our custom
        # kernel replaces the nn.Linear forward pass, making the module object redundant.
        fc_in_features = hidden_size * 2
        self.fc_weight = nn.Parameter(torch.empty(output_size, fc_in_features))
        self.fc_bias = nn.Parameter(torch.empty(output_size))
        
        # Initialize parameters with the same method as nn.Linear for correctness.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # This initialization mimics the default behavior of nn.Linear, ensuring
        # that our model is mathematically equivalent to the baseline.
        nn.init.kaiming_uniform_(self.fc_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.fc_bias, -bound, bound)
    
    def forward(self, x, h0, c0):
        # Forward propagate through the highly optimized cuDNN-backed LSTM.
        out, hn = self.lstm(x, (h0, c0))
        
        # Call our custom fused CUDA kernel. It handles both the slice `out[:, -1, :]`
        # and the linear layer `fc(...)` in a single, efficient, stride-aware operation.
        out = fused_slice_linear.fused_slice_linear_cuda(out, self.fc_weight, self.fc_bias)
        
        return out

# Test code
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    return [torch.randn(batch_size, sequence_length, input_size),torch.randn((num_layers*2, batch_size, hidden_size)),torch.randn((num_layers*2, batch_size, hidden_size))]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
# EVOLVE-BLOCK-END