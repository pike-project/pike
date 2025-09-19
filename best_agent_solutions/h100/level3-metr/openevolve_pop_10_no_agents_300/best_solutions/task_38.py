# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# This kernel fuses the slice and linear layer, building on the best prior designs.
# Improvements:
# 1. Optimized Block Size: A block size of 256 is used. While K_vec is 128, a larger
#    block can lead to better scheduling and occupancy on the GPU.
# 2. Launch Bounds: `__launch_bounds__(256)` is added to give the compiler hints for
#    register allocation, aiming for higher occupancy.
# 3. Proven Primitives: It retains the highly efficient vectorized memory access and
#    the warp-shuffle-based block reduction from the most promising prior attempts.
fused_slice_linear_optimized_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// This reduction uses shuffle-down instructions for fast intra-warp reduction.
__inline__ __device__ float warpReduceSum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Reduces a value across all threads in a block.
__inline__ __device__ float blockReduceSum(float val) {
  // Shared memory for partial sums from each warp.
  static __shared__ float shared[32]; 
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;

  // Each warp performs its own reduction.
  val = warpReduceSum(val); 

  // The first thread of each warp writes its result to shared memory.
  if (lane == 0) {
    shared[wid] = val;
  }

  // Wait for all warps to finish writing.
  __syncthreads(); 

  // The first warp reduces the partial sums from shared memory.
  val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
  if (wid == 0) {
    val = warpReduceSum(val);
  }

  return val;
}

__global__ void __launch_bounds__(256) fused_slice_linear_optimized_kernel(
    const float* __restrict__ lstm_out,   // Shape: (M, seq_len, K)
    const float* __restrict__ weight,     // Shape: (N, K)
    const float* __restrict__ bias,       // Shape: (N)
    float* __restrict__ output,           // Shape: (M, N)
    int M, int N, int K, int seq_len
) {
    // Each block computes one element of the output matrix C[m, n]
    const int m = blockIdx.y;
    const int n = blockIdx.x;

    // Boundary check for the grid
    if (m >= M || n >= N) {
        return;
    }

    // Pointer to the start of the last time step's data for the current batch item.
    // This performs the slicing operation: lstm_out[m, seq_len - 1, :]
    const float4* input_row = reinterpret_cast<const float4*>(
        lstm_out + (long long)m * seq_len * K + (long long)(seq_len - 1) * K);
    
    // Pointer to the start of the relevant row in the non-transposed weight matrix
    const float4* weight_row = reinterpret_cast<const float4*>(
        weight + (long long)n * K);

    float thread_sum = 0.0f;
    const int K_vec = K / 4;

    // Each thread computes a partial sum over a strided slice of the K dimension.
    // With blockDim.x=256 and K_vec=128, only threads 0-127 do work here.
    // The remaining threads will correctly contribute 0.0 to the reduction.
    for (int k_vec = threadIdx.x; k_vec < K_vec; k_vec += blockDim.x) {
        const float4 in_vec = input_row[k_vec];
        const float4 wt_vec = weight_row[k_vec];
        thread_sum += in_vec.x * wt_vec.x + in_vec.y * wt_vec.y + in_vec.z * wt_vec.z + in_vec.w * wt_vec.w;
    }
    
    // Reduce sums across all threads in the block.
    float total_sum = blockReduceSum(thread_sum);

    // Thread 0 of the block writes the final result.
    if (threadIdx.x == 0) {
        output[(long long)m * N + n] = total_sum + bias[n];
    }
}

torch::Tensor fused_slice_linear_optimized_cuda(
    torch::Tensor lstm_out,
    torch::Tensor weight,
    torch::Tensor bias
) {
    // Input validation
    TORCH_CHECK(lstm_out.is_cuda(), "lstm_out must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(lstm_out.scalar_type() == torch::kFloat32, "All inputs must be float32 tensors");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "All inputs must be float32 tensors");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "All inputs must be float32 tensors");

    // Ensure tensors are contiguous for correct memory access.
    lstm_out = lstm_out.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();
    
    // Get dimensions
    const int M = lstm_out.size(0);       // batch_size
    const int seq_len = lstm_out.size(1);
    const int K = lstm_out.size(2);       // hidden_size * 2 (in_features)
    const int N = weight.size(0);         // output_size (out_features)

    TORCH_CHECK(K == weight.size(1), "Inner dimensions of lstm_out and weight must match");
    TORCH_CHECK(N == bias.size(0), "Dimensions of weight and bias must match");
    TORCH_CHECK(K % 4 == 0, "K dimension must be divisible by 4 for vectorized kernel");

    // Create the output tensor
    auto output = torch::empty({M, N}, lstm_out.options());

    // Configure and launch the kernel
    // Use a larger block size of 256. This can improve scheduling and occupancy.
    const int block_size = 256;
    // Launch a 2D grid where each block computes one output element.
    const dim3 num_blocks(N, M);
    const dim3 threads_per_block(block_size);

    fused_slice_linear_optimized_kernel<<<num_blocks, threads_per_block>>>(
        lstm_out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K, seq_len
    );

    C10_CUDA_CHECK(cudaGetLastError());

    return output;
}
"""

# C++ source for PyTorch binding
fused_slice_linear_optimized_cpp_source = """
torch::Tensor fused_slice_linear_optimized_cuda(torch::Tensor lstm_out, torch::Tensor weight, torch::Tensor bias);
"""

# JIT compile the CUDA kernel and its C++ wrapper
fused_op = load_inline(
    name="fused_slice_linear_optimized",
    cpp_sources=fused_slice_linear_optimized_cpp_source,
    cuda_sources=fused_slice_linear_optimized_source,
    functions=["fused_slice_linear_optimized_cuda"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model.
        """
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        # Keep the nn.Linear layer to manage weight and bias parameters
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x, h0, c0):
        """
        Forward pass using the custom fused CUDA kernel.
        """
        # Forward propagate LSTM
        out_lstm, _ = self.lstm(x, (h0, c0))
        
        # Replace `self.fc(out_lstm[:, -1, :])` with the custom fused kernel
        out = fused_op.fused_slice_linear_optimized_cuda(out_lstm, self.fc.weight, self.fc.bias)
        
        return out

# Model and input configuration
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    # Generate random input tensors on the correct CUDA device
    return [
        torch.randn(batch_size, sequence_length, input_size).cuda(),
        torch.randn((num_layers*2, batch_size, hidden_size)).cuda(),
        torch.randn((num_layers*2, batch_size, hidden_size)).cuda()
    ]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
# EVOLVE-BLOCK-END