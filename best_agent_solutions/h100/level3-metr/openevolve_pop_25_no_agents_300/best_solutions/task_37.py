# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# This custom kernel fuses the LSTM output slicing `out[:, -1, :]` and the subsequent
# linear layer. It employs a "one output element per block" strategy to maximize
# parallelism, which is particularly effective for the small output dimensions
# in this model (batch_size=10, output_size=10). Previous tiled GEMM approaches
# suffered from underutilization because the output matrix (10x10) was smaller than
# a single tile (16x16), resulting in only one CUDA block being launched. This new
# approach launches 100 blocks (10x10), ensuring the GPU is fully engaged.
# Each block computes a single dot product using an efficient parallel reduction
# in shared memory.
slice_linear_fusion_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// This kernel computes one element of the output matrix C[b, o].
// C = matmul(A, W.T) + bias, where A is the sliced lstm_out.
// Each block is responsible for one (b, o) pair.
__global__ void fused_slice_linear_block_reduction_kernel(
    const float* __restrict__ lstm_out, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int B, int S, int H_in, int H_out) {

    // Block index identifies which output element to compute
    const int b = blockIdx.x; // Batch index
    const int o = blockIdx.y; // Output feature index

    // Thread index within the block
    const int tid = threadIdx.x;

    // Shared memory for the reduction
    extern __shared__ float s_data[];

    // Pointers to the input rows for the dot product
    const float* a_row = lstm_out + b * S * H_in + (S - 1) * H_in;
    const float* w_row = weight + o * H_in;

    // Each thread computes a partial product.
    // Since H_in is 256 and blockDim.x is 256, each thread computes one product.
    // A grid-stride loop is included for generality in case H_in > blockDim.x.
    float partial_sum = 0.0f;
    for (int k = tid; k < H_in; k += blockDim.x) {
        partial_sum += a_row[k] * w_row[k];
    }
    s_data[tid] = partial_sum;
    
    __syncthreads();

    // Parallel reduction in shared memory.
    // The loop is unrolled to improve instruction-level parallelism.
    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes the final result
    if (tid == 0) {
        output[b * H_out + o] = s_data[0] + bias[o];
    }
}


// C++ wrapper function to be called from Python
torch::Tensor slice_linear_fusion_cuda(
    torch::Tensor lstm_out, 
    torch::Tensor weight, 
    torch::Tensor bias) {
    
    // Input validation checks
    TORCH_CHECK(lstm_out.is_cuda(), "lstm_out must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(lstm_out.dim() == 3, "lstm_out must be a 3D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");

    const int B = lstm_out.size(0);
    const int S = lstm_out.size(1);
    const int H_in = lstm_out.size(2);
    const int H_out = weight.size(0);

    TORCH_CHECK(weight.size(1) == H_in, "weight dimension mismatch");
    TORCH_CHECK(bias.size(0) == H_out, "bias dimension mismatch");

    // Create the output tensor
    auto output = torch::empty({B, H_out}, lstm_out.options());

    // Define CUDA grid and block dimensions
    // One block per output element to maximize parallelism.
    const dim3 grid_size(B, H_out); 
    // Block size should be a power of two for efficient reduction.
    // 256 is a good choice as it matches H_in.
    const int block_size = 256; 
    const size_t shared_mem_size = block_size * sizeof(float);

    // Launch the kernel
    fused_slice_linear_block_reduction_kernel<<<grid_size, block_size, shared_mem_size>>>(
        lstm_out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, S, H_in, H_out
    );
    
    // Check for errors for robustness
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

slice_linear_fusion_cpp_source = (
    "torch::Tensor slice_linear_fusion_cuda(torch::Tensor lstm_out, torch::Tensor weight, torch::Tensor bias);"
)

# JIT compile the inline CUDA code. verbose=False for clean output.
# A unique name is used to prevent caching issues between different versions.
slice_linear_fusion = load_inline(
    name="slice_linear_fusion_block_reduce_v3",
    cpp_sources=slice_linear_fusion_cpp_source,
    cuda_sources=slice_linear_fusion_source,
    functions=["slice_linear_fusion_cuda"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        # The cuDNN-based LSTM is kept for its high performance.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        # The linear layer is kept to manage weight and bias parameters easily.
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0, c0):
        # Forward propagate LSTM
        out, state = self.lstm(x, (h0, c0))
        
        # Call the custom fused kernel, which replaces `self.fc(out[:, -1, :])`.
        # This single kernel call is more efficient than the original two-step process.
        # The result is calculated but not used, matching the original model's behavior.
        _ = slice_linear_fusion.slice_linear_fusion_cuda(out, self.fc.weight, self.fc.bias)
        
        # Return the final cell state, as in the original model.
        return state[1]

# Test code
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    return [
        torch.randn(batch_size, sequence_length, input_size).cuda(),
        torch.randn((num_layers, batch_size, hidden_size)).cuda(),
        torch.randn((num_layers, batch_size, hidden_size)).cuda()
    ]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
# EVOLVE-BLOCK-END