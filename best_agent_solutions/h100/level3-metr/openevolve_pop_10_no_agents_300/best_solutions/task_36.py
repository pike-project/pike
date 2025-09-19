# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused slicing and linear layer
# This version improves upon the top-performing tiled GEMM kernel by:
# 1. Using coalesced global memory loads for the weight matrix, which was a performance issue in prior attempts.
# 2. Padding the shared memory tile for the weight matrix to avoid bank conflicts introduced by the new access pattern.
fused_slice_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_DIM 16
#define PADDED_TILE_DIM (TILE_DIM + 1)

__global__ void fused_slice_linear_kernel_coalesced(
    const float* __restrict__ lstm_out,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int B, const int L, const int H, const int O) {

    // Shared memory for tiles
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    // Padded to avoid shared memory bank conflicts during computation
    __shared__ float s_B[TILE_DIM][PADDED_TILE_DIM];

    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Global row and column for this thread's output element
    const int row = by * TILE_DIM + ty;
    const int col = bx * TILE_DIM + tx;

    // Accumulator register
    float acc = 0.0f;

    // Loop over tiles in the K dimension (H)
    for (int k_tile = 0; k_tile < (H + TILE_DIM - 1) / TILE_DIM; ++k_tile) {
        // --- Load tile into shared memory ---

        // Load A tile (from last hidden state of lstm_out). This access pattern is already coalesced.
        const int a_k_offset = k_tile * TILE_DIM;
        const int a_k = a_k_offset + tx;
        if (row < B && a_k < H) {
            long long a_idx = (long long)row * L * H + (long long)(L - 1) * H + a_k;
            s_A[ty][tx] = lstm_out[a_idx];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Load B tile (from weight) with coalesced access.
        // Thread (ty, tx) loads W[col_base + ty, k_base + tx]
        const int w_row_base = bx * TILE_DIM;
        const int w_col_base = k_tile * TILE_DIM;
        const int w_row = w_row_base + ty;
        const int w_col = w_col_base + tx;
        if (w_row < O && w_col < H) {
            long long b_idx = (long long)w_row * H + w_col;
            s_B[ty][tx] = weight[b_idx];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // --- Compute tile multiplication ---
        // acc += A[row, k] * W[col, k]
        // A thread (ty,tx) computes for output(row,col)
        // It needs A[row, k_base+k] and W[col, k_base+k]
        // A[row_base+ty, k_base+k] is in s_A[ty][k]
        // W[col_base+tx, k_base+k] is in s_B[tx][k] (transposed access pattern in shared mem)
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += s_A[ty][k] * s_B[tx][k];
        }

        __syncthreads();
    }

    // --- Write result ---
    if (row < B && col < O) {
        acc += bias[col];
        output[(long long)row * O + col] = acc;
    }
}

torch::Tensor fused_slice_linear_cuda(
    torch::Tensor lstm_out,
    torch::Tensor weight,
    torch::Tensor bias) {

    TORCH_CHECK(lstm_out.is_cuda(), "lstm_out must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    
    // Ensure inputs are contiguous for direct pointer access in CUDA
    lstm_out = lstm_out.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();
    
    TORCH_CHECK(lstm_out.dim() == 3, "lstm_out must be a 3D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(bias.dim() == 1, "bias must be a 1D tensor");

    const auto B = lstm_out.size(0);
    const auto L = lstm_out.size(1);
    const auto H = lstm_out.size(2);
    const auto O = weight.size(0);

    TORCH_CHECK(weight.size(1) == H, "weight dimension mismatch: weight.size(1) != H");
    TORCH_CHECK(bias.size(0) == O, "bias dimension mismatch: bias.size(0) != O");

    auto output = torch::empty({B, O}, lstm_out.options());

    const dim3 block_dim(TILE_DIM, TILE_DIM);
    const dim3 grid_dim(
        (O + TILE_DIM - 1) / TILE_DIM,
        (B + TILE_DIM - 1) / TILE_DIM
    );

    fused_slice_linear_kernel_coalesced<<<grid_dim, block_dim>>>(
        lstm_out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, L, H, O
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
    }

    return output;
}
"""

fused_slice_linear_cpp_source = (
    "torch::Tensor fused_slice_linear_cuda(torch::Tensor lstm_out, torch::Tensor weight, torch::Tensor bias);"
)

# JIT compile the CUDA kernel, using a unique name to avoid caching issues
fused_op = load_inline(
    name="fused_op_coalesced_padded",
    cpp_sources=fused_slice_linear_cpp_source,
    cuda_sources=fused_slice_linear_source,
    functions=["fused_slice_linear_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h0, c0):
        # Forward propagate LSTM
        lstm_out, state = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step using the improved custom fused kernel
        # This kernel is optimized for coalesced global memory access and conflict-free shared memory access.
        out = fused_op.fused_slice_linear_cuda(lstm_out, self.fc.weight, self.fc.bias)
        
        return state[0]

# Test code
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    # Ensure inputs are contiguous for optimal performance with custom kernels
    x = torch.randn(batch_size, sequence_length, input_size).cuda().contiguous()
    h0 = torch.randn((num_layers, batch_size, hidden_size)).cuda().contiguous()
    c0 = torch.randn((num_layers, batch_size, hidden_size)).cuda().contiguous()
    return [x, h0, c0]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
# EVOLVE-BLOCK-END