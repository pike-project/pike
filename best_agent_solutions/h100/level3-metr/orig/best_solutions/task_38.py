import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for a high-performance, tiled, fused GEMM and bias addition.
# This kernel computes C = A @ B.T + bias, using shared memory to improve performance by reducing global memory traffic.
# A is the input tensor of shape [M, K]
# B is the weight tensor of shape [N, K]
# bias is the bias tensor of shape [N]
# C is the output tensor of shape [M, N]
tiled_gemm_bias_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Define the dimension of the tiles (e.g., 16x16). This must match in the C++ wrapper.
#define TILE_DIM 16

// High-performance tiled GEMM kernel using shared memory.
// It computes C = A @ B.T + bias.
// Each block computes one TILE_DIM x TILE_DIM tile of the output matrix C.
__global__ void tiled_gemm_bias_kernel(const float* A, const float* B, const float* bias, float* C, int M, int N, int K) {
    // Shared memory for tiles of A and B
    __shared__ float A_tile[TILE_DIM][TILE_DIM];
    __shared__ float B_tile[TILE_DIM][TILE_DIM];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Calculate the global row and column index for the thread's output element in C
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    // Accumulator for the result of a single element in C
    float C_value = 0.0f;

    // Loop over the tiles of A and B in the K dimension
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // --- Load a tile of A from global memory to shared memory ---
        // Calculate the global indices for the element of A to be loaded by this thread
        int A_load_row = by * TILE_DIM + ty;
        int A_load_col = t * TILE_DIM + tx;
        
        // Boundary check and load
        if (A_load_row < M && A_load_col < K) {
            A_tile[ty][tx] = A[A_load_row * K + A_load_col];
        } else {
            A_tile[ty][tx] = 0.0f;
        }

        // --- Load a tile of B from global memory to shared memory ---
        // We need B[col, k], so B_tile corresponds to a patch of B starting at [bx*TILE_DIM, t*TILE_DIM]
        // Calculate the global indices for the element of B to be loaded by this thread
        int B_load_row = bx * TILE_DIM + ty; // This corresponds to the N-dimension
        int B_load_col = t * TILE_DIM + tx; // This corresponds to the K-dimension
        
        // Boundary check and load
        if (B_load_row < N && B_load_col < K) {
            B_tile[ty][tx] = B[B_load_row * K + B_load_col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }

        // Synchronize to ensure all data is loaded into shared memory before computation
        __syncthreads();

        // --- Multiply the tiles from shared memory ---
        // Each thread accumulates its part of the dot product by iterating through the tile's inner dimension.
        for (int k = 0; k < TILE_DIM; ++k) {
            // A_tile[ty][k] corresponds to A[row, k_base + k]
            // B_tile[tx][k] corresponds to B[col, k_base + k]
            // This multiplication pattern correctly computes A @ B.T for the tile.
            C_value += A_tile[ty][k] * B_tile[tx][k];
        }

        // Synchronize before loading the next tile to avoid race conditions
        __syncthreads();
    }

    // --- Write the final result to the output matrix C ---
    // Boundary check and write, adding the bias
    if (row < M && col < N) {
        C[row * N + col] = C_value + bias[col];
    }
}

// C++ wrapper function that will be called from Python.
// It sets up the kernel launch configuration and calls the CUDA kernel.
torch::Tensor tiled_gemm_bias_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    // Input validation checks
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Weight tensor B must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "Input tensor A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Weight tensor B must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias tensor must be contiguous");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "All tensors must be of type float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "All tensors must be of type float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "All tensors must be of type float32");

    // Get tensor dimensions
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);

    // Dimension consistency checks
    TORCH_CHECK(A.dim() == 2, "Input tensor A must be 2D");
    TORCH_CHECK(B.dim() == 2, "Weight tensor B must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias tensor must be 1D");
    TORCH_CHECK(B.size(1) == K, "Inner dimensions of A and B must match (A.shape[1] == B.shape[1])");
    TORCH_CHECK(bias.size(0) == N, "Bias shape must match outer dimension of B (bias.shape[0] == B.shape[0])");

    // Create the output tensor C
    auto C = torch::empty({M, N}, A.options());

    // Define CUDA grid and block dimensions based on TILE_DIM
    const int TILE_DIM_CONST = 16;
    dim3 threads(TILE_DIM_CONST, TILE_DIM_CONST);
    dim3 blocks((N + TILE_DIM_CONST - 1) / TILE_DIM_CONST, (M + TILE_DIM_CONST - 1) / TILE_DIM_CONST);

    // Launch the CUDA kernel
    tiled_gemm_bias_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    // Check for errors during kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}
"""

tiled_gemm_bias_cpp_source = "torch::Tensor tiled_gemm_bias_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor bias);"

# Compile the inline CUDA code
tiled_gemm_bias_op = load_inline(
    name="tiled_gemm_bias_op",
    cpp_sources=tiled_gemm_bias_cpp_source,
    cuda_sources=tiled_gemm_bias_source,
    functions=["tiled_gemm_bias_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model with a custom high-performance GEMM kernel for the final linear layer.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        """
        super(ModelNew, self).__init__()
        # The LSTM layer remains unchanged as its optimization is highly complex.
        # We focus on significantly speeding up the final fully-connected layer.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        # Replace nn.Linear with our custom high-performance tiled GEMM op
        self.custom_gemm_bias = tiled_gemm_bias_op.tiled_gemm_bias_cuda
        
        # Create parameters for the linear layer that our custom kernel will use.
        # The weight shape is (output_size, hidden_size * 2), which corresponds to (N, K) in our kernel.
        self.fc_weight = nn.Parameter(torch.empty(output_size, hidden_size * 2))
        self.fc_bias = nn.Parameter(torch.empty(output_size))
        
        # Initialize the parameters using standard PyTorch linear layer initialization
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the weight and bias parameters to match the default behavior of nn.Linear.
        """
        nn.init.kaiming_uniform_(self.fc_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.fc_bias, -bound, bound)

    def forward(self, x, h0, c0):
        """
        Forward pass through the LSTM model using the custom high-performance GEMM kernel.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :param h0: The initial hidden state
        :param c0: The initial cell state
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size * 2)
        out, _ = self.lstm(x, (h0, c0))
        
        # Select the output of the last time step
        # lstm_out_last: tensor of shape (batch_size, hidden_size * 2)
        lstm_out_last = out[:, -1, :]
        
        # Use our custom tiled and fused GEMM+bias kernel instead of the standard nn.Linear layer.
        # The kernel computes: lstm_out_last @ self.fc_weight.T + self.fc_bias
        # This operation is memory-bound, and using shared memory tiling significantly reduces
        # the number of slow global memory accesses, leading to a substantial speedup over a naive kernel.
        fc_out = self.custom_gemm_bias(lstm_out_last, self.fc_weight, self.fc_bias)
        
        return fc_out