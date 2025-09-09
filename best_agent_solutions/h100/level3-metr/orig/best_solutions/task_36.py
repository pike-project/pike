import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math
from torch.nn import init

# Define an improved custom CUDA kernel for a tiled matrix multiplication (Linear Layer).
# This kernel is optimized to:
# 1. Handle strided inputs for `X` to fuse the slice + linear operation.
# 2. Perform coalesced reads from global memory for both `X` and `W`.
# 3. Avoid shared memory bank conflicts by loading the `W` tile in a transposed layout.
#
# It computes Y = X @ W.T + B
# X: (M, K), potentially strided. W: (N, K), contiguous. B: (N), contiguous. Y: (M, N), contiguous.
# M: batch_size, K: in_features, N: out_features
linear_forward_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 16

// Optimized CUDA kernel for tiled matrix multiplication with strided input: Y = X @ W.T + B
__global__ void strided_optimized_linear_forward_kernel(const float* X, const float* W, const float* B, float* Y, int M, int N, int K, int64_t X_stride0) {
    // Shared memory for tiles of X and W
    __shared__ float X_tile[TILE_DIM][TILE_DIM];
    __shared__ float W_tile[TILE_DIM][TILE_DIM]; // Stores a transposed tile of W

    // Thread's target output element in Y
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    // Accumulator for the output element
    float acc = 0.0f;

    // Loop over tiles along the K dimension
    for (int k_tile_idx = 0; k_tile_idx < (K + TILE_DIM - 1) / TILE_DIM; ++k_tile_idx) {
        int k_base = k_tile_idx * TILE_DIM;

        // --- Load a tile of X into shared memory (coalesced read) ---
        int x_load_row = blockIdx.y * TILE_DIM + threadIdx.y;
        int x_load_col = k_base + threadIdx.x;
        if (x_load_row < M && x_load_col < K) {
            // Use the provided stride for row access
            X_tile[threadIdx.y][threadIdx.x] = X[x_load_row * X_stride0 + x_load_col];
        } else {
            X_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // --- Load a tile of W and store it transposed in W_tile (coalesced read) ---
        int w_load_row = blockIdx.x * TILE_DIM + threadIdx.y; // N-dimension
        int w_load_col = k_base + threadIdx.x;              // K-dimension
        if (w_load_row < N && w_load_col < K) {
            // Read W[n,k] and store it at W_tile[k_local, n_local] to transpose
            W_tile[threadIdx.x][threadIdx.y] = W[w_load_row * K + w_load_col];
        } else {
            W_tile[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // --- Compute the dot product for the tiles ---
        // Y[row,col] = sum_k X[row,k] * W[col,k]
        // X[row, k_base+i] is in X_tile[threadIdx.y][i]
        // W[col, k_base+i] is in W_tile[i][threadIdx.x] due to transposed storage
        // This results in conflict-free row-wise access to W_tile.
        for (int i = 0; i < TILE_DIM; ++i) {
            acc += X_tile[threadIdx.y][i] * W_tile[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to the output tensor Y, adding the bias
    if (row < M && col < N) {
        Y[row * N + col] = acc + B[col];
    }
}

// C++ wrapper function to be called from PyTorch
torch::Tensor strided_optimized_linear_forward_cuda(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor B) {

    // Input validation
    TORCH_CHECK(X.is_cuda(), "Input X must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "Input W must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");

    TORCH_CHECK(W.is_contiguous(), "Input W must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Input B must be contiguous");

    TORCH_CHECK(X.dim() == 2, "Input X must be 2D");
    TORCH_CHECK(W.dim() == 2, "Input W must be 2D");
    TORCH_CHECK(B.dim() == 1, "Input B must be 1D");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be a float32 tensor");
    TORCH_CHECK(W.scalar_type() == torch::kFloat32, "W must be a float32 tensor");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be a float32 tensor");

    // Get dimensions
    const int M = X.size(0);
    const int K = X.size(1);
    const int N = W.size(0);

    TORCH_CHECK(W.size(1) == K, "Dimension mismatch between X and W");
    TORCH_CHECK(B.size(0) == N, "Dimension mismatch between W and B");
    
    TORCH_CHECK(X.stride(1) == 1, "The innermost dimension of X must be contiguous (stride of 1)");
    const int64_t X_stride0 = X.stride(0);

    // Create output tensor
    auto Y = torch::empty({M, N}, X.options());

    // Define grid and block dimensions
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Launch the kernel
    strided_optimized_linear_forward_kernel<<<blocks, threads>>>(
        X.data_ptr<float>(),
        W.data_ptr<float>(),
        B.data_ptr<float>(),
        Y.data_ptr<float>(),
        M, N, K,
        X_stride0
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return Y;
}
"""

linear_forward_cpp_source = """
torch::Tensor strided_optimized_linear_forward_cuda(torch::Tensor X, torch::Tensor W, torch::Tensor B);
"""

# Compile the inline CUDA code
custom_linear_op = load_inline(
    name="custom_linear_op_v2",
    cpp_sources=linear_forward_cpp_source,
    cuda_sources=linear_forward_source,
    functions=["strided_optimized_linear_forward_cuda"],
    verbose=True,
)

class CustomLinear(nn.Module):
    """
    Custom Linear layer using a specialized CUDA kernel that handles strided inputs
    and is optimized for memory access patterns.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(CustomLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        if self.bias is None:
            bias = torch.zeros(self.out_features, device=input.device, dtype=input.dtype)
        
        # Pass the input tensor directly, our kernel handles non-contiguous inputs.
        # The weight and bias must be contiguous for the kernel's assumptions.
        return custom_linear_op.strided_optimized_linear_forward_cuda(input, self.weight, bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model with a custom, highly-optimized Linear layer.
        """
        super(ModelNew, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = CustomLinear(hidden_size, output_size)
    
    def forward(self, x, h0, c0):
        """
        Forward pass through the LSTM model.
        """
        
        # Forward propagate LSTM
        out, state = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # We take the output of the last time step. This slice is not contiguous.
        # Our custom kernel handles this directly, avoiding a .contiguous() call.
        linear_input = out[:, -1, :] 
        
        # The result of this computation is not part of the final returned value,
        # but it is executed, and our optimizations will speed up this step.
        self.fc(linear_input)
        
        # Return the final hidden state as in the original model
        return state[0]