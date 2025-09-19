# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA C++ source for a fused linear layer (MatMul + Bias Addition).
# The primary optimization is fusing the GEMM (matrix multiplication) and bias addition
# into a single kernel launch, reducing overhead.
# The kernel implementation strategy uses one CUDA block to compute each element
# of the output matrix. This is efficient for cases like this one where the output
# dimensions (M, N) are small and the inner dimension (K) is large.
fused_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_kernel(
    const float* __restrict__ A, // Input tensor, shape (M, K)
    const float* __restrict__ B, // Weight tensor, shape (N, K)
    const float* __restrict__ bias, // Bias tensor, shape (N)
    float* __restrict__ C, // Output tensor, shape (M, N)
    int M, int N, int K) {

    // Each block computes one element of the output matrix C.
    // blockIdx.y corresponds to the row (M dimension), blockIdx.x to the column (N dimension).
    int row = blockIdx.y;
    int col = blockIdx.x;

    // Boundary check to prevent writing out of bounds.
    if (row >= M || col >= N) {
        return;
    }

    // Each thread computes a partial sum of the dot product.
    float sum = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        sum += A[row * K + k] * B[col * K + k];
    }

    // Use dynamically allocated shared memory for an efficient, parallel reduction.
    extern __shared__ float s_partials[];
    s_partials[threadIdx.x] = sum;
    __syncthreads();

    // Perform parallel reduction within the block.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_partials[threadIdx.x] += s_partials[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes the final sum plus bias to global memory.
    if (threadIdx.x == 0) {
        C[row * N + col] = s_partials[0] + bias[col];
    }
}

torch::Tensor fused_linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias) {

    // Basic tensor validation.
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    // Get tensor dimensions for kernel launch.
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);

    auto output = torch::zeros({M, N}, input.options());

    // Kernel launch configuration.
    const int block_size = 256; // Number of threads per block.
    dim3 threads(block_size);
    dim3 blocks(N, M); // Launch a 2D grid of blocks: one for each output element.

    size_t shared_mem_size = block_size * sizeof(float);

    // Launch the CUDA kernel.
    fused_linear_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K);
    
    // Check for any errors during kernel execution.
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

# C++ source for the function signature.
fused_linear_cpp_source = "torch::Tensor fused_linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

# JIT compile the CUDA kernel. This is done once at module load time.
# A try-except block handles cases where the CUDA toolkit is not available.
fused_op_module = None
try:
    fused_op_module = load_inline(
        name="fused_op_lstm_final", # A unique name to avoid build cache conflicts.
        cpp_sources=fused_linear_cpp_source,
        cuda_sources=fused_linear_source,
        functions=["fused_linear_forward"],
        verbose=False,
    )
except Exception:
    # If compilation fails, the module remains None, and the model will fall back
    # to the standard PyTorch implementation.
    fused_op_module = None


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        # The LSTM is the most computationally intensive part, but PyTorch's implementation
        # is already highly optimized using cuDNN. We will not replace it.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        # We define the original fc layer to hold the weight and bias parameters.
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
        # Store the compiled custom operator.
        self.fused_op = fused_op_module
    
    def forward(self, x, h0, c0):
        # Forward pass through the LSTM.
        out, hn = self.lstm(x, (h0, c0))
        
        # We only need the output of the last time step.
        last_step_out = out[:, -1, :]
        
        # Use the custom fused kernel if it was compiled successfully and inputs are on CUDA.
        if self.fused_op and last_step_out.is_cuda:
            return self.fused_op.fused_linear_forward(last_step_out, self.fc.weight, self.fc.bias)
        else:
            # Fallback to the standard PyTorch nn.Linear layer.
            return self.fc(last_step_out)

# Test code
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    # Ensure input tensors are on the GPU for the custom kernel.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return [
        torch.randn(batch_size, sequence_length, input_size, device=device),
        torch.randn((num_layers*2, batch_size, hidden_size), device=device),
        torch.randn((num_layers*2, batch_size, hidden_size), device=device)
    ]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
# EVOLVE-BLOCK-END