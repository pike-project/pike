# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for a fused bias-add and tanh operation.
# This version combines the best features of previous attempts:
# 1. float4 vectorization: Maximizes memory bandwidth, which is the bottleneck for this operation.
# 2. 2D grid/block structure: This maps naturally to the 2D tensor, simplifying indexing logic
#    and, crucially, eliminating the expensive modulo (%) operator used in 1D kernels for bias indexing.
# 3. Increased block size (512 threads): A larger block size (32x16) helps to maximize GPU occupancy.
# 4. __launch_bounds__: Provides hints to the CUDA compiler to optimize register usage for the
#    larger block size, preventing register spilling and improving performance.
fused_bias_tanh_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <c10/cuda/CUDAException.h>

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 16
#define BLOCK_SIZE (BLOCK_DIM_X * BLOCK_DIM_Y)

// Fused kernel: output = tanh(input + bias)
// Optimized with float4 vectorization and a 2D grid to avoid modulo arithmetic.
__global__ void __launch_bounds__(BLOCK_SIZE) fused_bias_tanh_vec4_2d_optimized(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int M,        // batch_size
    const int N_vec     // hidden_size / 4
) {
    // Calculate the vectorized column index and the row index from the 2D grid.
    const int col_vec = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check to ensure threads do not access out-of-bounds memory.
    if (row < M && col_vec < N_vec) {
        // Calculate the linear index for the vectorized load/store.
        const int idx_vec = row * N_vec + col_vec;
        
        // Vectorized, coalesced loads from global memory.
        const float4 in_val = reinterpret_cast<const float4*>(input)[idx_vec];
        // Bias is broadcast across the batch dimension. The 2D grid allows for a
        // direct, modulo-free lookup of the bias vector.
        const float4 bias_val = reinterpret_cast<const float4*>(bias)[col_vec];

        // Fused computation on the vector elements.
        float4 out_val;
        out_val.x = tanhf(in_val.x + bias_val.x);
        out_val.y = tanhf(in_val.y + bias_val.y);
        out_val.z = tanhf(in_val.z + bias_val.z);
        out_val.w = tanhf(in_val.w + bias_val.w);
        
        // Vectorized, coalesced store to global memory.
        reinterpret_cast<float4*>(output)[idx_vec] = out_val;
    }
}

torch::Tensor fused_bias_tanh_cuda_optimized(torch::Tensor input, torch::Tensor bias) {
    // Input validation checks for safety and correctness.
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias tensor must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    TORCH_CHECK(input.size(1) == bias.size(0), "Input dim 1 must match bias size");
    TORCH_CHECK((input.size(1) % 4) == 0, "Input feature dim must be divisible by 4 for vectorization");
    
    const auto M = input.size(0); // batch_size
    const auto N = input.size(1); // hidden_size
    auto out = torch::empty_like(input);

    // Vectorized dimension size.
    const int N_vec = N / 4;

    // Launch configuration with a 2D grid and 2D blocks of 32x16 threads.
    const dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    const dim3 blocks(
        (N_vec + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    fused_bias_tanh_vec4_2d_optimized<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        M,
        N_vec
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

fused_bias_tanh_cpp_source = """
torch::Tensor fused_bias_tanh_cuda_optimized(torch::Tensor input, torch::Tensor bias);
"""

# JIT compile the custom CUDA kernel with a unique name to avoid caching issues.
fused_op = load_inline(
    name="fused_bias_tanh_optimized_v6",
    cpp_sources=fused_bias_tanh_cpp_source,
    cuda_sources=fused_bias_tanh_source,
    functions=["fused_bias_tanh_cuda_optimized"],
    verbose=False,
)

batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256


class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Register the hidden state as a buffer. This ensures it's moved to the correct
        # device when model.to(device) is called and is included in the model's state_dict.
        self.register_buffer('hidden', torch.randn((batch_size, hidden_size)))
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        # The optimization strategy is to let PyTorch's highly optimized cuBLAS library
        # handle the compute-intensive matrix multiplication, while our custom kernel
        # efficiently fuses the subsequent memory-bound operations (bias addition and tanh activation).
        
        # Determine the input hidden state for this step.
        # self.hidden is already on the correct device because it's a registered buffer.
        h_in = self.hidden if initial_hidden is None else initial_hidden
        
        # Concatenate input and hidden state to enable a single large GEMM operation.
        combined = torch.cat((x, h_in), dim=1)
        
        # 1. Perform the matrix multiplication (GEMM) part of the linear layer (without bias).
        # This leverages PyTorch's cuBLAS backend for maximum performance.
        i2h_matmul_result = F.linear(combined, self.i2h.weight, None)
        
        # 2. Apply the fused bias-add and tanh operation using our optimized custom CUDA kernel.
        # This reduces kernel launch overhead and memory traffic by avoiding an intermediate tensor.
        # The result updates the model's persistent hidden state for the next step.
        self.hidden = fused_op.fused_bias_tanh_cuda_optimized(i2h_matmul_result, self.i2h.bias)
        
        # 3. Compute the output using the standard, optimized nn.Linear layer with the new hidden state.
        output = self.h2o(self.hidden)
        
        return output

def get_inputs():
    # Return tensors on the CUDA device to match the model and kernel expectations.
    return [torch.randn(batch_size, input_size).cuda(), torch.randn(batch_size, hidden_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, output_size]

# EVOLVE-BLOCK-END