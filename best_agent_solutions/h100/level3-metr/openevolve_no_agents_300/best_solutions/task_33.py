# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for a fused GEMM + bias + tanh kernel.
# This version combines several optimizations:
# 1. Virtual Concatenation: Avoids torch.cat overhead by reading from 'x' and 'hidden' directly.
# 2. float4 Vectorization: Maximizes memory bandwidth by loading/processing 4 floats at a time.
# 3. Read-Only Cache: Uses __ldg intrinsic to leverage the texture cache for read-only data.
# 4. Hybrid Reduction: Combines fast shared memory reduction with warp-level shuffles for the final step,
#    reducing synchronization overhead and improving warp utilization.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For tanhf

// Kernel to compute C = tanh(cat(x, hidden) @ W^T + bias)
// W is the weight matrix for the combined input.
// Grid: (N, M), Block: (BLOCK_SIZE, 1)
__global__ void fused_rnn_gemm_hybrid_reduction_vec4_kernel(
    const float* __restrict__ x,
    const float* __restrict__ hidden,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* C,
    const int M,      // batch_size
    const int N,      // hidden_size
    const int K_in,   // input_size
    const int K_hid   // hidden_size (again)
) {
    // Identify the output element (row, col) this block is responsible for.
    const int row = blockIdx.y;
    const int col = blockIdx.x;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Use dynamic shared memory for the reduction.
    extern __shared__ float sdata[];

    const int K_total = K_in + K_hid;
    
    // Pointers to the relevant rows.
    const float* x_row_ptr = x + row * K_in;
    const float* hidden_row_ptr = hidden + row * K_hid;
    const float* weight_row_ptr = weight + col * K_total;
    
    // Cast pointers to float4 for vectorized memory access.
    const float4* x_row_ptr_vec = reinterpret_cast<const float4*>(x_row_ptr);
    const float4* hidden_row_ptr_vec = reinterpret_cast<const float4*>(hidden_row_ptr);
    const float4* weight_row_ptr_vec = reinterpret_cast<const float4*>(weight_row_ptr);

    const int K_total_vec = K_total / 4;
    const int K_in_vec = K_in / 4;

    // Each thread computes a partial sum of the dot product using vectorized loads.
    float local_sum = 0.0f;
    for (int k_vec = tid; k_vec < K_total_vec; k_vec += block_size) {
        // Virtual concatenation: load from x or hidden based on k_vec
        // Using __ldg to explicitly fetch from the read-only cache.
        const float4 a_val_vec = (k_vec < K_in_vec) ? 
                           __ldg(&x_row_ptr_vec[k_vec]) : 
                           __ldg(&hidden_row_ptr_vec[k_vec - K_in_vec]);
        const float4 b_val_vec = __ldg(&weight_row_ptr_vec[k_vec]);

        // Dot product of the two float4 vectors.
        local_sum += a_val_vec.x * b_val_vec.x + a_val_vec.y * b_val_vec.y + a_val_vec.z * b_val_vec.z + a_val_vec.w * b_val_vec.w;
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Perform block-level reduction in shared memory down to warp size.
    for (int s = block_size / 2; s >= 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Perform final reduction within the first warp using shuffle instructions.
    if (tid < 32) {
        // Load the partial sum from shared memory into a register.
        float warp_sum = sdata[tid];
        
        // Unrolled warp reduction using __shfl_down_sync.
        // This is faster than using shared memory for the last 32 elements.
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 16);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 8);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 4);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 2);
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, 1);

        // Thread 0 of each block writes the final fused result.
        if (tid == 0) {
            C[row * N + col] = tanhf(warp_sum + bias[col]);
        }
    }
}


// C++ wrapper function that launches the CUDA kernel.
torch::Tensor rnn_cell_fused_hybrid_cuda(
    torch::Tensor x,
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor bias
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(hidden.is_cuda(), "hidden must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    const auto M = x.size(0);      // batch_size
    const auto K_in = x.size(1);   // input_size
    const auto N = hidden.size(1); // hidden_size
    const auto K_hid = N;

    // Add checks for vectorization compatibility.
    TORCH_CHECK(K_in % 4 == 0, "input_size must be divisible by 4 for float4 vectorization");
    TORCH_CHECK(K_hid % 4 == 0, "hidden_size must be divisible by 4 for float4 vectorization");

    auto hidden_out = torch::empty({M, N}, x.options());
    
    const int block_size = 256;
    const dim3 blockDim(block_size, 1, 1);
    const dim3 gridDim(N, M, 1);
    size_t shared_mem_size = block_size * sizeof(float);

    fused_rnn_gemm_hybrid_reduction_vec4_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        x.data_ptr<float>(),
        hidden.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        hidden_out.data_ptr<float>(),
        M, N, K_in, K_hid
    );
    
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return hidden_out;
}
"""

# C++ function signature for the inline compiler
cpp_source = """
torch::Tensor rnn_cell_fused_hybrid_cuda(
    torch::Tensor x,
    torch::Tensor hidden,
    torch::Tensor weight,
    torch::Tensor bias
);
"""

# JIT compile the CUDA and C++ code. Use a new name to avoid caching issues.
fused_rnn_cell = load_inline(
    name="fused_rnn_cell_hybrid_vec4",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["rnn_cell_fused_hybrid_cuda"],
    verbose=False
)

class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model with a custom high-performance fused CUDA kernel.
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Register 'hidden' as a buffer, so it's moved to the correct device with .to(device)
        self.register_buffer('hidden', torch.randn((batch_size, hidden_size)))
        
        # Create a single temporary linear layer to get correctly initialized weights
        # for the combined (input + hidden) to hidden transformation.
        _i2h = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Store its weights and bias as parameters for our fused kernel.
        self.weight_i2h = nn.Parameter(_i2h.weight.data.contiguous())
        self.bias_h = nn.Parameter(_i2h.bias.data.contiguous())
        
        # The second linear layer (hidden to output) remains a standard PyTorch module
        self.h2o = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        """
        Forward pass using the custom vectorized CUDA kernel with virtual concatenation and hybrid reduction.
        """
        # Determine the input hidden state for this forward pass.
        hidden_for_step = initial_hidden if initial_hidden is not None else self.hidden
        hidden_for_step = hidden_for_step.to(x.device)

        # Call the custom fused CUDA kernel which computes:
        # new_hidden = tanh(cat(x, hidden) @ weight_i2h.T + bias_h)
        # This is done in a single, highly optimized kernel call.
        new_hidden = fused_rnn_cell.rnn_cell_fused_hybrid_cuda(
            x,
            hidden_for_step,
            self.weight_i2h,
            self.bias_h
        )
        
        # Update the model's persistent hidden state
        self.hidden = new_hidden
        
        # Compute the final output using the standard linear layer
        output = self.h2o(self.hidden)
        return output

# Parameters from the original program description
batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128

def get_inputs():
    return [torch.randn(batch_size, input_size), torch.randn(batch_size, hidden_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]

# EVOLVE-BLOCK-END