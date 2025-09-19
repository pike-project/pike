# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fusing two additions and a tanh activation.
# This version is optimized with float4 vectorization to increase memory bandwidth,
# which is the primary bottleneck for this element-wise operation.
# It processes four float elements per thread, significantly improving throughput.
# The __restrict__ keyword is used to indicate to the compiler that the pointers
# do not alias, allowing for more aggressive optimizations.
fused_rnn_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__global__ void fused_add_bias_tanh_kernel_vec4(
    const float4* __restrict__ a, 
    const float4* __restrict__ b, 
    const float* __restrict__ bias, 
    float4* __restrict__ out, 
    int total_size_vec4, // total elements / 4
    int hidden_size
) {
    // Grid-stride loop for robustness and flexibility.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_size_vec4; 
         idx += blockDim.x * gridDim.x) 
    {
        // Load 4 floats at once from global memory.
        float4 a_val = a[idx];
        float4 b_val = b[idx];
        
        // Calculate the base index in the original float array.
        int base_float_idx = idx * 4;
        
        // The bias vector is broadcast across the batch dimension.
        // We calculate the corresponding bias index for each of the 4 elements.
        // Since hidden_size is a power of two (256), we can use a faster
        // bitwise AND instead of a more expensive modulo operation.
        const int mask = hidden_size - 1;
        int bias_idx0 = (base_float_idx + 0) & mask;
        int bias_idx1 = (base_float_idx + 1) & mask;
        int bias_idx2 = (base_float_idx + 2) & mask;
        int bias_idx3 = (base_float_idx + 3) & mask;
        
        // Load the 4 required bias values. This may cause uncoalesced reads from the bias
        // vector, but it's a small vector and likely in L1/L2 cache, so the impact is minimal
        // compared to the gains from vectorized reads/writes on the large a, b, and out tensors.
        float4 bias_val = make_float4(bias[bias_idx0], bias[bias_idx1], bias[bias_idx2], bias[bias_idx3]);

        // Perform the fused operation on all 4 elements.
        float4 result;
        result.x = tanhf(a_val.x + b_val.x + bias_val.x);
        result.y = tanhf(a_val.y + b_val.y + bias_val.y);
        result.z = tanhf(a_val.z + b_val.z + bias_val.z);
        result.w = tanhf(a_val.w + b_val.w + bias_val.w);

        // Store 4 floats at once to global memory.
        out[idx] = result;
    }
}

// C++ wrapper function to be called from Python.
torch::Tensor fused_add_bias_tanh_cuda(
    const torch::Tensor& a, 
    const torch::Tensor& b, 
    const torch::Tensor& bias
) {
    // --- Input Validation ---
    TORCH_CHECK(a.is_cuda(), "Input tensor 'a' must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "Input tensor 'b' must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input tensor 'bias' must be a CUDA tensor");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && bias.is_contiguous(), "All tensors must be contiguous");
    
    const int H = a.size(1);
    const int total_size = a.numel();
    
    // Check for vectorization compatibility.
    // Total elements must be divisible by 4 for float4 access.
    TORCH_CHECK(total_size % 4 == 0, "Total elements must be divisible by 4 for vec4 optimization");
    // The bitwise AND optimization for bias indexing requires hidden_size to be a power of two.
    TORCH_CHECK(H > 0 && (H & (H - 1)) == 0, "hidden_size must be a power of 2 for modulo optimization");
    // Pointers must be 16-byte aligned for efficient float4 access.
    TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0, "Tensor 'a' is not 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 == 0, "Tensor 'b' is not 16-byte aligned");

    // --- Kernel Launch ---
    auto out = torch::empty_like(a);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "Output tensor is not 16-byte aligned");

    const int total_size_vec4 = total_size / 4;

    // A block size of 256 is a common and safe choice that generally yields good occupancy.
    const int block_size = 256; 
    const int num_blocks = (total_size_vec4 + block_size - 1) / block_size;

    fused_add_bias_tanh_kernel_vec4<<<num_blocks, block_size>>>(
        reinterpret_cast<const float4*>(a.data_ptr<float>()), 
        reinterpret_cast<const float4*>(b.data_ptr<float>()), 
        bias.data_ptr<float>(), 
        reinterpret_cast<float4*>(out.data_ptr<float>()), 
        total_size_vec4,
        H
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return out;
}
"""

# Define the C++ function signature for the JIT compiler.
fused_rnn_elementwise_cpp_source = (
    "torch::Tensor fused_add_bias_tanh_cuda(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& bias);"
)

# JIT compile the CUDA code. Use a unique name to avoid caching issues.
fused_rnn_op = load_inline(
    name="fused_rnn_op_vec4_final",
    cpp_sources=fused_rnn_elementwise_cpp_source,
    cuda_sources=fused_rnn_elementwise_source,
    functions=["fused_add_bias_tanh_cuda"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model.
        
        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Register hidden state as a buffer. This ensures it's moved to the correct 
        # device with `.to(device)` calls but is not considered a model parameter.
        self.register_buffer('hidden', torch.randn((batch_size, hidden_size)))
        
        # Define the RNN cell components
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        """
        Forward pass of the Vanilla RNN using a custom fused and vectorized CUDA kernel.
        
        This hybrid approach leverages the best of both worlds:
        1. It avoids the expensive `torch.cat` by splitting the linear layer's computation,
           relying on PyTorch's highly optimized cuBLAS backend for the compute-intensive
           matrix multiplications (`F.linear`).
        2. It fuses the subsequent memory-bound operations (two additions, one bias add,
           and a tanh activation) into a single, custom CUDA kernel.
        3. This custom kernel is further optimized for memory bandwidth using `float4`
           vectorization, processing four elements per thread to maximize throughput.
        
        :param x: Input tensor of shape (batch_size, input_size).
        :param initial_hidden: Optional initial hidden state tensor.
        :return: Output tensor of shape (batch_size, output_size).
        """
        if initial_hidden is not None:
            self.hidden.copy_(initial_hidden, non_blocking=True)
        
        # Slicing the weight tensor is a metadata-only operation and does not incur a memory copy.
        i2h_weight_x = self.i2h.weight[:, :self.input_size]
        i2h_weight_h = self.i2h.weight[:, self.input_size:]
        
        # Perform two separate linear operations using PyTorch's optimized backend (cuBLAS).
        out_x = F.linear(x, i2h_weight_x)
        out_h = F.linear(self.hidden, i2h_weight_h)
        
        # Use the custom fused and vectorized kernel to perform: tanh(out_x + out_h + self.i2h.bias)
        self.hidden = fused_rnn_op.fused_add_bias_tanh_cuda(out_x, out_h, self.i2h.bias)
        
        # Compute the final output using the standard linear layer.
        output = self.h2o(self.hidden)
        return output

batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256

def get_inputs():
    return [torch.randn(batch_size, input_size),torch.randn(batch_size, hidden_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]

# EVOLVE-BLOCK-END