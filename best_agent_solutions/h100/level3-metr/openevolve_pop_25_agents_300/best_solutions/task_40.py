# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# This solution combines the best practices from the top-performing programs to create a highly optimized GRU implementation.
# The critical optimization is fixing the memory allocation flaw in the "Current Program".
# Instead of allocating new memory for the hidden-to-hidden GEMM result in every time step,
# this version pre-allocates a buffer (`gates_h_buffer`) and uses the `torch::addmm_out`
# function to perform the calculation in-place, eliminating thousands of costly memory allocations.
#
# Key Optimizations:
# 1.  **C++ Loop Orchestration**: The loops over layers and time-steps are in C++ to avoid Python interpreter overhead.
# 2.  **Batched Input GEMM**: The input-to-hidden matrix multiplication for the entire sequence
#     is computed in a single large, efficient `torch::addmm` call before the time-step loop begins.
# 3.  **Allocation-Free Time-Step Loop**: By using `torch::addmm_out`, we avoid any memory allocations
#     within the most performance-critical part of the code (the loop over the sequence length).
# 4.  **Fused Element-wise CUDA Kernel**: All element-wise operations (sigmoids, tanh, additions, multiplications)
#     of the GRU cell are fused into a single CUDA kernel, reducing kernel launch overhead and improving data locality.
# 5.  **Fast Math**: The CUDA kernel uses the `__expf` intrinsic, and the code is compiled with the
#     `--use_fast_math` flag, which enables faster, slightly less precise mathematical operations,
#     often leading to a performance gain.
# 6.  **Grid-Stride Loop**: The CUDA kernel employs a grid-stride loop, a robust pattern that ensures
#     high GPU utilization across a wide range of input sizes.

gru_source = """
#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h> // For C10_CUDA_KERNEL_LAUNCH_CHECK

// Use the fast, lower-precision __expf intrinsic for the sigmoid function.
__device__ __forceinline__ float sigmoidf(float x) {
    return 1.f / (1.f + __expf(-x));
}

// Fused GRU cell kernel with a grid-stride loop for robustness and performance.
__global__ void gru_cell_fused_kernel(
    const float* __restrict__ gates_x,      // Precomputed (x @ W_ih.T + b_ih) for current time step
    const float* __restrict__ gates_h,      // Precomputed (h_prev @ W_hh.T + b_hh) for current time step
    const float* __restrict__ h_prev,       // Previous hidden state
    float* __restrict__ h_new,              // Output new hidden state
    const int batch_size,
    const int hidden_size
) {
    const int num_elements = batch_size * hidden_size;
    // Grid-stride loop ensures all elements are processed regardless of grid size.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elements;
         i += gridDim.x * blockDim.x) {

        const int b = i / hidden_size;
        const int h = i % hidden_size;

        const int hidden_3_offset = b * 3 * hidden_size;
        
        // Gate indices for the current hidden feature 'h'
        const int r_idx = h;
        const int z_idx = h + hidden_size;
        const int n_idx = h + 2 * hidden_size;
        
        // --- Fused GRU Logic ---
        // r = sigmoid(gates_x_r + gates_h_r)
        float r = sigmoidf(gates_x[hidden_3_offset + r_idx] + gates_h[hidden_3_offset + r_idx]);
        
        // z = sigmoid(gates_x_z + gates_h_z)
        float z = sigmoidf(gates_x[hidden_3_offset + z_idx] + gates_h[hidden_3_offset + z_idx]);
        
        // n = tanh(gates_x_n + r * gates_h_n)
        float n = tanhf(gates_x[hidden_3_offset + n_idx] + r * gates_h[hidden_3_offset + n_idx]);
        
        // h' = (1 - z) * n + z * h_prev
        h_new[i] = (1.f - z) * n + z * h_prev[i];
    }
}

// C++ function that orchestrates the GRU forward pass, implementing all key optimizations.
torch::Tensor gru_forward_fast(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const std::vector<torch::Tensor>& params
) {
    const int num_layers = hx.size(0);
    const int seq_len = input.size(0);
    const int batch_size = input.size(1);
    const int hidden_size = hx.size(2);
    
    auto current_input = input;
    auto final_h_n = torch::empty_like(hx);

    // Pre-allocate buffers to avoid allocation inside the critical time-step loop.
    auto gates_h_buffer = torch::empty({batch_size, 3 * hidden_size}, input.options());
    auto temp_h_buffer = torch::empty({batch_size, hidden_size}, input.options());

    for (int l = 0; l < num_layers; ++l) {
        // Extract parameters for the current layer
        const auto& w_ih = params[l * 4 + 0];
        const auto& w_hh = params[l * 4 + 1];
        const auto& b_ih = params[l * 4 + 2];
        const auto& b_hh = params[l * 4 + 3];

        // Pre-compute input-gate matmuls for the entire sequence in one large, efficient GEMM call.
        auto gates_x = torch::addmm(b_ih, current_input.flatten(0, 1), w_ih.t()).view({seq_len, batch_size, 3 * hidden_size});
        
        // The output of this layer becomes the input for the next. Allocate it once.
        auto layer_output = (l < num_layers - 1)
            ? torch::empty({seq_len, batch_size, hidden_size}, input.options())
            : torch::Tensor();

        auto h_t = hx.select(0, l).contiguous();

        for (int t = 0; t < seq_len; ++t) {
            auto gates_x_t = gates_x.select(0, t);
            
            // CRITICAL OPTIMIZATION: Use an _out variant to write result into a pre-allocated buffer.
            torch::addmm_out(gates_h_buffer, b_hh, h_t, w_hh.t());

            // Write kernel output directly into the destination tensor (either the next layer's input or a temp buffer).
            auto h_new = (l < num_layers - 1) ? layer_output.select(0, t) : temp_h_buffer;
            
            const int block_size = 256;
            const int num_elements = batch_size * hidden_size;
            const int num_blocks = (num_elements + block_size - 1) / block_size;
            
            gru_cell_fused_kernel<<<num_blocks, block_size>>>(
                gates_x_t.data_ptr<float>(),
                gates_h_buffer.data_ptr<float>(),
                h_t.data_ptr<float>(),
                h_new.data_ptr<float>(),
                batch_size,
                hidden_size
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            h_t = h_new; // The new hidden state is the input for the next time step.
        }
        
        final_h_n.select(0, l).copy_(h_t); // Save the final hidden state of the layer.

        if (l < num_layers - 1) {
            current_input = layer_output;
        }
    }
    
    return final_h_n;
}
"""

gru_cpp_source = """
torch::Tensor gru_forward_fast(const torch::Tensor& input, const torch::Tensor& hx, const std::vector<torch::Tensor>& params);
"""

# JIT compile the CUDA code
custom_gru_impl = load_inline(
    name="custom_gru_impl_v5_fastmath", # Use a unique name to avoid caching issues
    cpp_sources=gru_cpp_source,
    cuda_sources=gru_source,
    functions=["gru_forward_fast"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--use_fast_math"] # Enable faster math intrinsics
)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(Model, self).__init__()
        
        if batch_first:
            raise NotImplementedError("batch_first=True is not supported by this custom kernel.")
        if not bias:
            raise NotImplementedError("bias=False is not supported by this custom kernel.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialize parameters from a standard nn.GRU to ensure correctness and allow
        # for easy loading of pretrained weights.
        temp_gru = nn.GRU(input_size, hidden_size, num_layers, bias=True, batch_first=False)

        self.all_params = nn.ParameterList()
        for i in range(num_layers):
            # The order must match what the C++ kernel expects: [w_ih, w_hh, b_ih, b_hh]
            self.all_params.append(nn.Parameter(getattr(temp_gru, f'weight_ih_l{i}').data.clone()))
            self.all_params.append(nn.Parameter(getattr(temp_gru, f'weight_hh_l{i}').data.clone()))
            self.all_params.append(nn.Parameter(getattr(temp_gru, f'bias_ih_l{i}').data.clone()))
            self.all_params.append(nn.Parameter(getattr(temp_gru, f'bias_hh_l{i}').data.clone()))
    
    def forward(self, x, h0):
        params_as_list = [p for p in self.all_params]

        # Call the custom CUDA GRU forward implementation.
        # Ensure inputs are contiguous, a common requirement for custom kernels.
        h_n = custom_gru_impl.gru_forward_fast(x.contiguous(), h0.contiguous(), params_as_list)
        return h_n

# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    # Ensure inputs are on CUDA and contiguous for the custom kernel.
    x = torch.randn(seq_len, batch_size, input_size).cuda().contiguous()
    h0 = torch.randn(num_layers, batch_size, hidden_size).cuda().contiguous()
    return [x, h0]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]
# EVOLVE-BLOCK-END