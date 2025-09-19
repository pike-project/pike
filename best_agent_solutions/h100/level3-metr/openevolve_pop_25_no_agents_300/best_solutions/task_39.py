# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- Custom CUDA/C++ GRU Implementation ---
#
# This solution optimizes the GRU layer by combining several high-performance strategies:
# 1.  **C++ Time-Step Loop**: The performance-critical loop over the sequence length is moved
#     from Python to C++, drastically reducing kernel launch overhead and eliminating
#     Python interpreter costs within the loop.
# 2.  **Fused Element-wise Kernel**: All element-wise operations within the GRU cell
#     (bias additions, sigmoid/tanh activations, and the final hidden state update)
#     are fused into a single CUDA kernel. This minimizes memory bandwidth usage and
#     maximizes arithmetic intensity by keeping intermediate values in registers.
# 3.  **Leverage cuBLAS**: The heavy matrix multiplications (GEMMs) are delegated to
#     PyTorch's `at::mm`, which calls the highly optimized cuBLAS library. This ensures
#     near-optimal performance for the compute-bound parts of the operation.
# 4.  **Optimized Memory Management**: The full output tensor for each layer is pre-allocated,
#     and the C++ loop writes directly into slices of this tensor. This avoids the overhead
#     of dynamic memory allocation (e.g., `torch::empty_like` in a loop) or costly
#     post-processing steps like `torch::stack`.

gru_layer_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAChecks.h>
#include <cmath>

// Inlined device function for sigmoid for maximum performance.
__device__ __forceinline__ float sigmoidf_dev(float x) {
    return 1.f / (1.f + expf(-x));
}

// Inlined device function for tanh.
__device__ __forceinline__ float tanhf_dev(float x) {
    return tanhf(x);
}

// The fused CUDA kernel for a single GRU cell's element-wise operations.
__global__ void gru_cell_kernel(
    const float* __restrict__ gates_ih, // Precomputed GEMM: x @ W_ih.t()
    const float* __restrict__ gates_hh, // Precomputed GEMM: h_prev @ W_hh.t()
    const float* __restrict__ h_prev,
    const float* __restrict__ bias_ih,
    const float* __restrict__ bias_hh,
    float* __restrict__ h_new,          // Output hidden state for this time step
    const int batch_size,
    const int hidden_size)
{
    // A 1D grid is used, where each thread computes one feature for one batch item.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * hidden_size;

    if (idx >= total_elements) {
        return;
    }

    const int batch_idx = idx / hidden_size;
    const int hidden_idx = idx % hidden_size;

    // Offsets for the reset, update, and new gates within the concatenated tensors.
    const int r_offset = hidden_idx;
    const int z_offset = hidden_idx + hidden_size;
    const int n_offset = hidden_idx + 2 * hidden_size;

    // Pointers to the start of the current batch item's data for clarity.
    const float* h_prev_ptr = h_prev + batch_idx * hidden_size;
    const float* gates_ih_ptr = gates_ih + batch_idx * 3 * hidden_size;
    const float* gates_hh_ptr = gates_hh + batch_idx * 3 * hidden_size;

    // --- Fused GRU Logic ---
    // Calculate reset gate: r_t = sigmoid(x_r + b_ir + h_r + b_hr)
    const float r_gate = sigmoidf_dev(
        gates_ih_ptr[r_offset] + bias_ih[r_offset] +
        gates_hh_ptr[r_offset] + bias_hh[r_offset]
    );

    // Calculate update gate: z_t = sigmoid(x_z + b_iz + h_z + b_hz)
    const float z_gate = sigmoidf_dev(
        gates_ih_ptr[z_offset] + bias_ih[z_offset] +
        gates_hh_ptr[z_offset] + bias_hh[z_offset]
    );

    // Calculate new gate candidate: n_t = tanh(x_n + b_in + r_t * (h_n + b_hn))
    const float n_gate = tanhf_dev(
        gates_ih_ptr[n_offset] + bias_ih[n_offset] +
        r_gate * (gates_hh_ptr[n_offset] + bias_hh[n_offset])
    );

    // Calculate final hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    h_new[idx] = (1.f - z_gate) * n_gate + z_gate * h_prev_ptr[hidden_idx];
}


// C++ wrapper function to orchestrate the forward pass for a single GRU layer.
torch::Tensor gru_layer_forward_cuda(
    torch::Tensor x,           // (seq_len, batch_size, input_size)
    torch::Tensor h0,          // (batch_size, hidden_size)
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh
) {
    // Ensure inputs are contiguous for optimal memory access patterns.
    x = x.contiguous();
    h0 = h0.contiguous();

    const auto seq_len = x.size(0);
    const auto batch_size = x.size(1);
    const auto hidden_size = h0.size(1);
    const auto input_size = x.size(2);

    // Pre-compute the input-to-hidden transformation for the entire sequence.
    // This is a single, large, and efficient matrix multiplication.
    auto x_reshaped = x.view({-1, input_size});
    auto gates_ih_all = at::mm(x_reshaped, weight_ih.t());
    gates_ih_all = gates_ih_all.view({seq_len, batch_size, 3 * hidden_size});

    // Pre-allocate the full output tensor to avoid overhead inside the loop.
    auto outputs = torch::empty({seq_len, batch_size, hidden_size}, x.options());

    auto h_prev = h0;
    
    // Configure kernel launch parameters for the 1D grid.
    const int total_elements = batch_size * hidden_size;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    // C++ loop over time steps - much faster than a Python loop.
    for (int t = 0; t < seq_len; ++t) {
        auto gates_ih_t = gates_ih_all[t];
        // Compute hidden-to-hidden transformation for the current time step.
        auto gates_hh_t = at::mm(h_prev, weight_hh.t());
        
        // Get a view into the output tensor for the current time step.
        auto h_new = outputs[t];

        // Launch the fused cell kernel to perform all element-wise ops.
        gru_cell_kernel<<<num_blocks, block_size>>>(
            gates_ih_t.data_ptr<float>(),
            gates_hh_t.data_ptr<float>(),
            h_prev.data_ptr<float>(),
            bias_ih.data_ptr<float>(),
            bias_hh.data_ptr<float>(),
            h_new.data_ptr<float>(),
            batch_size,
            hidden_size
        );
        
        // The output of this step becomes the hidden state for the next.
        h_prev = h_new;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Return the populated output tensor.
    return outputs;
}
"""

gru_layer_cpp_source = """
torch::Tensor gru_layer_forward_cuda(
    torch::Tensor x, torch::Tensor h0,
    torch::Tensor weight_ih, torch::Tensor weight_hh,
    torch::Tensor bias_ih, torch::Tensor bias_hh);
"""

# JIT compile the inline CUDA/C++ code.
try:
    custom_gru_module = load_inline(
        name="custom_gru_module_v3",
        cpp_sources=gru_layer_cpp_source,
        cuda_sources=gru_layer_source,
        functions=["gru_layer_forward_cuda"],
        verbose=False,
    )
except Exception as e:
    print(f"Skipping custom CUDA GRU kernel compilation due to: {e}")
    custom_gru_module = None


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(Model, self).__init__()
        
        # Instantiate a standard nn.GRU to borrow its perfectly initialized parameters.
        # This is a robust pattern that ensures numerical equivalence and proper handling
        # by the PyTorch ecosystem (e.g., state_dict, .to(device)).
        self.gru_ref = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
        self.custom_gru_layer = custom_gru_module
        self.num_layers = num_layers

        # Pre-extract weights and register them as parameters of this module.
        self.gru_weights = []
        if self.custom_gru_layer is not None:
            self.gru_weights_params = nn.ParameterList()
            named_params = dict(self.gru_ref.named_parameters())
            for i in range(num_layers):
                layer_weights = (
                    named_params[f"weight_ih_l{i}"],
                    named_params[f"weight_hh_l{i}"],
                    named_params[f"bias_ih_l{i}"],
                    named_params[f"bias_hh_l{i}"],
                )
                self.gru_weights.append(layer_weights)
                self.gru_weights_params.extend(layer_weights)


    def forward(self, x, h0):
        # Fallback to the native PyTorch GRU if the custom kernel failed to compile.
        if self.custom_gru_layer is None:
            output, _ = self.gru_ref(x, h0)
            return output

        # x shape: (seq_len, batch_size, input_size)
        # h0 shape: (num_layers, batch_size, hidden_size)
        
        layer_input = x

        # Python loop over layers (small loop, acceptable overhead).
        for layer_idx in range(self.num_layers):
            h_layer_0 = h0[layer_idx]
            weight_ih, weight_hh, bias_ih, bias_hh = self.gru_weights[layer_idx]
            
            # Call our custom C++/CUDA function that handles the entire sequence for one layer.
            layer_output = self.custom_gru_layer.gru_layer_forward_cuda(
                layer_input, h_layer_0, weight_ih, weight_hh, bias_ih, bias_hh
            )
            
            # The output of this layer becomes the input for the next.
            layer_input = layer_output

        return layer_input

# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    # Ensure inputs are on the correct CUDA device.
    return [torch.randn(seq_len, batch_size, input_size).cuda(), torch.randn((num_layers, batch_size, hidden_size)).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]
# EVOLVE-BLOCK-END