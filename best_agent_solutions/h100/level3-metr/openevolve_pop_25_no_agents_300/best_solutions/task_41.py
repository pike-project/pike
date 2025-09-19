# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# CUDA and C++ source code for a highly optimized custom GRU implementation.
# This version builds upon the best previous attempts with further refinements:
# 1. C++ Orchestration: The entire multi-layer, bidirectional GRU logic is handled in a single C++ function call to eliminate Python overhead.
# 2. Batched GEMMs: The input-to-hidden transformation (x @ W_ih.T) is performed as a single large GEMM for the entire sequence, which is highly efficient.
# 3. Fused CUDA Kernel: All element-wise operations within the GRU cell (gate additions, activations, hidden state update) are fused into a single CUDA kernel, minimizing kernel launch overhead and maximizing memory locality.
# 4. Optimized CUDA Intrinsics: Uses faster `__expf` and `__tanhf` device intrinsics for activation functions instead of the standard library versions.
# 5. Pre-allocation and Direct Writes: Instead of appending to std::vector and stacking, output tensors are pre-allocated, and each timestep's result is written directly into the correct slice, reducing memory management overhead in the C++ loop.
# 6. Optimized Compilation Flags: Uses -O3 and -ffast-math for aggressive compiler optimizations.

custom_gru_source = """
#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

// --- Fused GRU Cell CUDA Kernel with Optimizations ---

// Use faster CUDA intrinsics for activation functions
__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__global__ void fused_gru_cell_kernel(
    const float* __restrict__ gi, // input gate results (x @ W_i^T + b_i)
    const float* __restrict__ gh, // hidden gate results (h @ W_h^T + b_h)
    const float* __restrict__ h_in,
    float* __restrict__ h_out,
    const int total_elements, // batch_size * hidden_size
    const int hidden_size) {

    // Grid-stride loop for robustness and efficiency across different GPU architectures
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {

        const int batch_idx = idx / hidden_size;
        const int hidden_idx = idx % hidden_size;

        // Base address for this batch item's gates in the flattened tensor
        const int gate_base = batch_idx * 3 * hidden_size;

        // Calculate gates
        const float r_t = sigmoidf_fast(gi[gate_base + hidden_idx] + gh[gate_base + hidden_idx]);
        const float z_t = sigmoidf_fast(gi[gate_base + hidden_size + hidden_idx] + gh[gate_base + hidden_size + hidden_idx]);

        // Calculate candidate hidden state using the reset gate
        const float n_t = __tanhf(gi[gate_base + 2 * hidden_size + hidden_idx] + r_t * gh[gate_base + 2 * hidden_size + hidden_idx]);

        // Final hidden state update using the update gate
        h_out[idx] = (1.0f - z_t) * n_t + z_t * h_in[idx];
    }
}

// Host wrapper function to launch the CUDA kernel for the GRU cell
torch::Tensor fused_gru_cell(
    torch::Tensor gi,
    torch::Tensor gh,
    torch::Tensor h_in) {

    const auto batch_size = h_in.size(0);
    const auto hidden_size = h_in.size(1);
    const int total_elements = batch_size * hidden_size;
    auto out = torch::empty_like(h_in);

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_gru_cell_kernel<<<num_blocks, block_size>>>(
        gi.data_ptr<float>(),
        gh.data_ptr<float>(),
        h_in.data_ptr<float>(),
        out.data_ptr<float>(),
        total_elements,
        hidden_size
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return out;
}

// C++ implementation of the full multi-layer bidirectional GRU with pre-allocation
std::vector<torch::Tensor> gru_forward_cuda(
    torch::Tensor x,
    torch::Tensor h_0,
    std::vector<torch::Tensor> params,
    bool bias,
    int num_layers,
    int hidden_size
) {
    auto h_in_layers = h_0.unbind(0);
    auto current_input = x;

    std::vector<torch::Tensor> final_h_n_list;
    final_h_n_list.reserve(num_layers * 2);

    const int params_per_layer_direction = bias ? 4 : 2;
    const int total_params_per_layer = params_per_layer_direction * 2;

    for (int l = 0; l < num_layers; ++l) {
        const auto seq_len = current_input.size(0);
        const auto batch_size = current_input.size(1);
        const auto layer_input_size = current_input.size(2);
        
        // --- Forward direction ---
        const auto& w_ih_fwd = params[l * total_params_per_layer + 0];
        const auto& w_hh_fwd = params[l * total_params_per_layer + 1];
        const auto b_ih_fwd = bias ? params[l * total_params_per_layer + 2] : torch::zeros({3 * hidden_size}, x.options());
        const auto b_hh_fwd = bias ? params[l * total_params_per_layer + 3] : torch::zeros({3 * hidden_size}, x.options());

        auto gi_fwd = torch::addmm(b_ih_fwd, current_input.reshape({-1, layer_input_size}), w_ih_fwd.t()).view({seq_len, batch_size, -1});
        
        auto output_fwd = torch::empty({seq_len, batch_size, hidden_size}, current_input.options());
        auto h_t_fwd = h_in_layers[l * 2];
        for (int t = 0; t < seq_len; ++t) {
            auto gh_t_fwd = torch::addmm(b_hh_fwd, h_t_fwd, w_hh_fwd.t());
            h_t_fwd = fused_gru_cell(gi_fwd[t], gh_t_fwd, h_t_fwd);
            output_fwd[t] = h_t_fwd; // Write directly to slice
        }
        final_h_n_list.push_back(h_t_fwd);

        // --- Backward direction ---
        const auto& w_ih_bwd = params[l * total_params_per_layer + params_per_layer_direction + 0];
        const auto& w_hh_bwd = params[l * total_params_per_layer + params_per_layer_direction + 1];
        const auto b_ih_bwd = bias ? params[l * total_params_per_layer + params_per_layer_direction + 2] : torch::zeros({3 * hidden_size}, x.options());
        const auto b_hh_bwd = bias ? params[l * total_params_per_layer + params_per_layer_direction + 3] : torch::zeros({3 * hidden_size}, x.options());

        auto reversed_input = torch::flip(current_input, {0});
        auto gi_bwd = torch::addmm(b_ih_bwd, reversed_input.reshape({-1, layer_input_size}), w_ih_bwd.t()).view({seq_len, batch_size, -1});
        
        auto output_bwd_reversed = torch::empty({seq_len, batch_size, hidden_size}, current_input.options());
        auto h_t_bwd = h_in_layers[l * 2 + 1];
        for (int t = 0; t < seq_len; ++t) {
            auto gh_t_bwd = torch::addmm(b_hh_bwd, h_t_bwd, w_hh_bwd.t());
            h_t_bwd = fused_gru_cell(gi_bwd[t], gh_t_bwd, h_t_bwd);
            output_bwd_reversed[t] = h_t_bwd; // Write directly to slice
        }
        final_h_n_list.push_back(h_t_bwd);
        auto output_bwd = torch::flip(output_bwd_reversed, {0});

        current_input = torch::cat({output_fwd, output_bwd}, 2);
    }
    
    auto final_output = current_input;
    auto h_n = torch::stack(final_h_n_list, 0);

    return {final_output, h_n};
}
"""

# JIT compile the custom GRU implementation
custom_gru_cpp_source = """
std::vector<torch::Tensor> gru_forward_cuda(
    torch::Tensor x, torch::Tensor h_0, std::vector<torch::Tensor> params,
    bool bias, int num_layers, int hidden_size);
"""

try:
    custom_gru_module = load_inline(
        name="custom_gru_module_optimized_v4",
        cpp_sources=custom_gru_cpp_source,
        cuda_sources=custom_gru_source,
        functions=["gru_forward_cuda"],
        verbose=True,
        extra_cflags=["-std=c++17", "-O3", "-ffast-math"],
        extra_cuda_cflags=["-O3", "--use_fast_math"]
    )
except Exception as e:
    print("Failed to load custom CUDA kernel. Falling back to original model.")
    print(e)
    custom_gru_module = None


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        if custom_gru_module is None:
            self.use_custom_kernel = False
            self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
            return

        self.use_custom_kernel = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        
        # Create a temporary GRU to easily manage parameters with correct names and shapes
        with torch.no_grad():
            temp_gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
            self.params = nn.ParameterDict()
            for name, param in temp_gru.named_parameters():
                self.params[name.replace('.', '_')] = nn.Parameter(param.data.clone())

    def forward(self, x, h0):
        if not self.use_custom_kernel:
            output, _ = self.gru(x, h0)
            return output

        if self.batch_first and x.dim() == 3:
            x = x.transpose(0, 1)

        # Assemble parameters in the precise order expected by the C++ function
        param_list = []
        for l in range(self.num_layers):
            # Forward params
            param_list.append(self.params[f'weight_ih_l{l}'])
            param_list.append(self.params[f'weight_hh_l{l}'])
            if self.bias:
                param_list.append(self.params[f'bias_ih_l{l}'])
                param_list.append(self.params[f'bias_hh_l{l}'])
            
            # Backward params
            param_list.append(self.params[f'weight_ih_l{l}_reverse'])
            param_list.append(self.params[f'weight_hh_l{l}_reverse'])
            if self.bias:
                param_list.append(self.params[f'bias_ih_l{l}_reverse'])
                param_list.append(self.params[f'bias_hh_l{l}_reverse'])

        output_tuple = custom_gru_module.gru_forward_cuda(x, h0, param_list, self.bias, self.num_layers, self.hidden_size)
        output = output_tuple[0]
        
        if self.batch_first and output.dim() == 3:
            output = output.transpose(0, 1)

        return output

# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    # Ensure inputs are on CUDA device for the custom kernel
    return [
        torch.randn(seq_len, batch_size, input_size).cuda(),
        torch.randn((num_layers*2, batch_size, hidden_size)).cuda()
    ]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]
# EVOLVE-BLOCK-END