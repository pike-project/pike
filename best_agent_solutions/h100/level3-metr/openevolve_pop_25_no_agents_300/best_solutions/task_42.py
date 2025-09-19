# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define the custom CUDA kernel and C++ functions.
# This version combines a fully C++-orchestrated multi-layer loop with a
# highly optimized single-layer GRU implementation. It improves upon previous
# versions by adding crucial compiler flags and tuning kernel launch parameters.
custom_gru_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <utility> // For std::swap

// --- CUDA KERNEL ---
// Use CUDA intrinsics for faster activation functions, enabled by --use_fast_math
__device__ __forceinline__ float sigmoidf_dev(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Fused GRU cell kernel: computes the next hidden state and writes it out.
// This single kernel replaces multiple smaller kernels for element-wise ops,
// reducing launch overhead and improving memory locality.
__global__ void fused_gru_cell_kernel(
    const float* __restrict__ gi_t,     // Precomputed input gate values for this timestep
    const float* __restrict__ gh_t,     // Precomputed hidden gate values for this timestep
    const float* __restrict__ h_prev,   // Previous hidden state
    float* __restrict__ h_next,         // Output buffer for new hidden state
    const int B, // batch_size
    const int H  // hidden_size
) {
    const int total_size = B * H;
    // Grid-stride loop ensures all elements are processed, improving hardware utilization
    // and making the kernel robust to different launch configurations.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_size;
         idx += gridDim.x * blockDim.x) {

        const int b = idx / H; // batch index
        const int h = idx % H; // hidden index

        const int gate_offset = b * 3 * H;
        const int hidden_offset = b * H;

        // Compute gates: reset, update, and new candidate
        const float r = sigmoidf_dev(gi_t[gate_offset + h] + gh_t[gate_offset + h]);
        const float z = sigmoidf_dev(gi_t[gate_offset + H + h] + gh_t[gate_offset + H + h]);
        const float n = tanhf(gi_t[gate_offset + 2 * H + h] + r * gh_t[gate_offset + 2 * H + h]);

        // Compute next hidden state using the update gate
        h_next[hidden_offset + h] = (1.0f - z) * n + z * h_prev[hidden_offset + h];
    }
}


// --- C++ HELPERS ---
// This is the highly optimized forward pass for a *single* GRU layer (one direction).
// It is not exposed to Python but used by the top-level orchestrator.
// Key optimizations: zero-allocation time-step loop, pre-computation of input GEMM.
static std::vector<torch::Tensor> gru_layer_forward_internal(
    torch::Tensor x,
    torch::Tensor h_0,
    torch::Tensor weight_ih,
    torch::Tensor weight_hh,
    torch::Tensor bias_ih,
    torch::Tensor bias_hh)
{
    const auto seq_len = x.size(0);
    const auto batch_size = x.size(1);
    const auto hidden_size = h_0.size(1);

    // 1. Pre-compute the large, non-recurrent input-hidden projection for the entire sequence at once.
    auto gi_seq = torch::addmm(bias_ih, x.view({-1, x.size(2)}), weight_ih.t()).view({seq_len, batch_size, -1});

    // 2. Pre-allocate all tensors to avoid overhead inside the loop.
    auto outputs = torch::empty({seq_len, batch_size, hidden_size}, x.options());
    auto h_prev = h_0;
    auto gh_t = torch::empty({batch_size, 3 * hidden_size}, x.options());

    // 3. Tune kernel launch parameters for better occupancy.
    // A smaller block size and oversubscribing the grid often improves performance.
    const int block_size = 256;
    const int num_blocks = 128;

    // 4. Loop over time steps in C++ to avoid slow Python interpreter overhead.
    for (int t = 0; t < seq_len; ++t) {
        auto gi_t = gi_seq.select(0, t);
        auto h_next = outputs.select(0, t); // Zero-copy view into output tensor
        
        // Compute hidden-hidden projections in-place into the pre-allocated gh_t tensor.
        // This `_out` variant is critical for avoiding allocations in the loop.
        torch::addmm_out(gh_t, bias_hh, h_prev, weight_hh.t());
        
        fused_gru_cell_kernel<<<num_blocks, block_size>>>(
            gi_t.data_ptr<float>(), gh_t.data_ptr<float>(),
            h_prev.data_ptr<float>(), h_next.data_ptr<float>(),
            batch_size, hidden_size
        );
        h_prev = h_next; // Update h_prev to point to the slice just computed.
    }
    return {outputs, h_prev};
}


// --- TOP-LEVEL C++ ORCHESTRATOR ---
// This function, exposed to Python, orchestrates the entire multi-layer, bidirectional computation.
// This single-call approach from Python minimizes framework overhead.
std::vector<torch::Tensor> bidirectional_multilayer_gru_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    const std::vector<torch::Tensor>& params
) {
    const auto num_layers = params.size() / 8; // 8 tensors per layer (w_ih, w_hh, b_ih, b_hh for fwd/bwd)
    
    auto layer_input = x.contiguous();
    auto h0_reshaped = h0.contiguous().view({(long)num_layers, 2, x.size(1), h0.size(2)});

    std::vector<torch::Tensor> final_hidden_states;
    final_hidden_states.reserve(num_layers * 2);

    for (int i = 0; i < num_layers; ++i) {
        // --- Forward Direction ---
        auto fwd_results = gru_layer_forward_internal(
            layer_input, h0_reshaped[i][0], 
            params[i*8 + 0], params[i*8 + 1], params[i*8 + 2], params[i*8 + 3]);
        auto output_fwd = fwd_results[0];
        final_hidden_states.push_back(fwd_results[1]);
        
        // --- Backward Direction ---
        auto input_rev = torch::flip(layer_input, {0}); // Reverse sequence in C++
        auto bwd_results = gru_layer_forward_internal(
            input_rev, h0_reshaped[i][1],
            params[i*8 + 4], params[i*8 + 5], params[i*8 + 6], params[i*8 + 7]);
        auto output_bwd_rev = bwd_results[0];
        final_hidden_states.push_back(bwd_results[1]);
        
        auto output_bwd = torch::flip(output_bwd_rev, {0}); // Flip back

        // The concatenated output of this layer is the input for the next.
        layer_input = torch::cat({output_fwd, output_bwd}, 2);
    }
    
    auto h_n = torch::stack(final_hidden_states, 0);
    return {layer_input, h_n};
}
"""

custom_gru_cpp_source = """
#include <vector>
// Declaration of the top-level function to be exposed to Python
std::vector<torch::Tensor> bidirectional_multilayer_gru_forward_cuda(
    torch::Tensor x, torch::Tensor h0, const std::vector<torch::Tensor>& params);
"""

# JIT compile the custom CUDA/C++ code, using a unique name to prevent caching issues.
# CRITICAL: Added extra_cuda_cflags for performance.
custom_gru_module = load_inline(
    name="fused_gru_orchestrated_v_final_tuned",
    cpp_sources=custom_gru_cpp_source,
    cuda_sources=custom_gru_source,
    functions=["bidirectional_multilayer_gru_forward_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(Model, self).__init__()
        if batch_first:
            raise NotImplementedError("Custom GRU with batch_first=True is not implemented")
        
        # Use a standard nn.GRU layer as a convenient container for parameters.
        # Our custom implementation will use these parameters directly for compatibility.
        self.gru_ref = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
        self.num_layers = num_layers
        self.bias = bias
    
    def forward(self, x, h0):
        # Pack all parameters into a single list for the C++ backend.
        # This is fast and avoids multiple Python->C++ calls.
        params = []
        for layer in range(self.num_layers):
            # Forward direction params
            params.append(getattr(self.gru_ref, f'weight_ih_l{layer}'))
            params.append(getattr(self.gru_ref, f'weight_hh_l{layer}'))
            params.append(getattr(self.gru_ref, f'bias_ih_l{layer}'))
            params.append(getattr(self.gru_ref, f'bias_hh_l{layer}'))

            # Backward direction params
            params.append(getattr(self.gru_ref, f'weight_ih_l{layer}_reverse'))
            params.append(getattr(self.gru_ref, f'weight_hh_l{layer}_reverse'))
            params.append(getattr(self.gru_ref, f'bias_ih_l{layer}_reverse'))
            params.append(getattr(self.gru_ref, f'bias_hh_l{layer}_reverse'))

        # A single call to the fully-orchestrated C++ backend.
        # It returns {final_output_sequence, h_n}. We only need h_n for this problem.
        _, h_n = custom_gru_module.bidirectional_multilayer_gru_forward_cuda(x, h0, params)
        return h_n

# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    x = torch.randn(seq_len, batch_size, input_size, dtype=torch.float32).cuda()
    h0 = torch.randn((num_layers*2, batch_size, hidden_size), dtype=torch.float32).cuda()
    return [x, h0]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]
# EVOLVE-BLOCK-END