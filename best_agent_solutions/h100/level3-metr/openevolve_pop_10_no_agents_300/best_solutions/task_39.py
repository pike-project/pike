# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

# Set a unique name for the compilation to avoid caching issues with different versions
unique_build_name = "fused_gru_smem_pretransposed_v2"

# Combined CUDA and C++ source for a single GRU layer forward pass.
# This version merges the best optimizations: C++ loop, pre-transposed weights,
# in-place GEMMs, and a shared-memory-optimized fused kernel.
gru_layer_source = """
#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <string>

// Use CUDA's fast math intrinsic for sigmoid for better performance.
__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// Fused GRU cell kernel with shared memory optimization.
__global__ void gru_cell_kernel_smem(
    const float* __restrict__ gi,         // precomputed input gate activations (batch, 3*hidden)
    const float* __restrict__ gh,         // precomputed hidden gate activations (batch, 3*hidden)
    const float* __restrict__ h_prev,     // previous hidden state (batch, hidden)
    float* __restrict__ h_t_out,          // output hidden state for this time step (batch, hidden)
    const int batch_size,
    const int hidden_size
) {
    // Dynamically allocated shared memory for one batch item's data.
    // Total size: (3*h for gi) + (3*h for gh) + (h for h_prev) = 7*h
    extern __shared__ float s_data[];
    
    // Pointers to partitions of the shared memory for clarity.
    float* s_gi = s_data;
    float* s_gh = s_gi + 3 * hidden_size;
    float* s_h_prev = s_gh + 3 * hidden_size;

    int j = threadIdx.x; // Thread index within the block, maps to hidden_size dimension.
    int i = blockIdx.y;  // Block index, maps to batch_size dimension.

    if (i >= batch_size) {
        return;
    }

    // Each thread block processes one item in the batch.
    // Threads cooperatively load data from global memory into shared memory.
    int batch_offset_gates = i * 3 * hidden_size;
    int batch_offset_hidden = i * hidden_size;

    // Strided loop for loading, robust for any hidden_size vs. blockDim.x.
    // Each thread loads its share of gi, gh, and h_prev data.
    for (int k = j; k < hidden_size; k += blockDim.x) {
        // Load all 3 gate components for gi and gh.
        s_gi[k] = gi[batch_offset_gates + k];
        s_gi[k + hidden_size] = gi[batch_offset_gates + k + hidden_size];
        s_gi[k + 2 * hidden_size] = gi[batch_offset_gates + k + 2 * hidden_size];
        
        s_gh[k] = gh[batch_offset_gates + k];
        s_gh[k + hidden_size] = gh[batch_offset_gates + k + hidden_size];
        s_gh[k + 2 * hidden_size] = gh[batch_offset_gates + k + 2 * hidden_size];
        
        s_h_prev[k] = h_prev[batch_offset_hidden + k];
    }
    
    __syncthreads(); // Synchronize to ensure all data is loaded before proceeding.

    // Each thread computes one hidden unit.
    if (j < hidden_size) {
        // Indices into the shared memory arrays.
        int r_idx = j;
        int z_idx = j + hidden_size;
        int n_idx = j + 2 * hidden_size;

        // 1. Calculate reset (r_t) and update (z_t) gates from shared memory.
        float r_t = sigmoidf_fast(s_gi[r_idx] + s_gh[r_idx]);
        float z_t = sigmoidf_fast(s_gi[z_idx] + s_gh[z_idx]);

        // 2. Calculate new gate (n_t) from shared memory.
        float n_t = tanhf(s_gi[n_idx] + r_t * s_gh[n_idx]);

        // 3. Calculate next hidden state (h_t) and write directly to global memory output.
        h_t_out[batch_offset_hidden + j] = (1.0f - z_t) * n_t + z_t * s_h_prev[j];
    }
}

// C++ wrapper for a full GRU layer forward pass.
torch::Tensor gru_layer_forward(
    torch::Tensor x,           // (seq_len, batch, input_size)
    torch::Tensor h0,          // (batch, hidden_size)
    torch::Tensor w_ih,        // (3*hidden, input_size)
    torch::Tensor w_hh_t,      // PRE-TRANSPOSED (hidden_size, 3*hidden)
    torch::Tensor b_ih,        // Optional (3*hidden)
    torch::Tensor b_hh         // Optional (3*hidden)
) {
    const auto seq_len = x.size(0);
    const auto batch_size = x.size(1);
    const auto hidden_size = h0.size(1);
    const auto options = x.options();

    // Pre-compute input-hidden part for the whole sequence in one large, efficient GEMM.
    auto x_flat = x.view({-1, x.size(2)});
    torch::Tensor gi;
    if (b_ih.defined() && b_ih.numel() > 0) {
        gi = torch::addmm(b_ih, x_flat, w_ih.t());
    } else {
        gi = torch::mm(x_flat, w_ih.t());
    }
    gi = gi.view({seq_len, batch_size, -1});

    // Pre-allocate tensors to avoid allocation inside the loop.
    auto output_sequence = torch::empty({seq_len, batch_size, hidden_size}, options);
    auto gh = torch::empty({batch_size, 3 * hidden_size}, options);
    auto h_prev = h0;

    // Kernel launch configuration: one block per batch item.
    const int block_size_x = 256;
    const dim3 block_size(block_size_x, 1);
    const dim3 num_blocks(1, batch_size);
    
    // Calculate required shared memory size.
    const int smem_size = (7 * hidden_size) * sizeof(float);
    
    // Loop over sequence length in C++ to minimize Python overhead.
    for (long t = 0; t < seq_len; ++t) {
        auto gi_t = gi.select(0, t);
        auto h_next = output_sequence.select(0, t);

        // Compute hidden-hidden part using the pre-transposed weight and in-place GEMM.
        if (b_hh.defined() && b_hh.numel() > 0) {
            torch::addmm_out(gh, b_hh, h_prev, w_hh_t);
        } else {
            torch::mm_out(gh, h_prev, w_hh_t);
        }
        
        // Launch the fused cell kernel.
        gru_cell_kernel_smem<<<num_blocks, block_size, smem_size>>>(
            gi_t.data_ptr<float>(),
            gh.data_ptr<float>(),
            h_prev.data_ptr<float>(),
            h_next.data_ptr<float>(),
            batch_size,
            hidden_size
        );
        
        // The output of this step (h_next view) becomes the hidden state for the next step.
        // This is a pointer reassignment, not a memory copy.
        h_prev = h_next;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }
    
    return output_sequence;
}
"""

gru_layer_cpp_source = (
    "torch::Tensor gru_layer_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);"
)

# JIT compile the inline CUDA/C++ code with compiler optimizations.
fused_gru_module = load_inline(
    name=unique_build_name,
    cpp_sources=gru_layer_cpp_source,
    cuda_sources=gru_layer_source,
    functions=["gru_layer_forward"],
    verbose=False,
    extra_cflags=["-O3", "-ffast-math"],
)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(Model, self).__init__()
        
        if batch_first:
            raise NotImplementedError("batch_first=True is not supported by this custom GRU implementation.")
            
        # Instantiate original nn.GRU to manage parameters for compatibility.
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
        
        # Pre-transpose and store hidden-to-hidden weights to avoid re-computation in the forward pass.
        self.w_hh_t_list = nn.ParameterList()
        for i in range(self.gru.num_layers):
            w_hh = getattr(self.gru, f'weight_hh_l{i}')
            self.w_hh_t_list.append(nn.Parameter(w_hh.t().contiguous(), requires_grad=w_hh.requires_grad))

    def forward(self, x, h0):
        # Ensure inputs are contiguous for safe pointer access in CUDA.
        x = x.contiguous()
        h0 = h0.contiguous()
        
        current_x_sequence = x
        
        # Python loop is only over layers (a small number, not a bottleneck).
        for i in range(self.gru.num_layers):
            h_initial_layer = h0[i]
            
            # Extract weights and biases for the current layer.
            w_ih = getattr(self.gru, f'weight_ih_l{i}')
            w_hh_t = self.w_hh_t_list[i] # Use the pre-transposed weight
            b_ih = getattr(self.gru, f'bias_ih_l{i}') if self.gru.bias else torch.empty(0, device=x.device, dtype=x.dtype)
            b_hh = getattr(self.gru, f'bias_hh_l{i}') if self.gru.bias else torch.empty(0, device=x.device, dtype=x.dtype)
            
            # Call the custom C++/CUDA function for the entire layer.
            layer_output_sequence = fused_gru_module.gru_layer_forward(
                current_x_sequence,
                h_initial_layer,
                w_ih, w_hh_t, b_ih, b_hh
            )
            
            # The output sequence from this layer becomes the input for the next.
            current_x_sequence = layer_output_sequence
            
        return current_x_sequence

# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    # Generate random input tensors on the GPU and ensure they are contiguous.
    x = torch.randn(seq_len, batch_size, input_size, device='cuda').contiguous()
    h0 = torch.randn(num_layers, batch_size, hidden_size, device='cuda').contiguous()
    return [x, h0]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]
# EVOLVE-BLOCK-END