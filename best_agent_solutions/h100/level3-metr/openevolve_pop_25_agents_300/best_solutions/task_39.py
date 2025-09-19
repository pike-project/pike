# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This implementation enhances the "Current Program" by introducing float4 vectorization
# into the fused GRU cell kernel. This technique, inspired by prior successful attempts,
# is crucial for optimizing memory-bound operations. Each CUDA thread now processes
# four hidden units simultaneously, drastically improving memory bandwidth utilization
# by ensuring coalesced memory access. This is combined with the already effective
# strategies of C++-based looping and batched input GEMMs.

cuda_source = """
#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// Helper device functions to apply activations to float4 vectors component-wise.
__device__ __forceinline__ float4 sigmoidf4(float4 v) {
    v.x = 1.0f / (1.0f + expf(-v.x));
    v.y = 1.0f / (1.0f + expf(-v.y));
    v.z = 1.0f / (1.0f + expf(-v.z));
    v.w = 1.0f / (1.0f + expf(-v.w));
    return v;
}

__device__ __forceinline__ float4 tanhf4(float4 v) {
    v.x = tanhf(v.x);
    v.y = tanhf(v.y);
    v.z = tanhf(v.z);
    v.w = tanhf(v.w);
    return v;
}

// Fused GRU cell kernel optimized with float4 vectorization.
// Each thread processes 4 elements, improving memory coalescing.
__global__ void fused_gru_cell_kernel_vectorized(
    const float* __restrict__ gi,       // Precomputed input gates: x_t @ W_ih.T + b_ih
    const float* __restrict__ gh,       // Precomputed hidden gates: h_{t-1} @ W_hh.T + b_hh
    const float* __restrict__ h_prev,   // Hidden state from previous time step
    float* __restrict__ h_next,         // Output tensor for the new hidden state
    const int batch_size,
    const int hidden_size)
{
    const int hidden_size_vec = hidden_size / 4;
    const int total_work_vec = batch_size * hidden_size_vec;

    // Grid-stride loop over vectorized elements
    for (int i_vec = blockIdx.x * blockDim.x + threadIdx.x;
         i_vec < total_work_vec;
         i_vec += blockDim.x * gridDim.x) {

        const int batch_idx = i_vec / hidden_size_vec;
        const int hidden_idx_vec = i_vec % hidden_size_vec;

        // Reinterpret float pointers as float4 pointers for vectorized memory access
        const float4* gi_vec = reinterpret_cast<const float4*>(gi);
        const float4* gh_vec = reinterpret_cast<const float4*>(gh);
        const float4* h_prev_vec = reinterpret_cast<const float4*>(h_prev);
        float4* h_next_vec = reinterpret_cast<float4*>(h_next);

        // Calculate vectorized offsets for the current batch item
        const int gates_batch_stride_vec = batch_idx * 3 * hidden_size_vec;
        const int hidden_batch_stride_vec = batch_idx * hidden_size_vec;

        // Vectorized loads: load 4 gate values at once
        const float4 gi_r = gi_vec[gates_batch_stride_vec + hidden_idx_vec];
        const float4 gi_z = gi_vec[gates_batch_stride_vec + hidden_size_vec + hidden_idx_vec];
        const float4 gi_n = gi_vec[gates_batch_stride_vec + 2 * hidden_size_vec + hidden_idx_vec];

        const float4 gh_r = gh_vec[gates_batch_stride_vec + hidden_idx_vec];
        const float4 gh_z = gh_vec[gates_batch_stride_vec + hidden_size_vec + hidden_idx_vec];
        const float4 gh_n = gh_vec[gates_batch_stride_vec + 2 * hidden_size_vec + hidden_idx_vec];
        
        const float4 h_prev_val = h_prev_vec[hidden_batch_stride_vec + hidden_idx_vec];

        // Perform calculations component-wise on float4 vectors
        float4 r_in, z_in;
        r_in.x = gi_r.x + gh_r.x; r_in.y = gi_r.y + gh_r.y; r_in.z = gi_r.z + gh_r.z; r_in.w = gi_r.w + gh_r.w;
        z_in.x = gi_z.x + gh_z.x; z_in.y = gi_z.y + gh_z.y; z_in.z = gi_z.z + gh_z.z; z_in.w = gi_z.w + gh_z.w;

        const float4 r_val = sigmoidf4(r_in);
        const float4 z_val = sigmoidf4(z_in);

        float4 n_in;
        n_in.x = gi_n.x + r_val.x * gh_n.x; n_in.y = gi_n.y + r_val.y * gh_n.y;
        n_in.z = gi_n.z + r_val.z * gh_n.z; n_in.w = gi_n.w + r_val.w * gh_n.w;

        const float4 n_val = tanhf4(n_in);

        // Final hidden state calculation, component-wise
        float4 h_next_val;
        h_next_val.x = (1.0f - z_val.x) * n_val.x + z_val.x * h_prev_val.x;
        h_next_val.y = (1.0f - z_val.y) * n_val.y + z_val.y * h_prev_val.y;
        h_next_val.z = (1.0f - z_val.z) * n_val.z + z_val.z * h_prev_val.z;
        h_next_val.w = (1.0f - z_val.w) * n_val.w + z_val.w * h_prev_val.w;

        // Vectorized store: write 4 hidden state values at once
        h_next_vec[hidden_batch_stride_vec + hidden_idx_vec] = h_next_val;
    }
}


// C++ wrapper function to orchestrate the GRU forward pass.
torch::Tensor gru_forward(
    torch::Tensor x, torch::Tensor h_0,
    const std::vector<torch::Tensor>& weights_ih,
    const std::vector<torch::Tensor>& weights_hh,
    const std::vector<torch::Tensor>& biases_ih,
    const std::vector<torch::Tensor>& biases_hh)
{
    const int num_layers = weights_ih.size();
    const int seq_len = x.size(0);
    const int batch_size = x.size(1);
    const int hidden_size = h_0.size(2);

    TORCH_CHECK(hidden_size % 4 == 0, "hidden_size must be divisible by 4 for vectorized kernel");

    torch::Tensor current_input = x;
    auto final_output = torch::empty({seq_len, batch_size, hidden_size}, x.options());

    for (int layer = 0; layer < num_layers; ++layer) {
        auto layer_output = (layer == num_layers - 1) ? 
            final_output : 
            torch::empty({seq_len, batch_size, hidden_size}, x.options());

        auto h_prev = h_0[layer];

        // Pre-compute the input-to-hidden projections for the entire sequence.
        auto gi_full = at::linear(current_input.view({-1, current_input.size(-1)}), weights_ih[layer], biases_ih[layer]).view({seq_len, batch_size, -1});

        // Loop over the sequence length
        for (int t = 0; t < seq_len; ++t) {
            auto gi_t = gi_full[t];
            auto gh_t = at::linear(h_prev, weights_hh[layer], biases_hh[layer]);
            
            auto h_next = layer_output[t];

            // Launch configuration for vectorized kernel
            const int total_work_vec = (batch_size * hidden_size) / 4;
            const int block_size = 256;
            const int num_blocks = (total_work_vec + block_size - 1) / block_size;

            fused_gru_cell_kernel_vectorized<<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
                gi_t.data_ptr<float>(),
                gh_t.data_ptr<float>(),
                h_prev.data_ptr<float>(),
                h_next.data_ptr<float>(),
                batch_size,
                hidden_size);

            h_prev = h_next;
        }
        current_input = layer_output;
    }
    return final_output;
}
"""

cpp_source = """
#include <torch/extension.h>
#include <vector>

torch::Tensor gru_forward(
    torch::Tensor x, torch::Tensor h_0,
    const std::vector<torch::Tensor>& weights_ih,
    const std::vector<torch::Tensor>& weights_hh,
    const std::vector<torch::Tensor>& biases_ih,
    const std::vector<torch::Tensor>& biases_hh);
"""

# JIT compile the custom CUDA kernel
custom_gru_module = load_inline(
    name="custom_gru_vectorized_v4",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["gru_forward"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(Model, self).__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=False)
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.bias = bias
    
    def forward(self, x, h0):
        if self.batch_first:
            x = x.transpose(0, 1)

        x_contiguous = x.contiguous()
        h0_contiguous = h0.contiguous()

        weights_ih = [getattr(self.gru, f'weight_ih_l{i}') for i in range(self.num_layers)]
        weights_hh = [getattr(self.gru, f'weight_hh_l{i}') for i in range(self.num_layers)]
        
        if self.bias:
            biases_ih = [getattr(self.gru, f'bias_ih_l{i}') for i in range(self.num_layers)]
            biases_hh = [getattr(self.gru, f'bias_hh_l{i}') for i in range(self.num_layers)]
        else:
            biases_ih = [torch.Tensor()] * self.num_layers
            biases_hh = [torch.Tensor()] * self.num_layers
        
        output = custom_gru_module.gru_forward(x_contiguous, h0_contiguous, weights_ih, weights_hh, biases_ih, biases_hh)

        if self.batch_first:
            output = output.transpose(0, 1)
            
        return output

# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    return [torch.randn(seq_len, batch_size, input_size).cuda(),
            torch.randn((num_layers, batch_size, hidden_size)).cuda()]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]
# EVOLVE-BLOCK-END