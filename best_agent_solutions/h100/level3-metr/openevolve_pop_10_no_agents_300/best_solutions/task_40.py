# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# This version replaces the scalar kernel with a fully vectorized `float4` kernel
# to maximize memory bandwidth, a common bottleneck in GRU cells.
# 1. A single C++ function still encapsulates the entire multi-layer GRU to minimize Python overhead.
# 2. The core CUDA kernel now uses `float4` to load/process/store 4 floats at a time.
# 3. The kernel launch grid is 2D, mapping naturally to the (batch, hidden_dim) problem space.
stacked_gru_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// --- CUDA Device Helper Functions for Vectorized Operations ---

__device__ __forceinline__ float4 sigmoid_vec(const float4& v) {
    return make_float4(
        1.0f / (1.0f + __expf(-v.x)),
        1.0f / (1.0f + __expf(-v.y)),
        1.0f / (1.0f + __expf(-v.z)),
        1.0f / (1.0f + __expf(-v.w))
    );
}

__device__ __forceinline__ float4 tanh_vec(const float4& v) {
    return make_float4(tanhf(v.x), tanhf(v.y), tanhf(v.z), tanhf(v.w));
}

// Fused GRU cell forward kernel, vectorized with float4 for 4x memory bandwidth.
__global__ void fused_gru_cell_kernel_vectorized(
    const float4* __restrict__ gi,       // Pre-activations from input projection (batch, 3 * hidden_size / 4)
    const float4* __restrict__ gh,       // Pre-activations from hidden projection (batch, 3 * hidden_size / 4)
    const float4* __restrict__ h_prev,   // Previous hidden state (batch, hidden_size / 4)
    float4* __restrict__ h_next,         // Output new hidden state (batch, hidden_size / 4)
    const int batch_size,
    const int hidden_size_vec          // hidden_size / 4
) {
    // 2D grid: blockIdx.x corresponds to hidden_size_vec, blockIdx.y to batch_size
    const int hidden_vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (hidden_vec_idx >= hidden_size_vec || batch_idx >= batch_size) {
        return;
    }

    // Index for h_prev and h_next (ensures coalesced access)
    const int h_idx = batch_idx * hidden_size_vec + hidden_vec_idx;

    // Base pointer for the gates for this batch item
    const int gate_base_idx = batch_idx * 3 * hidden_size_vec;
    const float4* gi_b = gi + gate_base_idx;
    const float4* gh_b = gh + gate_base_idx;
    
    // Load all gate values using float4 (128-bit) memory transactions
    const float4 gi_r = gi_b[hidden_vec_idx];
    const float4 gi_z = gi_b[hidden_vec_idx + hidden_size_vec];
    const float4 gi_n = gi_b[hidden_vec_idx + 2 * hidden_size_vec];

    const float4 gh_r = gh_b[hidden_vec_idx];
    const float4 gh_z = gh_b[hidden_vec_idx + hidden_size_vec];
    const float4 gh_n = gh_b[hidden_vec_idx + 2 * hidden_size_vec];

    // Compute gates using vectorized math
    const float4 r_t = sigmoid_vec({gi_r.x + gh_r.x, gi_r.y + gh_r.y, gi_r.z + gh_r.z, gi_r.w + gh_r.w});
    const float4 z_t = sigmoid_vec({gi_z.x + gh_z.x, gi_z.y + gh_z.y, gi_z.z + gh_z.z, gi_z.w + gh_z.w});
    const float4 n_t = tanh_vec({
        gi_n.x + r_t.x * gh_n.x,
        gi_n.y + r_t.y * gh_n.y,
        gi_n.z + r_t.z * gh_n.z,
        gi_n.w + r_t.w * gh_n.w
    });

    const float4 h_prev_val = h_prev[h_idx];
    
    // Final hidden state update: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    const float4 h_next_val = {
        (1.0f - z_t.x) * n_t.x + z_t.x * h_prev_val.x,
        (1.0f - z_t.y) * n_t.y + z_t.y * h_prev_val.y,
        (1.0f - z_t.z) * n_t.z + z_t.z * h_prev_val.z,
        (1.0f - z_t.w) * n_t.w + z_t.w * h_prev_val.w
    };

    h_next[h_idx] = h_next_val;
}

// C++ function that orchestrates the entire multi-layer GRU forward pass.
torch::Tensor stacked_gru_forward(
    torch::Tensor input,                  // (seq_len, batch_size, input_size)
    torch::Tensor h_init,                 // (num_layers, batch_size, hidden_size)
    std::vector<torch::Tensor> weights_ih,
    std::vector<torch::Tensor> weights_hh,
    std::vector<torch::Tensor> biases_ih,
    std::vector<torch::Tensor> biases_hh
) {
    const auto seq_len = input.size(0);
    const auto batch_size = input.size(1);
    const auto hidden_size = h_init.size(2);
    const auto num_layers = h_init.size(0);

    TORCH_CHECK(hidden_size % 4 == 0, "hidden_size must be a multiple of 4 for the vectorized kernel.");

    auto current_input = input;
    std::vector<torch::Tensor> final_hidden_states;
    final_hidden_states.reserve(num_layers);

    auto gh_t = torch::empty({batch_size, 3 * hidden_size}, input.options());
    
    // Configure kernel launch parameters for the vectorized kernel
    const int hidden_size_vec = hidden_size / 4;
    const dim3 block_dim(256, 1, 1);
    const dim3 grid_dim(
        (hidden_size_vec + block_dim.x - 1) / block_dim.x,
        (batch_size + block_dim.y - 1) / block_dim.y,
        1
    );

    for (int i = 0; i < num_layers; ++i) {
        auto gi = at::linear(current_input, weights_ih[i], biases_ih[i]);
        auto h_prev = h_init.select(0, i).contiguous();
        auto layer_output = torch::empty({seq_len, batch_size, hidden_size}, input.options());
        auto w_hh_t = weights_hh[i].t().contiguous();

        for (int t = 0; t < seq_len; ++t) {
            auto gi_t = gi.select(0, t);
            auto h_next = layer_output.select(0, t);
            at::addmm_out(gh_t, biases_hh[i], h_prev, w_hh_t);
            
            fused_gru_cell_kernel_vectorized<<<grid_dim, block_dim>>>(
                reinterpret_cast<const float4*>(gi_t.data_ptr<float>()),
                reinterpret_cast<const float4*>(gh_t.data_ptr<float>()),
                reinterpret_cast<const float4*>(h_prev.data_ptr<float>()),
                reinterpret_cast<float4*>(h_next.data_ptr<float>()),
                batch_size,
                hidden_size_vec
            );
            
            h_prev = h_next;
        }

        final_hidden_states.push_back(h_prev);
        current_input = layer_output;
    }
    
    auto h_n = torch::stack(final_hidden_states, 0);
    return h_n;
}
"""

stacked_gru_cpp_source = """
torch::Tensor stacked_gru_forward(
    torch::Tensor input, torch::Tensor h_init,
    std::vector<torch::Tensor> weights_ih, std::vector<torch::Tensor> weights_hh,
    std::vector<torch::Tensor> biases_ih, std::vector<torch::Tensor> biases_hh
);
"""

custom_stacked_gru_module = load_inline(
    name="custom_stacked_gru_module_v6_vectorized",
    cpp_sources=stacked_gru_cpp_source,
    cuda_sources=stacked_gru_source,
    functions=["stacked_gru_forward"],
    verbose=False,
    extra_cuda_cflags=["--use_fast_math"],
)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(Model, self).__init__()
        if batch_first:
            raise NotImplementedError("batch_first=True is not supported by this custom implementation.")
        if hidden_size % 4 != 0:
            raise ValueError("hidden_size must be a multiple of 4 for the vectorized kernel.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.gru_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            layer = nn.Module()
            layer.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, layer_input_size))
            layer.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
            if bias:
                layer.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
                layer.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
            else:
                layer.register_parameter('bias_ih', None)
                layer.register_parameter('bias_hh', None)
            self.gru_layers.append(layer)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for param in self.parameters():
            if param.data is not None:
                nn.init.uniform_(param.data, -stdv, stdv)

    def forward(self, x, h0):
        weights_ih, weights_hh, biases_ih, biases_hh = [], [], [], []
        for layer in self.gru_layers:
            weights_ih.append(layer.weight_ih)
            weights_hh.append(layer.weight_hh)
            if self.bias:
                biases_ih.append(layer.bias_ih)
                biases_hh.append(layer.bias_hh)
            else:
                zeros = torch.zeros(3 * self.hidden_size, device=x.device, dtype=x.dtype)
                biases_ih.append(zeros)
                biases_hh.append(zeros)
        
        x_cont = x.contiguous()
        h0_cont = h0.contiguous()

        h_n = custom_stacked_gru_module.stacked_gru_forward(
            x_cont, h0_cont, weights_ih, weights_hh, biases_ih, biases_hh
        )
        return h_n

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