# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This version builds on the float4 vectorized kernel by adding compiler optimizations
# and a minor Python-level optimization.
#
# Key Improvements:
# 1. Compiler Flags: We add `-O3` and `-use_fast_math` to the CUDA compiler flags.
#    `-use_fast_math` enables faster, approximate versions of math functions like
#    `expf` (used in sigmoid) and `tanhf`, which can provide a significant speedup
#    for compute-bound parts of the kernel at a negligible cost to precision.
# 2. Python Loop Optimization: In the `forward` pass, the concatenation of layer
#    outputs is now skipped for the final layer, as its output is not used as an
#    input for a subsequent layer. This avoids an unnecessary `torch.cat` operation.
# 3. Kernel Readability: The internal calculations within the CUDA kernel have been
#    refactored to use `make_float4` for clarity, without changing the underlying
#    operations. This makes the vector-wise nature of the computations more explicit.
gru_looper_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// Device function for sigmoid, inlined for performance
__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Fused kernel for GRU cell element-wise ops, using float4 vectorization.
__global__ void gru_cell_kernel_v5(
    const float* __restrict__ gi,
    const float* __restrict__ gh,
    const float* __restrict__ h_prev,
    float* h_next,
    const int batch_size, const int hidden_size) {

    const int H4 = hidden_size / 4;
    const int N4 = batch_size * H4;
    const int H3_4 = 3 * H4;

    // Grid-stride loop over float4 elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N4; i += blockDim.x * gridDim.x) {
        const int b = i / H4;
        const int h4 = i % H4;
        
        const int gi_base_idx = b * H3_4;
        const int gh_base_idx = b * H3_4;
        
        const float4* gi4 = reinterpret_cast<const float4*>(gi);
        const float4* gh4 = reinterpret_cast<const float4*>(gh);
        const float4* h_prev4 = reinterpret_cast<const float4*>(h_prev);
        float4* h_next4 = reinterpret_cast<float4*>(h_next);

        // Load 4 floats at a time for each gate component
        const float4 gi_r = gi4[gi_base_idx + h4];
        const float4 gh_r = gh4[gh_base_idx + h4];

        const float4 gi_z = gi4[gi_base_idx + H4 + h4];
        const float4 gh_z = gh4[gh_base_idx + H4 + h4];

        const float4 gi_n = gi4[gi_base_idx + 2 * H4 + h4];
        const float4 gh_n = gh4[gh_base_idx + 2 * H4 + h4];
        
        const float4 h_prev_t = h_prev4[i];

        // --- Gate Computations (vectorized) ---
        const float4 r_t = make_float4(
            sigmoidf(gi_r.x + gh_r.x), sigmoidf(gi_r.y + gh_r.y),
            sigmoidf(gi_r.z + gh_r.z), sigmoidf(gi_r.w + gh_r.w)
        );
        const float4 z_t = make_float4(
            sigmoidf(gi_z.x + gh_z.x), sigmoidf(gi_z.y + gh_z.y),
            sigmoidf(gi_z.z + gh_z.z), sigmoidf(gi_z.w + gh_z.w)
        );
        const float4 n_t = make_float4(
            tanhf(gi_n.x + r_t.x * gh_n.x), tanhf(gi_n.y + r_t.y * gh_n.y),
            tanhf(gi_n.z + r_t.z * gh_n.z), tanhf(gi_n.w + r_t.w * gh_n.w)
        );

        // --- Final hidden state update ---
        h_next4[i] = make_float4(
            (1.0f - z_t.x) * n_t.x + z_t.x * h_prev_t.x,
            (1.0f - z_t.y) * n_t.y + z_t.y * h_prev_t.y,
            (1.0f - z_t.z) * n_t.z + z_t.z * h_prev_t.z,
            (1.0f - z_t.w) * n_t.w + z_t.w * h_prev_t.w
        );
    }
}

// C++ looper function remains structurally the same, but calls the new kernel.
torch::Tensor gru_layer_looper_v5(
    torch::Tensor gi_seq, torch::Tensor h_0, torch::Tensor w_hh, torch::Tensor b_hh,
    bool reverse, torch::Tensor output_seq
) {
    auto h_t = h_0.contiguous();
    w_hh = w_hh.contiguous();
    b_hh = b_hh.contiguous();

    const auto seq_len = gi_seq.size(0);
    const auto batch_size = h_t.size(0);
    const auto hidden_size = h_t.size(1);
    
    TORCH_CHECK(hidden_size % 4 == 0, "hidden_size must be divisible by 4 for the vectorized kernel.");

    auto gh = torch::empty({batch_size, 3 * hidden_size}, gi_seq.options());

    const int block_size = 256;
    const int N4 = (batch_size * hidden_size) / 4;
    const int num_blocks = (N4 + block_size - 1) / block_size;
    dim3 threads(block_size);
    dim3 blocks(num_blocks);

    auto stream = at::cuda::getCurrentCUDAStream();

    if (!reverse) {
        for (int64_t t = 0; t < seq_len; ++t) {
            auto gi_t = gi_seq.select(0, t);
            auto h_next = output_seq.select(0, t);
            at::linear_out(gh, h_t, w_hh, b_hh);
            gru_cell_kernel_v5<<<blocks, threads, 0, stream>>>(
                gi_t.data_ptr<float>(), gh.data_ptr<float>(), h_t.data_ptr<float>(),
                h_next.data_ptr<float>(), batch_size, hidden_size);
            h_t = h_next;
        }
    } else {
        for (int64_t t = seq_len - 1; t >= 0; --t) {
            auto gi_t = gi_seq.select(0, t);
            auto h_next = output_seq.select(0, t);
            at::linear_out(gh, h_t, w_hh, b_hh);
            gru_cell_kernel_v5<<<blocks, threads, 0, stream>>>(
                gi_t.data_ptr<float>(), gh.data_ptr<float>(), h_t.data_ptr<float>(),
                h_next.data_ptr<float>(), batch_size, hidden_size);
            h_t = h_next;
        }
    }
    
    return h_t;
}
"""

gru_looper_cpp_source = "torch::Tensor gru_layer_looper_v5(torch::Tensor gi_seq, torch::Tensor h_0, torch::Tensor w_hh, torch::Tensor b_hh, bool reverse, torch::Tensor output_seq);"

# JIT compile the custom C++/CUDA code
custom_gru_looper = load_inline(
    name="custom_gru_looper_v5",
    cpp_sources=gru_looper_cpp_source,
    cuda_sources=gru_looper_source,
    functions=["gru_layer_looper_v5"],
    verbose=False,
    extra_cuda_cflags=["-O3", "-use_fast_math"],
)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(Model, self).__init__()
        # Instantiate a standard GRU to "steal" its properly initialized weights.
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout=0, bidirectional=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        if self.batch_first:
            raise NotImplementedError("This custom GRU implementation does not support batch_first=True.")
        if self.hidden_size % 4 != 0:
            raise ValueError("This custom GRU implementation requires hidden_size to be divisible by 4.")

    def forward(self, x, h0):
        current_x = x
        final_hidden_states = []
        
        seq_len, batch_size, _ = current_x.shape

        for layer in range(self.num_layers):
            # Extract weights and biases for the current layer from the nn.GRU module.
            w_ih_fwd = getattr(self.gru, f'weight_ih_l{layer}')
            w_hh_fwd = getattr(self.gru, f'weight_hh_l{layer}')
            b_ih_fwd = getattr(self.gru, f'bias_ih_l{layer}')
            b_hh_fwd = getattr(self.gru, f'bias_hh_l{layer}')
            
            w_ih_bwd = getattr(self.gru, f'weight_ih_l{layer}_reverse')
            w_hh_bwd = getattr(self.gru, f'weight_hh_l{layer}_reverse')
            b_ih_bwd = getattr(self.gru, f'bias_ih_l{layer}_reverse')
            b_hh_bwd = getattr(self.gru, f'bias_hh_l{layer}_reverse')

            # Optimization: Pre-compute input-related linear transformations for the whole sequence.
            gi_fwd_seq = F.linear(current_x, w_ih_fwd, b_ih_fwd)
            gi_bwd_seq = F.linear(current_x, w_ih_bwd, b_ih_bwd)

            h_fwd_0 = h0[layer * 2]
            h_bwd_0 = h0[layer * 2 + 1]
            
            # Pre-allocate output tensors for the layer's forward and backward passes.
            outputs_fwd = torch.empty(seq_len, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            outputs_bwd = torch.empty(seq_len, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

            # Execute the optimized C++ time-sequence looper for both directions.
            h_fwd_n = custom_gru_looper.gru_layer_looper_v5(gi_fwd_seq, h_fwd_0, w_hh_fwd, b_hh_fwd, False, outputs_fwd)
            h_bwd_n = custom_gru_looper.gru_layer_looper_v5(gi_bwd_seq, h_bwd_0, w_hh_bwd, b_hh_bwd, True, outputs_bwd)
            
            final_hidden_states.append(h_fwd_n)
            final_hidden_states.append(h_bwd_n)

            # Optimization: The input to the next layer is the concatenated output, but this is
            # not needed for the final layer.
            if layer < self.num_layers - 1:
                current_x = torch.cat([outputs_fwd, outputs_bwd], dim=2)
        
        return torch.stack(final_hidden_states)

# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    # Ensure inputs are on the correct device (CUDA) for the custom kernel.
    x = torch.randn(seq_len, batch_size, input_size).cuda()
    h0 = torch.randn((num_layers*2, batch_size, hidden_size)).cuda()
    return [x, h0]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]
# EVOLVE-BLOCK-END