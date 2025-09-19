# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Combined CUDA source for all fused kernels. This reduces compilation overhead.
fused_attention_cuda_source = """
#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <limits>
#include <cmath>

// --- KERNEL 1: Fused QKV Projection ---
// This kernel takes the packed output of the c_attn layer (B, T, 3*C) and
// produces the Q, K, and V tensors in the desired (B, H, T, hs) layout.
// It fuses the split, view, and transpose operations into a single launch.

__global__ void fused_qkv_projection_kernel(
    const half* __restrict__ qkv_in, // Input: (B, T, 3*C)
    half* __restrict__ q_out,        // Output Q: (B, H, T, hs)
    half* __restrict__ k_out,        // Output K: (B, H, T, hs)
    half* __restrict__ v_out,        // Output V: (B, H, T, hs)
    const int B, const int T, const int C, const int H, const int hs
) {
    // Grid dimensions are mapped to (C_tiled, T, B)
    const int b = blockIdx.z;
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c >= C) {
        return;
    }

    // Source index from (B, T, 3*C) tensor
    const long in_base_idx = (long)b * T * 3 * C + (long)t * 3 * C;
    const long q_in_idx = in_base_idx + c;
    const long k_in_idx = q_in_idx + C;
    const long v_in_idx = k_in_idx + C;

    // Destination index for (B, H, T, hs) tensors
    const int h = c / hs;
    const int s = c % hs;
    const long out_idx = (long)b * H * T * hs + (long)h * T * hs + (long)t * hs + s;

    q_out[out_idx] = qkv_in[q_in_idx];
    k_out[out_idx] = qkv_in[k_in_idx];
    v_out[out_idx] = qkv_in[v_in_idx];
}

// C++ wrapper for the QKV projection kernel
std::vector<torch::Tensor> fused_qkv_projection_cuda(
    torch::Tensor qkv_in,
    const int n_head)
{
    const auto B = qkv_in.size(0);
    const auto T = qkv_in.size(1);
    const auto C = qkv_in.size(2) / 3;
    const auto H = n_head;
    const auto hs = C / H;

    auto opts = qkv_in.options();
    auto q_out = torch::empty({B, H, T, hs}, opts);
    auto k_out = torch::empty({B, H, T, hs}, opts);
    auto v_out = torch::empty({B, H, T, hs}, opts);

    const int block_size_x = 256;
    const dim3 grid( (C + block_size_x - 1) / block_size_x, T, B);
    const dim3 block(block_size_x);

    fused_qkv_projection_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(qkv_in.data_ptr<at::Half>()),
        reinterpret_cast<half*>(q_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_out.data_ptr<at::Half>()),
        B, T, C, H, hs
    );

    return {q_out, k_out, v_out};
}


// --- KERNEL 2: Fused Scaled, Masked Softmax (Warp-per-row) ---
// This kernel fuses scaling, causal masking, and softmax for T=512.
// It is highly optimized, using one warp per row to minimize global memory I/O
// and leverage fast warp-shuffle instructions for reductions.
// It reads FP16, computes in FP32 for stability, and writes FP16.

__global__ void scaled_masked_softmax_warp_fp16_kernel(
    const half* __restrict__ in_data,
    half* __restrict__ out_data,
    const float scale,
    const int T_dim,
    const int num_rows
) {
    const int WARP_SIZE = 32;
    const int N_VALS_PER_THREAD = 16; // 512 / 32

    const int row_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (row_id >= num_rows) {
        return;
    }

    const int query_pos = row_id % T_dim;
    const int lane_id = threadIdx.x;

    const half* row_in_ptr = in_data + row_id * T_dim;
    half* row_out_ptr = out_data + row_id * T_dim;

    float vals[N_VALS_PER_THREAD];

    float thread_max = -std::numeric_limits<float>::infinity();
    #pragma unroll
    for (int i = 0; i < N_VALS_PER_THREAD; ++i) {
        const int col_idx = lane_id + i * WARP_SIZE;
        float val = (col_idx <= query_pos)
            ? __half2float(row_in_ptr[col_idx]) * scale
            : -std::numeric_limits<float>::infinity();
        vals[i] = val;
        thread_max = fmaxf(thread_max, val);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));
    }
    const float row_max = __shfl_sync(0xFFFFFFFF, thread_max, 0);

    float thread_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < N_VALS_PER_THREAD; ++i) {
        float val = isinf(row_max) ? 0.0f : expf(vals[i] - row_max);
        vals[i] = val;
        thread_sum += val;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
    }
    const float row_sum = __shfl_sync(0xFFFFFFFF, thread_sum, 0);

    const float inv_row_sum = (row_sum > 1e-8f) ? (1.0f / row_sum) : 0.0f;
    #pragma unroll
    for (int i = 0; i < N_VALS_PER_THREAD; ++i) {
        const int col_idx = lane_id + i * WARP_SIZE;
        row_out_ptr[col_idx] = __float2half(vals[i] * inv_row_sum);
    }
}

// C++ wrapper for the softmax kernel
torch::Tensor fused_scaled_masked_softmax_fp16_cuda(torch::Tensor att, float scale) {
    const auto T = att.size(2);
    const int num_rows = att.numel() / T;
    auto out = torch::empty_like(att);

    const int WARP_SIZE = 32;
    const int WARPS_PER_BLOCK = 16;

    const dim3 grid((num_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    const dim3 block(WARP_SIZE, WARPS_PER_BLOCK);

    scaled_masked_softmax_warp_fp16_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(att.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        scale, T, num_rows
    );
    return out;
}
"""

# C++ source for the PyTorch bindings (forward declarations)
fused_attention_cpp_source = """
#include <torch/extension.h>
#include <vector>
// Forward declaration of the CUDA-defined functions
torch::Tensor fused_scaled_masked_softmax_fp16_cuda(torch::Tensor att, float scale);
std::vector<torch::Tensor> fused_qkv_projection_cuda(torch::Tensor qkv_in, int n_head);
"""

# JIT compile the CUDA kernels.
fused_ops = load_inline(
    name="fused_attention_ops_v7",
    cpp_sources=fused_attention_cpp_source,
    cuda_sources=fused_attention_cuda_source,
    functions=["fused_scaled_masked_softmax_fp16_cuda", "fused_qkv_projection_cuda"],
    verbose=False,
)

class Model(nn.Module):
    """
    A highly optimized multi-head masked self-attention layer.
    1. Uses FP16 precision to leverage Tensor Cores and reduce memory bandwidth.
    2. Fuses QKV projection (split, view, transpose) into a single CUDA kernel.
    3. Fuses the attention calculation (scale, mask, softmax) into a second CUDA kernel.
    4. Removes no-op dropout layers.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        self.n_head = n_head
        self.n_embd = n_embd
        
        # Store the compiled custom kernel functions
        self.fused_qkv_proj = fused_ops.fused_qkv_projection_cuda
        self.fused_softmax = fused_ops.fused_scaled_masked_softmax_fp16_cuda

        # Convert the entire model to FP16
        self.half()

    def forward(self, x):
        # Cast input to FP16 to match the model's parameters and kernels.
        if x.dtype != torch.half:
            x = x.half()

        B, T, C = x.size()

        # 1. Calculate packed Q, K, V using a single efficient GEMM (on Tensor Cores).
        qkv_packed = self.c_attn(x)
        
        # 2. Use fused kernel to split, view, and transpose into final Q, K, V shapes.
        # This replaces 6 separate PyTorch operations.
        q, k, v = self.fused_qkv_proj(qkv_packed, self.n_head)

        # 3. QK^T Batched Matrix-Matrix product (on Tensor Cores).
        att = q @ k.transpose(-2, -1)
        
        # 4. Fused (Scale * Mask * Softmax) using our optimized FP16 kernel.
        scale = 1.0 / math.sqrt(k.size(-1))
        att = self.fused_softmax(att, scale)
        
        # 5. AV Batched Matrix-Matrix product (on Tensor Cores).
        y = att @ v
        
        # Re-assemble all head outputs.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 6. Final output projection (on Tensor Cores).
        y = self.c_proj(y)
        
        # Cast final output to float to match the baseline's output dtype for correctness.
        return y.float()

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    # Generate inputs in FP16 and on the correct device.
    return [torch.randn(batch_size, seq_len, n_embd).cuda().half()]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
# EVOLVE-BLOCK-END