# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# --- Custom CUDA Kernel for an Optimized Layer Normalization ---
# This kernel uses a "one-pass" algorithm to compute mean and variance, which reduces
# global memory bandwidth. It is more efficient than the vectorized version for this problem's
# dimensions (C=768) because a block size of 256 leads to better thread utilization
# in the main loop compared to a vectorized approach where C/4 < block_size, causing
# many threads to be inactive. This implementation leverages efficient warp-level
# primitives (__shfl_down_sync) for reduction.

cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace { // Anonymous namespace for helper functions
// WARP_SIZE is 32 on NVIDIA GPUs. This function sums a value across all 32 threads in a warp.
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int d = 16; d > 0; d /= 2) {
        val += __shfl_down_sync(0xffffffff, val, d);
    }
    return val;
}
} // anonymous namespace

template <typename T>
__global__ void layer_norm_fwd_kernel_onepass(T* __restrict__ out,
                                      const T* __restrict__ inp,
                                      const T* __restrict__ gamma,
                                      const T* __restrict__ beta,
                                      int N, int C, float epsilon) {
    // Each block processes one row (one token embedding).
    int row_idx = blockIdx.x;
    if (row_idx >= N) return;

    const T* x_row = inp + row_idx * C;
    T* y_row = out + row_idx * C;

    // --- 1. Compute sum and sum_sq in a single pass ---
    float sum = 0.0f;
    float sum_sq = 0.0f;
    // Each thread in the block computes partial sums. This loop structure ensures
    // all threads are active when C > blockDim.x.
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float val = static_cast<float>(x_row[i]);
        sum += val;
        sum_sq += val * val;
    }

    // --- 2. Efficient Reduction ---
    // First, reduce within each warp.
    sum = warp_reduce_sum(sum);
    sum_sq = warp_reduce_sum(sum_sq);

    // The leader of each warp (thread 0, 32, 64, ...) writes its partial sum to shared memory.
    extern __shared__ float sdata[];
    int warp_id = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        sdata[warp_id] = sum;
        sdata[warp_id + blockDim.x / 32] = sum_sq;
    }
    __syncthreads();

    // The first thread of the block finishes the reduction across warps.
    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        float total_sum_sq = 0.0f;
        int num_warps = blockDim.x / 32;
        for (int i = 0; i < num_warps; ++i) {
            total_sum += sdata[i];
            total_sum_sq += sdata[i + num_warps];
        }
        // Compute mean and inverse standard deviation, storing them back to shared memory
        // for all threads in the block to access.
        float mean = total_sum / C;
        float var = total_sum_sq / C - mean * mean;
        sdata[0] = mean;
        sdata[1] = rsqrtf(var + epsilon);
    }
    __syncthreads();
    
    // All threads read the final mean and rstd.
    float mean = sdata[0];
    float rstd = sdata[1];

    // --- 3. Normalize, scale (gamma), and shift (beta) ---
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float norm_val = (static_cast<float>(x_row[i]) - mean) * rstd;
        y_row[i] = static_cast<T>(norm_val * static_cast<float>(gamma[i]) + static_cast<float>(beta[i]));
    }
}

torch::Tensor layer_norm_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float epsilon)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Beta must be a CUDA tensor");
    
    // Ensure contiguous memory layout for direct pointer access in CUDA.
    auto input_c = input.contiguous();
    auto gamma_c = gamma.contiguous();
    auto beta_c = beta.contiguous();

    const auto input_sizes = input_c.sizes();
    const int C = input_sizes.back();
    const int N = input_c.numel() / C;

    auto output = torch::empty_like(input_c);

    const int block_size = 256; // A common, well-performing block size.
    const int num_blocks = N;
    // Shared memory size: enough for two floats (sum, sum_sq) per warp.
    const int shared_mem_size = 2 * (block_size / 32) * sizeof(float);

    // Dispatch to the kernel, templated on the input data type (float or half).
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_c.scalar_type(), "layer_norm_forward_v_best", ([&] {
        layer_norm_fwd_kernel_onepass<scalar_t><<<num_blocks, block_size, shared_mem_size>>>(
            output.data_ptr<scalar_t>(),
            input_c.data_ptr<scalar_t>(),
            gamma_c.data_ptr<scalar_t>(),
            beta_c.data_ptr<scalar_t>(),
            N,
            C,
            epsilon
        );
    }));

    return output;
}
"""

cpp_source = "torch::Tensor layer_norm_forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float epsilon);"

# JIT compile the CUDA kernel.
custom_ops = load_inline(
    name="custom_ln_op_scalar_best",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["layer_norm_forward"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"] # Add compiler optimizations
)

# --- PyTorch Module wrapper for Custom LayerNorm ---

class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        return custom_ops.layer_norm_forward(x, self.weight, self.bias, self.eps)

# --- Fused Attention Module ---
# Using PyTorch's built-in fused scaled_dot_product_attention is critical for performance.
# Manual implementation is a major bottleneck. The built-in function fuses operations
# into a single, highly-optimized kernel (often FlashAttention).
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = attn_pdrop
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Use PyTorch's fused scaled dot-product attention for maximum efficiency.
        # The `is_causal=True` argument handles the causal masking automatically.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
    
# --- Main Model using Custom and Fused Components ---
class Model(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        # Use our highly optimized custom LayerNorm kernel.
        self.ln_1 = CustomLayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = CustomLayerNorm(n_embd)
        # Use a standard MLP structure with an approximate GELU for speed.
        # nn.GELU(approximate='tanh') is a fast and effective activation function.
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# --- Problem Configuration ---
batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.randn(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
# EVOLVE-BLOCK-END