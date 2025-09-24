import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# --- Optimized Fused Attention Kernel using Triton ---
# This solution improves upon the previous fused kernel with two key enhancements:
# 1. **Expanded Autotuning Search Space**: The performance of Triton kernels is highly sensitive
#    to launch parameters like block sizes, number of warps, and pipeline stages. This version
#    significantly expands the `triton.autotune` configuration space, allowing Triton to
#    more effectively search for the optimal parameters for the specific GPU architecture
#    it is running on. This is a pragmatic and powerful method for unlocking more performance
#    from the hardware.
# 2. **Generalization for Causal/Non-Causal Attention**: A boolean `is_causal` argument has been
#    added to the kernel. This allows the Triton compiler to generate specialized code based on
#    whether causal masking is needed. For non-causal attention, the entire masking logic
#    (offset calculation, range comparisons, `tl.where`) is compiled out, reducing instruction
#    count and eliminating branching in the inner loop, leading to a noticeable speedup in
#    that common use case. For the given model, `is_causal` is True.

@triton.autotune(
    configs=[
        # Base configs from previous version
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'num_stages': 4, 'num_warps': 4}),
        # Expanded search space
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'num_stages': 5, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_stages': 5, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_stages': 5, 'num_warps': 2}),
    ],
    key=['seq_len', 'head_dim', 'is_causal'],
)
@triton.jit
def _fused_qkv_attention_relu_kernel(
    X_QKV, Y,
    stride_x_b, stride_x_t, stride_x_c,
    stride_y_b, stride_y_t, stride_y_c,
    n_head, n_embd, seq_len,
    is_causal: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    
    batch_id = batch_head_id // n_head
    head_id = batch_head_id % n_head

    q_offset = head_id * HEAD_DIM
    k_offset = n_embd + head_id * HEAD_DIM
    v_offset = 2 * n_embd + head_id * HEAD_DIM
    
    x_batch_ptr = X_QKV + batch_id * stride_x_b
    q_base_ptr = x_batch_ptr + q_offset
    k_base_ptr = x_batch_ptr + k_offset
    v_base_ptr = x_batch_ptr + v_offset
    
    q_block_ptr = tl.make_block_ptr(
        base=q_base_ptr,
        shape=(seq_len, HEAD_DIM),
        strides=(stride_x_t, stride_x_c),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    q = tl.load(q_block_ptr, boundary_check=(0,))

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    scale = (1.0 / math.sqrt(HEAD_DIM))

    loop_end = seq_len if not is_causal else (start_m + 1) * BLOCK_M
    for start_n in range(0, loop_end, BLOCK_N):
        k_block_ptr = tl.make_block_ptr(
            base=k_base_ptr,
            shape=(HEAD_DIM, seq_len),
            strides=(stride_x_c, stride_x_t),
            offsets=(0, start_n),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1)
        )
        k = tl.load(k_block_ptr, boundary_check=(1,))

        v_block_ptr = tl.make_block_ptr(
            base=v_base_ptr,
            shape=(seq_len, HEAD_DIM),
            strides=(stride_x_t, stride_x_c),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0)
        )
        v = tl.load(v_block_ptr, boundary_check=(0,))
        
        qk = tl.dot(q, k)
        qk *= scale

        if is_causal:
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        att = tl.maximum(qk, 0.0)
        y_partial = tl.dot(att.to(v.dtype), v)
        acc += y_partial

    y_base_ptr = Y + batch_id * stride_y_b
    y_block_ptr = tl.make_block_ptr(
        base=y_base_ptr,
        shape=(seq_len, n_embd),
        strides=(stride_y_t, stride_y_c),
        offsets=(start_m * BLOCK_M, head_id * HEAD_DIM),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    tl.store(y_block_ptr, acc.to(Y.dtype.element_ty), boundary_check=(0,))

def fused_qkv_attention_relu(x_qkv, n_head, is_causal):
    B, seq_len, C3 = x_qkv.shape
    n_embd = C3 // 3
    head_dim = n_embd // n_head
    
    y = torch.empty((B, seq_len, n_embd), device=x_qkv.device, dtype=x_qkv.dtype)

    # Use a default BLOCK_M for grid calculation. Autotuner will find the best value.
    BLOCK_M = 128
    grid = (triton.cdiv(seq_len, BLOCK_M), B * n_head)

    _fused_qkv_attention_relu_kernel[grid](
        x_qkv, y,
        x_qkv.stride(0), x_qkv.stride(1), x_qkv.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        n_head, n_embd, seq_len,
        is_causal=is_causal,
        HEAD_DIM=head_dim,
    )
    return y

class ModelNew(nn.Module):
    """
    An optimized multi-head masked self-attention layer using a single fused, mixed-precision,
    and highly-tuned Triton kernel. Assumes half-precision (.half()) operation for performance.
    """
    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        # The causal mask is computed on-the-fly in the Triton kernel, so the buffer is not needed.

    def forward(self, x):
        # Assumes input `x` and model weights are in half precision.
        x_qkv = self.c_attn(x)
        
        # The fused Triton kernel handles the rest of the attention calculation.
        # We pass `is_causal=True` to match the original model's behavior.
        y = fused_qkv_attention_relu(x_qkv, self.n_head, is_causal=True)
        
        # To match the original model's forward pass, the final projection is omitted.
        # y = self.c_proj(y)
        
        return y

# --- Boilerplate for running the model ---

batch_size = 16
max_seqlen = 1024
n_embd = 768
n_head = 12

def get_inputs():
    # It is assumed the benchmarking harness will convert this to the appropriate precision (e.g., .half())
    return [torch.randn(batch_size, max_seqlen, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]