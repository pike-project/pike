# EVOLVE-BLOCK-START
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


@triton.jit
def _fully_fused_attention_kernel(
    QKV, O,
    stride_qkv_b, stride_qkv_t, stride_qkv_c,
    stride_o_b, stride_o_t, stride_o_c,
    B, H, T, C,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    """
    Triton kernel for fully fused attention.
    This kernel takes the packed QKV tensor from a linear layer and computes the final attention output.
    It fuses the following operations:
    1. Q, K, V preparation (split, reshape, transpose)
    2. Scaled dot-product attention (Q @ K.T)
    3. Causal masking
    4. ReLU activation
    5. Value mixing (Attn @ V)
    6. Output reassembly (transpose, reshape)
    This avoids materializing the large intermediate attention matrix and reduces memory I/O.
    """
    # Grid and program IDs
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H

    # Pointers to QKV and O for the current batch
    QKV_batch_ptr = QKV + off_b * stride_qkv_b
    O_batch_ptr = O + off_b * stride_o_b

    # Offsets for the current block
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Pointers to the Q tile
    # Q's data is at QKV[b, t, h*D + d]
    q_offset = off_h * HEAD_DIM * stride_qkv_c
    q_ptrs = QKV_batch_ptr + q_offset + (offs_m[:, None] * stride_qkv_t + offs_d[None, :] * stride_qkv_c)

    # Accumulator for the output tile
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Scale factor
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # Load Q tile
    q = tl.load(q_ptrs, mask=offs_m[:, None] < T, other=0.0)
    q = (q * sm_scale).to(tl.float16)

    # Loop over K and V tiles
    hi = (start_m + 1) * BLOCK_M
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # --- Load K tile (transposed) ---
        # K's data is at QKV[b, t, C + h*D + d]. We need K_transposed[d, t]
        k_offset = C * stride_qkv_c + off_h * HEAD_DIM * stride_qkv_c
        k_ptrs = QKV_batch_ptr + k_offset + (offs_d[:, None] * stride_qkv_c + offs_n[None, :] * stride_qkv_t)
        k = tl.load(k_ptrs, mask=offs_n[None, :] < T, other=0.0)
        
        # --- Load V tile ---
        # V's data is at QKV[b, t, 2*C + h*D + d]
        v_offset = 2 * C * stride_qkv_c + off_h * HEAD_DIM * stride_qkv_c
        v_ptrs = QKV_batch_ptr + v_offset + (offs_n[:, None] * stride_qkv_t + offs_d[None, :] * stride_qkv_c)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < T, other=0.0)
        
        # --- Compute attention scores ---
        qk = tl.dot(q, k, out_dtype=tl.float32)
        
        # --- Causal masking ---
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))

        # --- ReLU ---
        p = tl.maximum(qk, 0.0).to(tl.float16)
        
        # --- Accumulate output ---
        acc += tl.dot(p, v)

    # Write back the output tile to the final tensor O
    # O has shape (B, T, C). We write acc[m, d] to O[b, t, h*D + d]
    offs_m_global = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c_global = off_h * HEAD_DIM + tl.arange(0, HEAD_DIM)
    
    o_ptrs = O_batch_ptr + (offs_m_global[:, None] * stride_o_t + offs_c_global[None, :] * stride_o_c)
    tl.store(o_ptrs, acc, mask=offs_m_global[:, None] < T)


def _fully_fused_attention(qkv, n_head):
    B, T, C3 = qkv.shape
    C = C3 // 3
    H = n_head
    HEAD_DIM = C // H

    # Shape constraints
    assert HEAD_DIM in {16, 32, 64, 128}, "Head dimension must be one of 16, 32, 64, 128"
    assert qkv.is_cuda and qkv.is_contiguous(), "Input must be a contiguous CUDA tensor"
    
    original_dtype = qkv.dtype
    if original_dtype != torch.float16:
        qkv = qkv.to(torch.float16)
    
    # Output tensor `o` has the final shape (B, T, C)
    o = torch.empty((B, T, C), device=qkv.device, dtype=qkv.dtype)

    # Block size configuration
    BLOCK_M = 128
    BLOCK_N = 64
    num_warps = 4 if HEAD_DIM <= 64 else 8
    
    grid = (triton.cdiv(T, BLOCK_M), B * H)
    
    _fully_fused_attention_kernel[grid](
        qkv, o,
        qkv.stride(0), qkv.stride(1), qkv.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        B, H, T, C,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
    
    return o.to(original_dtype)


class Model(nn.Module):
    """
    A multi-head masked self-attention layer optimized with a single, fully-fused Triton kernel.
    This kernel replaces the entire sequence of PyTorch operations within the attention mechanism,
    from QKV preparation to final output reassembly, dramatically reducing memory overhead
    and kernel launch costs.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection (kept for compatibility, but not used in the original forward pass)
        self.c_proj = nn.Linear(n_embd, n_embd)
        # The causal mask logic is now fully contained within the Triton kernel
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        # 1. Project to packed Q, K, V with a single linear layer
        qkv = self.c_attn(x)

        # 2. Use the single fully-fused kernel for the entire attention computation
        y = _fully_fused_attention(qkv, self.n_head)

        return y

batch_size = 16
max_seqlen = 1024
n_embd = 768  # Hidden dimension, typical for BERT-base size
n_head = 12   # Number of attention heads, typical for BERT-base size

def get_inputs():
    return [torch.randn(batch_size, max_seqlen, n_embd).cuda()]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]
# EVOLVE-BLOCK-END