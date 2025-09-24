import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

# Triton JIT implementation of tanh, since it's not a native op.
@triton.jit
def tanh_triton(x):
    # tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    ex = tl.exp(x)
    emx = tl.exp(-x)
    return (ex - emx) / (ex + emx)

# Define NewGELU as a Triton JIT function for use inside kernels
@triton.jit
def gelu(x):
    """
    Triton implementation of the GELU activation function.
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    # math.sqrt(2.0 / math.pi) = 0.7978845608
    return 0.5 * x * (1.0 + tanh_triton(0.7978845608 * (x + 0.044715 * x * x * x)))

# Kernel 1: Fused LayerNorm -> Linear
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _layernorm_linear_kernel(
    X, ln_w, ln_b, W, B, Y,
    M, N, K,
    stride_xm, stride_xk,
    stride_ym, stride_yn,
    stride_wn, stride_wk,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    mean = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    var = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    for k_offset in range(0, K, BLOCK_SIZE_K):
        k_chunk = k_offset + offs_k
        mask = (offs_m[:, None] < M) & (k_chunk[None, :] < K)
        x_chunk = tl.load(X + offs_m[:, None] * stride_xm + k_chunk[None, :] * stride_xk, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x_chunk, axis=1)
        var += tl.sum(x_chunk * x_chunk, axis=1)
    
    mean /= K
    var = var / K - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_offset in range(0, K, BLOCK_SIZE_K):
        k_chunk = k_offset + offs_k
        x_mask = (offs_m[:, None] < M) & (k_chunk[None, :] < K)
        w_mask = (k_chunk[None, :] < K) & (offs_n[:, None] < N)
        
        x_tile = tl.load(X + offs_m[:, None] * stride_xm + k_chunk[None, :] * stride_xk, mask=x_mask, other=0.0)
        
        ln_w_tile = tl.load(ln_w + k_chunk, mask=(k_chunk < K), other=0.0)
        ln_b_tile = tl.load(ln_b + k_chunk, mask=(k_chunk < K), other=0.0)
        x_norm = (x_tile.to(tl.float32) - mean[:, None]) * rstd[:, None]
        x_ln = x_norm * ln_w_tile[None, :] + ln_b_tile[None, :]
        x_ln = x_ln.to(X.dtype.element_ty)
        
        w_tile = tl.load(W + offs_n[:, None] * stride_wn + k_chunk[None, :] * stride_wk, mask=w_mask, other=0.0)
        
        acc += tl.dot(x_ln, tl.trans(w_tile))
        
    if B is not None:
        b_tile = tl.load(B + offs_n, mask=(offs_n < N), other=0.0)
        acc += b_tile[None, :]
    
    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)

# Kernel 2: Fused LayerNorm -> Linear -> GELU
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _layernorm_linear_gelu_kernel(
    X, ln_w, ln_b, W, B, Y,
    M, N, K,
    stride_xm, stride_xk,
    stride_ym, stride_yn,
    stride_wn, stride_wk,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    mean = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    var = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    for k_offset in range(0, K, BLOCK_SIZE_K):
        k_chunk = k_offset + offs_k
        mask = (offs_m[:, None] < M) & (k_chunk[None, :] < K)
        x_chunk = tl.load(X + offs_m[:, None] * stride_xm + k_chunk[None, :] * stride_xk, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x_chunk, axis=1)
        var += tl.sum(x_chunk * x_chunk, axis=1)
    
    mean /= K
    var = var / K - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_offset in range(0, K, BLOCK_SIZE_K):
        k_chunk = k_offset + offs_k
        x_mask = (offs_m[:, None] < M) & (k_chunk[None, :] < K)
        w_mask = (k_chunk[None, :] < K) & (offs_n[:, None] < N)
        
        x_tile = tl.load(X + offs_m[:, None] * stride_xm + k_chunk[None, :] * stride_xk, mask=x_mask, other=0.0)
        
        ln_w_tile = tl.load(ln_w + k_chunk, mask=(k_chunk < K), other=0.0)
        ln_b_tile = tl.load(ln_b + k_chunk, mask=(k_chunk < K), other=0.0)
        x_norm = (x_tile.to(tl.float32) - mean[:, None]) * rstd[:, None]
        x_ln = x_norm * ln_w_tile[None, :] + ln_b_tile[None, :]
        x_ln = x_ln.to(X.dtype.element_ty)
        
        w_tile = tl.load(W + offs_n[:, None] * stride_wn + k_chunk[None, :] * stride_wk, mask=w_mask, other=0.0)
        
        acc += tl.dot(x_ln, tl.trans(w_tile))
        
    if B is not None:
        b_tile = tl.load(B + offs_n, mask=(offs_n < N), other=0.0)
        acc += b_tile[None, :]
    
    y = gelu(acc)
    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, y, mask=y_mask)

# Kernel 3: Fused Linear -> Dropout -> Residual Add
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_dropout_add_kernel(
    X, W, B, RES, Y,
    p, seed,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_resm, stride_resn,
    stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_offset in range(0, K, BLOCK_SIZE_K):
        offs_k = k_offset + tl.arange(0, BLOCK_SIZE_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        
        x_tile = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, mask=x_mask, other=0.0)
        w_tile = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, mask=w_mask, other=0.0)
        
        acc += tl.dot(x_tile, tl.trans(w_tile))
    
    if B is not None:
        b_tile = tl.load(B + offs_n, mask=(offs_n < N), other=0.0)
        acc += b_tile[None, :]
    
    if p > 0.0:
        rand_offset = offs_m[:, None] * N + offs_n[None, :]
        rand = tl.rand(seed, rand_offset)
        keep_mask = rand > p
        acc = tl.where(keep_mask, acc / (1.0 - p), 0.0)
    
    res_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    res_tile = tl.load(RES + offs_m[:, None] * stride_resm + offs_n[None, :] * stride_resn, mask=res_mask, other=0.0)
    acc += res_tile
    
    y_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=res_mask)

# Kernel 4: Fused Flash Attention for Packed QKV
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=5, num_warps=2),
    ],
    key=['T', 'C', 'H'],
)
@triton.jit
def _flash_attn_qkv_packed_kernel(
    QKV, O,
    stride_qkv_b, stride_qkv_t, stride_qkv_c,
    stride_ob, stride_ot, stride_oc,
    B, H, T, C,
    D_HEAD: tl.constexpr,
    PADDED_D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, PADDED_D_HEAD], dtype=tl.float32)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d_padded = tl.arange(0, PADDED_D_HEAD)
    d_mask = offs_d_padded[None, :] < D_HEAD

    q_ptrs = QKV + (b * stride_qkv_b + offs_m[:, None] * stride_qkv_t + 
                   (h * D_HEAD + offs_d_padded[None, :]) * stride_qkv_c)
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < T) & d_mask, other=0.0)
    
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(0, (pid_m + 1) * BLOCK_M, BLOCK_N):
        offs_t_kv = start_n + offs_n
        
        k_ptrs = QKV + (b * stride_qkv_b + offs_t_kv[:, None] * stride_qkv_t + 
                       (C + h * D_HEAD + offs_d_padded[None, :]) * stride_qkv_c)
        k = tl.load(k_ptrs, mask=(offs_t_kv[:, None] < T) & d_mask, other=0.0)
        
        v_ptrs = QKV + (b * stride_qkv_b + offs_t_kv[:, None] * stride_qkv_t + 
                       (2 * C + h * D_HEAD + offs_d_padded[None, :]) * stride_qkv_c)
        v = tl.load(v_ptrs, mask=(offs_t_kv[:, None] < T) & d_mask, other=0.0)
        
        s_ij = tl.dot(q, tl.trans(k))
        s_ij *= (1.0 / math.sqrt(D_HEAD))
        
        causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        s_ij = tl.where(causal_mask, s_ij, -float("inf"))
        
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        s_ij_exp = tl.exp(s_ij - m_new[:, None])
        l_prev_rescaled = tl.exp(m_i - m_new) * l_i
        
        l_new = l_prev_rescaled + tl.sum(s_ij_exp, axis=1)
        
        p_ij = s_ij_exp / l_new[:, None]
        
        acc_rescaled = acc * (l_prev_rescaled / l_new)[:, None]
        acc = acc_rescaled + tl.dot(p_ij.to(v.dtype), v)
        
        l_i = l_new
        m_i = m_new

    offs_o_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    o_ptrs = O + (b * stride_ob + offs_o_m[:, None] * stride_ot + 
                 (h * D_HEAD + offs_d_padded[None, :]) * stride_oc)
    tl.store(o_ptrs, acc, mask=(offs_o_m[:, None] < T) & d_mask)

# --- Python Wrappers for Triton Kernels ---

def layernorm_linear(x, ln_w, ln_b, w, b, eps):
    x_2d = x.view(-1, x.shape[-1])
    M, K = x_2d.shape
    N = w.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    _layernorm_linear_kernel[grid](
        x_2d, ln_w, ln_b, w, b, y, M, N, K,
        x_2d.stride(0), x_2d.stride(1), y.stride(0), y.stride(1),
        w.stride(0), w.stride(1), eps=eps
    )
    return y.view(*x.shape[:-1], N)

def layernorm_linear_gelu(x, ln_w, ln_b, w, b, eps):
    x_2d = x.view(-1, x.shape[-1])
    M, K = x_2d.shape
    N = w.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    _layernorm_linear_gelu_kernel[grid](
        x_2d, ln_w, ln_b, w, b, y, M, N, K,
        x_2d.stride(0), x_2d.stride(1), y.stride(0), y.stride(1),
        w.stride(0), w.stride(1), eps=eps
    )
    return y.view(*x.shape[:-1], N)

def linear_dropout_add(x, w, b, res, p, seed):
    x_2d = x.view(-1, x.shape[-1])
    res_2d = res.view(-1, res.shape[-1])
    M, K = x_2d.shape
    N = w.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    _linear_dropout_add_kernel[grid](
        x_2d, w, b, res_2d, y, p, seed, M, N, K,
        x_2d.stride(0), x_2d.stride(1), w.stride(0), w.stride(1),
        res_2d.stride(0), res_2d.stride(1), y.stride(0), y.stride(1)
    )
    return y.view(*x.shape[:-1], N)

def flash_attention_qkv_packed(qkv, n_head):
    B, T, C3 = qkv.shape
    C = C3 // 3
    D_HEAD = C // n_head
    PADDED_D_HEAD = triton.next_power_of_2(D_HEAD)
    O = torch.empty((B, T, C), device=qkv.device, dtype=qkv.dtype)
    grid = (triton.cdiv(T, 128), B * n_head) # Use BLOCK_M=128 for grid calculation
    _flash_attn_qkv_packed_kernel[grid](
        qkv, O,
        qkv.stride(0), qkv.stride(1), qkv.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        B, n_head, T, C, 
        D_HEAD=D_HEAD,
        PADDED_D_HEAD=PADDED_D_HEAD
    )
    return O

class ModelNew(nn.Module):
    """ A Transformer block with a custom Flash Attention kernel. """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn_c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.attn_c_proj = nn.Linear(n_embd, n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp_c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.mlp_c_proj = nn.Linear(4 * n_embd, n_embd)
        self.resid_dropout_p = resid_pdrop
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        # --- 1. Fused Attention Block with Custom Flash Attention ---
        attn_residual = x

        # Fused LayerNorm -> QKV projection
        qkv = layernorm_linear(
            x, self.ln_1.weight, self.ln_1.bias,
            self.attn_c_attn.weight, self.attn_c_attn.bias, self.ln_1.eps
        )
        
        # Custom Flash Attention on packed QKV tensor
        attn_output = flash_attention_qkv_packed(qkv, self.n_head)

        # Fused output projection -> dropout -> residual add
        resid_p = self.resid_dropout_p if self.training else 0.0
        seed_attn = torch.empty((), dtype=torch.int64, device=x.device).random_().item() if resid_p > 0 else 0
        x = linear_dropout_add(
            attn_output, self.attn_c_proj.weight, self.attn_c_proj.bias,
            attn_residual, resid_p, seed_attn
        )
        
        # --- 2. Fused MLP Block ---
        mlp_residual = x

        # Fused LayerNorm -> Linear -> GELU
        mlp_gelu_out = layernorm_linear_gelu(
            x, self.ln_2.weight, self.ln_2.bias,
            self.mlp_c_fc.weight, self.mlp_c_fc.bias, self.ln_2.eps,
        )

        # Fused linear projection -> dropout -> residual add
        seed_mlp = torch.empty((), dtype=torch.int64, device=x.device).random_().item() if resid_p > 0 else 0
        x = linear_dropout_add(
            mlp_gelu_out, self.mlp_c_proj.weight, self.mlp_c_proj.bias,
            mlp_residual, resid_p, seed_mlp
        )
        
        return x

# Boilerplate for testing
batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.randn(batch_size, seq_len, n_embd, device='cuda').half()]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]