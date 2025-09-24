import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_qkv_bmm_bias_kernel(
    X_ptr, W_ptr, B_ptr, Q_out_ptr, K_out_ptr, V_out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_qm, stride_qk,
    stride_km, stride_kk,
    stride_vm, stride_vk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < 3*K) & (offs_k[:, None] < K), other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b_ptrs = B_ptr + offs_n
    bias = tl.load(b_ptrs, mask=offs_n < 3 * K, other=0.0)
    acc += bias[None, :]

    q_offs_n = offs_n
    k_offs_n = offs_n - K
    v_offs_n = offs_n - 2 * K

    q_ptrs = Q_out_ptr + offs_m[:, None] * stride_qm + q_offs_n[None, :] * stride_qk
    k_ptrs = K_out_ptr + offs_m[:, None] * stride_km + k_offs_n[None, :] * stride_kk
    v_ptrs = V_out_ptr + offs_m[:, None] * stride_vm + v_offs_n[None, :] * stride_vk

    row_mask = offs_m[:, None] < M
    q_mask = row_mask & (offs_n[None, :] < K)
    k_mask = row_mask & (offs_n[None, :] >= K) & (offs_n[None, :] < 2 * K)
    v_mask = row_mask & (offs_n[None, :] >= 2 * K) & (offs_n[None, :] < 3 * K)

    tl.store(q_ptrs, acc.to(Q_out_ptr.dtype.element_ty), mask=q_mask)
    tl.store(k_ptrs, acc.to(K_out_ptr.dtype.element_ty), mask=k_mask)
    tl.store(v_ptrs, acc.to(V_out_ptr.dtype.element_ty), mask=v_mask)

def fused_qkv_projection(x, weight, bias):
    M, K = x.shape
    q = torch.empty_like(x)
    k = torch.empty_like(x)
    v = torch.empty_like(x)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(3 * K, META['BLOCK_N']))
    _fused_qkv_bmm_bias_kernel[grid](
        x, weight, bias, q, k, v,
        M, 3 * K, K,
        x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1)
    )
    return q, k, v

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
    ],
    key=['N_CTX', 'D_HEAD'],
)
@triton.jit
def _flash_attention_forward_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)
    off_h = tl.program_id(2)

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    
    Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(N_CTX, D_HEAD), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, D_HEAD), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=(D_HEAD, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(D_HEAD, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=(N_CTX, D_HEAD), strides=(stride_vn, stride_vk), offsets=(0, 0), block_shape=(BLOCK_N, D_HEAD), order=(1, 0))

    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0,))
    scale = (D_HEAD ** -0.5)
    q = (q * scale).to(q.dtype)

    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(1,))
        v = tl.load(V_block_ptr, boundary_check=(0,))
        
        s = tl.dot(q, k, allow_tf32=False)

        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(s - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=False)
        
        l_i = l_i_new
        m_i = m_i_new

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]

    o_offset = off_z * stride_oz + off_h * stride_oh
    O_block_ptr = tl.make_block_ptr(base=O + o_offset, shape=(N_CTX, D_HEAD), strides=(stride_om, stride_ok), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, D_HEAD), order=(1, 0))
    tl.store(O_block_ptr, acc.to(O.dtype.element_ty), boundary_check=(0,))

def flash_attention_forward(q, k, v):
    Z, H, N_CTX, D_HEAD = q.shape
    o = torch.empty_like(q, dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(N_CTX, meta['BLOCK_M']), Z, H)
    _flash_attention_forward_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H, N_CTX, D_HEAD=D_HEAD
    )
    return o

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_matmul_bias_add_kernel(
    A_ptr, B_ptr, BIAS_ptr, RESIDUAL_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_resm, stride_resn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias_ptrs = BIAS_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    res_ptrs = RESIDUAL_ptr + (offs_m[:, None] * stride_resm + offs_n[None, :] * stride_resn)
    res_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    residual = tl.load(res_ptrs, mask=res_mask, other=0.0)
    acc += residual

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)

def fused_matmul_bias_add(a, b_weight, bias, residual):
    M, K = a.shape
    N, _ = b_weight.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    b = b_weight.T.contiguous()
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    _fused_matmul_bias_add_kernel[grid](
        a, b, bias, residual, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        residual.stride(0), residual.stride(1),
        c.stride(0), c.stride(1)
    )
    return c

@triton.jit
def _layer_norm_fwd_kernel(
    X, Y, W, B,
    stride_x_row, stride_y_row,
    N_COLS, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X_row_ptr, Y_row_ptr = X + row * stride_x_row, Y + row * stride_y_row
    
    mean, var = 0., 0.
    for off in range(0, N_COLS, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.).to(tl.float32)
        mean += tl.sum(x)
        var += tl.sum(x * x)
    mean /= N_COLS
    var = var / N_COLS - mean * mean
    rstd = 1. / tl.sqrt(var + eps)

    for off in range(0, N_COLS, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        x = tl.load(X_row_ptr + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        output = (x - mean) * rstd * w + b
        tl.store(Y_row_ptr + cols, output.to(Y.dtype.element_ty), mask=mask)

def layer_norm(x, weight, bias, eps):
    output = torch.empty_like(x)
    x_reshaped, output_reshaped = x.reshape(-1, x.shape[-1]), output.reshape(-1, x.shape[-1])
    n_rows, n_cols = x_reshaped.shape
    grid = (n_rows,)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    _layer_norm_fwd_kernel[grid](
        x_reshaped, output_reshaped, weight, bias,
        x_reshaped.stride(0), output_reshaped.stride(0),
        n_cols, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    return output

class Model(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Model, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.attn.to(torch.float16).to('cuda')
        self.norm.to(torch.float16).to('cuda')

    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W
        
        x_flat = x.permute(0, 2, 3, 1).reshape(B * seq_len, C)
        x_flat_fp16 = x_flat.to(torch.float16)
        
        q_flat, k_flat, v_flat = fused_qkv_projection(
            x_flat_fp16, self.attn.in_proj_weight, self.attn.in_proj_bias
        )
        
        q = q_flat.view(B, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k_flat.view(B, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v_flat.view(B, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn_output = flash_attention_forward(q, k, v)
        
        attn_output_flat = attn_output.permute(0, 2, 1, 3).contiguous().view(B * seq_len, self.embed_dim)
        
        pre_norm_tensor = fused_matmul_bias_add(
            attn_output_flat, self.attn.out_proj.weight, self.attn.out_proj.bias, x_flat_fp16
        )

        x_out_fp16 = layer_norm(
            pre_norm_tensor, self.norm.weight, self.norm.bias, self.norm.eps
        )
        
        x_out = x_out_fp16.view(B, H, W, C).permute(0, 3, 1, 2)
        return x_out.to(torch.float32)

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    torch.manual_seed(0)
    return [torch.randn(batch_size, num_channels, image_height, image_width, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [embed_dim, num_heads]