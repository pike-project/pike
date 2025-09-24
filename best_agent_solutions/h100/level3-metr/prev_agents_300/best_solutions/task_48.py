import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import triton
import triton.language as tl

# Kernel 1: Y_diag (fused diagonal computation) - Retained from previous solution
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_L': 32, 'BLOCK_SIZE_P': 32, 'BLOCK_SIZE_S': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_L': 64, 'BLOCK_SIZE_P': 32, 'BLOCK_SIZE_S': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_L': 32, 'BLOCK_SIZE_P': 64, 'BLOCK_SIZE_S': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_L': 64, 'BLOCK_SIZE_P': 64, 'BLOCK_SIZE_S': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_L': 16, 'BLOCK_SIZE_P': 16, 'BLOCK_SIZE_S': 16}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE_L': 64, 'BLOCK_SIZE_P': 64, 'BLOCK_SIZE_S': 32}, num_warps=8, num_stages=3),
    ],
    key=['BLOCK_LEN', 'D_HEAD', 'D_STATE'],
)
@triton.jit
def mamba_diag_kernel(
    C_ptr, B_ptr, A_cumsum_ptr, X_ptr, Y_ptr,
    BLOCK_LEN, D_HEAD, N_HEADS, N_CHUNKS,
    stride_c_b, stride_c_c, stride_c_l, stride_c_h, stride_c_n,
    stride_b_b, stride_b_c, stride_b_l, stride_b_h, stride_b_n,
    stride_a_b, stride_a_h, stride_a_c, stride_a_l,
    stride_x_b, stride_x_c, stride_x_l, stride_x_h, stride_x_p,
    stride_y_b, stride_y_c, stride_y_l, stride_y_h, stride_y_p,
    BLOCK_SIZE_L: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_S: tl.constexpr,
    D_STATE: tl.constexpr,
):
    bhc = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_p = tl.program_id(2)

    c = bhc % N_CHUNKS
    bh = bhc // N_CHUNKS
    h = bh % N_HEADS
    b = bh // N_HEADS

    c_block_ptr = C_ptr + b * stride_c_b + c * stride_c_c + h * stride_c_h
    b_block_ptr = B_ptr + b * stride_b_b + c * stride_b_c + h * stride_b_h
    a_block_ptr = A_cumsum_ptr + b * stride_a_b + h * stride_a_h + c * stride_a_c
    x_block_ptr = X_ptr + b * stride_x_b + c * stride_x_c + h * stride_x_h
    y_block_ptr = Y_ptr + b * stride_y_b + c * stride_y_c + h * stride_y_h

    offs_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    offs_p = pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    
    acc = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_P), dtype=tl.float32)

    mask_l = offs_l < BLOCK_LEN
    mask_p = offs_p < D_HEAD

    a_l_ptrs = a_block_ptr + offs_l * stride_a_l
    a_l = tl.load(a_l_ptrs, mask=mask_l, other=0.0)

    offs_n = tl.arange(0, D_STATE)
    c_ptrs = c_block_ptr + offs_l[:, None] * stride_c_l + offs_n[None, :] * stride_c_n
    c_tile = tl.load(c_ptrs, mask=mask_l[:, None])

    for s_start in range(0, BLOCK_LEN, BLOCK_SIZE_S):
        offs_s = s_start + tl.arange(0, BLOCK_SIZE_S)
        mask_s = offs_s < BLOCK_LEN
        
        a_s_ptrs = a_block_ptr + offs_s * stride_a_l
        a_s = tl.load(a_s_ptrs, mask=mask_s, other=0.0)

        l_tile = tl.exp(a_l[:, None] - a_s[None, :])
        l_mask = offs_l[:, None] >= offs_s[None, :]
        l_tile = tl.where(l_mask, l_tile, 0.0)

        b_ptrs = b_block_ptr + offs_s[:, None] * stride_b_l + offs_n[None, :] * stride_b_n
        b_tile = tl.load(b_ptrs, mask=mask_s[:, None])

        k_tile = tl.dot(c_tile, tl.trans(b_tile), allow_tf32=False)
        m_tile = l_tile * k_tile

        x_ptrs = x_block_ptr + offs_s[:, None] * stride_x_l + offs_p[None, :] * stride_x_p
        x_tile = tl.load(x_ptrs, mask=mask_s[:, None] & mask_p[None, :])

        acc += tl.dot(m_tile.to(x_tile.dtype), x_tile, allow_tf32=False)

    y_ptrs = y_block_ptr + offs_l[:, None] * stride_y_l + offs_p[None, :] * stride_y_p
    y_mask = mask_l[:, None] & mask_p[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)

# Kernel 2: Intra-chunk states (fused decay + outer product) - Retained from previous solution
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_P': 32, 'BLOCK_SIZE_N': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_P': 64, 'BLOCK_SIZE_N': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_P': 16, 'BLOCK_SIZE_N': 16}, num_warps=2),
    ],
    key=['D_HEAD', 'D_STATE'],
)
@triton.jit
def compute_states_kernel(
    B_ptr, A_cumsum_ptr, X_ptr, States_ptr,
    BLOCK_LEN, D_HEAD, D_STATE, N_HEADS, N_CHUNKS,
    stride_b_b, stride_b_c, stride_b_l, stride_b_h, stride_b_n,
    stride_ac_b, stride_ac_h, stride_ac_c, stride_ac_l,
    stride_x_b, stride_x_c, stride_x_l, stride_x_h, stride_x_p,
    stride_s_b, stride_s_c, stride_s_h, stride_s_p, stride_s_n,
    BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    bhc = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    c = bhc % N_CHUNKS
    bh = bhc // N_CHUNKS
    h = bh % N_HEADS
    b = bh // N_HEADS
    
    b_block_ptr = B_ptr + b*stride_b_b + c*stride_b_c + h*stride_b_h
    ac_block_ptr = A_cumsum_ptr + b*stride_ac_b + h*stride_ac_h + c*stride_ac_c
    x_block_ptr = X_ptr + b*stride_x_b + c*stride_x_c + h*stride_x_h
    s_block_ptr = States_ptr + b*stride_s_b + c*stride_s_c + h*stride_s_h
    
    offs_p = pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    acc = tl.zeros((BLOCK_SIZE_P, BLOCK_SIZE_N), dtype=tl.float32)
    
    a_end = tl.load(ac_block_ptr + (BLOCK_LEN - 1) * stride_ac_l)
    
    for l in range(0, BLOCK_LEN):
        a_l = tl.load(ac_block_ptr + l * stride_ac_l)
        decay = tl.exp(a_end - a_l)
        
        mask_n = offs_n < D_STATE
        b_l_ptrs = b_block_ptr + l * stride_b_l + offs_n * stride_b_n
        b_l = tl.load(b_l_ptrs, mask=mask_n, other=0.0)
        
        mask_p = offs_p < D_HEAD
        x_l_ptrs = x_block_ptr + l * stride_x_l + offs_p * stride_x_p
        x_l = tl.load(x_l_ptrs, mask=mask_p, other=0.0)
        
        acc += decay * x_l[:, None] * b_l[None, :]
        
    s_ptrs = s_block_ptr + offs_p[:, None] * stride_s_p + offs_n[None, :] * stride_s_n
    s_mask = (offs_p[:, None] < D_HEAD) & (offs_n[None, :] < D_STATE)
    tl.store(s_ptrs, acc, mask=s_mask)

# Kernel 3: Y_off (fused decay + dot product) - Retained from previous solution
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_L': 32, 'BLOCK_SIZE_P': 32, 'BLOCK_SIZE_N': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_L': 64, 'BLOCK_SIZE_P': 64, 'BLOCK_SIZE_N': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_L': 16, 'BLOCK_SIZE_P': 16, 'BLOCK_SIZE_N': 16}, num_warps=2, num_stages=3),
    ],
    key=['BLOCK_LEN', 'D_HEAD', 'D_STATE'],
)
@triton.jit
def compute_y_off_kernel(
    C_ptr, States_ptr, A_cumsum_ptr, Y_off_ptr,
    BLOCK_LEN, D_HEAD, D_STATE, N_HEADS, N_CHUNKS,
    stride_c_b, stride_c_c, stride_c_l, stride_c_h, stride_c_n,
    stride_s_b, stride_s_c, stride_s_h, stride_s_p, stride_s_n,
    stride_ac_b, stride_ac_h, stride_ac_c, stride_ac_l,
    stride_y_b, stride_y_c, stride_y_l, stride_y_h, stride_y_p,
    BLOCK_SIZE_L: tl.constexpr, BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    bhc = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_p = tl.program_id(2)

    c = bhc % N_CHUNKS
    bh = bhc // N_CHUNKS
    h = bh % N_HEADS
    b = bh // N_HEADS
    
    c_base_ptr = C_ptr + b*stride_c_b + c*stride_c_c + h*stride_c_h
    s_base_ptr = States_ptr + b*stride_s_b + c*stride_s_c + h*stride_s_h
    ac_base_ptr = A_cumsum_ptr + b*stride_ac_b + h*stride_ac_h + c*stride_ac_c
    y_base_ptr = Y_off_ptr + b*stride_y_b + c*stride_y_c + h*stride_y_h
    
    offs_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    offs_p = pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    
    acc = tl.zeros((BLOCK_SIZE_L, BLOCK_SIZE_P), dtype=tl.float32)
    
    mask_l = offs_l < BLOCK_LEN
    mask_p = offs_p < D_HEAD

    for n_iter in range(0, tl.cdiv(D_STATE, BLOCK_SIZE_N)):
        offs_n = n_iter * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < D_STATE
        
        c_ptrs = c_base_ptr + offs_l[:, None] * stride_c_l + offs_n[None, :] * stride_c_n
        c_tile = tl.load(c_ptrs, mask=mask_l[:, None] & mask_n[None, :], other=0.0)
        
        s_ptrs = s_base_ptr + offs_p[:, None] * stride_s_p + offs_n[None, :] * stride_s_n
        s_tile = tl.load(s_ptrs, mask=mask_p[:, None] & mask_n[None, :], other=0.0)
        
        acc += tl.dot(c_tile, tl.trans(s_tile), allow_tf32=False)
    
    ac_ptrs = ac_base_ptr + offs_l * stride_ac_l
    decay = tl.exp(tl.load(ac_ptrs, mask=mask_l, other=0.0))
    
    y_ptrs = y_base_ptr + offs_l[:, None] * stride_y_l + offs_p[None, :] * stride_y_p
    y_mask = mask_l[:, None] & mask_p[None, :]
    tl.store(y_ptrs, decay[:, None] * acc, mask=y_mask)

# Kernel 4: Inter-chunk recurrence - NEW KERNEL
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_P': 32, 'BLOCK_SIZE_N': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_P': 64, 'BLOCK_SIZE_N': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_P': 16, 'BLOCK_SIZE_N': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE_P': 32, 'BLOCK_SIZE_N': 32}, num_warps=4),
    ],
    key=['D_HEAD', 'D_STATE'],
)
@triton.jit
def inter_chunk_recurrence_kernel(
    states_intra_chunk_ptr, chunk_sums_ptr, initial_states_ptr, states_inter_chunk_ptr,
    N_CHUNKS, D_HEAD, D_STATE, N_HEADS,
    stride_intra_s_b, stride_intra_s_c, stride_intra_s_h, stride_intra_s_p, stride_intra_s_n,
    stride_cs_b, stride_cs_h, stride_cs_c,
    stride_init_s_b, stride_init_s_h, stride_init_s_p, stride_init_s_n,
    stride_inter_s_b, stride_inter_s_c, stride_inter_s_h, stride_inter_s_p, stride_inter_s_n,
    BLOCK_SIZE_P: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    b = pid_bh // N_HEADS
    h = pid_bh % N_HEADS

    p_intra_base = states_intra_chunk_ptr + b * stride_intra_s_b + h * stride_intra_s_h
    p_cs_base = chunk_sums_ptr + b * stride_cs_b + h * stride_cs_h
    p_init_base = initial_states_ptr + b * stride_init_s_b + h * stride_init_s_h
    p_inter_base = states_inter_chunk_ptr + b * stride_inter_s_b + h * stride_inter_s_h

    offs_p = pid_p * BLOCK_SIZE_P + tl.arange(0, BLOCK_SIZE_P)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask_p = offs_p < D_HEAD
    mask_n = offs_n < D_STATE
    mask_pn = mask_p[:, None] & mask_n[None, :]

    init_ptrs = p_init_base + offs_p[:, None] * stride_init_s_p + offs_n[None, :] * stride_init_s_n
    current_state_tile = tl.load(init_ptrs, mask=mask_pn, other=0.0)

    for c in range(N_CHUNKS):
        inter_ptrs = p_inter_base + c * stride_inter_s_c + offs_p[:, None] * stride_inter_s_p + offs_n[None, :] * stride_inter_s_n
        tl.store(inter_ptrs, current_state_tile, mask=mask_pn)
        
        intra_ptrs = p_intra_base + c * stride_intra_s_c + offs_p[:, None] * stride_intra_s_p + offs_n[None, :] * stride_intra_s_n
        intra_state_tile = tl.load(intra_ptrs, mask=mask_pn, other=0.0)
        
        decay = tl.load(p_cs_base + c * stride_cs_c)
        
        current_state_tile = tl.exp(decay) * current_state_tile + intra_state_tile


# Wrapper functions for Triton kernels
def mamba_selective_scan_diag_triton(A_cumsum, B_blocks, C_blocks, X_blocks):
    batch_size, n_heads, n_chunks, block_len = A_cumsum.shape
    _, _, _, _, d_state = B_blocks.shape
    _, _, _, _, d_head = X_blocks.shape
    Y_diag = torch.empty_like(X_blocks)

    grid = lambda META: (
        batch_size * n_heads * n_chunks,
        triton.cdiv(block_len, META['BLOCK_SIZE_L']),
        triton.cdiv(d_head, META['BLOCK_SIZE_P'])
    )
    
    mamba_diag_kernel[grid](
        C_blocks, B_blocks, A_cumsum, X_blocks, Y_diag,
        block_len, d_head, n_heads, n_chunks,
        C_blocks.stride(0), C_blocks.stride(1), C_blocks.stride(2), C_blocks.stride(3), C_blocks.stride(4),
        B_blocks.stride(0), B_blocks.stride(1), B_blocks.stride(2), B_blocks.stride(3), B_blocks.stride(4),
        A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
        X_blocks.stride(0), X_blocks.stride(1), X_blocks.stride(2), X_blocks.stride(3), X_blocks.stride(4),
        Y_diag.stride(0), Y_diag.stride(1), Y_diag.stride(2), Y_diag.stride(3), Y_diag.stride(4),
        D_STATE=d_state,
    )
    return Y_diag

def compute_states_triton(B_blocks, A_cumsum, X_blocks):
    batch_size, n_chunks, _, n_heads, d_state = B_blocks.shape
    _, _, block_len, _, d_head = X_blocks.shape
    states = torch.empty((batch_size, n_chunks, n_heads, d_head, d_state), device=X_blocks.device, dtype=X_blocks.dtype)
    
    grid = lambda META: (
        batch_size * n_chunks * n_heads,
        triton.cdiv(d_head, META['BLOCK_SIZE_P']),
        triton.cdiv(d_state, META['BLOCK_SIZE_N'])
    )
    
    compute_states_kernel[grid](
        B_blocks, A_cumsum, X_blocks, states,
        block_len, d_head, d_state, n_heads, n_chunks,
        B_blocks.stride(0), B_blocks.stride(1), B_blocks.stride(2), B_blocks.stride(3), B_blocks.stride(4),
        A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
        X_blocks.stride(0), X_blocks.stride(1), X_blocks.stride(2), X_blocks.stride(3), X_blocks.stride(4),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
    )
    return states

def compute_y_off_triton(C_blocks, states_inter_chunk, A_cumsum):
    batch_size, n_chunks, block_len, n_heads, d_state = C_blocks.shape
    _, _, _, d_head, _ = states_inter_chunk.shape
    Y_off = torch.empty((batch_size, n_chunks, block_len, n_heads, d_head), device=C_blocks.device, dtype=C_blocks.dtype)

    grid = lambda META: (
        batch_size * n_chunks * n_heads,
        triton.cdiv(block_len, META['BLOCK_SIZE_L']),
        triton.cdiv(d_head, META['BLOCK_SIZE_P'])
    )
    
    compute_y_off_kernel[grid](
        C_blocks, states_inter_chunk, A_cumsum, Y_off,
        block_len, d_head, d_state, n_heads, n_chunks,
        C_blocks.stride(0), C_blocks.stride(1), C_blocks.stride(2), C_blocks.stride(3), C_blocks.stride(4),
        states_inter_chunk.stride(0), states_inter_chunk.stride(1), states_inter_chunk.stride(2), states_inter_chunk.stride(3), states_inter_chunk.stride(4),
        A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
        Y_off.stride(0), Y_off.stride(1), Y_off.stride(2), Y_off.stride(3), Y_off.stride(4),
    )
    return Y_off

def inter_chunk_recurrence_triton(states_intra_chunk, chunk_sums, initial_states):
    batch_size, n_chunks, n_heads, d_head, d_state = states_intra_chunk.shape
    states_inter_chunk = torch.empty_like(states_intra_chunk)

    grid = lambda META: (
        batch_size * n_heads,
        triton.cdiv(d_head, META['BLOCK_SIZE_P']),
        triton.cdiv(d_state, META['BLOCK_SIZE_N']),
    )

    inter_chunk_recurrence_kernel[grid](
        states_intra_chunk, chunk_sums, initial_states, states_inter_chunk,
        n_chunks, d_head, d_state, n_heads,
        states_intra_chunk.stride(0), states_intra_chunk.stride(1), states_intra_chunk.stride(2), states_intra_chunk.stride(3), states_intra_chunk.stride(4),
        chunk_sums.stride(0), chunk_sums.stride(1), chunk_sums.stride(2),
        initial_states.stride(0), initial_states.stride(1), initial_states.stride(2), initial_states.stride(3),
        states_inter_chunk.stride(0), states_inter_chunk.stride(1), states_inter_chunk.stride(2), states_inter_chunk.stride(3), states_inter_chunk.stride(4),
    )
    return states_inter_chunk


class Model(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(Model, self).__init__()
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    def forward(self, X, initial_states=None):
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # 1. Compute diagonal block outputs using fused Triton kernel
        Y_diag = mamba_selective_scan_diag_triton(A_cumsum, B_blocks, C_blocks, X_blocks)
        
        # 2. Compute intra-chunk states using fused Triton kernel
        states_intra_chunk = compute_states_triton(B_blocks, A_cumsum, X_blocks)
        
        # 3. Compute inter-chunk recurrence with a dedicated Triton kernel
        if initial_states is None:
            initial_states = torch.zeros(self.batch_size, self.n_heads, self.d_head, self.d_state, device=X.device, dtype=X.dtype)
        
        chunk_sums = A_cumsum[:, :, :, -1]
        states_inter_chunk = inter_chunk_recurrence_triton(states_intra_chunk, chunk_sums, initial_states)
            
        # 4. Compute state-to-output conversion (off-diagonal) using fused Triton kernel
        Y_off = compute_y_off_triton(C_blocks, states_inter_chunk, A_cumsum)
        
        # Combine diagonal and off-diagonal terms
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        
        return Y


# Test parameters
batch_size = 16
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    X = torch.randn(batch_size, seq_length, n_heads, d_head, dtype=torch.float32, device="cuda")
    return [X]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]