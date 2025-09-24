import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from itertools import repeat
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl

# --------------------------------------------------------
# Fused CUDA Kernels for Swin Transformer (Shift/Partition)
# --------------------------------------------------------

fused_swin_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Kernel to fuse Cyclic Shift and Window Partitioning
__global__ void fused_shift_window_partition_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int B, const int H, const int W, const int C,
    const int window_size,
    const int shift_size) {

    const int num_windows_h = H / window_size;
    const int num_windows_w = W / window_size;
    const int num_windows = num_windows_h * num_windows_w;
    const long long N_out = (long long)B * num_windows * window_size * window_size * C;

    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < N_out; idx += (long long)blockDim.x * gridDim.x) {
        // Decompose linear output index `idx` to multi-dimensional indices
        const int c = idx % C;
        long long temp = idx / C;
        const int win_w = temp % window_size;
        temp /= window_size;
        const int win_h = temp % window_size;
        temp /= window_size;
        const long long win_flat_idx_b = temp; // Flattened window index across all batches

        const int b = win_flat_idx_b / num_windows;
        const int win_flat_idx = win_flat_idx_b % num_windows;

        const int win_row_idx = win_flat_idx / num_windows_w;
        const int win_col_idx = win_flat_idx % num_windows_w;

        // Calculate coordinates in the conceptual 'shifted' tensor
        const int h_shifted = win_row_idx * window_size + win_h;
        const int w_shifted = win_col_idx * window_size + win_w;

        // Calculate source coordinates in the original unshifted tensor (reverse the roll)
        int h_src = h_shifted + shift_size;
        if (h_src >= H) h_src -= H;
        int w_src = w_shifted + shift_size;
        if (w_src >= W) w_src -= W;

        // Calculate linear index for input tensor (B, H, W, C)
        const long long in_idx = (long long)b * H * W * C +
                                 (long long)h_src * W * C +
                                 (long long)w_src * C + c;

        output[idx] = input[in_idx];
    }
}

// Kernel to fuse Window Reversing and Reverse Cyclic Shift
__global__ void fused_window_reverse_unshift_kernel(
    const float* __restrict__ input, // attn_windows
    float* __restrict__ output,      // x
    const int B, const int H, const int W, const int C,
    const int window_size,
    const int shift_size) {

    const int num_windows_h = H / window_size;
    const int num_windows_w = W / window_size;
    const int num_windows = num_windows_h * num_windows_w;
    const long long N_out = (long long)B * H * W * C;

    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < N_out; idx += (long long)blockDim.x * gridDim.x) {
        // Decompose linear output index `idx` into (b, h, w, c)
        const int c = idx % C;
        long long temp = idx / C;
        const int w = temp % W;
        temp /= W;
        const int h = temp % H;
        const int b = temp / H;

        // Calculate coordinates in the conceptual 'shifted' tensor (reverse the unshift)
        int h_shifted = h - shift_size;
        if (h_shifted < 0) h_shifted += H;
        int w_shifted = w - shift_size;
        if (w_shifted < 0) w_shifted += W;

        // Decompose shifted coordinates to find window and in-window coordinates
        const int win_row_idx = h_shifted / window_size;
        const int win_col_idx = w_shifted / window_size;
        const int win_h = h_shifted % window_size;
        const int win_w = w_shifted % window_size;

        // Calculate linear index for the source tensor (attn_windows)
        const int win_flat_idx = win_row_idx * num_windows_w + win_col_idx;
        const long long in_b_win = (long long)b * num_windows + win_flat_idx;
        const long long in_idx = in_b_win * window_size * window_size * C +
                                 (long long)win_h * window_size * C +
                                 (long long)win_w * C + c;

        output[idx] = input[in_idx];
    }
}

// Wrapper function for shift_window_partition
torch::Tensor fused_shift_window_partition(torch::Tensor x, int window_size, int shift_size) {
    const auto B = x.size(0);
    const auto H = x.size(1);
    const auto W = x.size(2);
    const auto C = x.size(3);

    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(torch::MemoryFormat::Contiguous), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");

    const auto num_windows_h = H / window_size;
    const auto num_windows_w = W / window_size;
    const auto num_windows = num_windows_h * num_windows_w;

    auto out = torch::empty({B * num_windows, window_size, window_size, C}, x.options());

    const long long N = out.numel();
    if (N == 0) return out;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    fused_shift_window_partition_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, H, W, C,
        window_size,
        shift_size
    );
    return out;
}

// Wrapper function for window_reverse_unshift
torch::Tensor fused_window_reverse_unshift(torch::Tensor windows, int window_size, int H, int W, int shift_size) {
    const auto B_win = windows.size(0);
    const auto C = windows.size(3);

    TORCH_CHECK(windows.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(windows.is_contiguous(torch::MemoryFormat::Contiguous), "Input tensor must be contiguous");
    TORCH_CHECK(windows.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(H > 0 && W > 0 && window_size > 0, "Dimensions must be positive");

    const int B = B_win / (H / window_size * W / window_size);
    auto out = torch::empty({B, H, W, C}, windows.options());

    const long long N = out.numel();
    if (N == 0) return out;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    fused_window_reverse_unshift_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        windows.data_ptr<float>(),
        out.data_ptr<float>(),
        B, H, W, C,
        window_size,
        shift_size
    );
    return out;
}
"""

fused_swin_ops_cpp_source = """
torch::Tensor fused_shift_window_partition(torch::Tensor x, int window_size, int shift_size);
torch::Tensor fused_window_reverse_unshift(torch::Tensor windows, int window_size, int H, int W, int shift_size);
"""

fused_swin_ops = load_inline(
    name="fused_swin_ops",
    cpp_sources=fused_swin_ops_cpp_source,
    cuda_sources=fused_swin_ops_source,
    functions=["fused_shift_window_partition", "fused_window_reverse_unshift"],
    verbose=True,
)


# --------------------------------------------------------
# Fused Triton Kernel for Window Attention
# --------------------------------------------------------
def _next_power_of_2(n):
    if n == 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n

@triton.jit
def _fused_attn_kernel(
    Q_ptr, K_ptr, V_ptr, RPB_ptr, MASK_ptr,
    LogitScale_ptr,
    O_ptr,
    q_stride_bh, q_stride_n, q_stride_d,
    k_stride_bh, k_stride_n, k_stride_d,
    v_stride_bh, v_stride_n, v_stride_d,
    rpb_stride_h, rpb_stride_n1, rpb_stride_n2,
    mask_stride_w, mask_stride_n1, mask_stride_n2,
    o_stride_bh, o_stride_n, o_stride_d,
    H, nW,
    N_CTX: tl.constexpr, D_HEAD: tl.constexpr,
    N_CTX_CEIL_POW2: tl.constexpr, D_HEAD_CEIL_POW2: tl.constexpr,
    HAS_MASK: tl.constexpr
):
    bh_idx = tl.program_id(0)

    q_start_ptr = Q_ptr + bh_idx * q_stride_bh
    k_start_ptr = K_ptr + bh_idx * k_stride_bh
    v_start_ptr = V_ptr + bh_idx * v_stride_bh
    o_start_ptr = O_ptr + bh_idx * o_stride_bh

    head_idx = bh_idx % H
    rpb_start_ptr = RPB_ptr + head_idx * rpb_stride_h

    offs_n = tl.arange(0, N_CTX_CEIL_POW2)
    offs_d = tl.arange(0, D_HEAD_CEIL_POW2)

    # Create masks for padded elements
    n_mask = offs_n < N_CTX
    d_mask = offs_d < D_HEAD
    
    # Load Q, K, V with masking to handle padding
    q_ptrs = q_start_ptr + offs_n[:, None] * q_stride_n + offs_d[None, :] * q_stride_d
    k_ptrs = k_start_ptr + offs_n[:, None] * k_stride_n + offs_d[None, :] * k_stride_d
    v_ptrs = v_start_ptr + offs_n[:, None] * v_stride_n + offs_d[None, :] * v_stride_d

    q = tl.load(q_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)
    v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0)

    # Cosine attention normalization
    q_norm_sq = tl.sum(q * q, axis=1)
    k_norm_sq = tl.sum(k * k, axis=1)
    q_norm = tl.sqrt(q_norm_sq)
    k_norm = tl.sqrt(k_norm_sq)
    
    # Add epsilon to prevent division by zero
    q = q / (q_norm[:, None] + 1e-6)
    k = k / (k_norm[:, None] + 1e-6)

    # Compute attention scores
    attn = tl.dot(q, tl.trans(k))

    # Apply logit scale
    logit_scale = tl.load(LogitScale_ptr + head_idx)
    attn = attn * logit_scale
    
    # Add relative position bias with masking
    offs_n1 = tl.arange(0, N_CTX_CEIL_POW2)
    offs_n2 = tl.arange(0, N_CTX_CEIL_POW2)
    n1_mask = offs_n1 < N_CTX
    n2_mask = offs_n2 < N_CTX
    
    rpb_ptrs = rpb_start_ptr + offs_n1[:, None] * rpb_stride_n1 + offs_n2[None, :] * rpb_stride_n2
    rpb = tl.load(rpb_ptrs, mask=n1_mask[:, None] & n2_mask[None, :], other=0.0)
    attn += rpb

    # Add attention mask if it exists
    if HAS_MASK:
        b_ = bh_idx // H
        w = b_ % nW
        mask_start_ptr = MASK_ptr + w * mask_stride_w
        mask_ptrs = mask_start_ptr + offs_n1[:, None] * mask_stride_n1 + offs_n2[None, :] * mask_stride_n2
        mask = tl.load(mask_ptrs, mask=n1_mask[:, None] & n2_mask[None, :], other=0.0)
        attn += mask

    # Mask out padding before softmax by setting scores to -inf
    attn = tl.where(n1_mask[:, None] & n2_mask[None, :], attn, -float('inf'))

    # Softmax
    row_max = tl.max(attn, axis=1)
    attn = attn - row_max[:, None]
    attn_exp = tl.exp(attn)
    row_sum = tl.sum(attn_exp, axis=1)
    softmax_attn = attn_exp / (row_sum[:, None] + 1e-6)

    # Compute output
    output = tl.dot(softmax_attn.to(v.dtype), v)

    # Store output with masking to write only valid elements
    o_ptrs = o_start_ptr + offs_n[:, None] * o_stride_n + offs_d[None, :] * o_stride_d
    tl.store(o_ptrs, output, mask=n_mask[:, None] & d_mask[None, :])


def fused_attention_forward(q, k, v, rpb, mask, logit_scale):
    B_, nH, N, D_HEAD = q.shape
    
    q_reshaped = q.reshape(B_ * nH, N, D_HEAD).contiguous()
    k_reshaped = k.reshape(B_ * nH, N, D_HEAD).contiguous()
    v_reshaped = v.reshape(B_ * nH, N, D_HEAD).contiguous()
    
    o = torch.empty_like(q_reshaped)
    
    grid = (B_ * nH, )
    
    nW = mask.shape[0] if mask is not None else 1

    N_CTX_CEIL_POW2 = _next_power_of_2(N)
    D_HEAD_CEIL_POW2 = _next_power_of_2(D_HEAD)

    _fused_attn_kernel[grid](
        q_reshaped, k_reshaped, v_reshaped, rpb, mask, logit_scale.reshape(nH), o,
        q_reshaped.stride(0), q_reshaped.stride(1), q_reshaped.stride(2),
        k_reshaped.stride(0), k_reshaped.stride(1), k_reshaped.stride(2),
        v_reshaped.stride(0), v_reshaped.stride(1), v_reshaped.stride(2),
        rpb.stride(0), rpb.stride(1), rpb.stride(2),
        mask.stride(0) if mask is not None else 0,
        mask.stride(1) if mask is not None else 0,
        mask.stride(2) if mask is not None else 0,
        o.stride(0), o.stride(1), o.stride(2),
        nH, nW,
        N_CTX=N, D_HEAD=D_HEAD,
        N_CTX_CEIL_POW2=N_CTX_CEIL_POW2, D_HEAD_CEIL_POW2=D_HEAD_CEIL_POW2,
        HAS_MASK=(mask is not None)
    )
    
    return o.reshape(B_, nH, N, D_HEAD)


# --------------------------------------------------------
# Swin Transformer V2 (Original + Modified Classes)
# --------------------------------------------------------

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttentionNew(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w], indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        # self.attn_drop = nn.Dropout(attn_drop) # NOTE: Dropout is not implemented in the fused kernel
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        logit_scale = torch.clamp(self.logit_scale, max=np.log(1. / 0.01)).exp()
        
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        
        # Use fused attention kernel
        attn_output = fused_attention_forward(q, k, v, relative_position_bias, mask, logit_scale)
        
        x = attn_output.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlockNew(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionNew(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x_4d = x.view(B, H, W, C)

        if self.shift_size > 0:
            x_windows = fused_swin_ops.fused_shift_window_partition(x_4d.contiguous(), self.window_size, self.shift_size)
        else:
            x_windows = window_partition(x_4d, self.window_size)
        
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        
        if self.shift_size > 0:
            x = fused_swin_ops.fused_window_reverse_unshift(attn_windows, self.window_size, H, W, self.shift_size)
        else:
            x = window_reverse(attn_windows, self.window_size, H, W)
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class BasicLayerNew(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlockNew(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class Model(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerNew(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

batch_size = 10
image_size = 224

def get_inputs():
    return [torch.randn(batch_size, 3, image_size, image_size).cuda()]

def get_init_inputs():
    return []