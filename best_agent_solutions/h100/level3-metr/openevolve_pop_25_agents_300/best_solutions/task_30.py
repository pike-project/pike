# EVOLVE-BLOCK-START
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import numpy as np
import collections
from itertools import repeat
import math

# --- KERNEL 1: Fused LayerNorm + Residual Add ---
layernorm_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int LN_BLOCK_SIZE = 256;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (tid < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void layernorm_add_fused_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ shortcut,
    const float* __restrict__ gamma, 
    const float* __restrict__ beta, 
    float* __restrict__ y, 
    float epsilon, 
    int N,
    int C
) {
    int row_idx = blockIdx.x;
    if (row_idx >= N) {
        return;
    }

    extern __shared__ float shared_data[];
    
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    const float* x_row = x + row_idx * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float val = x_row[i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    float total_sum = block_reduce_sum(local_sum, shared_data);
    __syncthreads();
    float total_sum_sq = block_reduce_sum(local_sum_sq, shared_data);

    if (threadIdx.x == 0) {
        float mean = total_sum / C;
        float var = total_sum_sq / C - mean * mean;
        shared_data[0] = mean;
        shared_data[1] = rsqrtf(var + epsilon);
    }
    
    __syncthreads();

    float mean = shared_data[0];
    float inv_std = shared_data[1];

    const float* shortcut_row = shortcut + row_idx * C;
    float* y_row = y + row_idx * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        y_row[i] = (x_row[i] - mean) * inv_std * gamma[i] + beta[i] + shortcut_row[i];
    }
}

torch::Tensor layernorm_add_cuda(
    const torch::Tensor& x,
    const torch::Tensor& shortcut,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    double epsilon) {

    const auto last_dim = x.dim() - 1;
    const int C = x.size(last_dim);
    const int N = x.numel() / C;

    auto out = torch::empty_like(x);

    const int block_size = LN_BLOCK_SIZE;
    const int num_blocks = N;
    const int shared_mem_size = (block_size / 32 + 2) * sizeof(float);

    layernorm_add_fused_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        shortcut.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<float>(epsilon),
        N, C
    );
    return out;
}
"""

layernorm_add_cpp_source = """
torch::Tensor layernorm_add_cuda(
    const torch::Tensor& x,
    const torch::Tensor& shortcut,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    double epsilon);
"""

fused_layernorm_add_op = load_inline(
    name="fused_layernorm_add_op",
    cpp_sources=layernorm_add_cpp_source,
    cuda_sources=layernorm_add_source,
    functions=["layernorm_add_cuda"],
    verbose=True,
)


# --- KERNEL 2: Fused Linear + GELU ---
fused_mlp_cuda_source = """
#include <torch/extension.h>
#include <ATen/Functions.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

__global__ void add_bias_gelu_kernel(float* matrix, const float* bias, long M, long N) {
    long row = blockIdx.y * blockDim.y + threadIdx.y;
    long col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        long idx = row * N + col;
        float val = matrix[idx] + bias[col];
        matrix[idx] = gelu_approx(val);
    }
}

torch::Tensor fused_linear_gelu(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto output = at::matmul(input, weight.t());
    
    auto output_sizes = output.sizes();
    long N = output_sizes.back();
    long M = output.numel() / N;
    
    auto output_reshaped = output.reshape({M, N});

    const int threads_x = 16;
    const int threads_y = 16;
    const dim3 threads(threads_x, threads_y);
    const dim3 blocks(
        (N + threads_x - 1) / threads_x,
        (M + threads_y - 1) / threads_y
    );
    
    add_bias_gelu_kernel<<<blocks, threads>>>(output_reshaped.data_ptr<float>(), bias.data_ptr<float>(), M, N);
    return output;
}
"""

fused_mlp_cpp_source = "torch::Tensor fused_linear_gelu(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"

fused_mlp = load_inline(
    name="fused_mlp",
    cpp_sources=fused_mlp_cpp_source,
    cuda_sources=fused_mlp_cuda_source,
    functions=["fused_linear_gelu"],
    verbose=True,
)

# --- KERNEL 3: Fused Bias + Mask + Softmax ---
fused_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void fused_bias_mask_softmax_kernel(
    float* __restrict__ scores, // in-place, (B_, H, N, N)
    const float* __restrict__ bias,   // (H, N, N)
    const float* __restrict__ mask,   // (nW, N, N) or nullptr
    int B_, int H, int N,
    int nW // num_windows for mask, 0 if no mask
) {
    const int rows_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    const int global_row_idx = blockIdx.x * rows_per_block + warp_id;
    if (global_row_idx >= B_ * H * N) return;

    const int mat_idx = global_row_idx / N;
    const int row_idx = global_row_idx % N;
    const int batch_idx = mat_idx / H;
    const int head_idx = mat_idx % H;
    
    float* scores_row = scores + mat_idx * N * N + row_idx * N;
    const float* bias_row = bias + head_idx * N * N + row_idx * N;
    
    float max_val = -FLT_MAX;
    for (int j = lane_id; j < N; j += 32) {
        float val = scores_row[j] + bias_row[j];
        if (nW > 0) {
            const float* mask_row = mask + (batch_idx % nW) * N * N + row_idx * N;
            val += mask_row[j];
        }
        scores_row[j] = val;
        max_val = max(max_val, val);
    }
    
    max_val = warp_reduce_max(max_val);
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);

    float sum_val = 0.0f;
    for (int j = lane_id; j < N; j += 32) {
        float val = expf(scores_row[j] - max_val);
        scores_row[j] = val;
        sum_val += val;
    }

    sum_val = warp_reduce_sum(sum_val);
    sum_val = __shfl_sync(0xFFFFFFFF, sum_val, 0);

    float inv_sum = 1.0f / (sum_val + 1e-12f);
    for (int j = lane_id; j < N; j += 32) {
        scores_row[j] *= inv_sum;
    }
}

torch::Tensor fused_softmax_inplace(
    torch::Tensor scores,
    torch::Tensor bias,
    torch::Tensor mask
) {
    const auto B_ = scores.size(0);
    const auto H = scores.size(1);
    const auto N = scores.size(2);
    const int nW = mask.numel() > 0 ? mask.size(0) : 0;

    const int block_size = 256;
    const int rows_per_block = block_size / 32;
    const int num_blocks = (B_ * H * N + rows_per_block - 1) / rows_per_block;

    fused_bias_mask_softmax_kernel<<<num_blocks, block_size>>>(
        scores.data_ptr<float>(),
        bias.data_ptr<float>(),
        nW > 0 ? mask.data_ptr<float>() : nullptr,
        B_, H, N, nW
    );
    return scores;
}
"""

fused_softmax_cpp_source = "torch::Tensor fused_softmax_inplace(torch::Tensor scores, torch::Tensor bias, torch::Tensor mask);"
fused_softmax_op = load_inline(
    name="fused_softmax_op",
    cpp_sources=fused_softmax_cpp_source,
    cuda_sources=fused_softmax_source,
    functions=["fused_softmax_inplace"],
    verbose=True,
)


# --- KERNEL 4: Fused Patch Merging ---
patch_merging_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void patch_merging_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int H, int W, int C
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total_output_elements = (long)B * (H / 2) * (W / 2) * (4 * C);
    if (idx >= total_output_elements) return;

    // Inverse mapping from output index to input index
    int c4 = idx % (4 * C);
    long temp = idx / (4 * C);
    int l_out = temp % ((H / 2) * (W / 2));
    int b = temp / ((H / 2) * (W / 2));

    int w_out = l_out % (W / 2);
    int h_out = l_out / (W / 2);

    int c_quadrant = c4 / C;
    int c_in = c4 % C;
    
    int h_off, w_off;
    if (c_quadrant == 0) { h_off = 0; w_off = 0; }
    else if (c_quadrant == 1) { h_off = 1; w_off = 0; }
    else if (c_quadrant == 2) { h_off = 0; w_off = 1; }
    else { h_off = 1; w_off = 1; }

    int h_in = 2 * h_out + h_off;
    int w_in = 2 * w_out + w_off;

    long in_idx = (long)b * H * W * C + (long)h_in * W * C + (long)w_in * C + c_in;
    
    output[idx] = input[in_idx];
}

torch::Tensor patch_merging_forward_cuda(torch::Tensor x) {
    const auto B = x.size(0);
    const auto H = x.size(1);
    const auto W = x.size(2);
    const auto C = x.size(3);

    auto output = torch::empty({B, (H / 2) * (W / 2), 4 * C}, x.options());
    const long total_elements = output.numel();
    const int block_size = 1024;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    patch_merging_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), B, H, W, C);
        
    return output;
}
"""
patch_merging_cpp_source = "torch::Tensor patch_merging_forward_cuda(torch::Tensor x);"
patch_merging_op = load_inline(
    name="patch_merging_op",
    cpp_sources=patch_merging_cpp_source,
    cuda_sources=patch_merging_source,
    functions=["patch_merging_forward_cuda"],
    verbose=True,
)


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
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = fused_mlp.fused_linear_gelu(x, self.fc1.weight, self.fc1.bias)
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


class WindowAttention(nn.Module):
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
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

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
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale.to(x.device), max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        
        if mask is None:
            # Create an empty tensor if no mask is provided
            mask_tensor = torch.empty(0, device=attn.device)
        else:
            # The mask needs to be broadcastable to attn.
            # Original: mask.unsqueeze(1).unsqueeze(0)
            # attn shape: (B_, H, N, N)
            # mask shape: (nW, N, N)
            # We pass the mask as is, kernel handles broadcasting.
            mask_tensor = mask

        # attn shape is (B_, num_heads, N, N), which matches kernel's expectation
        attn = fused_softmax_op.fused_softmax_inplace(attn, relative_position_bias, mask_tensor)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
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
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = nn.Identity()
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
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x_view = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x_view, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_view

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x_attn_out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_attn_out = shifted_x
        x_attn_out = x_attn_out.view(B, H * W, C)
        
        # Original: x = shortcut + self.drop_path(self.norm1(x))
        # With fusion: compute LayerNorm(x_attn_out) + shortcut
        x = fused_layernorm_add_op.layernorm_add_cuda(x_attn_out, shortcut, self.norm1.weight, self.norm1.bias, self.norm1.eps)

        # FFN
        # Original: x = x + self.drop_path(self.norm2(self.mlp(x)))
        # With fusion: compute LayerNorm(mlp(x)) + x
        shortcut2 = x
        mlp_out = self.mlp(shortcut2)
        x = fused_layernorm_add_op.layernorm_add_cuda(mlp_out, shortcut2, self.norm2.weight, self.norm2.bias, self.norm2.eps)
        
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
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        
        # Use fused kernel for memory operation (replaces slicing and cat)
        x = patch_merging_op.patch_merging_forward_cuda(x) # Output: B, H/2*W/2, 4*C
        
        # Apply reduction then norm, same as original
        x = self.reduction(x)
        x = self.norm(x)

        return x

class BasicLayer(nn.Module):
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
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

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
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
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

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
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
# EVOLVE-BLOCK-END