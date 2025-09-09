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
import numpy as np
import collections
from itertools import repeat
from torch.utils.cpp_extension import load_inline

# --- Custom CUDA Kernel 1: Fused Softmax for Attention ---
fused_softmax_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <c10/cuda/CUDAException.h>

__global__ void fused_softmax_kernel(
    const float* __restrict__ input,      // B_, H, N, N
    const float* __restrict__ scale,      // H
    const float* __restrict__ bias,       // H, N, N
    const float* __restrict__ mask,       // nW, N, N
    float* __restrict__ output,           // B_, H, N, N
    const int B_, const int H, const int N, const int nW
) {
    // Each block processes one row of length N
    const int row_idx_global = blockIdx.x;
    if (row_idx_global >= B_ * H * N) return;
    
    const int N_row_idx = row_idx_global % N;
    const int H_idx = (row_idx_global / N) % H;
    const int B_idx = row_idx_global / (N * H);

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    extern __shared__ float s_data[];

    // Step 1: Load data and apply transforms
    float val = -FLT_MAX;
    if (tid < N) {
        const int64_t input_offset = (int64_t)B_idx * H * N * N + (int64_t)H_idx * N * N + (int64_t)N_row_idx * N + tid;
        const int64_t bias_offset = (int64_t)H_idx * N * N + (int64_t)N_row_idx * N + tid;
        
        val = input[input_offset] * scale[H_idx] + bias[bias_offset];

        if (mask != nullptr) {
            const int64_t mask_offset = (int64_t)(B_idx % nW) * N * N + (int64_t)N_row_idx * N + tid;
            val += mask[mask_offset];
        }
    }
    s_data[tid] = val;
    __syncthreads();

    // Step 2: Find max value (reduction)
    if (tid >= N) s_data[tid] = -FLT_MAX;
    __syncthreads();

    for (int offset = block_size / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_data[tid] = max(s_data[tid], s_data[tid + offset]);
        }
        __syncthreads();
    }
    const float max_val = s_data[0];
    __syncthreads();

    // Step 3: exp and sum (reduction)
    val = expf(val - max_val);
    if (tid >= N) val = 0.0f;
    s_data[tid] = val;
    __syncthreads();

    for (int offset = block_size / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
    }
    const float sum_val = s_data[0];
    __syncthreads();

    // Step 4: Normalize and write output
    if (tid < N) {
        const int64_t output_offset = (int64_t)B_idx * H * N * N + (int64_t)H_idx * N * N + (int64_t)N_row_idx * N + tid;
        output[output_offset] = val / (sum_val + 1e-6f);
    }
}

torch::Tensor fused_softmax_forward(
    torch::Tensor input, 
    torch::Tensor scale, 
    torch::Tensor bias,  
    torch::Tensor mask,
    int nW
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "Scale must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    if (mask.defined() && mask.numel() > 0) {
        TORCH_CHECK(mask.is_contiguous(), "Mask must be contiguous");
    }

    const auto B_ = input.size(0);
    const auto H = input.size(1);
    const auto N = input.size(2);
    TORCH_CHECK(input.size(3) == N, "Input must be square in last two dims");
    
    auto output = torch::empty_like(input);
    
    int block_size = 1;
    while(block_size < N) block_size *= 2;
    if (block_size > 1024) block_size = 1024;
    TORCH_CHECK(N <= 1024, "N is too large for this kernel");

    const dim3 threads(block_size);
    const dim3 blocks(B_ * H * N);
    const size_t shared_mem_size = block_size * sizeof(float);
    
    const bool has_mask = mask.defined() && mask.numel() > 0;

    fused_softmax_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        has_mask ? mask.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        B_, H, N, nW
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return output;
}
"""

fused_softmax_cpp_source = """
#include <torch/extension.h>

torch::Tensor fused_softmax_forward(
    torch::Tensor input, 
    torch::Tensor scale, 
    torch::Tensor bias,  
    torch::Tensor mask,
    int nW
);
"""

fused_softmax_op = load_inline(
    name="fused_softmax_op_v2",
    cpp_sources=fused_softmax_cpp_source,
    cuda_sources=fused_softmax_cuda_source,
    functions=["fused_softmax_forward"],
    verbose=False
)
# --- End Custom CUDA Kernel 1 ---

# --- Custom CUDA Kernel 2: Vectorized Patch Merging ---
patch_merging_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__global__ void patch_merge_kernel_vectorized(const float* __restrict__ input, float* __restrict__ output, int B, int H, int W, int C) {
    // Each block processes one output pixel (h_out, w_out) for one batch item
    const int w_out = blockIdx.x;
    const int h_out = blockIdx.y;
    const int b = blockIdx.z;

    const int H_out = H / 2;
    const int W_out = W / 2;
    const int C_vec = C / 4;

    // Cast pointers to float4. This assumes input and output are aligned.
    const float4* p_in_base = reinterpret_cast<const float4*>(input + b * H * W * C);
    float4* p_out = reinterpret_cast<float4*>(output + (b * H_out * W_out + h_out * W_out + w_out) * (4 * C));

    // Each thread in the block processes multiple float4 vectors using a grid-stride loop
    for (int c_idx = threadIdx.x; c_idx < C_vec; c_idx += blockDim.x) {
        const int h_in = h_out * 2;
        const int w_in = w_out * 2;
        
        // Read from 4 locations in the input using vectorized loads
        float4 val00 = p_in_base[(h_in)   * W * C_vec + (w_in)   * C_vec + c_idx];
        float4 val10 = p_in_base[(h_in+1) * W * C_vec + (w_in)   * C_vec + c_idx];
        float4 val01 = p_in_base[(h_in)   * W * C_vec + (w_in+1) * C_vec + c_idx];
        float4 val11 = p_in_base[(h_in+1) * W * C_vec + (w_in+1) * C_vec + c_idx];

        // Write to 4 locations in the output vector using vectorized stores
        p_out[c_idx]           = val00;
        p_out[C_vec + c_idx]   = val10;
        p_out[2*C_vec + c_idx] = val01;
        p_out[3*C_vec + c_idx] = val11;
    }
}

torch::Tensor patch_merge_rearrange_cuda(torch::Tensor x, int H, int W) {
    const auto B = x.size(0);
    const auto C = x.size(2);
    
    TORCH_CHECK(C % 4 == 0, "Channel dimension C must be divisible by 4 for vectorized kernel.");

    auto x_view = x.view({B, H, W, C}).contiguous();
    
    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto output = torch::empty({B, (H / 2) * (W / 2), 4 * C}, options);

    dim3 threads(256);
    dim3 blocks(W / 2, H / 2, B);

    patch_merge_kernel_vectorized<<<blocks, threads>>>(
        x_view.data_ptr<float>(),
        output.data_ptr<float>(),
        B, H, W, C
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""

patch_merging_cpp_source = "torch::Tensor patch_merge_rearrange_cuda(torch::Tensor x, int H, int W);"

custom_patch_merging_op = load_inline(
    name="custom_patch_merging_op_v4",
    cpp_sources=patch_merging_cpp_source,
    cuda_sources=patch_merging_source,
    functions=["patch_merge_rearrange_cuda"],
    verbose=False,
)
# --- End Custom CUDA Kernel 2 ---

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

        attn_raw = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        
        logit_scale = torch.clamp(self.logit_scale, max=np.log(1. / 0.01)).exp()

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        
        nW = mask.shape[0] if mask is not None else 1
        
        attn = fused_softmax_op.fused_softmax_forward(
            attn_raw, 
            logit_scale.view(self.num_heads),
            relative_position_bias,
            mask if mask is not None else torch.empty(0, device=x.device),
            nW
        )
        
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
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class FusedPatchMerging(nn.Module):
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
        x = custom_patch_merging_op.patch_merge_rearrange_cuda(x, H, W)
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
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

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
                               downsample=FusedPatchMerging if (i_layer < self.num_layers - 1) else None,
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
    return [torch.randn(batch_size, 3, image_size, image_size)]

def get_init_inputs():
    return []
# EVOLVE-BLOCK-END