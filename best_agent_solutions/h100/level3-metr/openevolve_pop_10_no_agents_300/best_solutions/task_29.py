# EVOLVE-BLOCK-START
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
from torch.utils.cpp_extension import load_inline

# Fused CUDA kernel for Patch Merging and Layer Normalization, combining the best
# optimization techniques from previous attempts:
# 1. Coalesced vectorized data loading for maximum global memory bandwidth.
# 2. Single-pass statistics calculation (sum and sum-of-squares) to reduce shared memory traffic.
# 3. Warp-shuffle based reduction for the fastest possible block-level sum.
# 4. Vectorized normalization and store to global memory.
fused_patch_merging_layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

// Warp-level reduction using shuffle instructions for maximum speed.
__device__ inline float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory for inter-warp communication.
__device__ inline float blockReduceSum(float val, float* shared_reduction_buffer) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warpReduceSum(val); // Each warp computes its sum.

    if (lane == 0) {
        shared_reduction_buffer[warp_id] = val; // Warp leaders write to shared memory.
    }
    __syncthreads();

    // The first warp sums the results from other warps.
    val = (threadIdx.x < blockDim.x / 32) ? shared_reduction_buffer[lane] : 0.0f;
    if (warp_id == 0) {
        val = warpReduceSum(val);
    }
    return val;
}


__global__ void fused_patch_merging_layernorm_kernel_final(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const int B,
    const int H,
    const int W,
    const int C,
    const float epsilon) {

    const int L_out = (H / 2) * (W / 2);
    const int C_out = 4 * C;
    const int C_vec_size = C / 4; // C is guaranteed to be a multiple of 4

    // Each block processes one row for LayerNorm.
    const int row_idx = blockIdx.x;
    if (row_idx >= B * L_out) return;

    // Decompose row_idx into batch and spatial index.
    const int b = row_idx / L_out;
    const int l_out_idx = row_idx % L_out;
    
    const int W_out = W / 2;
    const int h_out = l_out_idx / W_out;
    const int w_out = l_out_idx % W_out;

    // Pre-calculate base pointers for the 4 input patches for coalesced access.
    const size_t L_in = (size_t)H * W;
    const float* x_base = x + (size_t)b * L_in * C;
    const float4* x0_ptr = (const float4*)(x_base + ((size_t)(h_out*2 + 0) * W + (w_out*2 + 0)) * C);
    const float4* x1_ptr = (const float4*)(x_base + ((size_t)(h_out*2 + 1) * W + (w_out*2 + 0)) * C);
    const float4* x2_ptr = (const float4*)(x_base + ((size_t)(h_out*2 + 0) * W + (w_out*2 + 1)) * C);
    const float4* x3_ptr = (const float4*)(x_base + ((size_t)(h_out*2 + 1) * W + (w_out*2 + 1)) * C);

    // Shared memory layout: [C_out for data] + [warps for reduction] + [2 for mean/inv_std]
    extern __shared__ float s_mem[];
    float* s_data_float = s_mem;
    float4* s_data_vec = reinterpret_cast<float4*>(s_mem);
    float* reduction_buffer = &s_data_float[C_out];
    float* mean_invstd_buffer = &reduction_buffer[blockDim.x / 32];
    
    // Step 1: Vectorized, coalesced load from global to shared memory.
    for (int i = threadIdx.x; i < C_vec_size; i += blockDim.x) {
        s_data_vec[i]                  = x0_ptr[i];
        s_data_vec[i + C_vec_size]     = x1_ptr[i];
        s_data_vec[i + 2 * C_vec_size] = x2_ptr[i];
        s_data_vec[i + 3 * C_vec_size] = x3_ptr[i];
    }
    __syncthreads();

    // Step 2: Single-pass statistics with optimized reduction.
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < C_out; i += blockDim.x) {
        float val = s_data_float[i];
        thread_sum += val;
        thread_sum_sq += val * val;
    }
    
    float total_sum = blockReduceSum(thread_sum, reduction_buffer);
    __syncthreads();
    float total_sum_sq = blockReduceSum(thread_sum_sq, reduction_buffer);
    
    if (threadIdx.x == 0) {
        const float mean = total_sum / C_out;
        const float var = total_sum_sq / C_out - mean * mean;
        mean_invstd_buffer[0] = mean;
        mean_invstd_buffer[1] = rsqrtf(var < 0 ? 0 : var + epsilon);
    }
    __syncthreads();
    const float mean = mean_invstd_buffer[0];
    const float inv_std = mean_invstd_buffer[1];

    // Step 3: Vectorized Normalize data and write to global output memory.
    const int C_out_vec_size = C_out / 4;
    for (int i = threadIdx.x; i < C_out_vec_size; i += blockDim.x) {
        float4 data_vec = s_data_vec[i];
        const int base_idx = i * 4;
        const float4 gamma_vec = *reinterpret_cast<const float4*>(&gamma[base_idx]);
        const float4 beta_vec = *reinterpret_cast<const float4*>(&beta[base_idx]);

        data_vec.x = (data_vec.x - mean) * inv_std * gamma_vec.x + beta_vec.x;
        data_vec.y = (data_vec.y - mean) * inv_std * gamma_vec.y + beta_vec.y;
        data_vec.z = (data_vec.z - mean) * inv_std * gamma_vec.z + beta_vec.z;
        data_vec.w = (data_vec.w - mean) * inv_std * gamma_vec.w + beta_vec.w;
        
        *reinterpret_cast<float4*>(&out[row_idx * C_out + base_idx]) = data_vec;
    }
}

torch::Tensor fused_patch_merging_layernorm_cuda(
    torch::Tensor x, 
    torch::Tensor gamma, 
    torch::Tensor beta,
    int H, int W, float epsilon) {
        
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input tensor must be a contiguous CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "Input tensor must be 3-dimensional");

    const auto B = x.size(0);
    const auto L = x.size(1);
    const auto C = x.size(2);
    const int C_out = 4 * C;
    const int L_out = L / 4;
    
    TORCH_CHECK(L == H * W, "Input tensor L dimension does not match H*W");
    TORCH_CHECK(H % 2 == 0 && W % 2 == 0, "H and W must be even");
    TORCH_CHECK(gamma.numel() == C_out, "Gamma has wrong size");
    TORCH_CHECK(beta.numel() == C_out, "Beta has wrong size");
    TORCH_CHECK(C % 4 == 0, "C must be a multiple of 4 for vectorization.");

    auto out = torch::empty({B, L_out, C_out}, x.options());
    
    const long long num_rows = B * L_out;
    if (num_rows == 0) return out;

    const int block_size = 256;
    const int num_blocks = num_rows;
    const int num_warps = block_size / 32;

    const int shared_mem_size = (C_out + num_warps + 2) * sizeof(float);

    fused_patch_merging_layernorm_kernel_final<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        B, H, W, C,
        epsilon
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

fused_patch_merging_layernorm_cpp_source = """
torch::Tensor fused_patch_merging_layernorm_cuda(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int H, int W, float epsilon);
"""

fused_patch_merging_op = load_inline(
    name="fused_patch_merging_op_final",
    cpp_sources=fused_patch_merging_layernorm_cpp_source,
    cuda_sources=fused_patch_merging_layernorm_source,
    functions=["fused_patch_merging_layernorm_cuda"],
    verbose=False,
)

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


class SwinMLPBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
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

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]

        self.norm1 = norm_layer(dim)
        self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2,
                                     self.num_heads * self.window_size ** 2,
                                     kernel_size=1,
                                     groups=self.num_heads)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x
        _, _H, _W, _ = shifted_x.shape

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        x_windows_heads = x_windows_heads.transpose(1, 2)
        x_windows_heads = x_windows_heads.reshape(-1, self.num_heads * self.window_size * self.window_size,
                                                  C // self.num_heads)
        spatial_mlp_windows = self.spatial_mlp(x_windows_heads)
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size,
                                                       C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size * self.window_size, C)

        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)

        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = fused_patch_merging_op.fused_patch_merging_layernorm_cuda(
            x, self.norm.weight, self.norm.bias, H, W, self.norm.eps)
        
        x = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinMLPBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer)
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

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

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
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
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
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
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