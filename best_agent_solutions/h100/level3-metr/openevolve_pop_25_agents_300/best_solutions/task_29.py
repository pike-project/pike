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

# --- Custom CUDA Kernels: Vectorized LayerNorm, Fused Patch Merging, and Fused Bias+Residual ---

combined_cuda_source_vectorized = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// --- Common Reduction Helpers ---
constexpr int BLOCK_SIZE = 256;

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[threadIdx.x] : (T)0.0;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// --- Kernel 1: Vectorized Standalone Layer Normalization ---
__global__ void layer_norm_kernel_vectorized(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int N, int C, float epsilon) {

    int row_idx = blockIdx.x;
    if (row_idx >= N) return;
    
    const int C_f4 = C / 4;
    __shared__ float s_reduc_buf[BLOCK_SIZE / 32];
    const float4* x_f4 = reinterpret_cast<const float4*>(x + row_idx * C);

    // Pass 1: Calculate mean using vectorized loads
    float sum = 0.0f;
    for (int i = threadIdx.x; i < C_f4; i += blockDim.x) {
        float4 val = x_f4[i];
        sum += val.x + val.y + val.z + val.w;
    }
    sum = block_reduce_sum(sum, s_reduc_buf);

    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = sum / C;
    __syncthreads();
    float mean = s_mean;

    // Pass 2: Calculate variance using vectorized loads
    float sum_sq_diff = 0.0f;
    for (int i = threadIdx.x; i < C_f4; i += blockDim.x) {
        float4 val = x_f4[i];
        sum_sq_diff += (val.x - mean) * (val.x - mean) +
                       (val.y - mean) * (val.y - mean) +
                       (val.z - mean) * (val.z - mean) +
                       (val.w - mean) * (val.w - mean);
    }
    sum_sq_diff = block_reduce_sum(sum_sq_diff, s_reduc_buf);

    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        s_inv_std = rsqrtf(sum_sq_diff / C + epsilon);
    }
    __syncthreads();
    float inv_std = s_inv_std;

    // Pass 3: Apply normalization using vectorized loads/stores
    const float4* gamma_f4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta_f4 = reinterpret_cast<const float4*>(beta);
    float4* out_f4 = reinterpret_cast<float4*>(out + row_idx * C);

    for (int i = threadIdx.x; i < C_f4; i += blockDim.x) {
        float4 val_x = x_f4[i];
        float4 val_gamma = gamma_f4[i];
        float4 val_beta = beta_f4[i];
        
        val_x.x = (val_x.x - mean) * inv_std * val_gamma.x + val_beta.x;
        val_x.y = (val_x.y - mean) * inv_std * val_gamma.y + val_beta.y;
        val_x.z = (val_x.z - mean) * inv_std * val_gamma.z + val_beta.z;
        val_x.w = (val_x.w - mean) * inv_std * val_gamma.w + val_beta.w;
        
        out_f4[i] = val_x;
    }
}

// --- Kernel 2: Vectorized Fused Patch Merging + Layer Normalization (Optimized Loading) ---
__global__ void fused_patch_merging_norm_kernel_vectorized(
    const float* __restrict__ input, // B, H, W, C
    const float* __restrict__ gamma, // 4*C
    const float* __restrict__ beta,  // 4*C
    float* __restrict__ output_norm, // B, H/2*W/2, 4*C
    int B, int H, int W, int C, float epsilon) {

    const int C4 = 4 * C;
    const int row_idx = blockIdx.x;
    if (row_idx >= B * (H / 2) * (W / 2)) return;

    const int C_f4 = C / 4;
    const int C4_f4 = C4 / 4;

    const int W_out = W / 2;
    const int H_out = H / 2;
    const int b = row_idx / (H_out * W_out);
    const int h_out = (row_idx / W_out) % H_out;
    const int w_out = row_idx % W_out;

    const int h_in = h_out * 2;
    const int w_in = w_out * 2;
    
    extern __shared__ float s_mem[];
    float* s_data = s_mem;
    float* s_reduce_buf = s_mem + C4;

    // Step 1: Vectorized load from 4 locations into shared memory (Optimized)
    const float4* input_f4 = reinterpret_cast<const float4*>(input);
    const float4* x_base_ptr = input_f4 + b * H * W * C_f4;
    const float4* x0_ptr_v4 = x_base_ptr + (h_in    ) * W * C_f4 + (w_in    ) * C_f4;
    const float4* x1_ptr_v4 = x_base_ptr + (h_in + 1) * W * C_f4 + (w_in    ) * C_f4;
    const float4* x2_ptr_v4 = x_base_ptr + (h_in    ) * W * C_f4 + (w_in + 1) * C_f4;
    const float4* x3_ptr_v4 = x_base_ptr + (h_in + 1) * W * C_f4 + (w_in + 1) * C_f4;
    float4* s_data_f4 = reinterpret_cast<float4*>(s_data);

    for (int i = threadIdx.x; i < C_f4; i += blockDim.x) {
        s_data_f4[i]              = x0_ptr_v4[i];
        s_data_f4[i + C_f4]       = x1_ptr_v4[i];
        s_data_f4[i + 2 * C_f4]   = x2_ptr_v4[i];
        s_data_f4[i + 3 * C_f4]   = x3_ptr_v4[i];
    }
    __syncthreads();


    // Step 2: LayerNorm on shared memory - Pass 1: Mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < C4_f4; i += blockDim.x) {
        float4 val = s_data_f4[i];
        sum += val.x + val.y + val.z + val.w;
    }
    sum = block_reduce_sum(sum, s_reduce_buf);
    
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = sum / C4;
    __syncthreads();
    float mean = s_mean;

    // Step 3: LayerNorm - Pass 2: Variance
    float sum_sq_diff = 0.0f;
    for (int i = threadIdx.x; i < C4_f4; i += blockDim.x) {
        float4 val = s_data_f4[i];
        sum_sq_diff += (val.x - mean) * (val.x - mean) + (val.y - mean) * (val.y - mean) +
                       (val.z - mean) * (val.z - mean) + (val.w - mean) * (val.w - mean);
    }
    sum_sq_diff = block_reduce_sum(sum_sq_diff, s_reduce_buf);
    
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        s_inv_std = rsqrtf(sum_sq_diff / C4 + epsilon);
    }
    __syncthreads();
    float inv_std = s_inv_std;

    // Step 4: Apply normalization and write to output using vectorization
    const float4* gamma_f4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta_f4 = reinterpret_cast<const float4*>(beta);
    float4* out_f4 = reinterpret_cast<float4*>(output_norm + row_idx * C4);

    for (int i = threadIdx.x; i < C4_f4; i += blockDim.x) {
        float4 val_s = s_data_f4[i];
        float4 val_gamma = gamma_f4[i];
        float4 val_beta = beta_f4[i];
        
        val_s.x = (val_s.x - mean) * inv_std * val_gamma.x + val_beta.x;
        val_s.y = (val_s.y - mean) * inv_std * val_gamma.y + val_beta.y;
        val_s.z = (val_s.z - mean) * inv_std * val_gamma.z + val_beta.z;
        val_s.w = (val_s.w - mean) * inv_std * val_gamma.w + val_beta.w;

        out_f4[i] = val_s;
    }
}

// --- Kernel 3: Fused Bias Add + Residual Add ---
__global__ void fused_bias_add_residual_kernel_v2(
    const float* __restrict__ linear_out,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int total_elements,
    int C) {

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += gridDim.x * blockDim.x) {
        int c_idx = idx % C;
        output[idx] = linear_out[idx] + bias[c_idx] + residual[idx];
    }
}


// --- C++ Bindings ---
torch::Tensor layer_norm_cuda_vectorized(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double epsilon) {
    const auto x_sizes = x.sizes();
    const int64_t N = x.numel() / x_sizes.back();
    const int64_t C = x_sizes.back();
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(C % 4 == 0, "C must be a multiple of 4 for vectorization");

    auto out = torch::empty_like(x);
    layer_norm_kernel_vectorized<<<N, BLOCK_SIZE>>>(
        x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        out.data_ptr<float>(), N, C, epsilon);
    return out;
}

torch::Tensor patch_merging_norm_forward_cuda_vectorized(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
    int H, int W, float epsilon) {

    const auto B = x.size(0);
    const auto C = x.size(3);
    TORCH_CHECK(C % 4 == 0, "C must be a multiple of 4 for vectorization");
    const auto H_out = H / 2;
    const auto W_out = W / 2;
    const auto L_out = H_out * W_out;
    const auto C4 = 4 * C;

    auto opts = x.options();
    auto y = torch::empty({B, L_out, C4}, opts);

    const int num_blocks = B * L_out;
    const int shared_mem_size = (C4 + BLOCK_SIZE / 32) * sizeof(float);

    fused_patch_merging_norm_kernel_vectorized<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
        x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        y.data_ptr<float>(), B, H, W, C, epsilon);
    return y;
}

torch::Tensor fused_bias_add_residual_cuda(torch::Tensor linear_out, torch::Tensor bias, torch::Tensor residual) {
    const auto linear_sizes = linear_out.sizes();
    const int64_t C = linear_sizes.back();
    const int64_t total_elements = linear_out.numel();

    auto output = torch::empty_like(linear_out);
    const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    fused_bias_add_residual_kernel_v2<<<num_blocks, BLOCK_SIZE>>>(
        linear_out.data_ptr<float>(),
        bias.data_ptr<float>(),
        residual.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        C
    );
    return output;
}
"""

combined_cpp_source_vectorized = """
torch::Tensor layer_norm_cuda_vectorized(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double epsilon);
torch::Tensor patch_merging_norm_forward_cuda_vectorized(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int H, int W, float epsilon);
torch::Tensor fused_bias_add_residual_cuda(torch::Tensor linear_out, torch::Tensor bias, torch::Tensor residual);
"""

custom_ops_vectorized = load_inline(
    name="custom_ops_vectorized_fused_v3",
    cpp_sources=combined_cpp_source_vectorized,
    cuda_sources=combined_cuda_source_vectorized,
    functions=["layer_norm_cuda_vectorized", "patch_merging_norm_forward_cuda_vectorized", "fused_bias_add_residual_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        original_shape = x.shape
        C = self.normalized_shape[0]
        # Vectorization requires C to be a multiple of 4
        if C % 4 != 0:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            
        x_reshaped = x.reshape(-1, C)
        out_reshaped = custom_ops_vectorized.layer_norm_cuda_vectorized(x_reshaped, self.weight, self.bias, self.eps)
        return out_reshaped.view(original_shape)

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
                 act_layer=nn.GELU, norm_layer=CustomLayerNorm):
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
        
        # Spatial MLP Part
        x_proc = self.norm1(x)
        x_proc = x_proc.view(B, H, W, C)
        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            shifted_x = F.pad(x_proc, [0, 0, P_l, P_r, P_t, P_b], "constant", 0)
        else:
            shifted_x = x_proc
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
            x_proc = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x_proc = shifted_x
        x_proc = x_proc.view(B, H * W, C)

        # First residual connection
        x_res1 = shortcut + self.drop_path(x_proc)

        # FFN
        x_norm2 = self.norm2(x_res1)
        x_mlp = self.mlp.fc1(x_norm2)
        x_mlp = self.mlp.act(x_mlp)
        x_mlp = self.mlp.drop(x_mlp)
        
        # Fused fc2 + residual add
        x_fc2_no_bias = F.linear(x_mlp, self.mlp.fc2.weight)
        x_fused = custom_ops_vectorized.fused_bias_add_residual_cuda(x_fc2_no_bias, self.mlp.fc2.bias, x_res1)
        
        # Final dropout from MLP
        x_out = self.mlp.drop(x_fused)

        return x_out


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=CustomLayerNorm):
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

        # Fallback for non-vectorizable C
        if C % 4 != 0:
            x_orig = x.view(B, H, W, C)
            x0 = x_orig[:, 0::2, 0::2, :]
            x1 = x_orig[:, 1::2, 0::2, :]
            x2 = x_orig[:, 0::2, 1::2, :]
            x3 = x_orig[:, 1::2, 1::2, :]
            x_cat = torch.cat([x0, x1, x2, x3], -1)
            x_cat = x_cat.view(B, -1, 4 * C)
            x_merged_norm = self.norm(x_cat)
        else:
            x = x.view(B, H, W, C)
            x_merged_norm = custom_ops_vectorized.patch_merging_norm_forward_cuda_vectorized(
                x.contiguous(), self.norm.weight, self.norm.bias, H, W, self.norm.eps)

        x = self.reduction(x_merged_norm)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=CustomLayerNorm, downsample=None, use_checkpoint=False):
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
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=CustomLayerNorm, patch_norm=True,
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
    return [torch.randn(batch_size, 3, image_size, image_size)]

def get_init_inputs():
    return []
# EVOLVE-BLOCK-END