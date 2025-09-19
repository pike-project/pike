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

# Combined CUDA source for Fused PatchMerging+LayerNorm and Fused Bias+GELU
# This approach combines the two most successful optimizations from previous attempts:
# 1. Fusing the memory-bound patch rearrangement with the LayerNorm operation to reduce memory bandwidth.
# 2. Fusing the bias addition from the MLP's linear layer with the GELU activation to reduce kernel launch overhead.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

// --- KERNEL 1: Fused Patch Merging + LayerNorm ---

// Helper for block-wide reduction using warp shuffle operations
template<typename T>
__device__ T block_reduce_sum(T val, T* shared) {
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    int wid = tid / warpSize;

    // Each warp performs reduction
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Warp leaders write to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // First warp reduces the results from shared memory
    val = (tid < blockDim.x / warpSize) ? shared[lane] : (T)0.0;
    if (wid == 0) {
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    return val;
}

__global__ void patch_merging_fused_norm_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output,
    const float* __restrict__ gamma, 
    const float* __restrict__ beta,
    const int B, const int H, const int W, const int C,
    const float epsilon)
{
    const int D = 4 * C;
    const int row_id = blockIdx.x;
    const int tid = threadIdx.x;

    const int H_out = H / 2;
    const int W_out = W / 2;
    const int L_out = H_out * W_out;
    const int b = row_id / L_out;
    const int h_out = (row_id % L_out) / W_out;
    const int w_out = (row_id % L_out) % W_out;

    extern __shared__ float s_mem[];
    float* s_data = s_mem;
    float* s_reduce = &s_mem[D];

    const int h_in_00 = 2 * h_out;
    const int w_in_00 = 2 * w_out;
    const float* in_ptr_base = input + b * H * W * C + h_in_00 * W * C + w_in_00 * C;
    
    for (int i = tid; i < D; i += blockDim.x) {
        int patch_idx = i / C;
        int c_in = i % C;
        
        float val;
        if (patch_idx == 0) val = __ldg(&in_ptr_base[c_in]);
        else if (patch_idx == 1) val = __ldg(&in_ptr_base[W * C + c_in]);
        else if (patch_idx == 2) val = __ldg(&in_ptr_base[C + c_in]);
        else val = __ldg(&in_ptr_base[W * C + C + c_in]);
        s_data[i] = val;
    }
    __syncthreads();

    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = s_data[i];
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    float total_sum = block_reduce_sum(thread_sum, s_reduce);
    float total_sum_sq = block_reduce_sum(thread_sum_sq, &s_reduce[blockDim.x / warpSize]);

    if (tid == 0) {
        float mean = total_sum / D;
        float var = total_sum_sq / D - mean * mean;
        s_reduce[0] = mean;
        s_reduce[1] = rsqrtf(var + epsilon);
    }
    __syncthreads();

    float mean = s_reduce[0];
    float rstd = s_reduce[1];

    float* out_ptr_row = output + row_id * D;

    for (int i = tid; i < D; i += blockDim.x) {
        float g = __ldg(&gamma[i]);
        float b = __ldg(&beta[i]);
        out_ptr_row[i] = g * (s_data[i] - mean) * rstd + b;
    }
}


torch::Tensor patch_merging_fused_norm_cuda(
    torch::Tensor x, int H, int W,
    torch::Tensor gamma, torch::Tensor beta, float epsilon) 
{
    const auto B = x.size(0);
    const auto C = x.size(2);
    
    auto x_view = x.view({B, H, W, C}).contiguous();

    const auto H_out = H / 2;
    const auto W_out = W / 2;
    const auto C_out = C * 4;
    const auto L_out = H_out * W_out;

    auto output = torch::empty({B, L_out, C_out}, x.options());

    const int D = C_out;
    const int num_rows = B * L_out;
    
    int block_size = 256;
    if (D > 256) block_size = 512;
    if (D > 512) block_size = 1024;
    
    int shared_mem_size = (D + 2 * (block_size / 32)) * sizeof(float);
    
    const dim3 grid(num_rows);
    const dim3 block(block_size);

    patch_merging_fused_norm_kernel<<<grid, block, shared_mem_size>>>(
        x_view.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.contiguous().data_ptr<float>(),
        beta.contiguous().data_ptr<float>(),
        B, H, W, C, epsilon
    );
    C10_CUDA_CHECK(cudaGetLastError());

    return output;
}

// --- KERNEL 2: Fused Bias + GELU ---
__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__global__ void bias_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M, // Total elements in batch dimensions
    int N  // Hidden dimension
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int idx = row * N + col;
        output[idx] = gelu_approx(input[idx] + bias[col]);
    }
}

torch::Tensor bias_gelu_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor");
    TORCH_CHECK(input.size(-1) == bias.size(0), "Last dim of input must match bias size");

    auto input_reshaped = input.reshape({-1, input.size(-1)});
    const auto M = input_reshaped.size(0);
    const auto N = input_reshaped.size(1);
    auto output = torch::empty_like(input_reshaped);

    const dim3 threads(16, 16);
    const dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    bias_gelu_kernel<<<blocks, threads>>>(
        input_reshaped.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N
    );
    C10_CUDA_CHECK(cudaGetLastError());
    
    return output.reshape(input.sizes());
}
"""

fused_ops_cpp_source = """
torch::Tensor patch_merging_fused_norm_cuda(
    torch::Tensor x, int H, int W,
    torch::Tensor gamma, torch::Tensor beta, float epsilon);

torch::Tensor bias_gelu_forward(torch::Tensor input, torch::Tensor bias);
"""

# Compile the inline CUDA code
swin_mlp_fused_ops = load_inline(
    name="swin_mlp_fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["patch_merging_fused_norm_cuda", "bias_gelu_forward"],
    verbose=True,
)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.act is removed as it's fused into the forward pass
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Use F.linear for matmul without bias, then our fused bias+gelu kernel
        x_matmul = F.linear(x, self.fc1.weight)
        x = swin_mlp_fused_ops.bias_gelu_forward(x_matmul, self.fc1.bias)
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

        # Replace the original sequence of rearrange and norm with a single fused kernel call
        x = swin_mlp_fused_ops.patch_merging_fused_norm_cuda(
            x, H, W, self.norm.weight, self.norm.bias, self.norm.eps
        )
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