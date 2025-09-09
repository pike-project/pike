import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------
# Fused CUDA Kernels
# --------------------------------------------------------

swin_mlp_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// KERNEL 1: Fused Pad + Window Partition
__global__ void fused_pad_window_partition_kernel(
    const float* __restrict__ x,
    float* __restrict__ windows,
    const int B, const int H, const int W, const int C,
    const int window_size, const int P_t, const int P_l,
    const int H_pad, const int W_pad,
    const long long total_elements) {

    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Deconstruct the flat output index `idx` to get coordinates in the `windows` tensor
    const int c = idx % C;
    const long long remaining = idx / C;
    const int w_win = remaining % window_size;
    const int h_win = (remaining / window_size) % window_size;
    const long long window_instance_idx = remaining / (window_size * window_size);

    // The layout of the flattened window dimension is (B, num_windows_h, num_windows_w),
    // so B is the slowest moving dimension.
    const int num_windows_h_pad = H_pad / window_size;
    const int num_windows_w_pad = W_pad / window_size;
    const int num_windows_per_batch = num_windows_h_pad * num_windows_w_pad;

    const int b = window_instance_idx / num_windows_per_batch;
    const int window_idx_in_batch = window_instance_idx % num_windows_per_batch;
    
    const int win_grid_w = window_idx_in_batch % num_windows_w_pad;
    const int win_grid_h = window_idx_in_batch / num_windows_w_pad;

    // Calculate the corresponding coordinate in the padded input
    const int h_pad = win_grid_h * window_size + h_win;
    const int w_pad = win_grid_w * window_size + w_win;

    // Calculate the coordinate in the original (unpadded) input
    const int h_orig = h_pad - P_t;
    const int w_orig = w_pad - P_l;

    // If the coordinate is within the bounds of the original tensor, copy the value.
    // Otherwise, it's padding, so write 0.
    if (h_orig >= 0 && h_orig < H && w_orig >= 0 && w_orig < W) {
        const long long in_idx = (long long)b * H * W * C +
                                 (long long)h_orig * W * C +
                                 (long long)w_orig * C +
                                 c;
        windows[idx] = x[in_idx];
    } else {
        windows[idx] = 0.0f;
    }
}

// KERNEL 2: Fused Window Reverse + Unpad (Slice)
__global__ void fused_window_reverse_unpad_kernel(
    const float* __restrict__ windows,
    float* __restrict__ x,
    const int B, const int H, const int W, const int C,
    const int window_size, const int P_t, const int P_l,
    const int H_pad, const int W_pad,
    const long long total_elements) {

    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Deconstruct the flat output index `idx` to get coordinates in the final `x` tensor
    const int c = idx % C;
    long long remaining = idx / C;
    const int w_orig = remaining % W;
    remaining /= W;
    const int h_orig = remaining % H;
    const int b = remaining / H;

    // Calculate the corresponding coordinate in the padded space
    const int h_pad = h_orig + P_t;
    const int w_pad = w_orig + P_l;

    // Determine which window and position within the window this pixel belongs to
    const int win_grid_h = h_pad / window_size;
    const int win_grid_w = w_pad / window_size;
    const int h_win = h_pad % window_size;
    const int w_win = w_pad % window_size;

    // Calculate the linear index into the source `windows` tensor
    const int num_windows_w_pad = W_pad / window_size;
    const int num_windows_per_batch = (H_pad / window_size) * num_windows_w_pad;
    const int window_idx_in_batch = win_grid_h * num_windows_w_pad + win_grid_w;
    
    // Correctly calculate the flattened window instance index (B is slowest-moving)
    const long long window_instance_idx = (long long)b * num_windows_per_batch + window_idx_in_batch;
    
    const long long pos_in_window = (long long)h_win * window_size + w_win;

    const long long in_idx = window_instance_idx * window_size * window_size * C +
                             pos_in_window * C +
                             c;

    x[idx] = windows[in_idx];
}

// KERNEL 3: Fused Patch Merging + LayerNorm (with Manual Reduction)
template <int BLOCK_SIZE>
__global__ void fused_patch_merging_norm_kernel(
    const float* __restrict__ x,  // Input: [B, H, W, C]
    float* __restrict__ out,      // Output: [B * H/2*W/2, 4*C]
    const float* __restrict__ gamma, // Layernorm weight [4*C]
    const float* __restrict__ beta,  // Layernorm bias [4*C]
    const int B, const int H, const int W, const int C,
    const float epsilon) {

    const int D = 4 * C;
    const int row_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;

    extern __shared__ float s_mem[];
    float* s_data = s_mem;
    float* s_partials_sum = &s_mem[D];
    float* s_partials_sum_sq = &s_mem[D + BLOCK_SIZE];

    if (thread_idx < D) {
      s_data[thread_idx] = 0.0f;
    }
    if (thread_idx < BLOCK_SIZE) {
      s_partials_sum[thread_idx] = 0.0f;
      s_partials_sum_sq[thread_idx] = 0.0f;
    }
     __syncthreads();


    const int H_half = H / 2;
    const int W_half = W / 2;
    const int num_rows_per_batch = H_half * W_half;
    const int b = row_idx / num_rows_per_batch;
    const int h_half = (row_idx % num_rows_per_batch) / W_half;
    const int w_half = (row_idx % num_rows_per_batch) % W_half;

    const int h_base = h_half * 2;
    const int w_base = w_half * 2;
    
    for (int i = thread_idx; i < D; i += BLOCK_SIZE) {
        int part = i / C;
        int c = i % C;
        // Corrected mapping from part index to 2x2 patch coordinates
        // Original PyTorch op: torch.cat([x0, x1, x2, x3], -1) where
        // x0 from (0::2, 0::2), x1 from (1::2, 0::2), x2 from (0::2, 1::2), x3 from (1::2, 1::2)
        // part=0 -> (h=0,w=0), part=1 -> (h=1,w=0), part=2 -> (h=0,w=1), part=3 -> (h=1,w=1)
        int h_offset = part & 1;
        int w_offset = part >> 1;

        long long in_idx = (long long)b * H * W * C +
                           (long long)(h_base + h_offset) * W * C +
                           (long long)(w_base + w_offset) * C +
                           c;
        s_data[i] = x[in_idx];
    }
    __syncthreads();

    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    for (int i = thread_idx; i < D; i += BLOCK_SIZE) {
        float val = s_data[i];
        thread_sum += val;
        thread_sum_sq += val * val;
    }
    
    s_partials_sum[thread_idx] = thread_sum;
    s_partials_sum_sq[thread_idx] = thread_sum_sq;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            s_partials_sum[thread_idx] += s_partials_sum[thread_idx + s];
            s_partials_sum_sq[thread_idx] += s_partials_sum_sq[thread_idx + s];
        }
        __syncthreads();
    }

    if (thread_idx == 0) {
        float total_sum = s_partials_sum[0];
        float total_sum_sq = s_partials_sum_sq[0];
        float mean = total_sum / D;
        float var = total_sum_sq / D - mean * mean;
        s_partials_sum[0] = mean;
        s_partials_sum_sq[0] = rsqrtf(var + epsilon);
    }
    __syncthreads();

    float mean = s_partials_sum[0];
    float inv_stddev = s_partials_sum_sq[0];

    for (int i = thread_idx; i < D; i += BLOCK_SIZE) {
        float val = s_data[i];
        out[row_idx * D + i] = (val - mean) * inv_stddev * gamma[i] + beta[i];
    }
}


// WRAPPER 1
torch::Tensor fused_pad_window_partition_cuda(
    torch::Tensor x, int window_size, int H, int W, int P_l, int P_r, int P_t, int P_b) {
    const auto B = x.size(0);
    const auto C = x.size(3);
    const int H_pad = H + P_t + P_b;
    const int W_pad = W + P_l + P_r;
    const int num_windows_h = H_pad / window_size;
    const int num_windows_w = W_pad / window_size;
    const int num_windows_total = num_windows_h * num_windows_w;
    auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto windows = torch::empty({(long)B * num_windows_total, (long)window_size, (long)window_size, (long)C}, options);
    const long long total_elements = windows.numel();
    if (total_elements == 0) return windows;
    const int block_size = 256;
    fused_pad_window_partition_kernel<<<(total_elements + block_size - 1) / block_size, block_size>>>(
        x.data_ptr<float>(), windows.data_ptr<float>(), B, H, W, C, window_size, P_t, P_l, H_pad, W_pad, total_elements);
    return windows;
}

// WRAPPER 2
torch::Tensor fused_window_reverse_unpad_cuda(
    torch::Tensor windows, int window_size, int H, int W, int P_l, int P_r, int P_t, int P_b) {
    const auto C = windows.size(3);
    const int H_pad = H + P_t + P_b;
    const int W_pad = W + P_l + P_r;
    const int num_windows_total = (H_pad / window_size) * (W_pad / window_size);
    const auto B = windows.size(0) / num_windows_total;
    auto options = torch::TensorOptions().device(windows.device()).dtype(windows.dtype());
    auto x = torch::empty({(long)B, (long)H, (long)W, (long)C}, options);
    const long long total_elements = x.numel();
    if (total_elements == 0) return x;
    const int block_size = 256;
    fused_window_reverse_unpad_kernel<<<(total_elements + block_size - 1) / block_size, block_size>>>(
        windows.data_ptr<float>(), x.data_ptr<float>(), B, H, W, C, window_size, P_t, P_l, H_pad, W_pad, total_elements);
    return x;
}

// WRAPPER 3
torch::Tensor fused_patch_merging_norm_cuda(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int H, int W, int C, float epsilon) {
    const auto B = x.size(0);
    const int H_half = H / 2;
    const int W_half = W / 2;
    const int D = 4 * C;
    auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto out = torch::empty({(long)B * H_half * W_half, (long)D}, options);
    const int num_rows = B * H_half * W_half;
    if (num_rows == 0) return out.view({(long)B, (long)(H_half*W_half), (long)D});
    const int BLOCK_SIZE = 256;
    int shared_mem_size = (D + 2 * BLOCK_SIZE) * sizeof(float);
    fused_patch_merging_norm_kernel<BLOCK_SIZE><<<num_rows, BLOCK_SIZE, shared_mem_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        B, H, W, C, epsilon);
    return out.view({(long)B, (long)(H_half*W_half), (long)D});
}
"""

swin_mlp_kernels_cpp_source = """
torch::Tensor fused_pad_window_partition_cuda(torch::Tensor x, int window_size, int H, int W, int P_l, int P_r, int P_t, int P_b);
torch::Tensor fused_window_reverse_unpad_cuda(torch::Tensor windows, int window_size, int H, int W, int P_l, int P_r, int P_t, int P_b);
torch::Tensor fused_patch_merging_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, int H, int W, int C, float epsilon);
"""

fused_swin_mlp_ops = load_inline(
    name="fused_swin_mlp_ops_v3",
    cpp_sources=swin_mlp_kernels_cpp_source,
    cuda_sources=swin_mlp_kernels_source,
    functions=[
        "fused_pad_window_partition_cuda",
        "fused_window_reverse_unpad_cuda",
        "fused_patch_merging_norm_cuda"
    ],
    verbose=True,
)

# --------------------------------------------------------
# Original Components & Helper Functions
# --------------------------------------------------------

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

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

# --------------------------------------------------------
# Optimized Model Components
# --------------------------------------------------------

class SwinMLPBlockNew(nn.Module):
    r""" Swin MLP Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

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
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

        self.norm1 = norm_layer(dim)
        # use group convolution to implement multi-head MLP
        self.spatial_mlp = nn.Conv1d(self.num_heads * self.window_size ** 2,
                                     self.num_heads * self.window_size ** 2,
                                     kernel_size=1,
                                     groups=self.num_heads)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.fused_ops = fused_swin_mlp_ops

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x_windows = self.fused_ops.fused_pad_window_partition_cuda(
                x.contiguous(), self.window_size, H, W, P_l, P_r, P_t, P_b)
        else:
            x_windows = window_partition(x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        x_windows_heads = x_windows.view(-1, self.window_size * self.window_size, self.num_heads, C // self.num_heads)
        x_windows_heads = x_windows_heads.transpose(1, 2).reshape(-1, self.num_heads * self.window_size * self.window_size, C // self.num_heads)
        spatial_mlp_windows = self.spatial_mlp(x_windows_heads)
        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.num_heads, self.window_size * self.window_size, C // self.num_heads).transpose(1, 2)
        spatial_mlp_windows = spatial_mlp_windows.reshape(-1, self.window_size, self.window_size, C)

        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = self.fused_ops.fused_window_reverse_unpad_cuda(
                spatial_mlp_windows.contiguous(), self.window_size, H, W, P_l, P_r, P_t, P_b)
        else:
            _H_pad = H
            _W_pad = W
            x = window_reverse(spatial_mlp_windows, self.window_size, _H_pad, _W_pad)

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMergingNew(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.fused_op = fused_swin_mlp_ops

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)

        x = self.fused_op.fused_patch_merging_norm_cuda(
            x.contiguous(), self.norm.weight, self.norm.bias, H, W, C, self.norm.eps)

        x = self.reduction(x)
        return x

class BasicLayerNew(nn.Module):
    """ A basic Swin MLP layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinMLPBlockNew(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
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
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size, patch_size = to_2tuple(img_size), to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size, self.patch_size = img_size, patch_size
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans, self.embed_dim = in_chans, embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class Model(nn.Module):
    r""" Swin MLP
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin MLP layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes, self.num_layers, self.embed_dim = num_classes, len(depths), embed_dim
        self.patch_norm, self.num_features = patch_norm, int(embed_dim * 2 ** (self.num_layers - 1))
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
            layer = BasicLayerNew(dim=int(embed_dim * 2 ** i_layer),
                                  input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                    patches_resolution[1] // (2 ** i_layer)),
                                  depth=depths[i_layer], num_heads=num_heads[i_layer],
                                  window_size=window_size, mlp_ratio=self.mlp_ratio,
                                  drop=drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMergingNew if (i_layer < self.num_layers - 1) else None,
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