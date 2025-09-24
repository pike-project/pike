import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl

# --------------------------------------------------------
# Fused CUDA Kernels
# --------------------------------------------------------

fused_cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
// #include <c10/cuda/CUDAMath.h> // This header is not found, replaced with static_cast

// Helper macros for tensor checking
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_DIM 16

// Overloaded helper function to convert various types to float for accumulation
__device__ inline float to_float(const c10::Half& val) {
    return static_cast<float>(val);
}
__device__ inline float to_float(const __half& val) {
    return __half2float(val);
}
__device__ inline float to_float(const float& val) {
    return val;
}
__device__ inline float to_float(const double& val) {
    return static_cast<float>(val);
}

// Kernel 1: Fused Spatial MLP
template <typename T>
__global__ void fused_spatial_mlp_kernel(
    const T* __restrict__ x_in, const T* __restrict__ weight, const T* __restrict__ bias,
    T* __restrict__ x_out, const int N_windows, const int C, const int nH, const int WS2)
{
    const int head_dim = C / nH;
    const int w_idx = blockIdx.x; // window index
    const int h_idx = blockIdx.y; // head index
    const int num_x_tiles = (head_dim + TILE_DIM - 1) / TILE_DIM;
    const int out_tile_y_idx = blockIdx.z / num_x_tiles;
    const int out_tile_x_idx = blockIdx.z % num_x_tiles;
    const int row = threadIdx.y;
    const int col = threadIdx.x;
    const int out_y = out_tile_y_idx * TILE_DIM + row;
    const int out_x = out_tile_x_idx * TILE_DIM + col;

    float acc = 0.0f; // Use float accumulator for precision
    __shared__ T s_W[TILE_DIM][TILE_DIM];
    __shared__ T s_X[TILE_DIM][TILE_DIM];

    for (int k_tile_start = 0; k_tile_start < WS2; k_tile_start += TILE_DIM) {
        if (out_y < WS2 && (k_tile_start + col < WS2)) {
            s_W[row][col] = weight[h_idx * WS2 * WS2 + out_y * WS2 + k_tile_start + col];
        } else {
            s_W[row][col] = (T)0.0f;
        }
        if ((k_tile_start + row < WS2) && out_x < head_dim) {
            s_X[row][col] = x_in[w_idx * WS2 * C + (k_tile_start + row) * C + h_idx * head_dim + out_x];
        } else {
            s_X[row][col] = (T)0.0f;
        }
        __syncthreads();
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += to_float(s_W[row][k]) * to_float(s_X[k][col]);
        }
        __syncthreads();
    }

    if (out_y < WS2 && out_x < head_dim) {
        acc += to_float(bias[h_idx * WS2 + out_y]);
        x_out[w_idx * WS2 * C + out_y * C + h_idx * head_dim + out_x] = (T)acc;
    }
}

// Kernel 2: Fused Patch Merging Rearrangement
template <typename T>
__global__ void patch_merging_rearrange_kernel(
    const T* x, T* y, int H, int W, int C) {
    
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int b     = blockIdx.z;

    if (w_out >= W/2 || h_out >= H/2) return;

    const T* x0_ptr = x + b * H * W * C + (h_out*2    ) * W * C + (w_out*2    ) * C;
    const T* x1_ptr = x + b * H * W * C + (h_out*2 + 1) * W * C + (w_out*2    ) * C;
    const T* x2_ptr = x + b * H * W * C + (h_out*2    ) * W * C + (w_out*2 + 1) * C;
    const T* x3_ptr = x + b * H * W * C + (h_out*2 + 1) * W * C + (w_out*2 + 1) * C;
    
    T* y_ptr = y + b * (H/2)*(W/2)*(4*C) + h_out * (W/2)*(4*C) + w_out * (4*C);

    for (int c = 0; c < C; ++c) {
        y_ptr[c]       = x0_ptr[c];
        y_ptr[c + C]   = x1_ptr[c];
        y_ptr[c + 2*C] = x2_ptr[c];
        y_ptr[c + 3*C] = x3_ptr[c];
    }
}

// C++ Wrapper Functions
torch::Tensor fused_spatial_mlp_forward(
    torch::Tensor x_in, torch::Tensor weight, torch::Tensor bias, int num_heads) {
    CHECK_INPUT(x_in); CHECK_INPUT(weight); CHECK_INPUT(bias);
    
    const int N_windows = x_in.size(0);
    const int WS2 = x_in.size(1);
    const int C = x_in.size(2);
    const int nH = num_heads;
    const int head_dim = C / nH;
    
    auto x_out = torch::empty_like(x_in);
    const int num_y_tiles = (WS2 + TILE_DIM - 1) / TILE_DIM;
    const int num_x_tiles = (head_dim + TILE_DIM - 1) / TILE_DIM;

    dim3 grid_dim(N_windows, nH, num_y_tiles * num_x_tiles);
    dim3 block_dim(TILE_DIM, TILE_DIM, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_in.scalar_type(), "fused_spatial_mlp_kernel", ([&] {
        fused_spatial_mlp_kernel<scalar_t><<<grid_dim, block_dim>>>(
            x_in.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
            x_out.data_ptr<scalar_t>(), N_windows, C, nH, WS2);
    }));
    return x_out;
}

torch::Tensor patch_merging_rearrange_forward(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor (B, H, W, C)");
    
    const int B = x.size(0);
    const int H = x.size(1);
    const int W = x.size(2);
    const int C = x.size(3);
    auto y = torch::empty({B, H/2, W/2, 4*C}, x.options());
    
    dim3 block_dim(16, 16, 1);
    dim3 grid_dim( (W/2 + block_dim.x - 1) / block_dim.x,
                   (H/2 + block_dim.y - 1) / block_dim.y,
                   B );
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "patch_merging_rearrange_kernel", ([&] {
        patch_merging_rearrange_kernel<scalar_t><<<grid_dim, block_dim>>>(
            x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), H, W, C);
    }));
    return y;
}
"""

fused_cuda_cpp_source = """
#include <torch/extension.h>

torch::Tensor fused_spatial_mlp_forward(torch::Tensor x_in, torch::Tensor weight, torch::Tensor bias, int num_heads);
torch::Tensor patch_merging_rearrange_forward(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_spatial_mlp_forward", &fused_spatial_mlp_forward, "Fused Spatial MLP forward (CUDA)");
  m.def("patch_merging_rearrange_forward", &patch_merging_rearrange_forward, "Fused Patch Merging Rearrangement forward (CUDA)");
}
"""

# Compile the inline CUDA code
custom_cuda_ops = load_inline(
    name="fused_swin_cuda_ops_v_fixed",
    cpp_sources=fused_cuda_cpp_source,
    cuda_sources=fused_cuda_source,
    verbose=False,
)

# --------------------------------------------------------
# Fused Triton Kernels
# --------------------------------------------------------

@triton.jit
def _layer_norm_fwd_kernel(
    X, Y, W, B,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    M, N, eps,
    BLOCK_SIZE_N: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_ptr = X + row_idx * stride_xm
    y_ptr = Y + row_idx * stride_ym
    
    _mean = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(x_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        _mean += x
    mean = tl.sum(_mean, axis=0) / N
    
    _var = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(x_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    
    for off in range(0, N, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        x = tl.load(x_ptr + cols * stride_xn, mask=mask, other=0.0)
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(y_ptr + cols * stride_yn, y, mask=mask)

def layer_norm_triton(x, weight, bias, eps):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    if BLOCK_SIZE_N > 4096: BLOCK_SIZE_N = 4096
    grid = (M, )
    _layer_norm_fwd_kernel[grid](
        x, y, weight, bias,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        M, N, eps,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    return y

class LayerNormTriton(nn.Module):
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

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x_reshaped = x.reshape(-1, shape[-1])
        y = layer_norm_triton(x_reshaped.contiguous(), self.weight.contiguous(), self.bias.contiguous(), self.eps)
        return y.view(shape)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_layernorm_matmul_fwd_kernel(
    X, Y, W, B, LN_W, LN_B,
    M, N, K,
    stride_xm, stride_xn,
    stride_ym, stride_yk,
    stride_wk, stride_wn, # Add strides for W
    eps,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    
    x_ptr_m = X + offs_m[:, None] * stride_xm
    mask_m = offs_m < M
    
    mean = tl.zeros([BLOCK_SIZE_M, 1], dtype=tl.float32)
    var = tl.zeros([BLOCK_SIZE_M, 1], dtype=tl.float32)
    
    for off in range(0, N, BLOCK_SIZE_N):
        offs_n_block = off + tl.arange(0, BLOCK_SIZE_N)
        mask_n_block = offs_n_block < N
        mask = mask_m[:, None] & mask_n_block[None, :]
        x = tl.load(x_ptr_m + offs_n_block[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=1)[:, None]
    mean = mean / N
    
    for off in range(0, N, BLOCK_SIZE_N):
        offs_n_block = off + tl.arange(0, BLOCK_SIZE_N)
        mask_n_block = offs_n_block < N
        mask = mask_m[:, None] & mask_n_block[None, :]
        x = tl.load(x_ptr_m + offs_n_block[None, :] * stride_xn, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, x - mean, 0.0)
        var += tl.sum(x * x, axis=1)[:, None]
    var = var / N
    rstd = 1 / tl.sqrt(var + eps)

    y_acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE_N):
        offs_n = off + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offs_n < N
        
        mask = mask_m[:, None] & mask_n[None, :]
        x = tl.load(x_ptr_m + offs_n[None, :] * stride_xn, mask=mask, other=0.0)
        
        ln_w = tl.load(LN_W + offs_n, mask=mask_n, other=0.0)
        ln_b = tl.load(LN_B + offs_n, mask=mask_n, other=0.0)
        
        x_norm = (x - mean.to(x.dtype)) * rstd.to(x.dtype)
        x_norm = x_norm * ln_w[None, :] + ln_b[None, :]
        
        w_ptr = W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptr, mask=mask_w, other=0.0)
        
        y_acc += tl.dot(x_norm, tl.trans(w))
    
    if B is not None:
        b = tl.load(B + offs_k, mask=(offs_k < K), other=0.0)
        y_acc += b[None, :]
    
    y_ptr = Y + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yk
    mask_y = mask_m[:, None] & (offs_k[None, :] < K)
    tl.store(y_ptr, y_acc.to(Y.dtype.element_ty), mask=mask_y)

class FusedLayerNormLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, eps=1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        self.ln_weight = nn.Parameter(torch.ones(in_features))
        self.ln_bias = nn.Parameter(torch.zeros(in_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, x):
        shape = x.shape
        x_reshaped = x.reshape(-1, self.in_features)
        M, N = x_reshaped.shape
        K = self.out_features
        
        y = torch.empty((M, K), dtype=x.dtype, device=x.device)
        
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']),)
        
        _fused_layernorm_matmul_fwd_kernel[grid](
            x_reshaped.contiguous(), y, self.weight, self.bias, self.ln_weight, self.ln_bias,
            M, N, K,
            x_reshaped.stride(0), x_reshaped.stride(1),
            y.stride(0), y.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            self.eps,
        )
        return y.view(*shape[:-1], K)

# --------------------------------------------------------
# Swin MLP with Custom Kernels
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

class SwinMLPBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=LayerNormTriton):
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
        
        self.WS2 = self.window_size * self.window_size
        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]

        self.norm1 = norm_layer(dim)
        self.spatial_mlp_weight = nn.Parameter(torch.empty(num_heads * self.WS2, self.WS2, 1))
        self.spatial_mlp_bias = nn.Parameter(torch.empty(num_heads * self.WS2))
        nn.init.kaiming_uniform_(self.spatial_mlp_weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.spatial_mlp_weight.squeeze(-1))
        bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.spatial_mlp_bias, -bound, bound)

        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2_fc1 = FusedLayerNormLinear(dim, mlp_hidden_dim, bias=True)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)


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
        x_windows = x_windows.view(-1, self.WS2, C)

        weight = self.spatial_mlp_weight.reshape(self.num_heads, self.WS2, self.WS2)
        bias = self.spatial_mlp_bias.view(self.num_heads, self.WS2)
        spatial_mlp_windows = custom_cuda_ops.fused_spatial_mlp_forward(x_windows, weight, bias, self.num_heads)

        spatial_mlp_windows = spatial_mlp_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(spatial_mlp_windows, self.window_size, _H, _W)

        if self.shift_size > 0:
            P_l, P_r, P_t, P_b = self.padding
            x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        residual = x
        x = self.norm2_fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = residual + self.drop_path(x)
        
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=LayerNormTriton):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.norm_reduction = FusedLayerNormLinear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = custom_cuda_ops.patch_merging_rearrange_forward(x)
        x = x.view(B, -1, 4 * C)
        
        x = self.norm_reduction(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 norm_layer=LayerNormTriton, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinMLPBlock(dim=dim, input_resolution=input_resolution,
                            num_heads=num_heads, window_size=window_size,
                            shift_size=0 if (i % 2 == 0) else window_size // 2,
                            mlp_ratio=mlp_ratio, drop=drop,
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
                 patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        
        norm_layer = LayerNormTriton

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
    return [torch.randn(batch_size, 3, image_size, image_size, device='cuda').half()]

def get_init_inputs():
    return []