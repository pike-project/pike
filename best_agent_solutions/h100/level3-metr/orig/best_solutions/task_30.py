import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import collections
from itertools import repeat
from torch.utils.cpp_extension import load_inline


# ----------------------------------------------------------------------------------
# Custom CUDA Kernels for Fused Operations
# ----------------------------------------------------------------------------------

fused_ops_cuda_source = """
#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <float.h>

// -------------------
// Fused LayerNorm + Add Residual
// -------------------

template <typename T>
__global__ void layer_norm_add_residual_kernel(
    T* __restrict__ out, const T* __restrict__ inp, const T* __restrict__ residual,
    const T* __restrict__ gamma, const T* __restrict__ beta,
    int M, int N, T epsilon) {
    int row_idx = blockIdx.x;
    if (row_idx >= M) return;

    extern __shared__ T sdata[];

    const T* x = inp + row_idx * N;
    const T* r = residual + row_idx * N;
    T* y = out + row_idx * N;

    T sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += x[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    if (blockDim.x >= 1024) { if (threadIdx.x < 512) sdata[threadIdx.x] += sdata[threadIdx.x + 512]; __syncthreads(); }
    if (blockDim.x >= 512) { if (threadIdx.x < 256) sdata[threadIdx.x] += sdata[threadIdx.x + 256]; __syncthreads(); }
    if (blockDim.x >= 256) { if (threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128]; __syncthreads(); }
    if (blockDim.x >= 128) { if (threadIdx.x < 64) sdata[threadIdx.x] += sdata[threadIdx.x + 64]; __syncthreads(); }
    if (threadIdx.x < 32) {
        volatile T* vsmem = sdata;
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
    }
    __syncthreads();
    T mean = sdata[0] / N;

    T sum_sq = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        T val = x[i] - mean;
        sum_sq += val * val;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();

    if (blockDim.x >= 1024) { if (threadIdx.x < 512) sdata[threadIdx.x] += sdata[threadIdx.x + 512]; __syncthreads(); }
    if (blockDim.x >= 512) { if (threadIdx.x < 256) sdata[threadIdx.x] += sdata[threadIdx.x + 256]; __syncthreads(); }
    if (blockDim.x >= 256) { if (threadIdx.x < 128) sdata[threadIdx.x] += sdata[threadIdx.x + 128]; __syncthreads(); }
    if (blockDim.x >= 128) { if (threadIdx.x < 64) sdata[threadIdx.x] += sdata[threadIdx.x + 64]; __syncthreads(); }
    if (threadIdx.x < 32) {
        volatile T* vsmem = sdata;
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
    }
    __syncthreads();
    T var = sdata[0] / N;
    T rstd = rsqrtf(var + epsilon);

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const T gamma_val = gamma ? gamma[i] : 1.0f;
        const T beta_val = beta ? beta[i] : 0.0f;
        y[i] = (x[i] - mean) * rstd * gamma_val + beta_val + r[i];
    }
}

torch::Tensor layer_norm_add_residual_forward_cuda(
    const torch::Tensor& input, const torch::Tensor& residual,
    const c10::optional<torch::Tensor>& gamma_opt, const c10::optional<torch::Tensor>& beta_opt,
    double epsilon) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type() == torch::kFloat32);
    TORCH_CHECK(residual.is_cuda() && residual.is_contiguous() && residual.scalar_type() == torch::kFloat32);
    const auto input_sizes = input.sizes();
    const int N = input_sizes.back();
    const int M = input.numel() / N;
    const torch::Tensor& gamma = gamma_opt.has_value() ? gamma_opt.value() : torch::Tensor();
    const torch::Tensor& beta = beta_opt.has_value() ? beta_opt.value() : torch::Tensor();

    auto output = torch::empty_like(input);
    const int block_size = (N < 1024) ? ((N < 512) ? ((N < 256) ? 128 : 256) : 512) : 1024;
    const int num_blocks = M;
    const int shared_mem_size = block_size * sizeof(float);

    layer_norm_add_residual_kernel<float><<<num_blocks, block_size, shared_mem_size>>>(
        output.data_ptr<float>(), input.data_ptr<float>(), residual.data_ptr<float>(),
        gamma.defined() ? gamma.data_ptr<float>() : nullptr,
        beta.defined() ? beta.data_ptr<float>() : nullptr,
        M, N, static_cast<float>(epsilon)
    );
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

// -------------------
// Fused Add Bias + GELU
// -------------------

__device__ __forceinline__ float gelu_forward_device(float x) {
    // Correct erf-based GELU approximation
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f)); // x / sqrt(2)
}

__global__ void add_bias_gelu_kernel(const float* __restrict__ input, const float* __restrict__ bias, float* __restrict__ output, int total_elements, int N) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int col = idx % N;
        float val = input[idx] + bias[col];
        output[idx] = gelu_forward_device(val);
    }
}

torch::Tensor add_bias_gelu_forward_cuda(const torch::Tensor& input, const torch::Tensor& bias) {
    TORCH_CHECK(input.is_cuda() && bias.is_cuda() && input.is_contiguous());
    const int N = input.size(-1);
    const int total_elements = input.numel();
    auto output = torch::empty_like(input);
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    add_bias_gelu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), total_elements, N);
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}


// -------------------
// Fused Attention v2 (Online Softmax + Matmul(V) + Transpose)
// -------------------

template <typename T, int BlockSize>
__global__ void fused_attention_v2_kernel(
    T* __restrict__ out, const T* __restrict__ q_in, const T* __restrict__ k_in, const T* __restrict__ v_in,
    const T* __restrict__ logit_scale, const T* __restrict__ relative_pos_bias, const T* mask,
    const int B_, const int nH, const int N, const int D_head, const int nW) {

    const int i = blockIdx.y; // Query token index (0 to N-1)
    const int head_batch_idx = blockIdx.x; // Head and batch index (0 to B_*nH-1)

    const int b_ = head_batch_idx / nH;
    const int h = head_batch_idx % nH;
    const int w = (mask != nullptr) ? (b_ % nW) : 0;

    extern __shared__ T s[];
    T* s_probs = s; // For storing the softmax row, size N
    T* s_reduce = s + N; // For reductions, size BlockSize

    // --- Part 1: Online Softmax ---
    // Pointers to the start of the row data
    const T* q_i = q_in + head_batch_idx * N * D_head + i * D_head;
    const T* k_base = k_in + head_batch_idx * N * D_head;
    const T* rpb_i = relative_pos_bias + (h * N * N) + (i * N);
    const T* mask_i = (mask != nullptr) ? (mask + (w * N * N) + (i * N)) : nullptr;
    const T scale_val = logit_scale[h];

    T q_norm_val = 0.0f;
    for(int d=0; d<D_head; ++d) q_norm_val += q_i[d] * q_i[d];
    q_norm_val = rsqrtf(q_norm_val + 1e-12f);

    T max_val = -FLT_MAX;
    for (int j = threadIdx.x; j < N; j += BlockSize) {
        const T* k_j = k_base + j * D_head;
        T k_norm_val = 0.0f;
        for(int d=0; d<D_head; ++d) k_norm_val += k_j[d] * k_j[d];
        k_norm_val = rsqrtf(k_norm_val + 1e-12f);

        T dot_prod = 0.0f;
        for(int d=0; d<D_head; ++d) dot_prod += q_i[d] * k_j[d];

        T val = (dot_prod * q_norm_val * k_norm_val) * scale_val + rpb_i[j];
        if (mask_i != nullptr) val += mask_i[j];

        max_val = max(val, max_val);
        s_probs[j] = val; // Temporarily store logits
    }
    s_reduce[threadIdx.x] = max_val;
    __syncthreads();
    for (unsigned int s = BlockSize / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_reduce[threadIdx.x] = max(s_reduce[threadIdx.x], s_reduce[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = s_reduce[0];

    T sum_exp = 0.0f;
    for (int j = threadIdx.x; j < N; j += BlockSize) {
        T val = __expf(s_probs[j] - max_val);
        s_probs[j] = val;
        sum_exp += val;
    }
    s_reduce[threadIdx.x] = sum_exp;
    __syncthreads();
    for (unsigned int s = BlockSize / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + s];
        __syncthreads();
    }
    const T inv_sum_exp = 1.0f / (s_reduce[0] + 1e-12f);

    for (int j = threadIdx.x; j < N; j += BlockSize) {
        s_probs[j] *= inv_sum_exp;
    }
    __syncthreads();

    // --- Part 2: Matmul with V and Fused Transpose ---
    const T* v_base = v_in + head_batch_idx * N * D_head;
    const int C = nH * D_head;
    T* out_i_base = out + b_ * N * C + i * C + h * D_head;

    for (int d = threadIdx.x; d < D_head; d += BlockSize) {
        T val = 0.0f;
        for (int j = 0; j < N; ++j) {
            val += s_probs[j] * v_base[j * D_head + d];
        }
        out_i_base[d] = val;
    }
}


torch::Tensor fused_attention_v2_forward_cuda(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& logit_scale, const torch::Tensor& relative_pos_bias,
    const c10::optional<torch::Tensor>& mask_opt, int nW) {

    const auto q_sizes = q.sizes();
    const int B_ = q_sizes[0], nH = q_sizes[1], N = q_sizes[2], D_head = q_sizes[3];
    const int C = nH * D_head;
    auto out = torch::empty({B_, N, C}, q.options());
    const torch::Tensor& mask = mask_opt.has_value() ? mask_opt.value() : torch::Tensor();

    const int BlockSize = 256;
    dim3 grid(B_ * nH, N);

    fused_attention_v2_kernel<float, BlockSize><<<grid, BlockSize, (N + BlockSize) * sizeof(float)>>>(
        out.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        logit_scale.squeeze().contiguous().data_ptr<float>(), relative_pos_bias.data_ptr<float>(),
        mask.defined() ? mask.data_ptr<float>() : nullptr,
        B_, nH, N, D_head, nW
    );
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

// -------------------
// Fused Patch Merging Reorder
// -------------------
__global__ void patch_merging_reorder_kernel(const float* x, float* y, int B, int H, int W, int C) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B * (H/2) * (W/2) * (4*C); i += blockDim.x * gridDim.x) {
        int b = i / ((H/2)*(W/2)*(4*C));
        int l_idx_reordered = (i / (4*C)) % ((H/2)*(W/2));
        int c_reordered = i % (4*C);

        int c = c_reordered % C;
        int quadrant = c_reordered / C;
        
        int h_reordered = l_idx_reordered / (W/2);
        int w_reordered = l_idx_reordered % (W/2);

        int h_offset, w_offset;
        if (quadrant == 0) { // x0: H even, W even
            h_offset = 0; w_offset = 0;
        } else if (quadrant == 1) { // x1: H odd, W even
            h_offset = 1; w_offset = 0;
        } else if (quadrant == 2) { // x2: H even, W odd
            h_offset = 0; w_offset = 1;
        } else { // quadrant == 3, x3: H odd, W odd
            h_offset = 1; w_offset = 1;
        }
        int h = h_reordered * 2 + h_offset;
        int w = w_reordered * 2 + w_offset;

        int src_i = b * H * W * C + h * W * C + w * C + c;
        y[i] = x[src_i];
    }
}

torch::Tensor patch_merging_reorder_cuda(const torch::Tensor& x) {
    const auto sizes = x.sizes();
    int B = sizes[0], H = sizes[1], W = sizes[2], C = sizes[3];
    auto y = torch::empty({B, H/2*W/2, 4*C}, x.options());
    const int total_elements = y.numel();
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    patch_merging_reorder_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), B, H, W, C);
    C10_CUDA_CHECK(cudaGetLastError());
    return y;
}
"""

fused_ops_cpp_source = """
#include <c10/util/Optional.h>
torch::Tensor layer_norm_add_residual_forward_cuda(const torch::Tensor&, const torch::Tensor&, const c10::optional<torch::Tensor>&, const c10::optional<torch::Tensor>&, double);
torch::Tensor add_bias_gelu_forward_cuda(const torch::Tensor&, const torch::Tensor&);
torch::Tensor fused_attention_v2_forward_cuda(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const c10::optional<torch::Tensor>&, int);
torch::Tensor patch_merging_reorder_cuda(const torch::Tensor&);
"""

_fused_ops_cache = None
def _get_fused_ops():
    global _fused_ops_cache
    if _fused_ops_cache is None:
        _fused_ops_cache = load_inline(
            name="fused_swin_ops_corrected_v2",
            cpp_sources=fused_ops_cpp_source,
            cuda_sources=fused_ops_cuda_source,
            functions=["layer_norm_add_residual_forward_cuda", "add_bias_gelu_forward_cuda", "fused_attention_v2_forward_cuda", "patch_merging_reorder_cuda"],
            verbose=False,
        )
    return _fused_ops_cache


class LayerNormNew(nn.LayerNorm):
    def forward(self, input: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        use_fused_kernel = (
            input.is_cuda and input.dtype == torch.float32 and input.is_contiguous() and
            self.elementwise_affine and len(self.normalized_shape) == 1
        )
        if use_fused_kernel and residual is not None and residual.is_contiguous() and residual.shape == input.shape:
            return _get_fused_ops().layer_norm_add_residual_forward_cuda(
                input, residual, self.weight, self.bias, self.eps)
        # Fallback to default if no residual or conditions not met
        out = F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
        if residual is not None:
            out += residual
        return out

def _ntuple(n):
    def parse(x):
        return tuple(x) if isinstance(x, collections.abc.Iterable) and not isinstance(x, str) else tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class MlpNew(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        use_fused_kernel = (x.is_cuda and x.dtype == torch.float32 and
                            isinstance(self.act, nn.GELU) and self.fc1.bias is not None and x.is_contiguous())
        
        if use_fused_kernel:
            x_matmul = F.linear(x, self.fc1.weight, None)
            x = _get_fused_ops().add_bias_gelu_forward_cuda(x_matmul, self.fc1.bias)
        else:
            x = self.fc1(x)
            x = self.act(x)
        
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

class WindowAttentionNew(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False))

        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0)
        
        scale_factor = torch.tensor([self.window_size[0] - 1, self.window_size[1] - 1], dtype=torch.float32) if pretrained_window_size[0] == 0 else torch.tensor([pretrained_window_size[0] - 1, pretrained_window_size[1] - 1], dtype=torch.float32)
        relative_coords_table /= scale_factor.view(1, 1, 1, 2)
        relative_coords_table *= 8.0
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / np.log2(8.0)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        coords = torch.stack(torch.meshgrid([torch.arange(s) for s in self.window_size], indexing='ij'))
        relative_coords = torch.flatten(coords, 1)[:, :, None] - torch.flatten(coords, 1)[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1), persistent=False)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias, self.v_bias = (nn.Parameter(torch.zeros(dim)), nn.Parameter(torch.zeros(dim))) if qkv_bias else (None, None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)) if self.q_bias is not None else None
        qkv = F.linear(x, self.qkv.weight, qkv_bias).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        logit_scale = torch.clamp(self.logit_scale.to(x.device), max=np.log(1. / 0.01)).exp()
        
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1).permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        
        use_fused_attn = x.is_cuda and x.dtype == torch.float32 and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()

        if use_fused_attn and self.training == False:
            fused_ops = _get_fused_ops()
            nW = mask.shape[0] if mask is not None else 0
            x = fused_ops.fused_attention_v2_forward_cuda(q, k, v, logit_scale, relative_position_bias, mask, nW)
        else:
            attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
            attn = attn * logit_scale
            attn = attn + relative_position_bias.unsqueeze(0)
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlockNew(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=LayerNormNew, pretrained_window_size=0):
        super().__init__()
        self.dim, self.input_resolution, self.window_size, self.shift_size, self.mlp_ratio = dim, input_resolution, window_size, shift_size, mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size, self.window_size = 0, min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionNew(dim, to_2tuple(self.window_size), num_heads, qkv_bias, attn_drop, drop, to_2tuple(pretrained_window_size))
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MlpNew(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            img_mask = torch.zeros((1, *self.input_resolution, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
            attn_mask = (mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)).masked_fill(mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) != 0, float(-100.0)).masked_fill(mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        
        x_view = x.view(B, H, W, C)
        shifted_x = torch.roll(x_view, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x_view
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, C)
        
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        attn_output = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x
        x = attn_output.view(B, L, C)
        
        x = shortcut + self.drop_path(self.norm1(x, residual=None))
        x = x + self.drop_path(self.norm2(self.mlp(x), residual=None))
        return x

class PatchMergingNew(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution, self.dim = input_resolution, dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x_view = x.view(B, H, W, C)

        use_fused_kernel = x.is_cuda and x.dtype == torch.float32 and x_view.is_contiguous()
        if use_fused_kernel:
            x_reordered = _get_fused_ops().patch_merging_reorder_cuda(x_view)
        else:
            x0, x1, x2, x3 = x_view[:, 0::2, 0::2, :], x_view[:, 1::2, 0::2, :], x_view[:, 0::2, 1::2, :], x_view[:, 1::2, 1::2, :]
            x_reordered = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)

        x = self.reduction(x_reordered)
        x = self.norm(x)
        return x

class BasicLayerNew(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=LayerNormNew, downsample=PatchMergingNew, use_checkpoint=False, pretrained_window_size=0):
        super().__init__()
        self.dim, self.input_resolution, self.depth, self.use_checkpoint = dim, input_resolution, depth, use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlockNew(dim, input_resolution, num_heads, window_size, 0 if (i % 2 == 0) else window_size // 2, mlp_ratio, qkv_bias, drop, attn_drop, drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer, pretrained_window_size=pretrained_window_size) for i in range(depth)])
        self.downsample = downsample(input_resolution, dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x, use_reentrant=False) if self.use_checkpoint and not torch.jit.is_scripting() else blk(x)
        return self.downsample(x) if self.downsample is not None else x

class PatchEmbedNew(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=LayerNormNew):
        super().__init__()
        self.patches_resolution = [s // p for s, p in zip(to_2tuple(img_size), to_2tuple(patch_size))]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x) if self.norm is not None else x

class ModelNew(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=LayerNormNew, patch_norm=True, use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()
        self.num_features = int(embed_dim * 2 ** (len(depths) - 1))
        self.patch_embed = PatchEmbedNew(img_size, patch_size, in_chans, embed_dim, norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([
            BasicLayerNew(dim=int(embed_dim * 2**i), input_resolution=([p // (2**i) for p in self.patch_embed.patches_resolution]), depth=depths[i], num_heads=num_heads[i], window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])], norm_layer=norm_layer, downsample=PatchMergingNew if i < len(depths) - 1 else None, use_checkpoint=use_checkpoint, pretrained_window_size=pretrained_window_sizes[i]) for i in range(len(depths))])
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
        return torch.flatten(x, 1)

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)