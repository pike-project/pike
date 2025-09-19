# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Combined CUDA source for vectorized NewGELU and LayerNorm kernels.
# The strategy builds upon the top-performing solution by further optimizing the
# custom kernels for memory-bound operations. By using `float4` vectorization,
# we can load and store 4 floating-point numbers at a time, effectively
# increasing the memory bandwidth utilization and reducing the number of
# memory instructions. This is a powerful technique for memory-bound kernels
# where the data layout is contiguous and dimensions are divisible by 4.
fused_ops_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Pre-calculated constant for performance: sqrt(2.0 / PI)
#define M_SQRT2_OVER_PI_VAL 0.7978845608028654f

// --------------------------
// --- NewGELU Kernel (Vec4) ---
// --------------------------
// Vectorized version of the GELU kernel. It processes 4 elements at a time
// using the float4 data type, which is ideal for this purely element-wise operation.
__global__ void new_gelu_kernel_vec(const float* __restrict__ in_ptr, float* __restrict__ out_ptr, int size) {
    const float4* in = reinterpret_cast<const float4*>(in_ptr);
    float4* out = reinterpret_cast<float4*>(out_ptr);
    const int vec_size = size / 4;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < vec_size) {
        float4 x4 = in[idx];

        // Manually unroll for each component of the float4 vector
        const float x_cubed_x = x4.x * x4.x * x4.x;
        const float inner_x = M_SQRT2_OVER_PI_VAL * (x4.x + 0.044715f * x_cubed_x);
        x4.x = 0.5f * x4.x * (1.0f + tanhf(inner_x));

        const float x_cubed_y = x4.y * x4.y * x4.y;
        const float inner_y = M_SQRT2_OVER_PI_VAL * (x4.y + 0.044715f * x_cubed_y);
        x4.y = 0.5f * x4.y * (1.0f + tanhf(inner_y));

        const float x_cubed_z = x4.z * x4.z * x4.z;
        const float inner_z = M_SQRT2_OVER_PI_VAL * (x4.z + 0.044715f * x_cubed_z);
        x4.z = 0.5f * x4.z * (1.0f + tanhf(inner_z));
        
        const float x_cubed_w = x4.w * x4.w * x4.w;
        const float inner_w = M_SQRT2_OVER_PI_VAL * (x4.w + 0.044715f * x_cubed_w);
        x4.w = 0.5f * x4.w * (1.0f + tanhf(inner_w));

        out[idx] = x4;
    }
}

torch::Tensor new_gelu_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(x.numel() % 4 == 0, "Input tensor numel must be divisible by 4 for vectorized kernel");

    auto out = torch::empty_like(x);
    auto size = x.numel();
    if (size == 0) return out;

    const int block_size = 256;
    const int num_blocks = (size / 4 + block_size - 1) / block_size;

    new_gelu_kernel_vec<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}


// -----------------------------
// --- LayerNorm Kernel (Vec4) ---
// -----------------------------
// Vectorized version of the LayerNorm kernel. Data is loaded and stored using float4,
// reducing memory instructions for both the initial reduction pass and the final
// normalization pass.
__global__ void layer_norm_kernel_vec(const float* __restrict__ x, float* __restrict__ y,
                                      const float* __restrict__ gamma, const float* __restrict__ beta,
                                      int rows, int cols, float epsilon) {
    const int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    
    const float4* x_vec = reinterpret_cast<const float4*>(x + row_idx * cols);
    const int vec_cols = cols / 4;

    // Step 1: Compute sum and sum of squares for the row in parallel using vectorized loads.
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = tid; i < vec_cols; i += blockDim.x) {
        const float4 val4 = x_vec[i];
        local_sum += val4.x + val4.y + val4.z + val4.w;
        local_sum_sq += val4.x * val4.x + val4.y * val4.y + val4.z * val4.z + val4.w * val4.w;
    }
    sdata[tid] = local_sum;
    sdata[tid + blockDim.x] = local_sum_sq;
    __syncthreads();

    // Step 2: Parallel reduction within the block (operates on scalar sums).
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
        }
        __syncthreads();
    }

    // Step 3: Thread 0 calculates mean and inv_stddev.
    if (tid == 0) {
        const float mean = sdata[0] / cols;
        const float variance = sdata[blockDim.x] / cols - mean * mean;
        const float inv_stddev = rsqrtf(variance + epsilon);
        sdata[0] = mean;
        sdata[1] = inv_stddev;
    }
    __syncthreads();

    // Step 4: All threads apply normalization using vectorized loads/stores.
    const float mean = sdata[0];
    const float inv_stddev = sdata[1];
    float4* y_vec = reinterpret_cast<float4*>(y + row_idx * cols);
    const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
    const float4* beta_vec = reinterpret_cast<const float4*>(beta);

    for (int i = tid; i < vec_cols; i += blockDim.x) {
        float4 val4 = x_vec[i];
        const float4 gamma4 = gamma_vec[i];
        const float4 beta4 = beta_vec[i];

        val4.x = (val4.x - mean) * inv_stddev * gamma4.x + beta4.x;
        val4.y = (val4.y - mean) * inv_stddev * gamma4.y + beta4.y;
        val4.z = (val4.z - mean) * inv_stddev * gamma4.z + beta4.z;
        val4.w = (val4.w - mean) * inv_stddev * gamma4.w + beta4.w;
        
        y_vec[i] = val4;
    }
}

torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float epsilon) {
    TORCH_CHECK(x.is_cuda() && gamma.is_cuda() && beta.is_cuda(), "All tensors must be CUDA tensors");
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(), "All tensors must be contiguous");
    TORCH_CHECK(x.size(-1) % 4 == 0, "Last dimension must be divisible by 4 for vectorized kernel");
    
    const auto C = x.size(-1);
    const int rows = x.numel() / C;
    const int cols = C;
    
    auto y = torch::empty_like(x);

    const int block_size = 512;
    const int num_blocks = rows;
    const int shared_mem_size = block_size * 2 * sizeof(float);

    layer_norm_kernel_vec<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        rows, cols, epsilon);

    return y;
}
"""

fused_ops_cpp_source = """
torch::Tensor new_gelu_cuda(torch::Tensor x);
torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float epsilon);
"""

# JIT compile the custom CUDA kernels
fused_ops = load_inline(
    name="fused_ops_vec",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_cuda_source,
    functions=["new_gelu_cuda", "layer_norm_cuda"],
    verbose=False,
)

class NewGELUCustom(nn.Module):
    def forward(self, x):
        return fused_ops.new_gelu_cuda(x)

class LayerNormCustom(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        return fused_ops.layer_norm_cuda(x, self.weight, self.bias, self.eps)

class CausalSelfAttention(nn.Module):
    """
    This version is optimized to use torch.nn.functional.scaled_dot_product_attention,
    which leverages fused kernels like FlashAttention under the hood in PyTorch 2.0+.
    This is significantly more performant than a manual implementation.
    """
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Use PyTorch's optimized scaled dot-product attention.
        # This fuses scaling, masking, softmax, and dropout into a single, highly-optimized kernel.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class Model(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = LayerNormCustom(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = LayerNormCustom(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELUCustom(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.randn(batch_size, seq_len, n_embd, device='cuda')]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
# EVOLVE-BLOCK-END