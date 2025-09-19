# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# This solution combines the most effective architectural fusion (Add+LayerNorm)
# with the most efficient kernel implementations (single-pass LayerNorm) and other
# proven fusions (Bias+GELU).

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <c10/cuda/CUDAException.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------------------
// KERNEL 1: Fused Bias + GELU
// Fuses the bias add and GELU activation in the MLP, saving a memory
// round-trip for the largest intermediate tensor in the model.
// -------------------
__device__ __forceinline__ float new_gelu_impl_device(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * x * x * x)));
}

__global__ void add_bias_gelu_kernel(
    const float* __restrict__ in,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int total_elements,
    const int N) {
    // Standard grid-stride loop for element-wise ops
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {
        const int col = idx % N;
        out[idx] = new_gelu_impl_device(in[idx] + bias[col]);
    }
}

torch::Tensor add_bias_gelu_cuda(torch::Tensor x, torch::Tensor bias) {
    auto out = torch::empty_like(x);
    const int total_elements = x.numel();
    const int N = x.size(-1);
    
    const int block_size = 256;
    int num_blocks = std::min((total_elements + block_size - 1) / block_size, 4096);

    add_bias_gelu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), total_elements, N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// -------------------
// KERNEL 2: Single-Pass Standalone LayerNorm
// A highly optimized LayerNorm that reads from global memory only once by
// caching the input row in shared memory. Used for the first LayerNorm.
// -------------------
template <unsigned int BLOCK_SIZE>
__global__ void layer_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float epsilon,
    int C) {

    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float sdata[];
    
    float* sh_input = sdata;
    float* sh_reducers = sdata + C;

    // Load input row from global to shared memory.
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        sh_input[i] = input[row_idx * C + i];
    }
    __syncthreads();
    
    // Compute sum and sum-of-squares from shared memory.
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        float val = sh_input[i];
        thread_sum += val;
        thread_sum_sq += val * val;
    }
    sh_reducers[tid] = thread_sum;
    sh_reducers[tid + BLOCK_SIZE] = thread_sum_sq;
    __syncthreads();
    
    // Parallel reduction.
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_reducers[tid] += sh_reducers[tid + s];
            sh_reducers[tid + BLOCK_SIZE] += sh_reducers[tid + BLOCK_SIZE + s];
        }
        __syncthreads();
    }
    
    // First thread calculates mean and rstd.
    if (tid == 0) {
        float mean = sh_reducers[0] / C;
        float var = (sh_reducers[BLOCK_SIZE] / C) - (mean * mean);
        sh_reducers[0] = mean;
        sh_reducers[1] = rsqrtf(var + epsilon);
    }
    __syncthreads();
    
    float mean = sh_reducers[0];
    float rstd = sh_reducers[1];
    
    // Normalize and write to global output.
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        output[row_idx * C + i] = (sh_input[i] - mean) * rstd * weight[i] + bias[i];
    }
}

torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double epsilon) {
    const int C = x.size(-1);
    const int num_rows = x.numel() / C;
    auto out = torch::empty_like(x);
    
    const int block_size = 256;
    const int shared_mem_size = (C + 2 * block_size) * sizeof(float);
    TORCH_CHECK(shared_mem_size < 48 * 1024, "Shared memory request exceeds common GPU limits.");

    layer_norm_kernel<block_size><<<num_rows, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), static_cast<float>(epsilon), C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


// -------------------
// KERNEL 3: Single-Pass Fused Add + LayerNorm
// The most impactful fusion: computes `residual = in1 + in2` and `ln_out = LN(in1+in2)`
// in a single pass, eliminating a major memory bottleneck.
// -------------------
template <unsigned int BLOCK_SIZE>
__global__ void fused_add_layernorm_kernel(
    const float* __restrict__ in1,
    const float* __restrict__ in2,
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ residual_out,
    float* __restrict__ ln_out,
    float epsilon,
    int C) {

    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float sdata[];
    
    float* sh_sum_input = sdata;
    float* sh_reducers = sdata + C;

    // Load, add, and store result in shared memory.
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        sh_sum_input[i] = in1[row_idx * C + i] + in2[row_idx * C + i];
    }
    __syncthreads();
    
    // Compute sum and sum-of-squares from shared memory.
    float thread_sum = 0.0f, thread_sum_sq = 0.0f;
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        float val = sh_sum_input[i];
        thread_sum += val;
        thread_sum_sq += val * val;
    }
    sh_reducers[tid] = thread_sum;
    sh_reducers[tid + BLOCK_SIZE] = thread_sum_sq;
    __syncthreads();
    
    // Parallel reduction.
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_reducers[tid] += sh_reducers[tid + s];
            sh_reducers[tid + BLOCK_SIZE] += sh_reducers[tid + BLOCK_SIZE + s];
        }
        __syncthreads();
    }
    
    // First thread calculates mean and rstd.
    if (tid == 0) {
        float mean = sh_reducers[0] / C;
        float var = (sh_reducers[BLOCK_SIZE] / C) - (mean * mean);
        sh_reducers[0] = mean;
        sh_reducers[1] = rsqrtf(var + epsilon);
    }
    __syncthreads();
    
    float mean = sh_reducers[0];
    float rstd = sh_reducers[1];
    
    // Write both outputs.
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        float residual_val = sh_sum_input[i];
        residual_out[row_idx * C + i] = residual_val;
        ln_out[row_idx * C + i] = (residual_val - mean) * rstd * weight[i] + bias[i];
    }
}

std::vector<torch::Tensor> fused_add_layernorm_cuda(torch::Tensor in1, torch::Tensor in2, torch::Tensor w, torch::Tensor b, double eps) {
    const int C = in1.size(-1);
    const int num_rows = in1.numel() / C;
    auto residual_out = torch::empty_like(in1);
    auto ln_out = torch::empty_like(in1);
    
    const int block_size = 256;
    const int shared_mem_size = (C + 2 * block_size) * sizeof(float);
    TORCH_CHECK(shared_mem_size < 48 * 1024, "Shared memory request exceeds common GPU limits.");

    fused_add_layernorm_kernel<block_size><<<num_rows, block_size, shared_mem_size>>>(
        in1.data_ptr<float>(), in2.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        residual_out.data_ptr<float>(), ln_out.data_ptr<float>(),
        static_cast<float>(eps), C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {residual_out, ln_out};
}

"""

cpp_source = """
torch::Tensor add_bias_gelu_cuda(torch::Tensor x, torch::Tensor bias);
torch::Tensor layer_norm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double epsilon);
std::vector<torch::Tensor> fused_add_layernorm_cuda(torch::Tensor in1, torch::Tensor in2, torch::Tensor w, torch::Tensor b, double eps);
"""

custom_ops = load_inline(
    name="fused_transformer_block_ops_v_final",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["add_bias_gelu_cuda", "layer_norm_cuda", "fused_add_layernorm_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return custom_ops.layer_norm_cuda(x.contiguous(), self.weight, self.bias, self.eps)

class FusedAddLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x1, x2):
        return custom_ops.fused_add_layernorm_cuda(x1.contiguous(), x2.contiguous(), self.weight, self.bias, self.eps)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head, self.n_embd, self.attn_pdrop = n_head, n_embd, attn_pdrop

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, 
                                           dropout_p=self.attn_pdrop if self.training else 0.0, 
                                           is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class FusedMLP(nn.Module):
    def __init__(self, n_embd, resid_pdrop):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        x_proj = F.linear(x, self.c_fc.weight)
        x_act = custom_ops.add_bias_gelu_cuda(x_proj, self.c_fc.bias)
        x_final = self.c_proj(x_act)
        x_dropped = self.dropout(x_final)
        return x_dropped

class Model(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = CustomLayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.fused_add_ln_2 = FusedAddLayerNorm(n_embd)
        self.mlp = FusedMLP(n_embd, resid_pdrop)

    def forward(self, x):
        attn_out = self.attn(self.ln_1(x))
        
        # Fused kernel computes both the first residual connection (x + attn_out)
        # and the second LayerNorm input in a single pass.
        x_after_attn, mlp_in = self.fused_add_ln_2(x, attn_out)
        
        mlp_out = self.mlp(mlp_in)
        
        # The final residual connection.
        x = x_after_attn + mlp_out
        return x

# Original NewGELU needed for correctness check against the baseline model.
class NewGELU(nn.Module):
    def __init__(self): super(NewGELU, self).__init__()
    def forward(self, x): return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

batch_size = 128
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.randn(batch_size, seq_len, n_embd).cuda().contiguous()]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
# EVOLVE-BLOCK-END