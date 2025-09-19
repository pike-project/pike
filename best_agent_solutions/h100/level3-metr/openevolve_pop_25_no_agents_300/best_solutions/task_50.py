# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernels for fused attention operations, with comprehensive vectorization.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For fmaxf

// Vectorized kernel to fuse the split, view, and transpose operations for Q, K, V using float4.
// This improves memory bandwidth utilization by loading and storing 4 floats at a time.
__global__ void fused_qkv_prep_kernel_vectorized(
    const float* __restrict__ qkv_in,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const int B, const int T, const int C,
    const int n_head, const int hs)
{
    // This kernel assumes hs and C are divisible by 4.
    const int hs_f4 = hs / 4;
    long long out_idx_f4 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    
    long long total_out_elements_f4 = (long long)B * n_head * T * hs_f4;
    if (out_idx_f4 >= total_out_elements_f4) return;

    // Cast pointers to float4 for vectorized access
    const float4* qkv_in_f4 = reinterpret_cast<const float4*>(qkv_in);
    float4* q_out_f4 = reinterpret_cast<float4*>(q_out);
    float4* k_out_f4 = reinterpret_cast<float4*>(k_out);
    float4* v_out_f4 = reinterpret_cast<float4*>(v_out);

    // Decode the flat float4 output index to multi-dimensional indices
    const int d4 = out_idx_f4 % hs_f4;
    const int t = (out_idx_f4 / hs_f4) % T;
    const int h = (out_idx_f4 / (hs_f4 * T)) % n_head;
    const int b = out_idx_f4 / (hs_f4 * T * n_head);

    // Calculate the corresponding C index for the start of the float4 vector
    const int c_start = h * hs + d4 * 4;

    // Calculate the source index in the input qkv_in tensor (B, T, 3*C)
    const int C_f4 = C / 4;
    const int C3_f4 = 3 * C_f4;
    const long long q_in_idx_f4 = (long long)b * T * C3_f4 + (long long)t * C3_f4 + (c_start / 4);
    const long long k_in_idx_f4 = q_in_idx_f4 + C_f4;
    const long long v_in_idx_f4 = q_in_idx_f4 + 2 * C_f4;

    // Perform coalesced reads and writes of float4 vectors
    q_out_f4[out_idx_f4] = qkv_in_f4[q_in_idx_f4];
    k_out_f4[out_idx_f4] = qkv_in_f4[k_in_idx_f4];
    v_out_f4[out_idx_f4] = qkv_in_f4[v_in_idx_f4];
}


// Vectorized kernel to fuse scaling, causal masking, and ReLU activation
__global__ void fused_scale_mask_relu_kernel_vectorized(
    const float* __restrict__ in, 
    float* __restrict__ out, 
    const float scale, 
    const int T,
    const long long total_elements_f4) 
{
    long long idx_f4 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_f4 >= total_elements_f4) return;

    const float4* in_f4 = reinterpret_cast<const float4*>(in);
    float4* out_f4 = reinterpret_cast<float4*>(out);

    float4 val_vec = in_f4[idx_f4];

    long long float_idx_start = idx_f4 * 4;
    const int j_start = float_idx_start % T;
    const int i = (float_idx_start / T) % T;

    float f1 = (j_start + 0 > i) ? 0.0f : fmaxf(val_vec.x * scale, 0.0f);
    float f2 = (j_start + 1 > i) ? 0.0f : fmaxf(val_vec.y * scale, 0.0f);
    float f3 = (j_start + 2 > i) ? 0.0f : fmaxf(val_vec.z * scale, 0.0f);
    float f4 = (j_start + 3 > i) ? 0.0f : fmaxf(val_vec.w * scale, 0.0f);

    out_f4[idx_f4] = make_float4(f1, f2, f3, f4);
}


// Vectorized kernel to fuse the final transpose and view operation for the output y
// Input: (B, n_head, T, hs) -> Output: (B, T, C)
__global__ void fused_y_reshape_kernel_vectorized(
    const float* __restrict__ y_in,
    float* __restrict__ y_out,
    const int B, const int T, const int C,
    const int n_head, const int hs)
{
    // hs must be divisible by 4
    const int hs_f4 = hs / 4;
    long long in_idx_f4 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_in_elements_f4 = (long long)B * n_head * T * hs_f4;
    if (in_idx_f4 >= total_in_elements_f4) return;

    // Decode vectorized input index to (b, h, t, d_f4)
    const int d_f4 = in_idx_f4 % hs_f4;
    const int t = (in_idx_f4 / hs_f4) % T;
    const int h = (in_idx_f4 / (hs_f4 * T)) % n_head;
    const int b = in_idx_f4 / (hs_f4 * T * n_head);

    // Calculate vectorized output index (b, t, c_f4)
    const int c_start = h * hs + d_f4 * 4;
    const int c_f4 = c_start / 4;
    const int C_f4 = C / 4;
    long long out_idx_f4 = (long long)b * T * C_f4 + (long long)t * C_f4 + c_f4;
    
    const float4* y_in_f4 = reinterpret_cast<const float4*>(y_in);
    float4* y_out_f4 = reinterpret_cast<float4*>(y_out);

    y_out_f4[out_idx_f4] = y_in_f4[in_idx_f4];
}

// C++ wrapper functions for the kernels
std::vector<torch::Tensor> fused_qkv_prep(torch::Tensor qkv_in, int n_head) {
    const auto B = qkv_in.size(0);
    const auto T = qkv_in.size(1);
    const auto C3 = qkv_in.size(2);
    const auto C = C3 / 3;
    const auto hs = C / n_head;

    TORCH_CHECK(hs % 4 == 0, "Head size must be divisible by 4 for vectorized kernel");

    auto opts = qkv_in.options();
    auto q_out = torch::empty({B, n_head, T, hs}, opts);
    auto k_out = torch::empty({B, n_head, T, hs}, opts);
    auto v_out = torch::empty({B, n_head, T, hs}, opts);

    const long long total_out_elements_f4 = q_out.numel() / 4;
    if (total_out_elements_f4 == 0) return {q_out, k_out, v_out};

    const int block_size = 512;
    const int num_blocks = (total_out_elements_f4 + block_size - 1) / block_size;

    fused_qkv_prep_kernel_vectorized<<<num_blocks, block_size>>>(
        qkv_in.data_ptr<float>(), q_out.data_ptr<float>(), k_out.data_ptr<float>(), v_out.data_ptr<float>(),
        B, T, C, n_head, hs);
    return {q_out, k_out, v_out};
}

torch::Tensor fused_scale_mask_relu(torch::Tensor att, float scale) {
    TORCH_CHECK(att.is_contiguous(), "Input tensor must be contiguous");
    const auto T = att.size(3);
    TORCH_CHECK(T % 4 == 0, "Last dim must be divisible by 4 for vectorization");

    auto out = torch::empty_like(att);
    const long long total_elements_f4 = att.numel() / 4;
    if (total_elements_f4 == 0) return out;

    const int block_size = 512;
    const int num_blocks = (total_elements_f4 + block_size - 1) / block_size;

    fused_scale_mask_relu_kernel_vectorized<<<num_blocks, block_size>>>(
        att.data_ptr<float>(), out.data_ptr<float>(), scale, T, total_elements_f4);
    return out;
}

torch::Tensor fused_y_reshape(torch::Tensor y_in) {
    TORCH_CHECK(y_in.is_contiguous(), "Input tensor must be contiguous");
    const auto B = y_in.size(0);
    const auto n_head = y_in.size(1);
    const auto T = y_in.size(2);
    const auto hs = y_in.size(3);
    const auto C = n_head * hs;
    TORCH_CHECK(hs % 4 == 0, "Head size must be divisible by 4 for vectorization");

    auto opts = y_in.options();
    auto y_out = torch::empty({B, T, C}, opts);

    const long long total_in_elements_f4 = y_in.numel() / 4;
    if (total_in_elements_f4 == 0) return y_out;
    
    const int block_size = 512;
    const int num_blocks = (total_in_elements_f4 + block_size - 1) / block_size;

    fused_y_reshape_kernel_vectorized<<<num_blocks, block_size>>>(
        y_in.data_ptr<float>(), y_out.data_ptr<float>(), B, T, C, n_head, hs);
    return y_out;
}
"""

cpp_source = """
std::vector<torch::Tensor> fused_qkv_prep(torch::Tensor qkv_in, int n_head);
torch::Tensor fused_scale_mask_relu(torch::Tensor att, float scale);
torch::Tensor fused_y_reshape(torch::Tensor y_in);
"""

# JIT compile the CUDA kernels
fused_attn_ops = load_inline(
    name="fused_attn_ops_fully_vectorized",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_qkv_prep", "fused_scale_mask_relu", "fused_y_reshape"],
    verbose=False,
)

class NewGELU(nn.Module):
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class Model(nn.Module):
    """
    A fully optimized multi-head masked self-attention layer using vectorized custom CUDA kernels.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.fused_ops = fused_attn_ops

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.n_head

        # Step 1: Project to Q, K, V and reshape using a single, vectorized fused kernel
        qkv = self.c_attn(x)
        q, k, v = self.fused_ops.fused_qkv_prep(qkv, self.n_head)

        # Step 2: Calculate raw attention scores
        att_raw = q @ k.transpose(-2, -1)
        
        # Step 3: Apply scaling, causal mask, and ReLU using a single vectorized fused kernel
        scale = 1.0 / math.sqrt(hs)
        att = self.fused_ops.fused_scale_mask_relu(att_raw.contiguous(), scale)

        # Step 4: Apply attention to values
        y_intermediate = att @ v
        
        # Step 5: Reshape output using a fused, vectorized kernel, replacing transpose().contiguous().view()
        y = self.fused_ops.fused_y_reshape(y_intermediate.contiguous())

        return y

batch_size = 16
max_seqlen = 1024
n_embd = 768
n_head = 12

def get_inputs():
    return [torch.randn(batch_size, max_seqlen, n_embd).cuda()]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]
# EVOLVE-BLOCK-END