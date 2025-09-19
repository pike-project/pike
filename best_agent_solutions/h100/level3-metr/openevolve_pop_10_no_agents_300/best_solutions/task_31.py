# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source for a vectorized fused linear + add + layernorm kernel.
# This version is optimized for memory bandwidth by using 4-element vector loads/stores.
# It uses one warp (32 threads) per row, with each thread handling 4 elements.
# The use of `if constexpr` requires C++17.
fused_linear_add_layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// A helper struct to handle half4 which is not a native CUDA type
struct alignas(8) half4 {
  __half x, y, z, w;
};

template <typename T>
__global__ void fused_linear_add_layernorm_kernel_vec4(
    const T* __restrict__ A,         // Input to GEMM, shape (M, K)
    const T* __restrict__ Wt,        // TRANSPOSED Weight of GEMM, shape (K, K)
    const T* __restrict__ B1,        // Bias of GEMM, shape (K)
    const T* __restrict__ X,         // Residual input, shape (M, K)
    const T* __restrict__ gamma,     // LN weight, shape (K)
    const T* __restrict__ beta,      // LN bias, shape (K)
    T* __restrict__ out,
    const float epsilon,
    const int M,                     // Total rows
    const int K                      // embed_dim
) {
    constexpr int VEC_SIZE = 4;
    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x; // vector lane index, 0 to K/VEC_SIZE - 1

    extern __shared__ float s_A[]; // Shared memory for one row of A, size K

    // --- Step 1: Vectorized Load of A into Shared Memory ---
    if constexpr (std::is_same_v<T, float>) {
        reinterpret_cast<float4*>(s_A)[tid] = reinterpret_cast<const float4*>(A + m * K)[tid];
    } else { // half
        const float2 val = reinterpret_cast<const float2*>(A + m * K)[tid];
        const __half2* vals_h = reinterpret_cast<const __half2*>(&val);
        s_A[tid * 4 + 0] = __half2float(vals_h[0].x);
        s_A[tid * 4 + 1] = __half2float(vals_h[0].y);
        s_A[tid * 4 + 2] = __half2float(vals_h[1].x);
        s_A[tid * 4 + 3] = __half2float(vals_h[1].y);
    }
    __syncthreads();

    // --- Step 2: Fused GEMM (A @ Wt) + Add + Add ---
    float gemm_acc[VEC_SIZE] = {0.0f};
    
    for (int k = 0; k < K; ++k) {
        const float s_a_val = s_A[k];
        if constexpr (std::is_same_v<T, float>) {
            const float4 wt_vec = reinterpret_cast<const float4*>(Wt + k * K)[tid];
            gemm_acc[0] += s_a_val * wt_vec.x;
            gemm_acc[1] += s_a_val * wt_vec.y;
            gemm_acc[2] += s_a_val * wt_vec.z;
            gemm_acc[3] += s_a_val * wt_vec.w;
        } else { // half
            const float2 wt_vec_f2 = reinterpret_cast<const float2*>(Wt + k * K)[tid];
            const __half2* wt_vec_h2 = reinterpret_cast<const __half2*>(&wt_vec_f2);
            gemm_acc[0] += s_a_val * __half2float(wt_vec_h2[0].x);
            gemm_acc[1] += s_a_val * __half2float(wt_vec_h2[0].y);
            gemm_acc[2] += s_a_val * __half2float(wt_vec_h2[1].x);
            gemm_acc[3] += s_a_val * __half2float(wt_vec_h2[1].y);
        }
    }

    float d_val[VEC_SIZE];
    if constexpr (std::is_same_v<T, float>) {
        const float4 b1_vec = reinterpret_cast<const float4*>(B1)[tid];
        const float4 x_vec  = reinterpret_cast<const float4*>(X + m * K)[tid];
        d_val[0] = gemm_acc[0] + b1_vec.x + x_vec.x;
        d_val[1] = gemm_acc[1] + b1_vec.y + x_vec.y;
        d_val[2] = gemm_acc[2] + b1_vec.z + x_vec.z;
        d_val[3] = gemm_acc[3] + b1_vec.w + x_vec.w;
    } else { // half
        const float2 b1_vec_f2 = reinterpret_cast<const float2*>(B1)[tid];
        const __half2* b1_vec_h2 = reinterpret_cast<const __half2*>(&b1_vec_f2);
        const float2 x_vec_f2 = reinterpret_cast<const float2*>(X + m * K)[tid];
        const __half2* x_vec_h2 = reinterpret_cast<const __half2*>(&x_vec_f2);
        d_val[0] = gemm_acc[0] + __half2float(b1_vec_h2[0].x) + __half2float(x_vec_h2[0].x);
        d_val[1] = gemm_acc[1] + __half2float(b1_vec_h2[0].y) + __half2float(x_vec_h2[0].y);
        d_val[2] = gemm_acc[2] + __half2float(b1_vec_h2[1].x) + __half2float(x_vec_h2[1].x);
        d_val[3] = gemm_acc[3] + __half2float(b1_vec_h2[1].y) + __half2float(x_vec_h2[1].y);
    }

    // --- Step 3: LayerNorm Reduction ---
    float local_sum = d_val[0] + d_val[1] + d_val[2] + d_val[3];
    float local_sum_sq = d_val[0]*d_val[0] + d_val[1]*d_val[1] + d_val[2]*d_val[2] + d_val[3]*d_val[3];

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
    }
    
    float mean = 0.0f, rstd = 0.0f;
    if (tid == 0) {
        mean = local_sum / K;
        float var = local_sum_sq / K - mean * mean;
        rstd = rsqrtf(var + epsilon);
    }
    mean = __shfl_sync(0xFFFFFFFF, mean, 0);
    rstd = __shfl_sync(0xFFFFFFFF, rstd, 0);

    // --- Step 4: Normalize and Store ---
    if constexpr (std::is_same_v<T, float>) {
        const float4 gamma_vec = reinterpret_cast<const float4*>(gamma)[tid];
        const float4 beta_vec = reinterpret_cast<const float4*>(beta)[tid];
        float4 out_vec;
        out_vec.x = (d_val[0] - mean) * rstd * gamma_vec.x + beta_vec.x;
        out_vec.y = (d_val[1] - mean) * rstd * gamma_vec.y + beta_vec.y;
        out_vec.z = (d_val[2] - mean) * rstd * gamma_vec.z + beta_vec.z;
        out_vec.w = (d_val[3] - mean) * rstd * gamma_vec.w + beta_vec.w;
        reinterpret_cast<float4*>(out + m * K)[tid] = out_vec;
    } else { // half
        const float2 gamma_vec_f2 = reinterpret_cast<const float2*>(gamma)[tid];
        const __half2* gamma_vec_h2 = reinterpret_cast<const __half2*>(&gamma_vec_f2);
        const float2 beta_vec_f2 = reinterpret_cast<const float2*>(beta)[tid];
        const __half2* beta_vec_h2 = reinterpret_cast<const __half2*>(&beta_vec_f2);
        
        __half val_h[VEC_SIZE];
        val_h[0] = __float2half((d_val[0] - mean) * rstd * __half2float(gamma_vec_h2[0].x) + __half2float(beta_vec_h2[0].x));
        val_h[1] = __float2half((d_val[1] - mean) * rstd * __half2float(gamma_vec_h2[0].y) + __half2float(beta_vec_h2[0].y));
        val_h[2] = __float2half((d_val[2] - mean) * rstd * __half2float(gamma_vec_h2[1].x) + __half2float(beta_vec_h2[1].x));
        val_h[3] = __float2half((d_val[3] - mean) * rstd * __half2float(gamma_vec_h2[1].y) + __half2float(beta_vec_h2[1].y));

        reinterpret_cast<half4*>(out + m * K)[tid] = *reinterpret_cast<half4*>(&val_h);
    }
}

torch::Tensor fused_linear_add_layernorm_cuda(
    torch::Tensor A, torch::Tensor Wt, torch::Tensor B1, torch::Tensor X,
    torch::Tensor gamma, torch::Tensor beta, double epsilon)
{
    TORCH_CHECK(A.is_cuda() && Wt.is_cuda() && B1.is_cuda() && X.is_cuda() && gamma.is_cuda() && beta.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && Wt.is_contiguous() && B1.is_contiguous() && X.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(), "All tensors must be contiguous");

    const auto M = A.size(0);
    const auto K = A.size(1);
    constexpr int VEC_SIZE = 4;

    TORCH_CHECK(K > 0 && K % VEC_SIZE == 0, "K must be a multiple of ", VEC_SIZE);
    TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32 for warp-level operations");

    auto out = torch::empty_like(A);
    const dim3 grid(M);
    const dim3 block(K / VEC_SIZE);
    const size_t shared_mem_size = K * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "fused_layernorm_vec_launcher", ([&] {
        fused_linear_add_layernorm_kernel_vec4<scalar_t><<<grid, block, shared_mem_size>>>(
            A.data_ptr<scalar_t>(), Wt.data_ptr<scalar_t>(), B1.data_ptr<scalar_t>(),
            X.data_ptr<scalar_t>(), gamma.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), static_cast<float>(epsilon), M, K
        );
    }));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed in fused_linear_add_layernorm: ", cudaGetErrorString(err));
    }
    
    return out;
}
"""

fused_linear_add_layernorm_cpp_source = """
torch::Tensor fused_linear_add_layernorm_cuda(
    torch::Tensor A, torch::Tensor Wt, torch::Tensor B1, torch::Tensor X,
    torch::Tensor gamma, torch::Tensor beta, double epsilon
);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op_vectorized",
    cpp_sources=fused_linear_add_layernorm_cpp_source,
    cuda_sources=fused_linear_add_layernorm_source,
    functions=["fused_linear_add_layernorm_cuda"],
    verbose=False,
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
)

class CustomMultiheadAttention(nn.Module):
    """
    A custom MHA implementation that exposes the tensor right before the output projection.
    This allows us to fuse the output projection with subsequent operations.
    It uses a standard nn.MultiheadAttention layer internally to hold the parameters,
    ensuring correct initialization and compatibility with PyTorch's ecosystem.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.mha_layer = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        
        self.in_proj_weight = self.mha_layer.in_proj_weight
        self.in_proj_bias = self.mha_layer.in_proj_bias
        self.out_proj_weight = self.mha_layer.out_proj.weight
        self.out_proj_bias = self.mha_layer.out_proj.bias

    def forward(self, x):
        seq_len, bsz, _ = x.shape
        
        q, k, v = F._in_projection_packed(x, x, x, self.in_proj_weight, self.in_proj_bias)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        attn_output = F.scaled_dot_product_attention(q, k, v)

        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(seq_len, bsz, self.embed_dim)
        
        return attn_output

class Model(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Model, self).__init__()
        self.attn = CustomMultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.fused_op = fused_op

    def forward(self, x):
        if self.attn.in_proj_weight.dtype != x.dtype:
            self.to(x.dtype)

        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, H * W).permute(2, 0, 1).contiguous()
        
        attn_output_pre_proj = self.attn(x_reshaped)
        
        M, K = attn_output_pre_proj.shape[0] * attn_output_pre_proj.shape[1], attn_output_pre_proj.shape[2]
        attn_input_flat = attn_output_pre_proj.view(M, K)
        x_reshaped_flat = x_reshaped.view(M, K)

        out_proj_weight_T = self.attn.out_proj_weight.T.contiguous()

        x_norm_flat = self.fused_op.fused_linear_add_layernorm_cuda(
            attn_input_flat,
            out_proj_weight_T,
            self.attn.out_proj_bias,
            x_reshaped_flat,
            self.norm.weight,
            self.norm.bias,
            self.norm.eps
        )
        
        x_norm = x_norm_flat.view(x_reshaped.shape)
        x = x_norm.permute(1, 2, 0).view(B, C, H, W)
        return x

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    return [torch.randn(batch_size, num_channels, image_height, image_width, dtype=torch.half).cuda()]

def get_init_inputs():
    return [embed_dim, num_heads]
# EVOLVE-BLOCK-END