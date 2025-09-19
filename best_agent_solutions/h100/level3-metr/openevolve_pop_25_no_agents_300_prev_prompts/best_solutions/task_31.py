# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# A highly optimized one-pass, vectorized, warp-level CUDA kernel for Add + LayerNorm.
# This version combines the best features from previous attempts:
# 1. SMEM-Free Design: Avoids shared memory entirely, using registers and warp-shuffle
#    intrinsics for maximum performance and to reduce latency.
# 2. Vectorized Memory I/O: Uses float4 to load/store 4 elements per thread, saturating
#    memory bandwidth for the C=128 case.
# 3. Fused `float2` Reduction: Reduces sum and sum-of-squares simultaneously using a
#    custom float2 warp reduction, improving instruction-level parallelism.
# 4. 64-bit Broadcast: Broadcasts mean and rstd from lane 0 to the entire warp in a single
#    64-bit shuffle instruction, which is more efficient than two separate 32-bit shuffles.
# 5. Launch Bounds: `__launch_bounds__(32)` provides hints to the compiler to optimize
#    register allocation for the known block size.
# 6. Compiler Optimizations: JIT compilation uses "-O3 --use_fast_math" for maximum speed.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Reduces a float2 value across a warp (32 threads) using shuffle instructions.
__device__ __forceinline__ float2 warp_reduce_sum_float2(float2 val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(0xffffffff, val.x, offset);
        val.y += __shfl_down_sync(0xffffffff, val.y, offset);
    }
    return val;
}

template<typename T>
__global__ void __launch_bounds__(32) add_layernorm_fwd_kernel_smem_free(
    T* __restrict__ output,
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    int C, // Embedding dimension (must be 128)
    float epsilon
) {
    // Each block processes one row (one token) using a single warp (32 threads).
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // --- Step 1: Vectorized Load and Add. Result stays in registers. ---
    const float4* input1_ptr = reinterpret_cast<const float4*>(input1 + row_idx * C);
    const float4* input2_ptr = reinterpret_cast<const float4*>(input2 + row_idx * C);
    const float4 v1 = input1_ptr[tid];
    const float4 v2 = input2_ptr[tid];
    const float4 sum_val = {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w};

    // --- Step 2: One-Pass Reduction from Registers using float2 ---
    float2 local_sums; // .x for sum, .y for sum_sq
    local_sums.x = sum_val.x + sum_val.y + sum_val.z + sum_val.w;
    local_sums.y = (sum_val.x * sum_val.x) + (sum_val.y * sum_val.y) + (sum_val.z * sum_val.z) + (sum_val.w * sum_val.w);
    
    // Reduce sum and sum_sq across the warp in a single combined operation.
    float2 total_sums = warp_reduce_sum_float2(local_sums);

    // --- Step 3: Compute Mean/Rstd and Broadcast using a single 64-bit Warp Shuffle ---
    float2 norm_params; // .x = mean, .y = rstd
    if (tid == 0) {
        norm_params.x = total_sums.x / C;
        float var = (total_sums.y / C) - (norm_params.x * norm_params.x);
        norm_params.y = rsqrtf(var + epsilon);
    }
    // Broadcast both mean and rstd from thread 0 in a single 64-bit operation.
    *(unsigned long long*)&norm_params = __shfl_sync(0xffffffff, *(unsigned long long*)&norm_params, 0);

    // --- Step 4: Final Normalization and Vectorized Store ---
    const float4* gamma_ptr = reinterpret_cast<const float4*>(gamma);
    const float4* beta_ptr = reinterpret_cast<const float4*>(beta);
    const float4 g = gamma_ptr[tid];
    const float4 b = beta_ptr[tid];

    float4 out_val;
    out_val.x = (sum_val.x - norm_params.x) * norm_params.y * g.x + b.x;
    out_val.y = (sum_val.y - norm_params.x) * norm_params.y * g.y + b.y;
    out_val.z = (sum_val.z - norm_params.x) * norm_params.y * g.z + b.z;
    out_val.w = (sum_val.w - norm_params.x) * norm_params.y * g.w + b.w;
    
    reinterpret_cast<float4*>(output + row_idx * C)[tid] = out_val;
}

// C++ wrapper to handle tensor operations and launch the kernel
torch::Tensor add_layernorm_cuda(
    torch::Tensor attn_output, 
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    double epsilon
) {
    const auto N = attn_output.size(0); // Total number of tokens
    const auto C = attn_output.size(1); // Embedding dimension
    
    TORCH_CHECK(C == 128, "This kernel is specialized for embed_dim=128");
    TORCH_CHECK(attn_output.is_contiguous() && x.is_contiguous(), "Input tensors must be contiguous");

    auto out = torch::empty_like(attn_output);
    
    // Launch configuration: One block per token, one warp per block. No shared memory needed.
    const int block_size = 32; // One warp
    const int grid_size = N;
    
    add_layernorm_fwd_kernel_smem_free<float><<<grid_size, block_size>>>(
        out.data_ptr<float>(),
        attn_output.data_ptr<float>(),
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        C,
        static_cast<float>(epsilon)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

cpp_source = "torch::Tensor add_layernorm_cuda(torch::Tensor attn_output, torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double epsilon);"

# JIT compile the CUDA kernel, using a unique name to avoid caching conflicts
# and enabling compiler optimizations for performance.
fused_add_layernorm_op = load_inline(
    name="fused_add_layernorm_smem_free_v4",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["add_layernorm_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class Model(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Model, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reshape for attention: (B, C, H*W) -> (S, B, C). S=H*W
        x_reshaped = x.view(B, C, H * W).permute(2, 0, 1).contiguous()
        
        # Multi-head attention. need_weights=False is a small optimization for inference.
        attn_output, _ = self.attn(x_reshaped, x_reshaped, x_reshaped, need_weights=False)
        
        seq_len, batch_size, embed_dim = attn_output.shape
        
        # Reshape to 2D (Token, Embedding) for the kernel.
        attn_output_flat = attn_output.view(-1, embed_dim)
        x_reshaped_flat = x_reshaped.view(-1, embed_dim)

        # Call the custom fused CUDA kernel for Add + LayerNorm
        x_norm_flat = fused_add_layernorm_op.add_layernorm_cuda(
            attn_output_flat, 
            x_reshaped_flat, 
            self.norm.weight, 
            self.norm.bias,
            self.norm.eps
        )
        
        # Reshape back to the 3D attention format
        x_out = x_norm_flat.view(seq_len, batch_size, embed_dim)
        
        # Reshape back to the original 4D image format
        x_out = x_out.permute(1, 2, 0).view(B, C, H, W)
        return x_out

embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128

def get_inputs():
    # Ensure input tensor is on the correct device for the CUDA kernel
    return [torch.randn(batch_size, num_channels, image_height, image_width).cuda()]

def get_init_inputs():
    return [embed_dim, num_heads]
# EVOLVE-BLOCK-END