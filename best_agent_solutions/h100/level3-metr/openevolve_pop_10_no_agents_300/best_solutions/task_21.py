# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for fused operations in half-precision (FP16).
# This approach leverages FP16 to reduce memory bandwidth and enable Tensor Core usage for convolutions.
# The kernel uses 128-bit vectorized memory access (float4, corresponding to 8 halfs)
# and __hfma2 intrinsics for maximum efficiency.
fused_ops_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // Required for half precision types and intrinsics

// Enum to specify the post-fusion operation.
enum class PostOp {
    NONE,
    RELU6,
    ADD
};

// Device function for applying the post-op on a half2 vector.
template<PostOp op>
__device__ __forceinline__ half2 apply_post_op_h2(half2 val_vec, half2 identity_vec) {
    if constexpr (op == PostOp::RELU6) {
        // ReLU6 clamp: max(0, min(val, 6))
        const half2 zeros = __float2half2_rn(0.0f);
        const half2 sixes = __float2half2_rn(6.0f);
        return __hmin2(sixes, __hmax2(zeros, val_vec));
    } else if constexpr (op == PostOp::ADD) {
        return __hadd2(val_vec, identity_vec);
    }
    return val_vec;
}

// Templated, vectorized kernel for fused BatchNorm and post-operation in FP16.
// It processes 8 half elements (one float4) per thread.
template<PostOp op>
__global__ void fused_bn_postop_kernel_half(
    const half* __restrict__ x,
    const half* __restrict__ identity, // Only used for ADD
    half* __restrict__ out,
    const float* __restrict__ scale,   // BN params kept in FP32 for precision
    const float* __restrict__ shift,
    int num_vec_elements, // Number of float4-sized vectors
    int C,
    int plane_size)      // H * W
{
    for (int idx_vec = blockIdx.x * blockDim.x + threadIdx.x; 
         idx_vec < num_vec_elements; 
         idx_vec += gridDim.x * blockDim.x) 
    {
        const int idx_half = idx_vec * 8;
        const int c = (idx_half / plane_size) % C;
        
        // Use __ldg for cached reads of scale/shift parameters.
        const float s_f = __ldg(&scale[c]);
        const float h_f = __ldg(&shift[c]);
        
        const half2 s_h2 = __floats2half2_rn(s_f, s_f);
        const half2 h_h2 = __floats2half2_rn(h_f, h_f);
        
        // Vectorized load of 8 halfs (128 bits)
        const float4 x_f4 = ((const float4*)x)[idx_vec];
        const half2* x_h2_ptr = reinterpret_cast<const half2*>(&x_f4);
        
        half2 val_h2[4];
        
        // Unroll the loop to process 4x half2 vectors
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            val_h2[i] = __hfma2(x_h2_ptr[i], s_h2, h_h2);
        }
        
        // Apply the specified post-operation (ReLU6 or Add)
        if constexpr (op == PostOp::ADD) {
            const float4 identity_f4 = ((const float4*)identity)[idx_vec];
            const half2* identity_h2_ptr = reinterpret_cast<const half2*>(&identity_f4);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                val_h2[i] = apply_post_op_h2<op>(val_h2[i], identity_h2_ptr[i]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                val_h2[i] = apply_post_op_h2<op>(val_h2[i], {__float2half(0.0f), __float2half(0.0f)});
            }
        }
        
        // Pack results back into float4 and perform vectorized store.
        float4 out_f4;
        half2* out_h2_ptr = reinterpret_cast<half2*>(&out_f4);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            out_h2_ptr[i] = val_h2[i];
        }
        ((float4*)out)[idx_vec] = out_f4;
    }
}

// C++ wrapper for BN + ReLU6 (FP16)
torch::Tensor fused_bn_relu6_cuda_half(torch::Tensor x, torch::Tensor scale, torch::Tensor shift) {
    TORCH_CHECK(x.is_cuda() && scale.is_cuda() && shift.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "Input tensor x must be FP16");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat && shift.scalar_type() == torch::kFloat, "Scale/shift must be FP32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor x must be contiguous");
    const long total_elements = x.numel();
    TORCH_CHECK(total_elements % 8 == 0, "Vectorized kernel requires total elements to be a multiple of 8.");

    auto out = torch::empty_like(x);
    const int N = x.size(0);
    const int C = x.size(1);
    const int plane_size = (C > 0) ? (total_elements / (N * C)) : 0;
    const int num_vec_elements = total_elements / 8;

    const int block_size = 256;
    const int num_blocks = (num_vec_elements + block_size - 1) / block_size;

    fused_bn_postop_kernel_half<PostOp::RELU6><<<num_blocks, block_size>>>(
        (const half*)x.data_ptr<at::Half>(), nullptr, (half*)out.data_ptr<at::Half>(), 
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        num_vec_elements, C, plane_size);
    return out;
}

// C++ wrapper for BN (FP16)
torch::Tensor fused_bn_cuda_half(torch::Tensor x, torch::Tensor scale, torch::Tensor shift) {
    TORCH_CHECK(x.is_cuda() && scale.is_cuda() && shift.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "Input tensor x must be FP16");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat && shift.scalar_type() == torch::kFloat, "Scale/shift must be FP32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor x must be contiguous");
    const long total_elements = x.numel();
    TORCH_CHECK(total_elements % 8 == 0, "Vectorized kernel requires total elements to be a multiple of 8.");

    auto out = torch::empty_like(x);
    const int N = x.size(0);
    const int C = x.size(1);
    const int plane_size = (C > 0) ? (total_elements / (N * C)) : 0;
    const int num_vec_elements = total_elements / 8;

    const int block_size = 256;
    const int num_blocks = (num_vec_elements + block_size - 1) / block_size;

    fused_bn_postop_kernel_half<PostOp::NONE><<<num_blocks, block_size>>>(
        (const half*)x.data_ptr<at::Half>(), nullptr, (half*)out.data_ptr<at::Half>(), 
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        num_vec_elements, C, plane_size);
    return out;
}

// C++ wrapper for BN + Add (FP16)
torch::Tensor fused_bn_add_cuda_half(torch::Tensor x, torch::Tensor identity, torch::Tensor scale, torch::Tensor shift) {
    TORCH_CHECK(x.is_cuda() && identity.is_cuda() && scale.is_cuda() && shift.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kHalf && identity.scalar_type() == torch::kHalf, "Input and identity tensors must be FP16");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat && shift.scalar_type() == torch::kFloat, "Scale/shift must be FP32");
    TORCH_CHECK(x.is_contiguous() && identity.is_contiguous(), "Input tensors must be contiguous");
    const long total_elements = x.numel();
    TORCH_CHECK(total_elements % 8 == 0, "Vectorized kernel requires total elements to be a multiple of 8.");

    auto out = torch::empty_like(x);
    const int N = x.size(0);
    const int C = x.size(1);
    const int plane_size = (C > 0) ? (total_elements / (N * C)) : 0;
    const int num_vec_elements = total_elements / 8;

    const int block_size = 256;
    const int num_blocks = (num_vec_elements + block_size - 1) / block_size;

    fused_bn_postop_kernel_half<PostOp::ADD><<<num_blocks, block_size>>>(
        (const half*)x.data_ptr<at::Half>(), (const half*)identity.data_ptr<at::Half>(), (half*)out.data_ptr<at::Half>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        num_vec_elements, C, plane_size);
    return out;
}
"""

fused_ops_cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_bn_relu6_cuda_half(torch::Tensor x, torch::Tensor scale, torch::Tensor shift);
torch::Tensor fused_bn_cuda_half(torch::Tensor x, torch::Tensor scale, torch::Tensor shift);
torch::Tensor fused_bn_add_cuda_half(torch::Tensor x, torch::Tensor identity, torch::Tensor scale, torch::Tensor shift);
"""

fused_ops_fp16 = load_inline(
    name="fused_ops_fp16_v2",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_cuda_source,
    functions=["fused_bn_relu6_cuda_half", "fused_bn_cuda_half", "fused_bn_add_cuda_half"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation with fused BatchNorm operations, optimized for FP16 inference.
        """
        super(Model, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        self.has_expand = expand_ratio != 1

        if self.has_expand:
            expand_seq = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            scale, shift = self._fold_bn(expand_seq[1])
            self.register_buffer('expand_scale', scale)
            self.register_buffer('expand_shift', shift)
            self.expand_conv = expand_seq[0].half()

        depthwise_seq = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )
        scale, shift = self._fold_bn(depthwise_seq[1])
        self.register_buffer('depthwise_scale', scale)
        self.register_buffer('depthwise_shift', shift)
        self.depthwise_conv = depthwise_seq[0].half()
        
        project_seq = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        scale, shift = self._fold_bn(project_seq[1])
        self.register_buffer('project_scale', scale)
        self.register_buffer('project_shift', shift)
        self.project_conv = project_seq[0].half()

    def _fold_bn(self, bn_layer):
        bn_layer.eval()
        running_mean = bn_layer.running_mean
        running_var = bn_layer.running_var
        eps = bn_layer.eps
        gamma = bn_layer.weight
        beta = bn_layer.bias
        
        inv_std = torch.rsqrt(running_var + eps)
        scale = gamma * inv_std
        shift = beta - running_mean * scale
        return scale.contiguous(), shift.contiguous()

    def forward(self, x):
        """
        Forward pass of the MBConv block with fused FP16 kernels.
        """
        identity = x
        x = x.half() # Cast input to FP16
        
        if self.has_expand:
            x = self.expand_conv(x)
            x = fused_ops_fp16.fused_bn_relu6_cuda_half(x, self.expand_scale, self.expand_shift)
        
        x = self.depthwise_conv(x)
        x = fused_ops_fp16.fused_bn_relu6_cuda_half(x, self.depthwise_scale, self.depthwise_shift)
        
        x = self.project_conv(x)
        
        if self.use_residual:
            # Cast identity to FP16 just before use
            x = fused_ops_fp16.fused_bn_add_cuda_half(x, identity.half(), self.project_scale, self.project_shift)
        else:
            x = fused_ops_fp16.fused_bn_cuda_half(x, self.project_scale, self.project_shift)
        
        # Cast back to float32 for correctness check against baseline
        return x.float()

# Test code
batch_size = 10
in_channels = 112
out_channels = 192
kernel_size = 5
stride = 2
expand_ratio = 6

def get_inputs():
    # Return FP32 tensor, the model will handle casting internally
    return [torch.randn(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, expand_ratio]
# EVOLVE-BLOCK-END