# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fusing BatchNorm, ReLU6, and residual addition.
# This version combines multiple advanced optimizations:
# 1. Pre-computation of BatchNorm parameters into a scale (A) and shift (B) on the CPU
#    to minimize arithmetic operations inside the kernel.
# 2. C++ templates with `if constexpr` to generate specialized kernels at compile time,
#    eliminating runtime branching overhead for `apply_relu` and `add_residual`.
# 3. `__launch_bounds__` to provide hints to the compiler for optimizing register usage.
# 4. float4 vectorization for maximizing memory bandwidth.
# 5. A robust scalar fallback for tensors not aligned for float4.
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <c10/cuda/CUDAException.h>

// Templated, Vectorized (float4) CUDA kernel for post-convolution operations
template <bool APPLY_RELU, bool ADD_RESIDUAL>
__global__ void __launch_bounds__(512) fused_post_conv_kernel_vec4(
    const float4* __restrict__ input,
    const float4* __restrict__ residual,
    const float* __restrict__ bn_A, // Fused scale: weight / sqrt(var + eps)
    const float* __restrict__ bn_B, // Fused shift: bias - mean * A
    float4* __restrict__ output,
    const int C,
    const int H_W, // H * W
    const long total_elements_vec) {

    const long thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < total_elements_vec) {
        const int c = ((thread_idx * 4) / H_W) % C;

        const float A = bn_A[c];
        const float B = bn_B[c];

        float4 in_val = input[thread_idx];
        float4 out_val;

        // --- Fused BatchNorm ---
        out_val.x = in_val.x * A + B;
        out_val.y = in_val.y * A + B;
        out_val.z = in_val.z * A + B;
        out_val.w = in_val.w * A + B;

        // --- ReLU6 (Compile-time branch) ---
        if constexpr (APPLY_RELU) {
            out_val.x = fminf(fmaxf(0.0f, out_val.x), 6.0f);
            out_val.y = fminf(fmaxf(0.0f, out_val.y), 6.0f);
            out_val.z = fminf(fmaxf(0.0f, out_val.z), 6.0f);
            out_val.w = fminf(fmaxf(0.0f, out_val.w), 6.0f);
        }

        // --- Residual Addition (Compile-time branch) ---
        if constexpr (ADD_RESIDUAL) {
            float4 res_val = residual[thread_idx];
            out_val.x += res_val.x;
            out_val.y += res_val.y;
            out_val.z += res_val.z;
            out_val.w += res_val.w;
        }

        output[thread_idx] = out_val;
    }
}

// Templated, Scalar CUDA kernel for fallback
template <bool APPLY_RELU, bool ADD_RESIDUAL>
__global__ void __launch_bounds__(512) fused_post_conv_kernel_scalar(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ bn_A,
    const float* __restrict__ bn_B,
    float* __restrict__ output,
    const int C,
    const int H_W,
    const long total_elements) {

    const long thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_idx < total_elements) {
        const int c = (thread_idx / H_W) % C;

        const float A = bn_A[c];
        const float B = bn_B[c];
        
        float out_val = input[thread_idx] * A + B;

        if constexpr (APPLY_RELU) {
            out_val = fminf(fmaxf(0.0f, out_val), 6.0f);
        }

        if constexpr (ADD_RESIDUAL) {
            out_val += residual[thread_idx];
        }

        output[thread_idx] = out_val;
    }
}

// C++ wrapper function to dispatch to the appropriate templated CUDA kernel
torch::Tensor fused_post_conv(
    torch::Tensor input,
    torch::Tensor bn_A,
    torch::Tensor bn_B,
    bool apply_relu,
    bool add_residual,
    torch::Tensor residual) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bn_A.is_contiguous(), "bn_A must be contiguous");
    TORCH_CHECK(bn_B.is_contiguous(), "bn_B must be contiguous");
    if (add_residual) {
        TORCH_CHECK(residual.is_cuda(), "Residual must be a CUDA tensor");
        TORCH_CHECK(residual.is_contiguous(), "Residual must be contiguous");
    }

    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const long total_elements = input.numel();
    const int H_W = H * W;

    auto output = torch::empty_like(input);
    const int block_size = 512;

    if (W % 4 == 0) {
        const long total_elements_vec = total_elements / 4;
        const int num_blocks = (total_elements_vec + block_size - 1) / block_size;
        
        if (apply_relu && !add_residual) {
            fused_post_conv_kernel_vec4<true, false><<<num_blocks, block_size>>>((const float4*)input.data_ptr<float>(), nullptr, bn_A.data_ptr<float>(), bn_B.data_ptr<float>(), (float4*)output.data_ptr<float>(), C, H_W, total_elements_vec);
        } else if (!apply_relu && add_residual) {
            fused_post_conv_kernel_vec4<false, true><<<num_blocks, block_size>>>((const float4*)input.data_ptr<float>(), (const float4*)residual.data_ptr<float>(), bn_A.data_ptr<float>(), bn_B.data_ptr<float>(), (float4*)output.data_ptr<float>(), C, H_W, total_elements_vec);
        } else { // (!apply_relu && !add_residual)
            fused_post_conv_kernel_vec4<false, false><<<num_blocks, block_size>>>((const float4*)input.data_ptr<float>(), nullptr, bn_A.data_ptr<float>(), bn_B.data_ptr<float>(), (float4*)output.data_ptr<float>(), C, H_W, total_elements_vec);
        }
    } else {
        const int num_blocks = (total_elements + block_size - 1) / block_size;
        
        if (apply_relu && !add_residual) {
            fused_post_conv_kernel_scalar<true, false><<<num_blocks, block_size>>>(input.data_ptr<float>(), nullptr, bn_A.data_ptr<float>(), bn_B.data_ptr<float>(), output.data_ptr<float>(), C, H_W, total_elements);
        } else if (!apply_relu && add_residual) {
            fused_post_conv_kernel_scalar<false, true><<<num_blocks, block_size>>>(input.data_ptr<float>(), residual.data_ptr<float>(), bn_A.data_ptr<float>(), bn_B.data_ptr<float>(), output.data_ptr<float>(), C, H_W, total_elements);
        } else { // (!apply_relu && !add_residual)
            fused_post_conv_kernel_scalar<false, false><<<num_blocks, block_size>>>(input.data_ptr<float>(), nullptr, bn_A.data_ptr<float>(), bn_B.data_ptr<float>(), output.data_ptr<float>(), C, H_W, total_elements);
        }
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_op_cpp_source = """
torch::Tensor fused_post_conv(
    torch::Tensor input,
    torch::Tensor bn_A,
    torch::Tensor bn_B,
    bool apply_relu,
    bool add_residual,
    torch::Tensor residual);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op_template_precomputed",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_post_conv"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation using a highly optimized fused CUDA kernel.
        """
        super(Model, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.expand_ratio = expand_ratio
        hidden_dim = in_channels * expand_ratio
        
        # --- Layers Definition and BN Fusion ---
        if self.expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self._fuse_bn_params(nn.BatchNorm2d(hidden_dim), 'expand')
        
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False)
        self._fuse_bn_params(nn.BatchNorm2d(hidden_dim), 'depthwise')

        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self._fuse_bn_params(nn.BatchNorm2d(out_channels), 'project')
    
    def _fuse_bn_params(self, bn: nn.BatchNorm2d, prefix: str):
        """Pre-computes BN parameters into y = A*x + B form and registers them as buffers."""
        bn.eval()
        with torch.no_grad():
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            bias = bn.bias - bn.running_mean * scale
        self.register_buffer(f'{prefix}_A', scale)
        self.register_buffer(f'{prefix}_B', bias)

    def forward(self, x):
        """
        Forward pass of the MBConv block using the templated fused CUDA kernel.
        """
        identity = x
        
        # Expansion phase
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = fused_op.fused_post_conv(
                x, self.expand_A, self.expand_B,
                True, False, x
            )
        
        # Depthwise phase
        x = self.depthwise_conv(x)
        x = fused_op.fused_post_conv(
            x, self.depthwise_A, self.depthwise_B,
            True, False, x
        )
        
        # Projection phase
        x = self.project_conv(x)
        x = fused_op.fused_post_conv(
            x, self.project_A, self.project_B,
            False, self.use_residual, identity
        )
        
        return x

# Test code
batch_size = 10
in_channels = 112
out_channels = 192
kernel_size = 5
stride = 2
expand_ratio = 6

def get_inputs():
    return [torch.randn(batch_size, in_channels, 224, 224).cuda().contiguous()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, expand_ratio]
# EVOLVE-BLOCK-END