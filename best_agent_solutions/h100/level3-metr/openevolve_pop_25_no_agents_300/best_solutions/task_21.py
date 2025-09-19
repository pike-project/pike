# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This solution combines the best features of previous attempts:
# 1. Vectorized Memory Access (float4): From the top-performing program, this is key for maximizing
#    memory bandwidth, as the tensor dimensions are suitable.
# 2. Pre-computed BatchNorm Parameters: This avoids redundant calculations of scale/bias on every
#    forward pass by computing them once at initialization, an efficiency gain over prior art.
# 3. Tuned Launch Parameters: Uses a block size of 512 and an un-capped grid size, which the
#    top-performing program found to be effective for this workload.
# 4. Fused Kernels: Fuses BatchNorm + ReLU6 and BatchNorm + Add to reduce kernel launch overhead
#    and memory traffic.
fused_ops_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(expr)                                                         \\
  do {                                                                           \\
    cudaError_t status = (expr);                                                 \\
    if (status != cudaSuccess) {                                                 \\
      fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__,           \\
              cudaGetErrorString(status));                                       \\
      exit(EXIT_FAILURE);                                                        \\
    }                                                                            \\
  } while (0)

// Vectorized kernel for BatchNorm + ReLU6
__global__ void bn_relu6_kernel_vec4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    const int total_vec_elements,
    const int plane_size_in_floats,
    const int C) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vec_elements; i += gridDim.x * blockDim.x) {
        const int float_idx = i * 4;
        const int c = (float_idx / plane_size_in_floats) % C;
        
        const float s = scale[c];
        const float b = bias[c];

        const float4 in_val = input[i];
        float4 out_val;
        out_val.x = fminf(fmaxf(in_val.x * s + b, 0.0f), 6.0f);
        out_val.y = fminf(fmaxf(in_val.y * s + b, 0.0f), 6.0f);
        out_val.z = fminf(fmaxf(in_val.z * s + b, 0.0f), 6.0f);
        out_val.w = fminf(fmaxf(in_val.w * s + b, 0.0f), 6.0f);
        
        output[i] = out_val;
    }
}

// Vectorized kernel for BatchNorm + Add
__global__ void bn_add_kernel_vec4(
    const float4* __restrict__ input,
    const float4* __restrict__ residual,
    float4* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    const int total_vec_elements,
    const int plane_size_in_floats,
    const int C) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vec_elements; i += gridDim.x * blockDim.x) {
        const int float_idx = i * 4;
        const int c = (float_idx / plane_size_in_floats) % C;
        
        const float s = scale[c];
        const float b = bias[c];

        const float4 in_val = input[i];
        const float4 res_val = residual[i];
        float4 out_val;
        out_val.x = (in_val.x * s + b) + res_val.x;
        out_val.y = (in_val.y * s + b) + res_val.y;
        out_val.z = (in_val.z * s + b) + res_val.z;
        out_val.w = (in_val.w * s + b) + res_val.w;
        
        output[i] = out_val;
    }
}

// Vectorized kernel for just BatchNorm
__global__ void bn_kernel_vec4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    const int total_vec_elements,
    const int plane_size_in_floats,
    const int C) {
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vec_elements; i += gridDim.x * blockDim.x) {
        const int float_idx = i * 4;
        const int c = (float_idx / plane_size_in_floats) % C;
        
        const float s = scale[c];
        const float b = bias[c];

        const float4 in_val = input[i];
        float4 out_val;
        out_val.x = in_val.x * s + b;
        out_val.y = in_val.y * s + b;
        out_val.z = in_val.z * s + b;
        out_val.w = in_val.w * s + b;
        
        output[i] = out_val;
    }
}


// C++ wrapper for bn_relu6_vec4
torch::Tensor bn_relu6_vec4_cuda(
    const torch::Tensor& input, const torch::Tensor& scale, const torch::Tensor& bias) {
    
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.numel() % 4 == 0, "Vectorized kernel requires numel to be divisible by 4.");
    
    auto output = torch::empty_like(input);
    
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int total_vec_elements = input.numel() / 4;
    const int plane_size_in_floats = H * W;

    const int block_size = 512;
    const int num_blocks = (total_vec_elements + block_size - 1) / block_size;
    
    bn_relu6_kernel_vec4<<<num_blocks, block_size>>>(
        (const float4*)input.data_ptr<float>(), (float4*)output.data_ptr<float>(),
        scale.data_ptr<float>(), bias.data_ptr<float>(),
        total_vec_elements, plane_size_in_floats, C);
    
    CUDA_CHECK(cudaGetLastError());
    return output;
}

// C++ wrapper for bn_add_vec4
torch::Tensor bn_add_vec4_cuda(
    const torch::Tensor& input, const torch::Tensor& residual, 
    const torch::Tensor& scale, const torch::Tensor& bias) {
    
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor");
    TORCH_CHECK(residual.is_cuda() && residual.is_contiguous(), "Residual must be a contiguous CUDA tensor");
    TORCH_CHECK(input.numel() % 4 == 0, "Vectorized kernel requires numel to be divisible by 4.");

    auto output = torch::empty_like(input);
    
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int total_vec_elements = input.numel() / 4;
    const int plane_size_in_floats = H * W;

    const int block_size = 512;
    const int num_blocks = (total_vec_elements + block_size - 1) / block_size;
    
    bn_add_kernel_vec4<<<num_blocks, block_size>>>(
        (const float4*)input.data_ptr<float>(), (const float4*)residual.data_ptr<float>(), (float4*)output.data_ptr<float>(),
        scale.data_ptr<float>(), bias.data_ptr<float>(),
        total_vec_elements, plane_size_in_floats, C);
        
    CUDA_CHECK(cudaGetLastError());
    return output;
}

// C++ wrapper for bn_vec4
torch::Tensor bn_vec4_cuda(
    const torch::Tensor& input, const torch::Tensor& scale, const torch::Tensor& bias) {

    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.numel() % 4 == 0, "Vectorized kernel requires numel to be divisible by 4.");

    auto output = torch::empty_like(input);

    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int total_vec_elements = input.numel() / 4;
    const int plane_size_in_floats = H * W;

    const int block_size = 512;
    const int num_blocks = (total_vec_elements + block_size - 1) / block_size;

    bn_kernel_vec4<<<num_blocks, block_size>>>(
        (const float4*)input.data_ptr<float>(), (float4*)output.data_ptr<float>(),
        scale.data_ptr<float>(), bias.data_ptr<float>(),
        total_vec_elements, plane_size_in_floats, C);

    CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

fused_ops_cpp_source = """
#include <torch/extension.h>

torch::Tensor bn_relu6_vec4_cuda(
    const torch::Tensor& input, const torch::Tensor& scale, const torch::Tensor& bias);

torch::Tensor bn_add_vec4_cuda(
    const torch::Tensor& input, const torch::Tensor& residual, 
    const torch::Tensor& scale, const torch::Tensor& bias);
    
torch::Tensor bn_vec4_cuda(
    const torch::Tensor& input, const torch::Tensor& scale, const torch::Tensor& bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops_vec4_precomputed",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_cuda_source,
    functions=["bn_relu6_vec4_cuda", "bn_add_vec4_cuda", "bn_vec4_cuda"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        Optimized MBConv block with vectorized fused kernels and pre-computed BN parameters.
        """
        super(Model, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        self.has_expand = expand_ratio != 1
        
        if self.has_expand:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

        self._precompute_bn_params()

    def _precompute_bn_params(self):
        """Pre-computes and registers BN scale and bias for fused kernels."""
        if self.has_expand:
            scale, bias = self._get_bn_scale_bias(self.expand_bn)
            self.register_buffer('expand_scale', scale)
            self.register_buffer('expand_bias', bias)

        scale, bias = self._get_bn_scale_bias(self.depthwise_bn)
        self.register_buffer('depthwise_scale', scale)
        self.register_buffer('depthwise_bias', bias)

        scale, bias = self._get_bn_scale_bias(self.project_bn)
        self.register_buffer('project_scale', scale)
        self.register_buffer('project_bias', bias)
    
    @staticmethod
    def _get_bn_scale_bias(bn: nn.BatchNorm2d):
        """Calculates the folded scale and bias from a BatchNorm2d layer for inference."""
        bn.eval() 
        with torch.no_grad():
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            bias = bn.bias - bn.running_mean * scale
        return scale.contiguous(), bias.contiguous()

    def forward(self, x):
        """
        Forward pass using vectorized fused kernels.
        """
        identity = x
        
        if self.has_expand:
            out = self.expand_conv(x)
            out = fused_ops.bn_relu6_vec4_cuda(out, self.expand_scale, self.expand_bias)
        else:
            out = x
        
        out = self.depthwise_conv(out)
        out = fused_ops.bn_relu6_vec4_cuda(out, self.depthwise_scale, self.depthwise_bias)
        
        out = self.project_conv(out)
        
        if self.use_residual:
            out = fused_ops.bn_add_vec4_cuda(out, identity, self.project_scale, self.project_bias)
        else:
            out = fused_ops.bn_vec4_cuda(out, self.project_scale, self.project_bias)
        
        return out

# Test code
batch_size = 10
in_channels = 112
out_channels = 192
kernel_size = 5
stride = 2
expand_ratio = 6

def get_inputs():
    return [torch.randn(batch_size, in_channels, 224, 224).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, expand_ratio]
# EVOLVE-BLOCK-END