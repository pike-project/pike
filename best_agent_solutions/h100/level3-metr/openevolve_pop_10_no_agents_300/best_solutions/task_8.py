# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Fusing the two parallel BatchNorms, the Add, and the final ReLU is the key optimization.
# This version builds on the top-performing solution by:
# 1. Adopting the aggressive `bn_bn_add_relu` fusion strategy.
# 2. Incorporating `__launch_bounds__` from other attempts to guide the compiler
#    in optimizing register usage and maximizing thread occupancy, which is crucial for
#    memory-bound kernels.
# 3. Cleaning up the C++ code with `reinterpret_cast` and ensuring tensors are contiguous.
# 4. Removing the unused `bn_add_relu` kernel to simplify the code.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimal block size for memory-bound kernels to maximize occupancy.
constexpr int BLOCK_SIZE = 512;

// Vectorized kernel to fuse BatchNorm and ReLU using float4.
// Added __launch_bounds__ for better performance tuning by the compiler.
__global__ void __launch_bounds__(BLOCK_SIZE)
bn_relu_kernel_optimized(const float4* __restrict__ in, float4* __restrict__ out,
                         const float* __restrict__ mean, const float* __restrict__ var,
                         const float* __restrict__ weight, const float* __restrict__ bias,
                         float eps, int C, int HW, int total_f4_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_f4_elements) {
        const int c = ((idx * 4) / HW) % C;

        // Pre-calculate scale and shift for FMA (Fused Multiply-Add) optimization.
        const float inv_std_c = rsqrtf(var[c] + eps);
        const float scale = weight[c] * inv_std_c;
        const float shift = bias[c] - mean[c] * scale;

        const float4 in_val = in[idx];
        float4 out_val;
        
        out_val.x = fmaxf(0.f, in_val.x * scale + shift);
        out_val.y = fmaxf(0.f, in_val.y * scale + shift);
        out_val.z = fmaxf(0.f, in_val.z * scale + shift);
        out_val.w = fmaxf(0.f, in_val.w * scale + shift);
        
        out[idx] = out_val;
    }
}

// Aggressively fused kernel: Fuses two parallel BatchNorms, an Add, and a ReLU using float4.
// Added __launch_bounds__ for performance.
__global__ void __launch_bounds__(BLOCK_SIZE)
bn_bn_add_relu_kernel_optimized(
    const float4* __restrict__ in1, const float4* __restrict__ in2, float4* __restrict__ out,
    const float* __restrict__ mean1, const float* __restrict__ var1,
    const float* __restrict__ weight1, const float* __restrict__ bias1, float eps1,
    const float* __restrict__ mean2, const float* __restrict__ var2,
    const float* __restrict__ weight2, const float* __restrict__ bias2, float eps2,
    int C, int HW, int total_f4_elements) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_f4_elements) {
        const int c = ((idx * 4) / HW) % C;

        // Pre-calculate scale and shift for the first BatchNorm
        const float inv_std1_c = rsqrtf(var1[c] + eps1);
        const float scale1 = weight1[c] * inv_std1_c;
        const float shift1 = bias1[c] - mean1[c] * scale1;

        // Pre-calculate scale and shift for the second BatchNorm
        const float inv_std2_c = rsqrtf(var2[c] + eps2);
        const float scale2 = weight2[c] * inv_std2_c;
        const float shift2 = bias2[c] - mean2[c] * scale2;

        const float4 in1_val = in1[idx];
        const float4 in2_val = in2[idx];
        float4 out_val;
        
        // Apply first BN
        const float bn1_x = in1_val.x * scale1 + shift1;
        const float bn1_y = in1_val.y * scale1 + shift1;
        const float bn1_z = in1_val.z * scale1 + shift1;
        const float bn1_w = in1_val.w * scale1 + shift1;
        
        // Apply second BN
        const float bn2_x = in2_val.x * scale2 + shift2;
        const float bn2_y = in2_val.y * scale2 + shift2;
        const float bn2_z = in2_val.z * scale2 + shift2;
        const float bn2_w = in2_val.w * scale2 + shift2;

        // Add results and apply ReLU
        out_val.x = fmaxf(0.f, bn1_x + bn2_x);
        out_val.y = fmaxf(0.f, bn1_y + bn2_y);
        out_val.z = fmaxf(0.f, bn1_z + bn2_z);
        out_val.w = fmaxf(0.f, bn1_w + bn2_w);

        out[idx] = out_val;
    }
}

// C++ wrapper for bn_relu_cuda.
torch::Tensor bn_relu_cuda(torch::Tensor in, torch::Tensor weight, torch::Tensor bias,
                           torch::Tensor running_mean, torch::Tensor running_var, double eps) {
    TORCH_CHECK(in.is_cuda() && in.is_contiguous(), "Input tensor must be a contiguous CUDA tensor");
    const auto total_elements = in.numel();
    TORCH_CHECK(total_elements % 4 == 0, "Input tensor size must be divisible by 4 for vectorized kernel");
    
    auto out = torch::empty_like(in);
    const int total_f4_elements = total_elements / 4;
    const int num_blocks = (total_f4_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    bn_relu_kernel_optimized<<<num_blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const float4*>(in.data_ptr<float>()), 
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        (float)eps, in.size(1), in.size(2) * in.size(3), total_f4_elements);
    return out;
}

// C++ wrapper for the new bn_bn_add_relu kernel
torch::Tensor bn_bn_add_relu_cuda(
    torch::Tensor in1, torch::Tensor in2,
    torch::Tensor weight1, torch::Tensor bias1, torch::Tensor running_mean1, torch::Tensor running_var1, double eps1,
    torch::Tensor weight2, torch::Tensor bias2, torch::Tensor running_mean2, torch::Tensor running_var2, double eps2) {
    
    TORCH_CHECK(in1.is_cuda() && in1.is_contiguous(), "Input tensor 1 must be a contiguous CUDA tensor");
    TORCH_CHECK(in2.is_cuda() && in2.is_contiguous(), "Input tensor 2 must be a contiguous CUDA tensor");
    TORCH_CHECK(in1.sizes() == in2.sizes(), "Input tensors must have the same size");
    const auto total_elements = in1.numel();
    TORCH_CHECK(total_elements % 4 == 0, "Input tensor size must be divisible by 4 for vectorized kernel");

    auto out = torch::empty_like(in1);
    const int total_f4_elements = total_elements / 4;
    const int num_blocks = (total_f4_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    bn_bn_add_relu_kernel_optimized<<<num_blocks, BLOCK_SIZE>>>(
        reinterpret_cast<const float4*>(in1.data_ptr<float>()), 
        reinterpret_cast<const float4*>(in2.data_ptr<float>()), 
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        running_mean1.data_ptr<float>(), running_var1.data_ptr<float>(),
        weight1.data_ptr<float>(), bias1.data_ptr<float>(), (float)eps1,
        running_mean2.data_ptr<float>(), running_var2.data_ptr<float>(),
        weight2.data_ptr<float>(), bias2.data_ptr<float>(), (float)eps2,
        in1.size(1), in1.size(2) * in1.size(3), total_f4_elements);
    return out;
}
"""

fused_ops_cpp_source = """
torch::Tensor bn_relu_cuda(torch::Tensor in, torch::Tensor weight, torch::Tensor bias,
                           torch::Tensor running_mean, torch::Tensor running_var, double eps);
torch::Tensor bn_bn_add_relu_cuda(
    torch::Tensor in1, torch::Tensor in2,
    torch::Tensor weight1, torch::Tensor bias1, torch::Tensor running_mean1, torch::Tensor running_var1, double eps1,
    torch::Tensor weight2, torch::Tensor bias2, torch::Tensor running_mean2, torch::Tensor running_var2, double eps2);
"""

# Compile the inline CUDA code, using a unique name to avoid caching conflicts.
fused_ops = load_inline(
    name="fused_ops_ultimate",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["bn_relu_cuda", "bn_bn_add_relu_cuda"],
    verbose=True,
)


class Model(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride
        # All ReLU layers are removed as their functionality is fused into our custom kernels.

    def forward(self, x):
        # Block 1: Conv -> Fused BN-ReLU
        out = self.conv1(x)
        out = fused_ops.bn_relu_cuda(
            out,
            self.bn1.weight,
            self.bn1.bias,
            self.bn1.running_mean,
            self.bn1.running_var,
            self.bn1.eps
        )

        # Get the outputs of the convolutions from both parallel paths
        out_conv2 = self.conv2(out)
        
        downsample_conv = self.downsample[0]
        downsample_bn = self.downsample[1]
        identity_conv_out = downsample_conv(x)

        # Block 2: Apply the ultimate kernel that fuses the BN from both paths, the add, and the ReLU
        out = fused_ops.bn_bn_add_relu_cuda(
            out_conv2,                     # Input from main path (already contiguous from conv)
            identity_conv_out,             # Input from shortcut path (already contiguous from conv)
            self.bn2.weight,               # BN params for main path
            self.bn2.bias,
            self.bn2.running_mean,
            self.bn2.running_var,
            self.bn2.eps,
            downsample_bn.weight,          # BN params for shortcut path
            downsample_bn.bias,
            downsample_bn.running_mean,
            downsample_bn.running_var,
            downsample_bn.eps
        )

        return out
    
# Test code
in_channels = 3
out_channels = 64
stride = 1
batch_size = 10
num_classes = 1000

def get_inputs():
    # Input tensor must be on GPU and contiguous for vectorized access.
    return [torch.randn(batch_size, in_channels, 224, 224).cuda().contiguous()]

def get_init_inputs():
    return [in_channels, out_channels, stride]
# EVOLVE-BLOCK-END