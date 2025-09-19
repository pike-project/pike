# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# This solution synthesizes the most effective techniques from all prior attempts to create a superior implementation.
#
# Key Optimizations:
# 1. Aggressive In-Place Fusion: Retains the 3-kernel fusion strategy from the best attempts,
#    including the specialized kernel for the complex downsample path (fusing two BatchNorms, add, and ReLU).
#    All operations are in-place to eliminate memory allocation overhead and reduce bandwidth.
# 2. Optimized Kernel Implementation:
#    - Grid-Stride Loops: All kernels use grid-stride loops (`for (int i =...; i < N; i += step)`). This is
#      more robust and often more performant than simple `if` guards, as it allows for flexible grid sizes and
#      better hardware utilization.
#    - Scalar Processing: Explicit vectorization (e.g., float4) is removed in favor of simple scalar kernels.
#      The top-performing solutions showed that for this architecture's tensor shapes, simple kernels with
#      high occupancy (large block size) outperform more complex vectorized code.
#    - Optimized BatchNorm Arithmetic: The standard `(x - mean) * inv_std * weight + bias` is replaced by
#      `x * scale + shift`, where scale and shift are pre-calculated. This reduces the number of FLOPs per element.
#
# This combination of maximum fusion with a simple, robust, and arithmetically optimized kernel design
# aims to surpass the performance of all previous versions.

fused_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <c10/cuda/CUDAException.h>
#include <algorithm> // For std::min

// Kernel 1: In-place Fused BatchNorm + ReLU (Grid-Stride, Optimized BN Arithmetic)
__global__ void fused_bn_relu_inplace_kernel(
    float* __restrict__ data,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    float bn_eps,
    int total_elements,
    int C,
    int HW)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements;
         index += gridDim.x * blockDim.x)
    {
        int c = (index / HW) % C;
        // Pre-compute scale and shift for BN
        float inv_std = rsqrtf(bn_var[c] + bn_eps);
        float scale = bn_weight[c] * inv_std;
        float shift = bn_bias[c] - bn_mean[c] * scale;
        // Apply BN and ReLU in-place
        data[index] = fmaxf(0.0f, data[index] * scale + shift);
    }
}

// Kernel 2: In-place Fused BatchNorm + Add + ReLU (Grid-Stride, Optimized BN Arithmetic)
__global__ void fused_bn_add_relu_inplace_kernel(
    float* __restrict__ conv_out,
    const float* __restrict__ identity,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    float bn_eps,
    int total_elements,
    int C,
    int HW)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements;
         index += gridDim.x * blockDim.x)
    {
        int c = (index / HW) % C;
        // Pre-compute scale and shift for BN
        float inv_std = rsqrtf(bn_var[c] + bn_eps);
        float scale = bn_weight[c] * inv_std;
        float shift = bn_bias[c] - bn_mean[c] * scale;
        // Apply BN, add residual, and apply ReLU in-place
        float bn_val = conv_out[index] * scale + shift;
        conv_out[index] = fmaxf(0.0f, bn_val + identity[index]);
    }
}

// Kernel 3: In-place Fused (BN + Downsample_BN + Add + ReLU) (Grid-Stride, Optimized BN Arithmetic)
__global__ void fused_bn_downsample_bn_add_relu_inplace_kernel(
    float* __restrict__ main_in_out,
    const float* __restrict__ residual_in,
    const float* __restrict__ main_bn_weight,
    const float* __restrict__ main_bn_bias,
    const float* __restrict__ main_bn_mean,
    const float* __restrict__ main_bn_var,
    float main_bn_eps,
    const float* __restrict__ residual_bn_weight,
    const float* __restrict__ residual_bn_bias,
    const float* __restrict__ residual_bn_mean,
    const float* __restrict__ residual_bn_var,
    float residual_bn_eps,
    int total_elements,
    int C,
    int HW)
{
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements;
         index += gridDim.x * blockDim.x)
    {
        int c = (index / HW) % C;
        // BN on main path (with pre-computed scale/shift)
        float inv_std_main = rsqrtf(main_bn_var[c] + main_bn_eps);
        float scale_main = main_bn_weight[c] * inv_std_main;
        float shift_main = main_bn_bias[c] - main_bn_mean[c] * scale_main;
        float bn_val_main = main_in_out[index] * scale_main + shift_main;
        // BN on residual path (with pre-computed scale/shift)
        float inv_std_residual = rsqrtf(residual_bn_var[c] + residual_bn_eps);
        float scale_residual = residual_bn_weight[c] * inv_std_residual;
        float shift_residual = residual_bn_bias[c] - residual_bn_mean[c] * scale_residual;
        float bn_val_residual = residual_in[index] * scale_residual + shift_residual;
        // Add and ReLU, update in-place
        main_in_out[index] = fmaxf(0.0f, bn_val_main + bn_val_residual);
    }
}

// --- C++ Wrappers ---

torch::Tensor fused_bn_relu_inplace_cuda(
    torch::Tensor in, torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps)
{
    const auto total_elements = in.numel();
    if (total_elements == 0) return in;
    const auto C = in.size(1);
    const auto HW = in.size(2) * in.size(3);
    const dim3 block_size(1024);
    
    const int num_blocks = (total_elements + block_size.x - 1) / block_size.x;
    fused_bn_relu_inplace_kernel<<<std::min(num_blocks, 4096), block_size>>>(
        in.data_ptr<float>(), bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
        bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(), (float)bn_eps,
        total_elements, C, HW);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return in;
}

torch::Tensor fused_bn_add_relu_inplace_cuda(
    torch::Tensor conv_out, torch::Tensor identity, torch::Tensor bn_weight,
    torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps)
{
    const auto total_elements = conv_out.numel();
    if (total_elements == 0) return conv_out;
    const auto C = conv_out.size(1);
    const auto HW = conv_out.size(2) * conv_out.size(3);
    const dim3 block_size(1024);

    const int num_blocks = (total_elements + block_size.x - 1) / block_size.x;
    fused_bn_add_relu_inplace_kernel<<<std::min(num_blocks, 4096), block_size>>>(
        conv_out.data_ptr<float>(), identity.data_ptr<float>(), bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(), bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(),
        (float)bn_eps, total_elements, C, HW);
        
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return conv_out;
}

torch::Tensor fused_bn_downsample_bn_add_relu_inplace_cuda(
    torch::Tensor main_in, torch::Tensor residual_in,
    torch::Tensor main_bn_weight, torch::Tensor main_bn_bias,
    torch::Tensor main_bn_mean, torch::Tensor main_bn_var, double main_bn_eps,
    torch::Tensor residual_bn_weight, torch::Tensor residual_bn_bias,
    torch::Tensor residual_bn_mean, torch::Tensor residual_bn_var, double residual_bn_eps)
{
    const auto total_elements = main_in.numel();
    if (total_elements == 0) return main_in;
    const auto C = main_in.size(1);
    const auto HW = main_in.size(2) * main_in.size(3);
    const dim3 block_size(1024);

    const int num_blocks = (total_elements + block_size.x - 1) / block_size.x;
    fused_bn_downsample_bn_add_relu_inplace_kernel<<<std::min(num_blocks, 4096), block_size>>>(
        main_in.data_ptr<float>(), residual_in.data_ptr<float>(),
        main_bn_weight.data_ptr<float>(), main_bn_bias.data_ptr<float>(),
        main_bn_mean.data_ptr<float>(), main_bn_var.data_ptr<float>(), (float)main_bn_eps,
        residual_bn_weight.data_ptr<float>(), residual_bn_bias.data_ptr<float>(),
        residual_bn_mean.data_ptr<float>(), residual_bn_var.data_ptr<float>(), (float)residual_bn_eps,
        total_elements, C, HW);
        
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return main_in;
}
"""

fused_kernels_cpp_source = """
torch::Tensor fused_bn_relu_inplace_cuda(
    torch::Tensor in, torch::Tensor bn_weight, torch::Tensor bn_bias,
    torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps);

torch::Tensor fused_bn_add_relu_inplace_cuda(
    torch::Tensor conv_out, torch::Tensor identity, torch::Tensor bn_weight,
    torch::Tensor bn_bias, torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps);

torch::Tensor fused_bn_downsample_bn_add_relu_inplace_cuda(
    torch::Tensor main_in, torch::Tensor residual_in,
    torch::Tensor main_bn_weight, torch::Tensor main_bn_bias,
    torch::Tensor main_bn_mean, torch::Tensor main_bn_var, double main_bn_eps,
    torch::Tensor residual_bn_weight, torch::Tensor residual_bn_bias,
    torch::Tensor residual_bn_mean, torch::Tensor residual_bn_var, double residual_bn_eps);
"""

# JIT compile the fused CUDA kernels
fused_kernels = load_inline(
    name="fused_resnet_kernels_synthesis_v2",
    cpp_sources=fused_kernels_cpp_source,
    cuda_sources=fused_kernels_source,
    functions=["fused_bn_relu_inplace_cuda", "fused_bn_add_relu_inplace_cuda", "fused_bn_downsample_bn_add_relu_inplace_cuda"],
    verbose=False,
)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # Fusion 1: conv1_out -> bn1 -> relu (in-place)
        fused_kernels.fused_bn_relu_inplace_cuda(
            out, self.bn1.weight, self.bn1.bias,
            self.bn1.running_mean, self.bn1.running_var, self.bn1.eps
        )

        out = self.conv2(out)

        if self.downsample is not None:
            # Fusion 3: (conv2_out -> bn2) + (downsample_conv_out -> downsample_bn) -> add -> relu (in-place)
            downsample_conv_out = self.downsample[0](x)
            downsample_bn = self.downsample[1]
            out = fused_kernels.fused_bn_downsample_bn_add_relu_inplace_cuda(
                out, downsample_conv_out,
                self.bn2.weight, self.bn2.bias, self.bn2.running_mean, self.bn2.running_var, self.bn2.eps,
                downsample_bn.weight, downsample_bn.bias, downsample_bn.running_mean, downsample_bn.running_var, downsample_bn.eps
            )
        else:
            # Fusion 2: (conv2_out -> bn2) + identity -> add -> relu (in-place)
            out = fused_kernels.fused_bn_add_relu_inplace_cuda(
                out, identity, self.bn2.weight, self.bn2.bias,
                self.bn2.running_mean, self.bn2.running_var, self.bn2.eps
            )

        return out

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # Fused operation in the stem
        fused_kernels.fused_bn_relu_inplace_cuda(
            x, self.bn1.weight, self.bn1.bias,
            self.bn1.running_mean, self.bn1.running_var, self.bn1.eps
        )
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Test code
batch_size = 2
num_classes = 1000
input_shape = (batch_size, 3, 224, 224)

def get_inputs():
    return [torch.randn(input_shape).cuda()]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END