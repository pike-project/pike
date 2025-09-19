# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for fused operations:
# 1. (BatchNorm + ReLU): For the first part of a residual block.
# 2. (BatchNorm + Add + ReLU): For the second part of a residual block.
# 3. (BatchNorm + ReLU + MaxPool): A new, aggressive fusion for the initial network stem,
#    which operates on the largest feature maps, yielding significant memory bandwidth savings.
# Micro-optimizations like using `rsqrtf` and `fmaxf` are used for maximum performance.
fused_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath> // For fmaxf
#include <cfloat> // For FLT_MAX

// Kernel 1: Fused operation for out = relu(BN(a))
__global__ void fused_bn_relu_kernel(
    const float* __restrict__ a,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ mean, const float* __restrict__ var,
    float* __restrict__ out,
    int size, int C, int plane_size, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c = (idx / plane_size) % C;
        float scale = weight[c] * rsqrtf(var[c] + eps);
        float shift = bias[c] - mean[c] * scale;
        float bn_val = a[idx] * scale + shift;
        out[idx] = fmaxf(bn_val, 0.0f);
    }
}

// Kernel 2: Fused operation for out = relu(BN(a) + b)
__global__ void fused_bn_add_relu_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ mean, const float* __restrict__ var,
    float* __restrict__ out,
    int size, int C, int plane_size, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int c = (idx / plane_size) % C;
        float scale = weight[c] * rsqrtf(var[c] + eps);
        float shift = bias[c] - mean[c] * scale;
        float bn_val = a[idx] * scale + shift;
        float add_val = bn_val + b[idx];
        out[idx] = fmaxf(add_val, 0.0f);
    }
}

// Kernel 3: Fused operation for out = MaxPool(ReLU(BN(a)))
// This kernel handles a 3x3 kernel, stride 2, padding 1 maxpool.
__global__ void fused_bn_relu_maxpool_kernel(
    const float* __restrict__ a,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ mean, const float* __restrict__ var,
    float* __restrict__ out,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    float eps)
{
    const int K = 3, S = 2, P = 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_size = N * C * H_out * W_out;

    if (idx < out_size) {
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int n = idx / (C * W_out * H_out);

        float scale = weight[c] * rsqrtf(var[c] + eps);
        float shift = bias[c] - mean[c] * scale;

        int h_start = h_out * S - P;
        int w_start = w_out * S - P;

        float max_val = -FLT_MAX;

        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h = h_start + kh;
                int w = w_start + kw;

                if (h >= 0 && h < H_in && w >= 0 && w < W_in) {
                    int in_idx = n * C * H_in * W_in + c * H_in * W_in + h * W_in + w;
                    float bn_val = a[in_idx] * scale + shift;
                    float relu_val = fmaxf(bn_val, 0.0f);
                    max_val = fmaxf(max_val, relu_val);
                }
            }
        }
        out[idx] = max_val;
    }
}


// C++ wrapper for BN+ReLU kernel
torch::Tensor fused_bn_relu_cuda(
    torch::Tensor a, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mean, torch::Tensor var, double eps)
{
    auto out = torch::empty_like(a);
    if (a.numel() == 0) return out;
    const auto shapes = a.sizes();
    const int C = shapes[1];
    const int plane_size = shapes[2] * shapes[3];
    const int size = a.numel();
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_bn_relu_kernel<<<num_blocks, block_size, 0, stream>>>(
        a.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        mean.data_ptr<float>(), var.data_ptr<float>(), out.data_ptr<float>(),
        size, C, plane_size, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// C++ wrapper for BN+Add+ReLU kernel
torch::Tensor fused_bn_add_relu_cuda(
    torch::Tensor a, torch::Tensor b, torch::Tensor weight,
    torch::Tensor bias, torch::Tensor mean, torch::Tensor var, double eps)
{
    auto out = torch::empty_like(a);
    if (a.numel() == 0) return out;
    const auto shapes = a.sizes();
    const int C = shapes[1];
    const int plane_size = shapes[2] * shapes[3];
    const int size = a.numel();
    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_bn_add_relu_kernel<<<num_blocks, block_size, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        mean.data_ptr<float>(), var.data_ptr<float>(), out.data_ptr<float>(),
        size, C, plane_size, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// C++ wrapper for BN+ReLU+MaxPool kernel
torch::Tensor fused_bn_relu_maxpool_cuda(
    torch::Tensor a, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mean, torch::Tensor var, double eps)
{
    const auto shapes = a.sizes();
    const int N = shapes[0], C = shapes[1], H_in = shapes[2], W_in = shapes[3];
    const int H_out = (H_in + 2 * 1 - 3) / 2 + 1;
    const int W_out = (W_in + 2 * 1 - 3) / 2 + 1;

    auto out = torch::empty({N, C, H_out, W_out}, a.options());
    const int size = out.numel();
    if (size == 0) return out;

    const int block_size = 1024;
    const int num_blocks = (size + block_size - 1) / block_size;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_bn_relu_maxpool_kernel<<<num_blocks, block_size, 0, stream>>>(
        a.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        mean.data_ptr<float>(), var.data_ptr<float>(), out.data_ptr<float>(),
        N, C, H_in, W_in, H_out, W_out, (float)eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

fused_kernels_cpp_source = """
torch::Tensor fused_bn_relu_cuda(
    torch::Tensor a, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mean, torch::Tensor var, double eps);

torch::Tensor fused_bn_add_relu_cuda(
    torch::Tensor a, torch::Tensor b, torch::Tensor weight,
    torch::Tensor bias, torch::Tensor mean, torch::Tensor var,
    double eps);

torch::Tensor fused_bn_relu_maxpool_cuda(
    torch::Tensor a, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mean, torch::Tensor var, double eps);
"""

# JIT compile the inline CUDA code. A unique name is used to avoid caching issues.
fused_ops = load_inline(
    name="fused_resnet_kernels_v_final",
    cpp_sources=fused_kernels_cpp_source,
    cuda_sources=fused_kernels_source,
    functions=["fused_bn_relu_cuda", "fused_bn_add_relu_cuda", "fused_bn_relu_maxpool_cuda"],
    verbose=True,
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
        # Fuse BN1 + ReLU
        out = fused_ops.fused_bn_relu_cuda(
            out, self.bn1.weight, self.bn1.bias,
            self.bn1.running_mean, self.bn1.running_var, self.bn1.eps
        )

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Fuse BN2 + Add + ReLU
        out = fused_ops.fused_bn_add_relu_cuda(
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
        # MaxPool is now fused into a custom kernel, so the nn.MaxPool2d layer is removed.

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
        # Use the new aggressive fusion for the initial BN + ReLU + MaxPool
        x = fused_ops.fused_bn_relu_maxpool_cuda(
            x, self.bn1.weight, self.bn1.bias,
            self.bn1.running_mean, self.bn1.running_var, self.bn1.eps
        )

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