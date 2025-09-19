# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This solution combines the most effective strategies from all high-performing prior attempts:
# 1.  In-Place Fusion: Fuses `BatchNorm + ReLU` and `BatchNorm + Add + ReLU` to reduce
#     kernel launches and memory traffic. In-place operations are used to avoid memory
#     allocation overhead, which proved to be the fastest approach.
# 2.  Hybrid Vectorization: Employs `float4` vectorized kernels for maximum memory bandwidth
#     on aligned data with compatible dimensions, and dynamically dispatches to a scalar
#     kernel otherwise. This ensures high performance on all feature map sizes.
# 3.  Optimized Launch Parameters: The block size for the critical scalar fallback path
#     is set to 512, matching the configuration of the top-performing prior solution. This
#     is often a sweet spot for occupancy on modern GPUs. The grid size calculation is
#     unconstrained, allowing the scheduler to manage parallelism effectively.
# 4.  Grid-Stride Loops: All kernels use a grid-stride loop, a robust CUDA programming
#     pattern that improves hardware utilization and makes kernels independent of grid size.

fused_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

// --- Kernel 1: In-place BatchNorm + ReLU ---

// Vectorized version with grid-stride loop
__global__ void bn_relu_inplace_kernel_vec4(
    float* __restrict__ data,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ mean, const float* __restrict__ var,
    const float eps, const int C, const int HW, const int n_float4) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_float4;
         i += gridDim.x * blockDim.x) {
        const int c = ((i * 4) / HW) % C;

        const float scale = weight[c] * rsqrtf(var[c] + eps);
        const float shift = bias[c] - mean[c] * scale;

        float4 val_data = ((float4*)data)[i];
        val_data.x = fmaxf(0.0f, val_data.x * scale + shift);
        val_data.y = fmaxf(0.0f, val_data.y * scale + shift);
        val_data.z = fmaxf(0.0f, val_data.z * scale + shift);
        val_data.w = fmaxf(0.0f, val_data.w * scale + shift);
        ((float4*)data)[i] = val_data;
    }
}

// Scalar fallback with grid-stride loop
__global__ void bn_relu_inplace_kernel_scalar(
    float* __restrict__ data,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ mean, const float* __restrict__ var,
    const float eps, const int C, const int HW, const long total_elements) {

    for (long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {
        const int c = (idx / HW) % C;
        const float scale = weight[c] * rsqrtf(var[c] + eps);
        const float shift = bias[c] - mean[c] * scale;
        data[idx] = fmaxf(0.0f, data[idx] * scale + shift);
    }
}

// --- Kernel 2: In-place BatchNorm + Add + ReLU ---

// Vectorized version with grid-stride loop
__global__ void bn_add_relu_inplace_kernel_vec4(
    float* __restrict__ data, const float* __restrict__ identity,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ mean, const float* __restrict__ var,
    const float eps, const int C, const int HW, const int n_float4) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_float4;
         i += gridDim.x * blockDim.x) {
        const int c = ((i * 4) / HW) % C;

        const float scale = weight[c] * rsqrtf(var[c] + eps);
        const float shift = bias[c] - mean[c] * scale;

        float4 val_data = ((float4*)data)[i];
        const float4 val_identity = ((const float4*)identity)[i];

        val_data.x = fmaxf(0.0f, (val_data.x * scale + shift) + val_identity.x);
        val_data.y = fmaxf(0.0f, (val_data.y * scale + shift) + val_identity.y);
        val_data.z = fmaxf(0.0f, (val_data.z * scale + shift) + val_identity.z);
        val_data.w = fmaxf(0.0f, (val_data.w * scale + shift) + val_identity.w);
        ((float4*)data)[i] = val_data;
    }
}

// Scalar fallback with grid-stride loop
__global__ void bn_add_relu_inplace_kernel_scalar(
    float* __restrict__ data, const float* __restrict__ identity,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ mean, const float* __restrict__ var,
    const float eps, const int C, const int HW, const long total_elements) {

    for (long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {
        const int c = (idx / HW) % C;
        const float scale = weight[c] * rsqrtf(var[c] + eps);
        const float shift = bias[c] - mean[c] * scale;
        data[idx] = fmaxf(0.0f, (data[idx] * scale + shift) + identity[idx]);
    }
}

// --- C++ Wrappers with dispatch logic ---

torch::Tensor bn_relu_inplace_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mean, torch::Tensor var, double eps) {
    
    TORCH_CHECK(x.is_cuda() && x.dim() == 4, "Input tensor x must be a 4D CUDA tensor");
    auto x_cont = x.contiguous();

    const int C = x_cont.size(1), H = x_cont.size(2), W = x_cont.size(3);
    const long total_elements = x_cont.numel();
    const int HW = H * W;
    float* data_ptr = x_cont.data_ptr<float>();

    bool can_vectorize = (total_elements % 4 == 0) && (HW % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(data_ptr) % 16 == 0);
    
    if (can_vectorize) {
        const int n_float4 = total_elements / 4;
        const int block_size = 256;
        const int grid_size = (n_float4 + block_size - 1) / block_size;
        bn_relu_inplace_kernel_vec4<<<grid_size, block_size>>>(
            data_ptr, weight.data_ptr<float>(), bias.data_ptr<float>(),
            mean.data_ptr<float>(), var.data_ptr<float>(), (float)eps, C, HW, n_float4);
    } else {
        const int block_size = 512; // Optimized block size for scalar path
        const int grid_size = (total_elements + block_size - 1) / block_size;
        bn_relu_inplace_kernel_scalar<<<grid_size, block_size>>>(
            data_ptr, weight.data_ptr<float>(), bias.data_ptr<float>(),
            mean.data_ptr<float>(), var.data_ptr<float>(), (float)eps, C, HW, total_elements);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(err)); }
    return x_cont;
}

torch::Tensor bn_add_relu_inplace_cuda(
    torch::Tensor x, torch::Tensor identity, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mean, torch::Tensor var, double eps) {
    
    TORCH_CHECK(x.is_cuda() && identity.is_cuda(), "Input tensors must be CUDA tensors");
    TORCH_CHECK(x.sizes() == identity.sizes(), "Input tensors must have the same shape");

    auto x_cont = x.contiguous();
    auto identity_cont = identity.contiguous();

    const int C = x_cont.size(1), H = x_cont.size(2), W = x_cont.size(3);
    const long total_elements = x_cont.numel();
    const int HW = H * W;
    float* data_ptr = x_cont.data_ptr<float>();
    const float* identity_ptr = identity_cont.data_ptr<float>();

    bool can_vectorize = (total_elements % 4 == 0) && (HW % 4 == 0) &&
                         (reinterpret_cast<uintptr_t>(data_ptr) % 16 == 0) &&
                         (reinterpret_cast<uintptr_t>(identity_ptr) % 16 == 0);

    if (can_vectorize) {
        const int n_float4 = total_elements / 4;
        const int block_size = 256;
        const int grid_size = (n_float4 + block_size - 1) / block_size;
        bn_add_relu_inplace_kernel_vec4<<<grid_size, block_size>>>(
            data_ptr, identity_ptr, weight.data_ptr<float>(), bias.data_ptr<float>(),
            mean.data_ptr<float>(), var.data_ptr<float>(), (float)eps, C, HW, n_float4);
    } else {
        const int block_size = 512; // Optimized block size for scalar path
        const int grid_size = (total_elements + block_size - 1) / block_size;
        bn_add_relu_inplace_kernel_scalar<<<grid_size, block_size>>>(
            data_ptr, identity_ptr, weight.data_ptr<float>(), bias.data_ptr<float>(),
            mean.data_ptr<float>(), var.data_ptr<float>(), (float)eps, C, HW, total_elements);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(err)); }
    return x_cont;
}
"""

fused_kernels_cpp_source = """
torch::Tensor bn_relu_inplace_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor mean, torch::Tensor var, double eps);
torch::Tensor bn_add_relu_inplace_cuda(torch::Tensor x, torch::Tensor identity, torch::Tensor weight, torch::Tensor bias, torch::Tensor mean, torch::Tensor var, double eps);
"""

try:
    fused_kernels = load_inline(
        name="fused_resnet_kernels_v_final", 
        cpp_sources=fused_kernels_cpp_source,
        cuda_sources=fused_kernels_source,
        functions=["bn_relu_inplace_cuda", "bn_add_relu_inplace_cuda"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
except Exception as e:
    print(f"Warning: Failed to load custom CUDA kernels. Falling back to PyTorch. Error: {e}")
    class FallbackModule:
        def bn_relu_inplace_cuda(self, x, weight, bias, mean, var, eps):
            x = F.batch_norm(x, mean, var, weight, bias, training=False, eps=eps)
            return F.relu(x, inplace=True)
        def bn_add_relu_inplace_cuda(self, x, identity, weight, bias, mean, var, eps):
            x = F.batch_norm(x, mean, var, weight, bias, training=False, eps=eps)
            x += identity
            return F.relu(x, inplace=True)
    fused_kernels = FallbackModule()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        # nn.ReLU is removed as its function is fused into the custom kernels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = fused_kernels.bn_relu_inplace_cuda(
            out, self.bn1.weight, self.bn1.bias, self.bn1.running_mean, self.bn1.running_var, self.bn1.eps
        )

        out = self.conv2(out)
        out = fused_kernels.bn_relu_inplace_cuda(
            out, self.bn2.weight, self.bn2.bias, self.bn2.running_mean, self.bn2.running_var, self.bn2.eps
        )

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = fused_kernels.bn_add_relu_inplace_cuda(
            out, identity, self.bn3.weight, self.bn3.bias, self.bn3.running_mean, self.bn3.running_var, self.bn3.eps
        )

        return out

class Model(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(Model, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        block = Bottleneck
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        x = fused_kernels.bn_relu_inplace_cuda(
            x, self.bn1.weight, self.bn1.bias, self.bn1.running_mean, self.bn1.running_var, self.bn1.eps
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
batch_size = 10
height = 224
width = 224
layers = [3, 4, 23, 3]
num_classes = 1000

def get_inputs():
    # Ensure input is contiguous for safe pointer access by CUDA kernels
    return [torch.randn(batch_size, 3, height, width).cuda().contiguous()]

def get_init_inputs():
    return [layers, num_classes]
# EVOLVE-BLOCK-END