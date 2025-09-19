# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

# -----------------------------------------------------------------------------
# 1. CUDA/C++ Source Code for Fused Operators
# -----------------------------------------------------------------------------
# This source combines two highly effective fusion strategies:
#   1. A set of specialized, non-templated kernels for post-convolution operations
#      (fused folded BatchNorm, optional residual add, and activation). This
#      approach, inspired by the top-performing prior solution, uses a large
#      block size (1024) and avoids GPU-side branching entirely.
#   2. A kernel for fused adaptive average pooling and flattening, which avoids
#      writing an intermediate tensor to global memory.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// =====================================================================================
// KERNEL SET 1: Specialized Post-Convolution Kernels (Folded BN + Activation + Add)
// =====================================================================================

// Kernel for Affine Transform (Input * scale + bias)
__global__ void fused_affine_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int total_elements,
    int channels,
    int spatial_dim)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += blockDim.x * gridDim.x) {
        int c = (i / spatial_dim) % channels;
        output[i] = input[i] * scale[c] + bias[c];
    }
}

// Kernel for Affine Transform + ReLU
__global__ void fused_affine_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int total_elements,
    int channels,
    int spatial_dim)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += blockDim.x * gridDim.x) {
        int c = (i / spatial_dim) % channels;
        output[i] = fmaxf(0.0f, input[i] * scale[c] + bias[c]);
    }
}

// Kernel for Affine Transform + ReLU6
__global__ void fused_affine_relu6_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int total_elements,
    int channels,
    int spatial_dim)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += blockDim.x * gridDim.x) {
        int c = (i / spatial_dim) % channels;
        output[i] = fminf(fmaxf(0.0f, input[i] * scale[c] + bias[c]), 6.0f);
    }
}

// Kernel for Affine Transform + Add (for residual connections)
__global__ void fused_affine_add_kernel(
    const float* __restrict__ input,
    const float* __restrict__ identity,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int total_elements,
    int channels,
    int spatial_dim)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += blockDim.x * gridDim.x) {
        int c = (i / spatial_dim) % channels;
        output[i] = (input[i] * scale[c] + bias[c]) + identity[i];
    }
}


// =====================================================================================
// KERNEL 2: Fused Adaptive Average Pooling + Flatten
// =====================================================================================

__global__ void fused_adaptive_avg_pool_flatten_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C, const int HW)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= N || c >= C) return;

    float sum = 0.0f;
    const float* feature_map_start = input + (n * C + c) * HW;
    for (int i = 0; i < HW; ++i) {
        sum += feature_map_start[i];
    }
    output[n * C + c] = sum / (float)HW;
}

// =====================================================================================
// C++ WRAPPERS
// =====================================================================================

// Generic launcher for the affine kernels
template<typename KernelFunc>
void launch_affine_kernel(torch::Tensor input, torch::Tensor scale, torch::Tensor bias,
                          c10::optional<torch::Tensor> identity_opt, torch::Tensor output, KernelFunc kernel)
{
    const int total_elements = input.numel();
    const int channels = input.size(1);
    const int spatial_dim = (input.dim() > 2) ? input.size(2) * input.size(3) : 1;
    const int block_size = 1024;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    const float* identity_ptr = identity_opt.has_value() ? identity_opt.value().data_ptr<float>() : nullptr;

    kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), identity_ptr, scale.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), total_elements, channels, spatial_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


torch::Tensor fused_affine_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias) {
    auto out = torch::empty_like(input);
    launch_affine_kernel(input, scale, bias, c10::nullopt, out,
        (void (*)(const float*, const float*, const float*, const float*, float*, int, int, int))fused_affine_kernel);
    return out;
}

torch::Tensor fused_affine_relu_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias) {
    auto out = torch::empty_like(input);
    launch_affine_kernel(input, scale, bias, c10::nullopt, out,
        (void (*)(const float*, const float*, const float*, const float*, float*, int, int, int))fused_affine_relu_kernel);
    return out;
}

torch::Tensor fused_affine_relu6_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias) {
    auto out = torch::empty_like(input);
    launch_affine_kernel(input, scale, bias, c10::nullopt, out,
        (void (*)(const float*, const float*, const float*, const float*, float*, int, int, int))fused_affine_relu6_kernel);
    return out;
}

torch::Tensor fused_affine_add_cuda(torch::Tensor input, torch::Tensor identity, torch::Tensor scale, torch::Tensor bias) {
    auto out = torch::empty_like(input);
    launch_affine_kernel(input, scale, bias, identity, out, fused_affine_add_kernel);
    return out;
}

torch::Tensor fused_adaptive_avg_pool_flatten_cuda(torch::Tensor input) {
    input = input.contiguous();
    const auto N = input.size(0);
    const auto C = input.size(1);
    const int HW = input.size(2) * input.size(3);

    auto output = torch::empty({N, C}, input.options());
    const dim3 block_size(16, 16);
    const dim3 num_blocks((C + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    fused_adaptive_avg_pool_flatten_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N, C, HW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

"""

fused_ops_cpp_source = """
torch::Tensor fused_affine_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias);
torch::Tensor fused_affine_relu_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias);
torch::Tensor fused_affine_relu6_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias);
torch::Tensor fused_affine_add_cuda(torch::Tensor input, torch::Tensor identity, torch::Tensor scale, torch::Tensor bias);
torch::Tensor fused_adaptive_avg_pool_flatten_cuda(torch::Tensor input);
"""

# -----------------------------------------------------------------------------
# 2. JIT Compilation of the Fused Kernels
# -----------------------------------------------------------------------------
build_dir = f"/tmp/torch_extensions/fused_efficientnet_{os.getpid()}"
os.makedirs(build_dir, exist_ok=True)

fused_ops_module = load_inline(
    name="fused_efficientnet_hybrid",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_affine_cuda", "fused_affine_relu_cuda", "fused_affine_relu6_cuda", "fused_affine_add_cuda", "fused_adaptive_avg_pool_flatten_cuda"],
    verbose=False,
    build_directory=build_dir,
)

# -----------------------------------------------------------------------------
# 3. Rewritten Model using Fused Kernels
# -----------------------------------------------------------------------------

class FusedMBConv(nn.Module):
    """ MBConv block with fused operations for inference. """
    def __init__(self, original_mbconv):
        super().__init__()
        self.use_residual = original_mbconv.use_residual
        self.has_expand_conv = hasattr(original_mbconv, 'expand_conv')

        if self.has_expand_conv:
            self.expand_conv = original_mbconv.expand_conv[0]
            scale, bias = self._fuse_bn_params(original_mbconv.expand_conv[1])
            self.register_buffer('expand_scale', scale)
            self.register_buffer('expand_bias', bias)

        self.depthwise_conv = original_mbconv.depthwise_conv[0]
        scale, bias = self._fuse_bn_params(original_mbconv.depthwise_conv[1])
        self.register_buffer('depthwise_scale', scale)
        self.register_buffer('depthwise_bias', bias)
        
        self.project_conv = original_mbconv.project_conv[0]
        scale, bias = self._fuse_bn_params(original_mbconv.project_conv[1])
        self.register_buffer('project_scale', scale)
        self.register_buffer('project_bias', bias)
        
    def _fuse_bn_params(self, bn):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bias = bn.bias - bn.running_mean * scale
        return scale, bias

    def forward(self, x):
        identity = x
        
        if self.has_expand_conv:
            out = self.expand_conv(x)
            out = fused_ops_module.fused_affine_relu6_cuda(out, self.expand_scale, self.expand_bias)
        else:
            out = x

        out = self.depthwise_conv(out)
        out = fused_ops_module.fused_affine_relu6_cuda(out, self.depthwise_scale, self.depthwise_bias)
        
        out = self.project_conv(out)
        if self.use_residual:
            out = fused_ops_module.fused_affine_add_cuda(out, identity, self.project_scale, self.project_bias)
        else:
            out = fused_ops_module.fused_affine_cuda(out, self.project_scale, self.project_bias)
        
        return out


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        original_model = OriginalModel(num_classes)
        original_model.eval()

        self.conv1 = original_model.conv1
        scale1, bias1 = self._fuse_bn_params(original_model.bn1)
        self.register_buffer('scale1', scale1)
        self.register_buffer('bias1', bias1)
        
        fused_blocks = [FusedMBConv(block) for block in original_model.blocks]
        self.blocks = nn.Sequential(*fused_blocks)
        
        self.conv2 = original_model.conv2
        scale2, bias2 = self._fuse_bn_params(original_model.bn2)
        self.register_buffer('scale2', scale2)
        self.register_buffer('bias2', bias2)
        
        self.fc = original_model.fc
        
    def _fuse_bn_params(self, bn):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bias = bn.bias - bn.running_mean * scale
        return scale, bias
    
    def forward(self, x):
        x = self.conv1(x)
        x = fused_ops_module.fused_affine_relu_cuda(x, self.scale1, self.bias1)
        
        x = self.blocks(x)
        
        x = self.conv2(x)
        x = fused_ops_module.fused_affine_relu_cuda(x, self.scale2, self.bias2)
        
        x = fused_ops_module.fused_adaptive_avg_pool_flatten_cuda(x)
        
        x = self.fc(x)
        return x

# -----------------------------------------------------------------------------
# 4. Original Model Definition (for initialization)
# -----------------------------------------------------------------------------
class OriginalMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(OriginalMBConv, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        if self.use_residual:
            x += identity
        return x

class OriginalModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(OriginalModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.blocks = nn.Sequential(
            OriginalMBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            OriginalMBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            OriginalMBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            OriginalMBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            OriginalMBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            OriginalMBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            OriginalMBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            OriginalMBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            OriginalMBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            OriginalMBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            OriginalMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            OriginalMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            OriginalMBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -----------------------------------------------------------------------------
# 5. Input Generation Functions (Unchanged)
# -----------------------------------------------------------------------------
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END