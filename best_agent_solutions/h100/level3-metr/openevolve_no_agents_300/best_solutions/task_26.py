# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Combined CUDA and C++ source for all custom kernels
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For fmaxf

// Kernel for Channel Shuffle (Gather-based for coalesced writes)
__global__ void channel_shuffle_kernel(
    const float* input,
    float* output,
    int total_elements,
    int channels,
    int spatial_dim,
    int groups
) {
    int C_per_group = channels / groups;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += blockDim.x * gridDim.x) {
        int n = idx / (channels * spatial_dim);
        int c_out = (idx / spatial_dim) % channels;
        int s = idx % spatial_dim;

        // Map output channel 'c_out' to input channel 'c_in'
        int c_pg = c_out / groups;
        int g = c_out % groups;
        int c_in = g * C_per_group + c_pg;

        int in_idx = n * (channels * spatial_dim) + c_in * spatial_dim + s;
        output[idx] = input[in_idx];
    }
}


// Kernel for Fused BatchNorm (eval) + ReLU
__global__ void bn_relu_kernel(const float* x, float* y, const float* scale, const float* shift, int N, int C, int plane_size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        int c = (i / plane_size) % C;
        float val = x[i] * scale[c] + shift[c];
        y[i] = fmaxf(0.f, val);
    }
}

// Kernel for 3-way Fused BatchNorm (eval) + ReLU + Add
__global__ void bn_relu_add_kernel(const float* x, float* y, const float* scale, const float* shift, const float* shortcut, int N, int C, int plane_size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        int c = (i / plane_size) % C;
        float val = x[i] * scale[c] + shift[c]; // BatchNorm
        val = fmaxf(0.f, val);                 // ReLU
        y[i] = val + shortcut[i];              // Add
    }
}


// Wrapper for Channel Shuffle
torch::Tensor channel_shuffle_cuda(torch::Tensor x, int64_t groups) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "Input must be a contiguous CUDA tensor");
    const auto channels = x.size(1);
    const auto height = x.size(2);
    const auto width = x.size(3);
    const auto numel = x.numel();
    const int spatial_dim = height * width;

    auto output = torch::empty_like(x);

    const int threads_per_block = 256;
    const int num_blocks = (numel + threads_per_block - 1) / threads_per_block;

    channel_shuffle_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), numel, channels, spatial_dim, groups);
    return output;
}

// Wrapper for Fused BatchNorm + ReLU
torch::Tensor bn_relu_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor shift) {
    const auto N = x.numel();
    const auto C = x.size(1);
    const auto plane_size = x.size(2) * x.size(3);
    auto y = torch::empty_like(x);
    const int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    bn_relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(), N, C, plane_size);
    return y;
}

// Wrapper for Fused BatchNorm + ReLU + Add
torch::Tensor bn_relu_add_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor shift, torch::Tensor shortcut) {
    const auto N = x.numel();
    const auto C = x.size(1);
    const auto plane_size = x.size(2) * x.size(3);
    auto y = torch::empty_like(x);
    const int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    bn_relu_add_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), scale.data_ptr<float>(), shift.data_ptr<float>(), shortcut.data_ptr<float>(), N, C, plane_size);
    return y;
}
"""

cpp_source = """
torch::Tensor channel_shuffle_cuda(torch::Tensor x, int64_t groups);
torch::Tensor bn_relu_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor shift);
torch::Tensor bn_relu_add_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor shift, torch::Tensor shortcut);
"""

# Compile the inline CUDA code for all custom operations
custom_ops = load_inline(
    name="custom_shufflenet_v2_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["channel_shuffle_cuda", "bn_relu_cuda", "bn_relu_add_cuda"],
    verbose=False,
)


class FusedBnReLU(nn.Module):
    """ Fuses BatchNorm2d (in eval mode) and ReLU. """
    def __init__(self, bn_layer):
        super().__init__()
        self.bn_layer = bn_layer.eval()
        with torch.no_grad():
            gamma, beta, mean, var, eps = bn_layer.weight, bn_layer.bias, bn_layer.running_mean, bn_layer.running_var, bn_layer.eps
            inv_std = 1.0 / torch.sqrt(var + eps)
            scale = gamma * inv_std
            shift = beta - mean * inv_std * gamma
            self.register_buffer('scale', scale)
            self.register_buffer('shift', shift)

    def forward(self, x):
        return custom_ops.bn_relu_cuda(x, self.scale, self.shift)


class FusedBnReLUAdd(nn.Module):
    """ Fuses BatchNorm2d (eval), ReLU, and Add operations. """
    def __init__(self, bn_layer):
        super().__init__()
        self.bn_layer = bn_layer.eval()
        with torch.no_grad():
            gamma, beta, mean, var, eps = bn_layer.weight, bn_layer.bias, bn_layer.running_mean, bn_layer.running_var, bn_layer.eps
            inv_std = 1.0 / torch.sqrt(var + eps)
            scale = gamma * inv_std
            shift = beta - mean * inv_std * gamma
            self.register_buffer('scale', scale)
            self.register_buffer('shift', shift)

    def forward(self, x, shortcut):
        return custom_ops.bn_relu_add_cuda(x, self.scale, self.shift, shortcut)


class ChannelShuffle(nn.Module):
    """ Channel shuffle operation using a custom CUDA kernel. """
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        return custom_ops.channel_shuffle_cuda(x, self.groups)


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnit, self).__init__()
        
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, groups=groups, bias=False)
        self.fused_bn_relu1 = FusedBnReLU(nn.BatchNorm2d(mid_channels))
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1, 0, groups=groups, bias=False)
        self.fused_bn_relu_add3 = FusedBnReLUAdd(nn.BatchNorm2d(out_channels))
        
        self.shuffle = ChannelShuffle(groups)
        
        self.is_identity_shortcut = (in_channels == out_channels)
        if not self.is_identity_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        shortcut_val = x if self.is_identity_shortcut else self.shortcut(x)
        
        out = self.conv1(x)
        out = self.fused_bn_relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.shuffle(out)
        
        out = self.conv3(out)
        out = self.fused_bn_relu_add3(out, shortcut_val)
        
        return out


class Model(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.fused_bn_relu1 = FusedBnReLU(nn.BatchNorm2d(stages_out_channels[0]))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.fused_bn_relu5 = FusedBnReLU(nn.BatchNorm2d(1024))
        
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = [ShuffleNetUnit(in_channels, out_channels, groups)]
        for _ in range(repeats - 1):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.fused_bn_relu1(x)
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.conv5(x)
        x = self.fused_bn_relu5(x)
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def get_inputs():
    return [torch.randn(10, 3, 224, 224)]

def get_init_inputs():
    return [1000]
# EVOLVE-BLOCK-END