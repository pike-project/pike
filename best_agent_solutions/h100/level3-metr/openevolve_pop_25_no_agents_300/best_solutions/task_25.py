# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# --- Fused CUDA Kernels ---
# This solution combines the most effective strategies from previous attempts:
# 1. Maximal Fusion: Three kernels are used to fuse all sequential memory-bound operations:
#    (BN+ReLU), (BN+Shuffle), and (BN+ReLU+Add), minimizing kernel launches and memory I/O.
# 2. Pre-computed BatchNorm: The BN parameters are pre-calculated on the CPU into a simple
#    scale and bias, reducing the in-kernel math to a single efficient FMA instruction.
# 3. Coalesced Shuffle: We use the theoretically optimal "one-block-per-channel" shuffle
#    kernel, which ensures both memory reads and writes are perfectly coalesced. Combining
#    this with pre-computed BN parameters is expected to yield the best performance.
# 4. Vectorization & Tuning: All kernels are vectorized using float4 and use a 1024-thread
#    block size to maximize memory bandwidth and GPU occupancy.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm> // For std::min

// Kernel 1: Fused BatchNorm + ReLU (Pre-computed, Vectorized)
__global__ void fused_bn_relu_kernel_precomputed(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    int total_vec_count,
    int channels,
    int spatial_dim)
{
    for (int vec_idx = blockIdx.x * blockDim.x + threadIdx.x; vec_idx < total_vec_count; vec_idx += blockDim.x * gridDim.x) {
        int base_idx = vec_idx * 4;
        const int c = (base_idx / spatial_dim) % channels;

        const float s = scale[c];
        const float b = bias[c];

        float4 val_vec = *(reinterpret_cast<const float4*>(&input[base_idx]));
        
        val_vec.x = fmaxf(0.f, val_vec.x * s + b);
        val_vec.y = fmaxf(0.f, val_vec.y * s + b);
        val_vec.z = fmaxf(0.f, val_vec.z * s + b);
        val_vec.w = fmaxf(0.f, val_vec.w * s + b);

        *(reinterpret_cast<float4*>(&output[base_idx])) = val_vec;
    }
}

// Kernel 2: Fused BatchNorm + Channel Shuffle (Pre-computed, Coalesced, Vectorized)
__global__ void fused_bn_shuffle_kernel_precomputed_coalesced(
    const float* __restrict__ in,
    float* __restrict__ out,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    int N, int C, int HW, int G)
{
    const int channels_per_group = C / G;
    const int c_in = blockIdx.x; // Each block processes one input channel

    // Pre-calculate the output channel `c_out` based on shuffle logic.
    const int g_in = c_in / channels_per_group;
    const int cpg_in = c_in % channels_per_group;
    const int c_out = cpg_in * G + g_in;

    const float scale_val = scale[c_in];
    const float bias_val = bias[c_in];

    const long long c_in_plane_offset = (long long)c_in * HW;
    const long long c_out_plane_offset = (long long)c_out * HW;
    const long long C_x_HW = (long long)C * HW;
    const int HW_vec = HW / 4;

    for (int i = threadIdx.x + blockIdx.y * blockDim.x; i < N * HW_vec; i += gridDim.y * blockDim.x) {
        const int n = i / HW_vec;
        const int pos_vec = i % HW_vec;

        const long long base_offset = (long long)n * C_x_HW;
        const long long input_idx = base_offset + c_in_plane_offset + (pos_vec * 4);
        const long long output_idx = base_offset + c_out_plane_offset + (pos_vec * 4);

        float4 val_vec = *(reinterpret_cast<const float4*>(&in[input_idx]));

        val_vec.x = val_vec.x * scale_val + bias_val;
        val_vec.y = val_vec.y * scale_val + bias_val;
        val_vec.z = val_vec.z * scale_val + bias_val;
        val_vec.w = val_vec.w * scale_val + bias_val;

        *(reinterpret_cast<float4*>(&out[output_idx])) = val_vec;
    }
}

// Kernel 3: Fused BatchNorm + ReLU + Add (Pre-computed, Vectorized)
__global__ void fused_bn_relu_add_kernel_precomputed(
    const float* __restrict__ main_path_in,
    const float* __restrict__ shortcut_in,
    float* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    int total_vec_count,
    int channels,
    int spatial_dim)
{
    for (int vec_idx = blockIdx.x * blockDim.x + threadIdx.x; vec_idx < total_vec_count; vec_idx += blockDim.x * gridDim.x) {
        int base_idx = vec_idx * 4;
        const int c = (base_idx / spatial_dim) % channels;
        
        const float s = scale[c];
        const float b = bias[c];

        float4 main_vec = *(reinterpret_cast<const float4*>(&main_path_in[base_idx]));
        float4 shortcut_vec = *(reinterpret_cast<const float4*>(&shortcut_in[base_idx]));
        
        main_vec.x = fmaxf(0.f, main_vec.x * s + b) + shortcut_vec.x;
        main_vec.y = fmaxf(0.f, main_vec.y * s + b) + shortcut_vec.y;
        main_vec.z = fmaxf(0.f, main_vec.z * s + b) + shortcut_vec.z;
        main_vec.w = fmaxf(0.f, main_vec.w * s + b) + shortcut_vec.w;

        *(reinterpret_cast<float4*>(&output[base_idx])) = main_vec;
    }
}

// --- C++ Wrappers for PyTorch ---

torch::Tensor fused_bn_relu_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor bias) {
    const auto channels = x.size(1);
    const auto spatial_dim = x.size(2) * x.size(3);
    TORCH_CHECK(spatial_dim % 4 == 0, "Spatial dimension must be divisible by 4 for vectorization.");
    auto out = torch::empty_like(x);
    if (x.numel() == 0) return out;
    const int total_vec_count = x.numel() / 4;
    const int block_size = 1024;
    const int grid_size = (total_vec_count + block_size - 1) / block_size;
    fused_bn_relu_kernel_precomputed<<<grid_size, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), scale.data_ptr<float>(), bias.data_ptr<float>(),
        total_vec_count, channels, spatial_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_bn_shuffle_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor bias, int groups) {
    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    const int HW = H * W;
    TORCH_CHECK(HW % 4 == 0, "Spatial dimension must be divisible by 4 for vectorization.");
    auto out = torch::empty_like(x);
    if (x.numel() == 0) return out;
    const long long total_spatial_elements_vec = (long long)N * HW / 4;
    const int block_size = 1024;
    const long grid_y = (total_spatial_elements_vec + block_size - 1) / block_size;
    dim3 grid(C, std::min(grid_y, 65535L));
    dim3 block(block_size);
    fused_bn_shuffle_kernel_precomputed_coalesced<<<grid, block>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), scale.data_ptr<float>(), bias.data_ptr<float>(),
        N, C, HW, groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_bn_relu_add_cuda(torch::Tensor main, torch::Tensor shortcut, torch::Tensor scale, torch::Tensor bias) {
    const auto channels = main.size(1);
    const auto spatial_dim = main.size(2) * main.size(3);
    TORCH_CHECK(spatial_dim % 4 == 0, "Spatial dimension must be divisible by 4 for vectorization.");
    auto out = torch::empty_like(main);
    if (main.numel() == 0) return out;
    const int total_vec_count = main.numel() / 4;
    const int block_size = 1024;
    const int grid_size = (total_vec_count + block_size - 1) / block_size;
    fused_bn_relu_add_kernel_precomputed<<<grid_size, block_size>>>(
        main.data_ptr<float>(), shortcut.data_ptr<float>(), out.data_ptr<float>(),
        scale.data_ptr<float>(), bias.data_ptr<float>(),
        total_vec_count, channels, spatial_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_bn_relu_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor bias);
torch::Tensor fused_bn_shuffle_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor bias, int groups);
torch::Tensor fused_bn_relu_add_cuda(torch::Tensor main, torch::Tensor shortcut, torch::Tensor scale, torch::Tensor bias);
"""

# JIT compile the CUDA kernels
fused_ops = load_inline(
    name="fused_shufflenet_ops_hybrid_max_fusion",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_bn_relu_cuda", "fused_bn_shuffle_cuda", "fused_bn_relu_add_cuda"],
    verbose=True,
)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(Model, self).__init__()
        
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.groups = groups
        
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Pre-compute BN scale and bias for all layers to simplify kernel operations
        with torch.no_grad():
            var_eps1 = self.bn1.running_var + self.bn1.eps
            scale1 = self.bn1.weight / torch.sqrt(var_eps1)
            bias1 = self.bn1.bias - self.bn1.running_mean * scale1

            var_eps2 = self.bn2.running_var + self.bn2.eps
            scale2 = self.bn2.weight / torch.sqrt(var_eps2)
            bias2 = self.bn2.bias - self.bn2.running_mean * scale2

            var_eps3 = self.bn3.running_var + self.bn3.eps
            scale3 = self.bn3.weight / torch.sqrt(var_eps3)
            bias3 = self.bn3.bias - self.bn3.running_mean * scale3

        shortcut_out = self.shortcut(x)
        
        # Block 1: Conv -> Fused(BN + ReLU)
        out = self.conv1(x)
        out = fused_ops.fused_bn_relu_cuda(out, scale1, bias1)
        
        # Block 2: Conv -> Fused(BN + Shuffle)
        out = self.conv2(out)
        out = fused_ops.fused_bn_shuffle_cuda(out, scale2, bias2, self.groups)

        # Block 3: Conv -> Fused(BN + ReLU + Add)
        out = self.conv3(out)
        out = fused_ops.fused_bn_relu_add_cuda(out, shortcut_out, scale3, bias3)
        
        return out

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x

# --- Boilerplate ---
batch_size = 10
input_channels = 240
out_channels = 480
groups = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width, dtype=torch.float32).cuda()]

def get_init_inputs():
    return [input_channels, out_channels, groups]
# EVOLVE-BLOCK-END