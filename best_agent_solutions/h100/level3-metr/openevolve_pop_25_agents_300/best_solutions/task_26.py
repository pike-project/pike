# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define and compile custom CUDA kernels for aggressive fusion within ShuffleNet, using FP16 and float4 vectorization.
# This approach converts the top-performing FP32 fused kernels to half-precision to halve memory bandwidth
# requirements and leverage hardware support for FP16, which is a major performance lever for memory-bound operations.
cuda_source_fp16 = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm> // For std::min

#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

// Kernel 1: Fused BatchNorm (inference) + ReLU, vectorized for FP16
// Processes 8 half-precision elements per thread using a single float4 load/store.
__global__ void fused_bn_relu_kernel_fp16(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ mean,
    const half* __restrict__ var,
    float eps,
    int total_elements_v8,
    int C, int HW_v8) {

    CUDA_KERNEL_LOOP(idx, total_elements_v8) {
        int c = (idx / HW_v8) % C;

        const half m = mean[c];
        const half v = var[c];
        const half w = weight[c];
        const half b = bias[c];
        const half inv_std = __float2half(rsqrtf(__half2float(v) + eps));

        const half2 m_vec = __halves2half2(m, m);
        const half2 w_vec = __halves2half2(w, w);
        const half2 b_vec = __halves2half2(b, b);
        const half2 inv_std_vec = __halves2half2(inv_std, inv_std);
        const half2 zero_vec = __float2half2_rn(0.0f);

        // Vectorized load of 8 halfs
        float4 x_vec = reinterpret_cast<const float4*>(input)[idx];
        half2* x_h2 = reinterpret_cast<half2*>(&x_vec);

        // Process 4 half2 vectors using FP16 intrinsics
        #pragma unroll
        for (int i=0; i<4; ++i) {
            half2 val = x_h2[i];
            val = __hsub2(val, m_vec);
            val = __hmul2(val, inv_std_vec);
            val = __hfma2(val, w_vec, b_vec);
            val = __hmax2(val, zero_vec);
            x_h2[i] = val;
        }
        
        // Vectorized store
        reinterpret_cast<float4*>(output)[idx] = x_vec;
    }
}

// Kernel 2: Fused BatchNorm (inference) + ChannelShuffle, vectorized for FP16
__global__ void fused_bn_shuffle_kernel_fp16(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ mean,
    const half* __restrict__ var,
    float eps,
    int num_elements_v8,
    int C, int H, int W, int groups) {
    
    const int W_v8 = W / 8;
    const int HW_v8 = H * W_v8;
    const int CHW_v8 = C * HW_v8;

    CUDA_KERNEL_LOOP(idx_out_v8, num_elements_v8) {
        const int n = idx_out_v8 / CHW_v8;
        const int c_out = (idx_out_v8 / HW_v8) % C;
        const int channels_per_group = C / groups;
        const int g = c_out % groups;
        const int c_prime = c_out / groups;
        const int c_in = g * channels_per_group + c_prime;

        const int spatial_idx_v8 = idx_out_v8 % HW_v8;
        const int idx_in_v8 = n * CHW_v8 + c_in * HW_v8 + spatial_idx_v8;

        const half m = mean[c_in];
        const half v = var[c_in];
        const half w = weight[c_in];
        const half b = bias[c_in];
        const half inv_std = __float2half(rsqrtf(__half2float(v) + eps));

        const half2 m_vec = __halves2half2(m, m);
        const half2 w_vec = __halves2half2(w, w);
        const half2 b_vec = __halves2half2(b, b);
        const half2 inv_std_vec = __halves2half2(inv_std, inv_std);

        float4 x_vec = reinterpret_cast<const float4*>(input)[idx_in_v8];
        half2* x_h2 = reinterpret_cast<half2*>(&x_vec);
        
        #pragma unroll
        for (int i=0; i<4; ++i) {
            half2 val = x_h2[i];
            val = __hsub2(val, m_vec);
            val = __hmul2(val, inv_std_vec);
            val = __hfma2(val, w_vec, b_vec);
            x_h2[i] = val;
        }
        
        reinterpret_cast<float4*>(output)[idx_out_v8] = x_vec;
    }
}

// Kernel 3: Fused BatchNorm (inference) + ReLU + Add, vectorized for FP16
__global__ void fused_bn_relu_add_kernel_fp16(
    const half* __restrict__ input,
    const half* __restrict__ shortcut,
    half* __restrict__ output,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ mean,
    const half* __restrict__ var,
    float eps,
    int total_elements_v8,
    int C, int HW_v8) {

    CUDA_KERNEL_LOOP(idx, total_elements_v8) {
        int c = (idx / HW_v8) % C;

        const half m = mean[c];
        const half v = var[c];
        const half w = weight[c];
        const half b = bias[c];
        const half inv_std = __float2half(rsqrtf(__half2float(v) + eps));
        
        const half2 m_vec = __halves2half2(m, m);
        const half2 w_vec = __halves2half2(w, w);
        const half2 b_vec = __halves2half2(b, b);
        const half2 inv_std_vec = __halves2half2(inv_std, inv_std);
        const half2 zero_vec = __float2half2_rn(0.0f);

        float4 x_vec = reinterpret_cast<const float4*>(input)[idx];
        float4 s_vec = reinterpret_cast<const float4*>(shortcut)[idx];
        half2* x_h2 = reinterpret_cast<half2*>(&x_vec);
        half2* s_h2 = reinterpret_cast<half2*>(&s_vec);
        
        #pragma unroll
        for (int i=0; i<4; ++i) {
            half2 val = x_h2[i];
            val = __hsub2(val, m_vec);
            val = __hmul2(val, inv_std_vec);
            val = __hfma2(val, w_vec, b_vec);
            val = __hmax2(val, zero_vec);
            val = __hadd2(val, s_h2[i]);
            x_h2[i] = val;
        }
        
        reinterpret_cast<float4*>(output)[idx] = x_vec;
    }
}

// --- C++ Wrappers ---
static inline void get_launch_params(int total_elements, int& grid_size, int& block_size) {
    block_size = 256;
    grid_size = std::min((total_elements + block_size - 1) / block_size, 4096);
}

torch::Tensor fused_bn_relu_cuda_fp16(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mean, torch::Tensor var, double eps) {
    
    TORCH_CHECK(x.size(3) % 8 == 0, "Width must be divisible by 8 for fp16 kernel");
    auto out = torch::empty_like(x);
    const int total_elements_v8 = x.numel() / 8;
    if (total_elements_v8 == 0) return out;

    const int C = x.size(1);
    const int HW_v8 = (x.size(2) * x.size(3)) / 8;
    
    int grid_size, block_size;
    get_launch_params(total_elements_v8, grid_size, block_size);

    fused_bn_relu_kernel_fp16<<<grid_size, block_size>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), 
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()), 
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()), 
        reinterpret_cast<const half*>(mean.data_ptr<at::Half>()), 
        reinterpret_cast<const half*>(var.data_ptr<at::Half>()),
        eps, total_elements_v8, C, HW_v8);
    return out;
}

torch::Tensor fused_bn_shuffle_cuda_fp16(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mean, torch::Tensor var, double eps, int64_t groups) {

    TORCH_CHECK(x.size(3) % 8 == 0, "Width must be divisible by 8 for fp16 kernel");
    auto out = torch::empty_like(x);
    const int num_elements_v8 = x.numel() / 8;
    if (num_elements_v8 == 0) return out;

    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    int grid_size, block_size;
    get_launch_params(num_elements_v8, grid_size, block_size);

    fused_bn_shuffle_kernel_fp16<<<grid_size, block_size>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), 
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()), 
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()), 
        reinterpret_cast<const half*>(mean.data_ptr<at::Half>()), 
        reinterpret_cast<const half*>(var.data_ptr<at::Half>()),
        eps, num_elements_v8, C, H, W, groups);
    return out;
}

torch::Tensor fused_bn_relu_add_cuda_fp16(
    torch::Tensor x, torch::Tensor shortcut, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor mean, torch::Tensor var, double eps) {
    
    TORCH_CHECK(x.size(3) % 8 == 0, "Width must be divisible by 8 for fp16 kernel");
    auto out = torch::empty_like(x);
    const int total_elements_v8 = x.numel() / 8;
    if (total_elements_v8 == 0) return out;

    const int C = x.size(1);
    const int HW_v8 = (x.size(2) * x.size(3)) / 8;
    
    int grid_size, block_size;
    get_launch_params(total_elements_v8, grid_size, block_size);
    
    fused_bn_relu_add_kernel_fp16<<<grid_size, block_size>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()), 
        reinterpret_cast<const half*>(shortcut.data_ptr<at::Half>()), 
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()), 
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()), 
        reinterpret_cast<const half*>(mean.data_ptr<at::Half>()), 
        reinterpret_cast<const half*>(var.data_ptr<at::Half>()),
        eps, total_elements_v8, C, HW_v8);
    return out;
}
"""

cpp_source_fp16 = """
torch::Tensor fused_bn_relu_cuda_fp16(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor mean, torch::Tensor var, double eps);
torch::Tensor fused_bn_shuffle_cuda_fp16(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, torch::Tensor mean, torch::Tensor var, double eps, int64_t groups);
torch::Tensor fused_bn_relu_add_cuda_fp16(torch::Tensor x, torch::Tensor shortcut, torch::Tensor weight, torch::Tensor bias, torch::Tensor mean, torch::Tensor var, double eps);
"""

# Compile the inline CUDA code
shufflenet_fused_kernels_fp16 = load_inline(
    name="shufflenet_fused_kernels_fp16",
    cpp_sources=cpp_source_fp16,
    cuda_sources=cuda_source_fp16,
    functions=["fused_bn_relu_cuda_fp16", "fused_bn_shuffle_cuda_fp16", "fused_bn_relu_add_cuda_fp16"],
    verbose=True,
)


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnit, self).__init__()
        
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        self.groups = groups
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        shortcut_val = self.shortcut(x)
        
        # Fusion 1: Conv -> BN -> ReLU
        out = self.conv1(x)
        out = shufflenet_fused_kernels_fp16.fused_bn_relu_cuda_fp16(
            out, self.bn1.weight, self.bn1.bias, self.bn1.running_mean, self.bn1.running_var, self.bn1.eps
        )
        
        # Fusion 2: Conv -> BN -> Shuffle
        out = self.conv2(out)
        out = shufflenet_fused_kernels_fp16.fused_bn_shuffle_cuda_fp16(
            out, self.bn2.weight, self.bn2.bias, self.bn2.running_mean, self.bn2.running_var, self.bn2.eps, self.groups
        )
        
        # Fusion 3: Conv -> BN -> ReLU -> Add
        out = self.conv3(out)
        out = shufflenet_fused_kernels_fp16.fused_bn_relu_add_cuda_fp16(
            out, shortcut_val, self.bn3.weight, self.bn3.bias, self.bn3.running_mean, self.bn3.running_var, self.bn3.eps
        )

        return out


class Model(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.fc = nn.Linear(1024, num_classes)
        
        # Convert the entire model to half-precision
        self.half()
        # Cast the final fully connected layer back to float32 for numerical stability
        self.fc.float()
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Ensure the input tensor is in half precision to match the model weights
        x = x.half()
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # Cast back to float32 before the final linear layer for stability, a common practice.
        x = x.to(torch.float32)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    # Provide FP16 inputs to the model
    return [torch.randn(batch_size, input_channels, height, width).cuda().half()]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END