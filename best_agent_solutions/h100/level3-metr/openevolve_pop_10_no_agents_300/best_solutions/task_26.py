# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- COMBINED CUDA KERNELS ---
# This block combines and enhances fusion strategies.
# 1. fused_bn_relu: Fuses BatchNorm and ReLU.
# 2. fused_bn_shuffle: Fuses BatchNorm with ChannelShuffle.
# 3. fused_residual: A generalized residual connection fusion. It handles two cases:
#    a) MainPath(BN->ReLU) + IdentityShortcut
#    b) MainPath(BN->ReLU) + ShortcutPath(BN)
#    This eliminates the separate BN kernel launch on the shortcut path when channels change.
# All kernels use float4 vectorization and __ldg for cached reads.
combined_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CUDA_KERNEL_LOOP_VEC(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Kernel 1: Vectorized Fused BatchNorm + ReLU
__global__ void fused_bn_relu_kernel_vec4(
    const float4* __restrict__ in,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float4* __restrict__ out,
    int total_vectors,
    int channels,
    int spatial_dim_vec4)
{
    CUDA_KERNEL_LOOP_VEC(idx, total_vectors) {
        int c = (idx / spatial_dim_vec4) % channels;
        float s = scale[c];
        float b = bias[c];

        float4 val = __ldg(&in[idx]);
        
        val.x = fmaxf(0.0f, val.x * s + b);
        val.y = fmaxf(0.0f, val.y * s + b);
        val.z = fmaxf(0.0f, val.z * s + b);
        val.w = fmaxf(0.0f, val.w * s + b);

        out[idx] = val;
    }
}

// Kernel 2: Vectorized Fused BatchNorm + ChannelShuffle
__global__ void fused_bn_shuffle_kernel_vec4(
    const float4* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    int total_vec_elements,
    int channels,
    int spatial_dim_vec,
    int groups)
{
    CUDA_KERNEL_LOOP_VEC(idx, total_vec_elements) {
        int hw_vec = idx % spatial_dim_vec;
        int temp = idx / spatial_dim_vec;
        int c_out = temp % channels;
        int n = temp / channels;

        int channels_per_group = channels / groups;
        int g_idx = c_out % groups;
        int c_group_idx = c_out / groups;
        int c_in = g_idx * channels_per_group + c_group_idx;

        int src_idx = n * channels * spatial_dim_vec + c_in * spatial_dim_vec + hw_vec;

        float4 in_val = __ldg(&input[src_idx]);
        
        float scale_val = scale[c_in];
        float bias_val = bias[c_in];

        output[idx].x = in_val.x * scale_val + bias_val;
        output[idx].y = in_val.y * scale_val + bias_val;
        output[idx].z = in_val.z * scale_val + bias_val;
        output[idx].w = in_val.w * scale_val + bias_val;
    }
}


// Kernel 3: Generalized Fused Residual Connection (BN+ReLU+Add or BN+ReLU+Add+BN)
__global__ void fused_residual_kernel_vec4(
    const float4* __restrict__ main_in,
    const float4* __restrict__ shortcut_in,
    const float* __restrict__ main_scale,
    const float* __restrict__ main_bias,
    const float* __restrict__ shortcut_scale, // Can be nullptr for identity shortcut
    const float* __restrict__ shortcut_bias,  // Can be nullptr for identity shortcut
    float4* __restrict__ out,
    int total_vectors,
    int channels,
    int spatial_dim_vec4)
{
    CUDA_KERNEL_LOOP_VEC(idx, total_vectors) {
        int c = (idx / spatial_dim_vec4) % channels;
        
        // Main path: BN -> ReLU
        float ms = main_scale[c];
        float mb = main_bias[c];
        float4 main_val = __ldg(&main_in[idx]);
        main_val.x = fmaxf(0.0f, main_val.x * ms + mb);
        main_val.y = fmaxf(0.0f, main_val.y * ms + mb);
        main_val.z = fmaxf(0.0f, main_val.z * ms + mb);
        main_val.w = fmaxf(0.0f, main_val.w * ms + mb);

        // Shortcut path: Optional BN
        float4 shortcut_val = __ldg(&shortcut_in[idx]);
        if (shortcut_scale != nullptr) {
            float ss = shortcut_scale[c];
            float sb = shortcut_bias[c];
            shortcut_val.x = shortcut_val.x * ss + sb;
            shortcut_val.y = shortcut_val.y * ss + sb;
            shortcut_val.z = shortcut_val.z * ss + sb;
            shortcut_val.w = shortcut_val.w * ss + sb;
        }
        
        // Add and store
        out[idx].x = main_val.x + shortcut_val.x;
        out[idx].y = main_val.y + shortcut_val.y;
        out[idx].z = main_val.z + shortcut_val.z;
        out[idx].w = main_val.w + shortcut_val.w;
    }
}


// --- C++ Wrappers ---

torch::Tensor fused_bn_relu_cuda(torch::Tensor in, torch::Tensor scale, torch::Tensor bias) {
    in = in.contiguous();
    const auto channels = in.size(1);
    const auto height = in.size(2);
    const auto width = in.size(3);
    TORCH_CHECK(width % 4 == 0, "Tensor width must be divisible by 4 for vectorization");
    
    auto out = torch::empty_like(in);
    const int total_vectors = in.numel() / 4;
    if (total_vectors == 0) return out;
    
    const int spatial_dim_vec4 = (height * width) / 4;
    const int block_size = 1024;
    const int num_blocks = (total_vectors + block_size - 1) / block_size;

    fused_bn_relu_kernel_vec4<<<num_blocks, block_size>>>(
        reinterpret_cast<const float4*>(in.data_ptr<float>()), scale.data_ptr<float>(), bias.data_ptr<float>(),
        reinterpret_cast<float4*>(out.data_ptr<float>()), total_vectors, static_cast<int>(channels), spatial_dim_vec4);
    return out;
}

torch::Tensor fused_bn_shuffle_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor bias, int64_t groups) {
    x = x.contiguous();
    const auto channels = x.size(1);
    const auto height = x.size(2);
    const auto width = x.size(3);
    TORCH_CHECK(width % 4 == 0, "Tensor width must be divisible by 4 for float4 vectorization");

    auto out = torch::empty_like(x);
    const int total_elements = x.numel();
    if (total_elements == 0) return out;
    
    const int spatial_dim_vec = (height * width) / 4;
    const int total_vec_elements = total_elements / 4;

    const int block_size = 1024;
    const int num_blocks = (total_vec_elements + block_size - 1) / block_size;

    fused_bn_shuffle_kernel_vec4<<<num_blocks, block_size>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()), scale.data_ptr<float>(), bias.data_ptr<float>(),
        reinterpret_cast<float4*>(out.data_ptr<float>()), total_vec_elements, static_cast<int>(channels), spatial_dim_vec, static_cast<int>(groups));
    return out;
}

torch::Tensor fused_residual_cuda(
    torch::Tensor main_in, torch::Tensor shortcut_in, torch::Tensor main_scale, torch::Tensor main_bias,
    torch::Tensor shortcut_scale, torch::Tensor shortcut_bias) {
    
    main_in = main_in.contiguous();
    shortcut_in = shortcut_in.contiguous();
    const auto channels = main_in.size(1);
    const auto height = main_in.size(2);
    const auto width = main_in.size(3);
    TORCH_CHECK(width % 4 == 0, "Tensor width must be divisible by 4 for vectorization");

    auto out = torch::empty_like(main_in);
    const int total_vectors = main_in.numel() / 4;
    if (total_vectors == 0) return out;

    const int spatial_dim_vec4 = (height * width) / 4;
    const int block_size = 1024;
    const int num_blocks = (total_vectors + block_size - 1) / block_size;

    const float* shortcut_scale_ptr = shortcut_scale.defined() ? shortcut_scale.data_ptr<float>() : nullptr;
    const float* shortcut_bias_ptr = shortcut_bias.defined() ? shortcut_bias.data_ptr<float>() : nullptr;

    fused_residual_kernel_vec4<<<num_blocks, block_size>>>(
        reinterpret_cast<const float4*>(main_in.data_ptr<float>()), reinterpret_cast<const float4*>(shortcut_in.data_ptr<float>()),
        main_scale.data_ptr<float>(), main_bias.data_ptr<float>(), shortcut_scale_ptr, shortcut_bias_ptr,
        reinterpret_cast<float4*>(out.data_ptr<float>()), total_vectors, static_cast<int>(channels), spatial_dim_vec4);
    return out;
}
"""

combined_kernels_cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_bn_relu_cuda(torch::Tensor in, torch::Tensor scale, torch::Tensor bias);
torch::Tensor fused_bn_shuffle_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor bias, int64_t groups);
torch::Tensor fused_residual_cuda(
    torch::Tensor main_in, torch::Tensor shortcut_in, torch::Tensor main_scale, torch::Tensor main_bias,
    torch::Tensor shortcut_scale, torch::Tensor shortcut_bias);
"""

# JIT compile all kernels in one go with optimization flags
shufflenet_ops = load_inline(
    name="shufflenet_ops_fused_residual",
    cpp_sources=combined_kernels_cpp_source,
    cuda_sources=combined_kernels_source,
    functions=["fused_bn_relu_cuda", "fused_bn_shuffle_cuda", "fused_residual_cuda"],
    verbose=False,
    extra_cflags=["-O3"]
)

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

class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnit, self).__init__()
        
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1, 0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shuffle = ChannelShuffle(groups)
        self.groups = groups
        
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)
            self.shortcut = nn.Sequential(self.shortcut_conv, self.shortcut_bn)
        
        self.register_buffer('inf_scale1', None, persistent=False)
        self.register_buffer('inf_bias1', None, persistent=False)
        self.register_buffer('inf_scale2', None, persistent=False)
        self.register_buffer('inf_bias2', None, persistent=False)
        self.register_buffer('inf_scale3', None, persistent=False)
        self.register_buffer('inf_bias3', None, persistent=False)
        self.register_buffer('inf_shortcut_scale', None, persistent=False)
        self.register_buffer('inf_shortcut_bias', None, persistent=False)

    def _precompute_bn_params(self):
        with torch.no_grad():
            if self.inf_scale1 is None:
                std1 = torch.sqrt(self.bn1.running_var + self.bn1.eps)
                self.inf_scale1 = self.bn1.weight / std1
                self.inf_bias1 = self.bn1.bias - self.bn1.running_mean * self.inf_scale1
            if self.inf_scale2 is None:
                std2 = torch.sqrt(self.bn2.running_var + self.bn2.eps)
                self.inf_scale2 = self.bn2.weight / std2
                self.inf_bias2 = self.bn2.bias - self.bn2.running_mean * self.inf_scale2
            if self.inf_scale3 is None:
                std3 = torch.sqrt(self.bn3.running_var + self.bn3.eps)
                self.inf_scale3 = self.bn3.weight / std3
                self.inf_bias3 = self.bn3.bias - self.bn3.running_mean * self.inf_scale3
            if self.in_channels != self.out_channels and self.inf_shortcut_scale is None:
                std_s = torch.sqrt(self.shortcut_bn.running_var + self.shortcut_bn.eps)
                self.inf_shortcut_scale = self.shortcut_bn.weight / std_s
                self.inf_shortcut_bias = self.shortcut_bn.bias - self.shortcut_bn.running_mean * self.inf_shortcut_scale
                
    def forward(self, x):
        if not self.training:
            self._precompute_bn_params()

            conv1_out = self.conv1(x)
            out = shufflenet_ops.fused_bn_relu_cuda(conv1_out, self.inf_scale1, self.inf_bias1)
            
            conv2_out = self.conv2(out)
            out = shufflenet_ops.fused_bn_shuffle_cuda(conv2_out, self.inf_scale2, self.inf_bias2, self.groups)

            conv3_out = self.conv3(out)
            if self.in_channels == self.out_channels:
                return shufflenet_ops.fused_residual_cuda(
                    conv3_out, x, self.inf_scale3, self.inf_bias3, 
                    torch.Tensor(), torch.Tensor()
                )
            else:
                shortcut_conv_out = self.shortcut_conv(x)
                return shufflenet_ops.fused_residual_cuda(
                    conv3_out, shortcut_conv_out, self.inf_scale3, self.inf_bias3,
                    self.inf_shortcut_scale, self.inf_shortcut_bias
                )
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.shuffle(out)
            out = F.relu(self.bn3(self.conv3(out)))
            out += self.shortcut(x)
            return out

class FusedConvBNReLU(nn.Module):
    def __init__(self, conv, bn):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.register_buffer('inf_scale', None, persistent=False)
        self.register_buffer('inf_bias', None, persistent=False)

    def forward(self, x):
        conv_out = self.conv(x)
        if not self.training:
            if self.inf_scale is None:
                with torch.no_grad():
                    std = torch.sqrt(self.bn.running_var + self.bn.eps)
                    self.inf_scale = self.bn.weight / std
                    self.inf_bias = self.bn.bias - self.bn.running_mean * self.inf_scale
            return shufflenet_ops.fused_bn_relu_cuda(conv_out, self.inf_scale, self.inf_bias)
        else:
            return F.relu(self.bn(conv_out))

class Model(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(Model, self).__init__()
        
        conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.conv_bn_relu1 = FusedConvBNReLU(conv1, bn1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        bn5 = nn.BatchNorm2d(1024)
        self.conv_bn_relu5 = FusedConvBNReLU(conv5, bn5)
        
        self.fc = nn.Linear(1024, num_classes)
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_bn_relu1(x)
        x = self.maxpool(x)
        
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.conv_bn_relu5(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
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
    return [torch.randn(batch_size, input_channels, height, width).cuda().contiguous()]

def get_init_inputs():
    return [num_classes]

# EVOLVE-BLOCK-END