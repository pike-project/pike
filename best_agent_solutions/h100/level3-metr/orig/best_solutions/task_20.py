import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# JIT compile the CUDA kernels.
# This solution adds a fully fused depthwise convolution kernel to the existing library.

cpp_source = """
#include <torch/extension.h>
#include <c10/util/Optional.h>

// Forward declarations of the functions that will be defined in the CUDA source.
torch::Tensor bias_relu6_inplace(torch::Tensor input, torch::Tensor bias);
torch::Tensor final_add(torch::Tensor input, torch::Tensor bias, c10::optional<torch::Tensor> residual_opt);
torch::Tensor bias_relu6_adaptive_avg_pool(torch::Tensor input, torch::Tensor bias);
torch::Tensor fused_depthwise_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride);
"""

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>


// --- Kernel 1: Bias + ReLU6 Fusion (for pointwise conv) ---
__global__ void bias_relu6_kernel(
    float* __restrict__ data, 
    const float* __restrict__ bias, 
    int num_elements, 
    int num_channels, 
    int channel_size) {
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements; i += blockDim.x * gridDim.x) {
        int channel_idx = (i / channel_size) % num_channels;
        float val = data[i] + bias[channel_idx];
        val = fminf(fmaxf(0.0f, val), 6.0f);
        data[i] = val;
    }
}


// --- Kernel 2: Bias Add + Optional Residual Add Fusion ---
__global__ void final_op_kernel(
    float* __restrict__ out,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    int num_elements,
    int num_channels,
    int channel_size) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements; i += blockDim.x * gridDim.x) {
        int channel_idx = (i / channel_size) % num_channels;
        float val = out[i] + bias[channel_idx];
        if (residual != nullptr) {
            val += residual[i];
        }
        out[i] = val;
    }
}

// --- Kernel 3: BiasAdd + ReLU6 + Global Average Pooling Fusion ---
__global__ void bias_relu6_adaptive_avg_pool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int H, int W, int C) {

    extern __shared__ float sdata[];
    const int n = blockIdx.x;
    const int c = blockIdx.y;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    const float* plane_in = input + n * C * H * W + c * H * W;
    const int plane_size = H * W;

    const float b = bias[c];
    float partial_sum = 0.0f;

    for (int i = tid; i < plane_size; i += block_size) {
        float val = plane_in[i] + b;
        val = fminf(fmaxf(0.0f, val), 6.0f);
        partial_sum += val;
    }
    sdata[tid] = partial_sum;
    __syncthreads();

    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float* out_ptr = output + n * C + c;
        *out_ptr = sdata[0] / plane_size;
    }
}

// --- NEW Kernel 4: Fused 3x3 Depthwise Conv + Bias + ReLU6 ---
__global__ void fused_depthwise_conv_kernel(
    const float* __restrict__ input,      // NCHW
    const float* __restrict__ weight,     // C x 9 (flat 3x3)
    const float* __restrict__ bias,       // C
    float* __restrict__ output,         // NCHW_out
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int stride) {

    // Grid: (H_out * W_out, N * C)
    // One thread per output pixel per plane.
    const int xy_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (xy_out >= H_out * W_out) return;

    const int plane_idx = blockIdx.y;
    const int n = plane_idx / C;
    const int c = plane_idx % C;

    const int x_out = xy_out % W_out;
    const int y_out = xy_out / W_out;

    float acc = 0.0f;
    const float* w = weight + c * 9;

    // 3x3 kernel loop
    for (int ky = 0; ky < 3; ++ky) {
        const int y_in = y_out * stride + ky - 1; // padding=1
        for (int kx = 0; kx < 3; ++kx) {
            const int x_in = x_out * stride + kx - 1; // padding=1

            // Boundary check
            if (y_in >= 0 && y_in < H_in && x_in >= 0 && x_in < W_in) {
                const int input_idx = n * C * H_in * W_in + c * H_in * W_in + y_in * W_in + x_in;
                acc += input[input_idx] * w[ky * 3 + kx];
            }
        }
    }
    
    acc += bias[c];
    acc = fminf(fmaxf(0.0f, acc), 6.0f);

    const int output_idx = plane_idx * H_out * W_out + xy_out;
    output[output_idx] = acc;
}

// --- C++ Wrappers ---

torch::Tensor bias_relu6_inplace(torch::Tensor input, torch::Tensor bias) {
    const int num_elements = input.numel();
    const int num_channels = input.size(1);
    const int channel_size = input.size(2) * input.size(3);
    const int threads = 256;
    bias_relu6_kernel<<<(num_elements + threads - 1) / threads, threads>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(),
        num_elements, num_channels, channel_size);
    AT_CUDA_CHECK(cudaGetLastError());
    return input;
}

torch::Tensor final_add(torch::Tensor input, torch::Tensor bias, c10::optional<torch::Tensor> residual_opt) {
    const float* residual_ptr = residual_opt ? residual_opt.value().data_ptr<float>() : nullptr;
    const int num_elements = input.numel();
    const int num_channels = input.size(1);
    const int channel_size = input.size(2) * input.size(3);
    const int threads = 256;
    final_op_kernel<<<(num_elements + threads - 1) / threads, threads>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), residual_ptr,
        num_elements, num_channels, channel_size);
    AT_CUDA_CHECK(cudaGetLastError());
    return input;
}

torch::Tensor bias_relu6_adaptive_avg_pool(torch::Tensor input, torch::Tensor bias) {
    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    auto output = torch::empty({N, C, 1, 1}, input.options());
    const int threads = 512;
    const dim3 grid(N, C);
    const dim3 block(threads);
    bias_relu6_adaptive_avg_pool_kernel<<<grid, block, threads * sizeof(float)>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), H, W, C);
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor fused_depthwise_conv(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride) {
    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);
    const auto H_out = (H_in + 2 * 1 - 3) / stride + 1;
    const auto W_out = (W_in + 2 * 1 - 3) / stride + 1;
    
    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    
    const int threads = 256;
    dim3 grid((H_out * W_out + threads - 1) / threads, N * C);
    dim3 block(threads);

    fused_depthwise_conv_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H_in, W_in, H_out, W_out, stride);
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

# Suppress verbose compilation output
fused_ops = load_inline(
    name="fused_ops_mobilenet_v2_final",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["bias_relu6_inplace", "final_add", "bias_relu6_adaptive_avg_pool", "fused_depthwise_conv"],
    verbose=False
)

class FusedConvBNReLU6(nn.Module):
    """Fuses Conv2d, BatchNorm2d, and ReLU6 for pointwise convolutions."""
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        super().__init__()
        bn_mean, bn_var, bn_gamma, bn_beta, bn_eps = bn.running_mean, bn.running_var, bn.weight, bn.bias, bn.eps
        conv_w, conv_b = conv.weight, conv.bias if conv.bias is not None else torch.zeros_like(bn_mean)
        scale = bn_gamma / torch.sqrt(bn_var + bn_eps)
        w_folded = conv_w * scale.view(-1, 1, 1, 1)
        b_folded = (conv_b - bn_mean) * scale + bn_beta
        
        self.conv = conv
        self.conv.weight = nn.Parameter(w_folded)
        self.conv.bias = None 
        self.register_buffer('bias_folded', b_folded)

    def forward(self, x):
        x = self.conv(x)
        return fused_ops.bias_relu6_inplace(x, self.bias_folded)

class FusedDepthwiseConvBNReLU6(nn.Module):
    """Fuses 3x3 DepthwiseConv, BatchNorm, and ReLU6 into a single kernel."""
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        super().__init__()
        bn_mean, bn_var, bn_gamma, bn_beta, bn_eps = bn.running_mean, bn.running_var, bn.weight, bn.bias, bn.eps
        conv_w = conv.weight
        scale = bn_gamma / torch.sqrt(bn_var + bn_eps)
        w_folded = conv_w * scale.view(-1, 1, 1, 1)
        b_folded = (torch.zeros_like(bn_mean) - bn_mean) * scale + bn_beta
        
        self.stride = conv.stride[0]
        # Store folded weights/biases, no nn.Conv2d layer needed at runtime
        self.register_buffer('weight_folded', w_folded.contiguous().view(conv.out_channels, -1))
        self.register_buffer('bias_folded', b_folded)

    def forward(self, x):
        return fused_ops.fused_depthwise_conv(x, self.weight_folded, self.bias_folded, self.stride)

class FusedConvBNReLU6AvgPool(nn.Module):
    """Fuses Conv2d, BatchNorm2d, ReLU6, and AdaptiveAvgPool2d."""
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        super().__init__()
        bn_mean, bn_var, bn_gamma, bn_beta, bn_eps = bn.running_mean, bn.running_var, bn.weight, bn.bias, bn.eps
        conv_w = conv.weight
        scale = bn_gamma / torch.sqrt(bn_var + bn_eps)
        w_folded = conv_w * scale.view(-1, 1, 1, 1)
        b_folded = (torch.zeros_like(bn_mean) - bn_mean) * scale + bn_beta
        
        self.conv = conv
        self.conv.weight = nn.Parameter(w_folded)
        self.conv.bias = None
        self.register_buffer('bias_folded', b_folded)

    def forward(self, x):
        x = self.conv(x)
        return fused_ops.bias_relu6_adaptive_avg_pool(x, self.bias_folded)

class OptimizedInvertedResidual(nn.Module):
    """Optimized inverted residual block with heavily fused operations."""
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.use_res_connect = stride == 1 and inp == oup
        hidden_dim = int(round(inp * expand_ratio))

        convs = []
        # Expansion: 1x1 pointwise conv + bn + relu6
        if expand_ratio != 1:
            convs.append(
                FusedConvBNReLU6(
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), 
                    nn.BatchNorm2d(hidden_dim)))
        
        # Depthwise: Fused 3x3 depthwise conv + bn + relu6.
        convs.append(
            FusedDepthwiseConvBNReLU6(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), 
                nn.BatchNorm2d(hidden_dim)))
        
        self.features = nn.Sequential(*convs)
        
        # Projection: 1x1 pointwise conv + bn (no activation)
        pw_linear_conv = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        pw_linear_bn = nn.BatchNorm2d(oup)
        bn_mean, bn_var, bn_gamma, bn_beta, bn_eps = pw_linear_bn.running_mean, pw_linear_bn.running_var, pw_linear_bn.weight, pw_linear_bn.bias, pw_linear_bn.eps
        scale = bn_gamma / torch.sqrt(bn_var + bn_eps)
        w_folded = pw_linear_conv.weight * scale.view(-1, 1, 1, 1)
        b_folded = (torch.zeros_like(bn_mean) - bn_mean) * scale + bn_beta
        
        self.pw_linear_conv = pw_linear_conv
        self.pw_linear_conv.weight = nn.Parameter(w_folded)
        self.pw_linear_conv.bias = None
        self.register_buffer('pw_linear_bias', b_folded)

    def forward(self, x):
        identity = x
        out = self.features(x)
        out = self.pw_linear_conv(out)
        residual = identity if self.use_res_connect else None
        return fused_ops.final_add(out, self.pw_linear_bias, residual)


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None: min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v: new_v += divisor
            return new_v

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2],
            [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1],
        ]
        
        # Initial Conv layer
        features = [FusedConvBNReLU6(nn.Conv2d(3, input_channel, 3, 2, 1, bias=False), nn.BatchNorm2d(input_channel))]

        # Inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(OptimizedInvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Final Conv-BN-ReLU6 and AdaptiveAvgPool
        features.append(FusedConvBNReLU6AvgPool(
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel)))
        
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]