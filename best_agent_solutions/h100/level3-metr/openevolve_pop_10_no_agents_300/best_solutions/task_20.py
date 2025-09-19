# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

# Define custom CUDA kernels.
# This version enhances the top-performing solution (Program 1) by introducing a more advanced
# Adaptive Average Pooling kernel that uses warp-shuffle reductions. This technique is often
# faster than pure shared memory or global reductions by leveraging high-speed intra-warp communication.
# The highly efficient vectorized BatchNorm kernels from Program 1 are retained.
custom_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm> // For std::min

// --- KERNEL 1 & 2: Fused BatchNorm (+ ReLU6) with vectorization (from Program 1) ---

template<bool with_relu>
__device__ inline float apply_activation(float val) {
    if (with_relu) {
        return fminf(fmaxf(0.0f, val), 6.0f);
    }
    return val;
}

template<bool with_relu>
__global__ void fused_batchnorm_kernel_scalar(
    const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta,
    const float* __restrict__ running_mean, const float* __restrict__ running_var, float eps,
    float* __restrict__ out, const int num_elements, const int C, const int spatium) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        const int c = (i / spatium) % C;
        
        const float mean = running_mean[c];
        const float inv_std = rsqrtf(running_var[c] + eps);
        const float g = gamma[c];
        const float b = beta[c];

        const float normalized = g * (x[i] - mean) * inv_std + b;
        out[i] = apply_activation<with_relu>(normalized);
    }
}

template<bool with_relu>
__global__ void fused_batchnorm_kernel_vec4(
    const float4* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta,
    const float* __restrict__ running_mean, const float* __restrict__ running_var, float eps,
    float4* __restrict__ out, const int num_vec_elements, const int C, const int spatium) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_vec_elements) {
        // All 4 floats in the vector are guaranteed to be in the same channel
        // because we only vectorize on the W dimension (last dim of NCHW)
        const int float_idx_base = i * 4;
        const int c = (float_idx_base / spatium) % C;
        
        const float mean = running_mean[c];
        const float inv_std = rsqrtf(running_var[c] + eps);
        const float g = gamma[c];
        const float b = beta[c];

        float4 x_vec = x[i];
        float4 out_vec;
        
        out_vec.x = apply_activation<with_relu>(g * (x_vec.x - mean) * inv_std + b);
        out_vec.y = apply_activation<with_relu>(g * (x_vec.y - mean) * inv_std + b);
        out_vec.z = apply_activation<with_relu>(g * (x_vec.z - mean) * inv_std + b);
        out_vec.w = apply_activation<with_relu>(g * (x_vec.w - mean) * inv_std + b);

        out[i] = out_vec;
    }
}

template<bool with_relu>
void launch_fused_bn_kernels(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor running_mean, torch::Tensor running_var, double eps, torch::Tensor out) {
    
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    
    const long num_elements = x.numel();
    if (num_elements == 0) return;

    const int spatium = H * W;
    const int block_size = 256;
    
    // Use vectorized kernel if W is a multiple of 4 and tensor is contiguous
    if (W % 4 == 0 && x.is_contiguous() && out.is_contiguous()) {
        const int num_vec_elements = num_elements / 4;
        const int num_blocks = (num_vec_elements + block_size - 1) / block_size;
        fused_batchnorm_kernel_vec4<with_relu><<<num_blocks, block_size>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(), (float)eps,
            reinterpret_cast<float4*>(out.data_ptr<float>()), num_vec_elements, C, spatium);
    } else { // Fallback to scalar kernel
        const int num_blocks = (num_elements + block_size - 1) / block_size;
        fused_batchnorm_kernel_scalar<with_relu><<<num_blocks, block_size>>>(
            x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(), (float)eps,
            out.data_ptr<float>(), num_elements, C, spatium);
    }
}

torch::Tensor batchnorm_cuda(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor running_mean, torch::Tensor running_var, double eps) {
    auto out = torch::empty_like(x);
    launch_fused_bn_kernels<false>(x, gamma, beta, running_mean, running_var, eps, out);
    return out;
}

torch::Tensor fused_batchnorm_relu6_cuda(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor running_mean, torch::Tensor running_var, double eps) {
    auto out = torch::empty_like(x);
    launch_fused_bn_kernels<true>(x, gamma, beta, running_mean, running_var, eps, out);
    return out;
}


// --- KERNEL 3: Fused Adaptive Average Pool + Flatten with Warp Shuffle Reduction ---
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template <unsigned int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
adaptive_avg_pool_flatten_kernel(const float* __restrict__ input, float* __restrict__ output, int N, int C, int H, int W) {
    __shared__ float sdata[BLOCK_SIZE / 32]; 

    const int n = blockIdx.y;
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int spatial_size = H * W;
    if (spatial_size == 0) return;

    const float* channel_data = input + (n * C * spatial_size) + (c * spatial_size);

    float thread_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += BLOCK_SIZE) {
        thread_sum += channel_data[i];
    }
    
    float warp_sum = warpReduceSum(thread_sum);

    if (lane_id == 0) {
        sdata[warp_id] = warp_sum;
    }
    
    __syncthreads();
    
    if (warp_id == 0) {
        float final_sum = (lane_id < (BLOCK_SIZE / 32)) ? sdata[lane_id] : 0.0f;
        final_sum = warpReduceSum(final_sum);

        if (lane_id == 0) {
            output[n * C + c] = final_sum / static_cast<float>(spatial_size);
        }
    }
}

torch::Tensor adaptive_avg_pool_flatten_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input tensor must be 4-dimensional (N, C, H, W)");
    input = input.contiguous();

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    auto output = torch::empty({N, C}, input.options());
    if (input.numel() == 0) return output;

    const int block_size = 256;
    dim3 grid_dim(C, N);
    dim3 block_dim(block_size);

    adaptive_avg_pool_flatten_kernel<block_size><<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N, C, H, W
    );
    
    return output;
}
"""

custom_kernels_cpp_source = """
torch::Tensor batchnorm_cuda(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor running_mean, torch::Tensor running_var, double eps);
    
torch::Tensor fused_batchnorm_relu6_cuda(
    torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
    torch::Tensor running_mean, torch::Tensor running_var, double eps);

torch::Tensor adaptive_avg_pool_flatten_cuda(torch::Tensor input);
"""

# JIT compile the CUDA code
fused_ops = load_inline(
    name="fused_ops_mobilenetv2_warp_shuffle",
    cpp_sources=custom_kernels_cpp_source,
    cuda_sources=custom_kernels_source,
    functions=["batchnorm_cuda", "fused_batchnorm_relu6_cuda", "adaptive_avg_pool_flatten_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class CustomBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(CustomBatchNorm2d, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                  True, 0.1, self.eps)
        else:
            return fused_ops.batchnorm_cuda(
                x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
            )

class FusedBatchNormReLU6(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(FusedBatchNormReLU6, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            bn_out = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                  True, 0.1, self.eps)
            return F.relu6(bn_out, inplace=True)
        else:
            return fused_ops.fused_batchnorm_relu6_cuda(
                x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
            )

class FusedAdaptiveAvgPoolFlatten(nn.Module):
    def forward(self, x):
        return fused_ops.adaptive_avg_pool_flatten_cuda(x)


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            hidden_dim = int(inp * expand_ratio)
            layers = []
            if expand_ratio != 1:
                # Pointwise convolution
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(FusedBatchNormReLU6(hidden_dim))

            layers.extend([
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                FusedBatchNormReLU6(hidden_dim),
                # Pointwise linear convolution
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                CustomBatchNorm2d(oup),
            ])
            return nn.Sequential(*layers)

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # Building first layer
        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    FusedBatchNormReLU6(input_channel)]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(_inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Building last several layers
        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(FusedBatchNormReLU6(last_channel))

        # Final layer
        features.append(FusedAdaptiveAvgPoolFlatten())

        self.features = nn.Sequential(*features)

        # Linear layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (CustomBatchNorm2d, FusedBatchNormReLU6)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END