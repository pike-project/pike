# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Rationale: This solution builds upon the top-performing program (Program 1, runtime 1.55ms)
# which correctly identified memory bandwidth as the key bottleneck and used float4 vectorization
# to achieve significant speedups. The current program's pre-calculation strategy has been shown
# to be slower due to kernel launch overhead.
# This refined solution improves upon the best vectorized approach by:
# 1. Incorporating grid-stride loops into the CUDA kernels. This is a best practice that
#    makes kernels more robust and can improve performance by ensuring full GPU utilization,
#    regardless of the number of blocks launched.
# 2. Making the vectorized nature of the kernels more explicit in the function signatures by
#    using float4 pointers where appropriate.
# 3. Retaining the highly effective fusion strategy: (Conv -> BN -> ReLU) and
#    (Conv -> BN -> ReLU -> MaxPool), where the post-convolution operations are handled
#    by a single, high-bandwidth custom kernel.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// --- Kernel 1: Fused BatchNorm + ReLU (Vectorized with float4 and Grid-Stride Loop) ---
__global__ void batch_norm_relu_kernel_v4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float eps,
    const int total_f4_elements,
    const int C,
    const int HxW) {

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_f4_elements;
         idx += gridDim.x * blockDim.x) {

        const int element_idx = idx * 4;
        const int c = (element_idx / HxW) % C;

        const float mean = running_mean[c];
        const float inv_std = rsqrtf(running_var[c] + eps);
        const float w = weight[c];
        const float b = bias[c];

        float4 data = input[idx];

        data.x = fmaxf(0.f, ((data.x - mean) * inv_std * w + b));
        data.y = fmaxf(0.f, ((data.y - mean) * inv_std * w + b));
        data.z = fmaxf(0.f, ((data.z - mean) * inv_std * w + b));
        data.w = fmaxf(0.f, ((data.w - mean) * inv_std * w + b));

        output[idx] = data;
    }
}

// Wrapper for Kernel 1
torch::Tensor batch_norm_relu_forward_cuda_v4(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps) {

    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor");
    const int total_elements = input.numel();
    TORCH_CHECK(total_elements % 4 == 0, "Input numel must be divisible by 4 for v4 kernel");

    const int C = input.size(1);
    const int HxW = input.size(2) * input.size(3);
    auto output = torch::empty_like(input);

    const int total_f4_elements = total_elements / 4;
    const int threads = 512;
    const int blocks = std::min((total_f4_elements + threads - 1) / threads, 4096);


    batch_norm_relu_kernel_v4<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        (float)eps, total_f4_elements, C, HxW);
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

// --- Kernel 2: Fused BatchNorm + ReLU + MaxPool2d (Vectorized with float4 and Grid-Stride Loop) ---

__device__ __forceinline__ void apply_bn_relu_f4(
    float4& data, const float mean, const float inv_std, const float w, const float b) {
    data.x = fmaxf(0.f, (data.x - mean) * inv_std * w + b);
    data.y = fmaxf(0.f, (data.y - mean) * inv_std * w + b);
    data.z = fmaxf(0.f, (data.z - mean) * inv_std * w + b);
    data.w = fmaxf(0.f, (data.w - mean) * inv_std * w + b);
}

__global__ void bn_relu_maxpool_kernel_v4(
    const float* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ running_mean, const float* __restrict__ running_var,
    float eps, int N, int C, int InH, int InW, int OutH, int OutW) {

    const int total_out_f4_elements = (N * C * OutH * OutW) / 4;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_out_f4_elements;
         idx += gridDim.x * blockDim.x) {

        // Decompose float4 index to get n, c, h, w for the first element
        long linear_idx = (long)idx * 4;
        int out_w_start = linear_idx % OutW;
        long temp = linear_idx / OutW;
        int out_h = temp % OutH;
        temp /= OutH;
        int c = temp % C;
        int n = temp / C;

        const float mean = running_mean[c];
        const float inv_std = rsqrtf(running_var[c] + eps);
        const float w = weight[c];
        const float b = bias[c];

        int in_start_h = out_h * 2; // stride 2
        int in_start_w = out_w_start * 2; // stride 2

        const float* in_ptr = input + ((long)n * C + c) * InH * InW + (long)in_start_h * InW + in_start_w;
        
        // Read 2x8 input window using 4 float4 loads
        const float4* in_ptr_v4 = reinterpret_cast<const float4*>(in_ptr);
        float4 in_r1_p1 = in_ptr_v4[0];
        float4 in_r1_p2 = in_ptr_v4[1];

        const float4* in_ptr_v4_row2 = reinterpret_cast<const float4*>(in_ptr + InW);
        float4 in_r2_p1 = in_ptr_v4_row2[0];
        float4 in_r2_p2 = in_ptr_v4_row2[1];

        apply_bn_relu_f4(in_r1_p1, mean, inv_std, w, b);
        apply_bn_relu_f4(in_r1_p2, mean, inv_std, w, b);
        apply_bn_relu_f4(in_r2_p1, mean, inv_std, w, b);
        apply_bn_relu_f4(in_r2_p2, mean, inv_std, w, b);

        float4 out_val;
        out_val.x = fmaxf(fmaxf(in_r1_p1.x, in_r1_p1.y), fmaxf(in_r2_p1.x, in_r2_p1.y));
        out_val.y = fmaxf(fmaxf(in_r1_p1.z, in_r1_p1.w), fmaxf(in_r2_p1.z, in_r2_p1.w));
        out_val.z = fmaxf(fmaxf(in_r1_p2.x, in_r1_p2.y), fmaxf(in_r2_p2.x, in_r2_p2.y));
        out_val.w = fmaxf(fmaxf(in_r1_p2.z, in_r1_p2.w), fmaxf(in_r2_p2.z, in_r2_p2.w));

        output[idx] = out_val;
    }
}

// Wrapper for Kernel 2
torch::Tensor bn_relu_maxpool_forward_cuda_v4(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps,
    int kernel_h, int kernel_w, int stride_h, int stride_w) {
    
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor");

    const int N = input.size(0);
    const int C = input.size(1);
    const int InH = input.size(2);
    const int InW = input.size(3);

    const int OutH = (InH - kernel_h) / stride_h + 1;
    const int OutW = (InW - kernel_w) / stride_w + 1;

    TORCH_CHECK(OutW % 4 == 0, "Output width must be divisible by 4 for v4 kernel");
    TORCH_CHECK(InW % 8 == 0, "Input width must be divisible by 8 for v4 kernel");

    auto output = torch::empty({N, C, OutH, OutW}, input.options());
    const int total_out_elements = output.numel();
    if (total_out_elements == 0) return output;

    const int total_out_f4_elements = total_out_elements / 4;
    const int threads = 256;
    const int blocks = std::min((total_out_f4_elements + threads - 1) / threads, 4096);

    bn_relu_maxpool_kernel_v4<<<blocks, threads>>>(
        input.data_ptr<float>(),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        (float)eps, N, C, InH, InW, OutH, OutW);
    
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

fused_ops_cpp_source = """
#include <algorithm> // For std::min

torch::Tensor batch_norm_relu_forward_cuda_v4(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps);

torch::Tensor bn_relu_maxpool_forward_cuda_v4(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps,
    int kernel_h, int kernel_w, int stride_h, int stride_w);
"""

# Compile the inline CUDA code
fused_ops_v4 = load_inline(
    name="fused_regnet_ops_v4_gridstride",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["batch_norm_relu_forward_cuda_v4", "bn_relu_maxpool_forward_cuda_v4"],
    verbose=True,
)

class FusedConvBNReLUv4(nn.Module):
    """ Fuses Conv -> BN -> ReLU. The BN+ReLU part is a vectorized custom kernel. """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(FusedConvBNReLUv4, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return fused_ops_v4.batch_norm_relu_forward_cuda_v4(
            x, self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps
        )

class FusedConvBNReLUMaxPoolv4(nn.Module):
    """ Fuses Conv -> BN -> ReLU -> MaxPool. The BN+ReLU+MaxPool part is a vectorized custom kernel. """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(FusedConvBNReLUMaxPoolv4, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool_kernel_size = 2
        self.pool_stride = 2

    def forward(self, x):
        x = self.conv(x)
        return fused_ops_v4.bn_relu_maxpool_forward_cuda_v4(
            x, self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps,
            self.pool_kernel_size, self.pool_kernel_size, self.pool_stride, self.pool_stride
        )

class Model(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        super(Model, self).__init__()
        self.stages = stages
        self.block_widths = block_widths
        
        layers = []
        current_channels = input_channels
        
        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        
        self.feature_extractor = nn.Sequential(*layers)
        self.fc = nn.Linear(block_widths[-1], output_classes)
    
    def _make_stage(self, in_channels, out_channels):
        """ Creates a stage using fused and vectorized operation modules. """
        return nn.Sequential(
            FusedConvBNReLUv4(in_channels, out_channels, kernel_size=3, padding=1),
            FusedConvBNReLUMaxPoolv4(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.mean(x, dim=[2, 3])
        x = self.fc(x)
        return x

# Model parameters
batch_size = 8
input_channels = 3
image_height, image_width = 224, 224
stages = 3
block_widths = [64, 128, 256]
output_classes = 10

def get_inputs():
    """ Generates random input tensor. """
    return [torch.randn(batch_size, input_channels, image_height, image_width)]

def get_init_inputs():
    """ Initializes model parameters. """
    return [input_channels, stages, block_widths, output_classes]
# EVOLVE-BLOCK-END