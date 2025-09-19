# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# This solution combines the most effective optimizations from prior attempts:
# 1. A general-purpose, robust BN+ReLU fusion kernel using float4 vectorization and a grid-stride loop.
#    This is used within each layer of the DenseBlocks.
# 2. A specialized BN+ReLU+AvgPool(2x2) fusion kernel. This enables an algorithmic optimization in the
#    TransitionLayer by reordering operations to (BN+ReLU+Pool)->Conv, reducing the computational
#    cost of the 1x1 convolution.
# 3. A highly-optimized BN+ReLU+GlobalAvgPool fusion kernel for the final classification head. This
#    avoids materializing the large final feature map, performing the reduction in-place and
#    significantly reducing memory traffic.
# This holistic approach targets all major bottlenecks in the DenseNet architecture.

fused_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>

// --- Kernel 1: Fused BatchNorm2d + ReLU (Vectorized, Grid-Stride) ---
// The workhorse for layers inside DenseBlocks. High throughput via float4 and robust via grid-stride.
__global__ void bn_relu_inference_kernel_vectorized_gridstride(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float eps,
    const int num_vec_elements,
    const int C,
    const int HW) {

    for (int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
         vec_idx < num_vec_elements;
         vec_idx += blockDim.x * gridDim.x) {

        const int base_idx = vec_idx * 4;
        const int c = (base_idx / HW) % C;

        const float scale = weight[c] * rsqrtf(var[c] + eps);
        const float shift = bias[c] - mean[c] * scale;

        const float4 in_val = input[vec_idx];
        float4 out_val;
        out_val.x = fmaxf(0.f, in_val.x * scale + shift);
        out_val.y = fmaxf(0.f, in_val.y * scale + shift);
        out_val.z = fmaxf(0.f, in_val.z * scale + shift);
        out_val.w = fmaxf(0.f, in_val.w * scale + shift);
        output[vec_idx] = out_val;
    }
}

// --- Kernel 2: Fused BatchNorm2d + ReLU + AvgPool2d(kernel=2, stride=2) ---
// For use in the TransitionLayer. Each thread computes one output pixel.
__global__ void bn_relu_avgpool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float eps,
    const int total_out_elements,
    const int C,
    const int H_in, const int W_in,
    const int H_out, const int W_out) {

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_out_elements;
         idx += gridDim.x * blockDim.x) {
        
        const int w_out = idx % W_out;
        const int h_out = (idx / W_out) % H_out;
        const int c = (idx / (W_out * H_out)) % C;
        const int n = idx / (C * W_out * H_out);

        const float scale = weight[c] * rsqrtf(var[c] + eps);
        const float shift = bias[c] - mean[c] * scale;

        const int h_in = h_out * 2;
        const int w_in = w_out * 2;
        
        const int base_in_idx = n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in;

        float sum = 0.0f;
        sum += fmaxf(0.f, input[base_in_idx] * scale + shift);
        sum += fmaxf(0.f, input[base_in_idx + 1] * scale + shift);
        sum += fmaxf(0.f, input[base_in_idx + W_in] * scale + shift);
        sum += fmaxf(0.f, input[base_in_idx + W_in + 1] * scale + shift);

        output[idx] = sum * 0.25f;
    }
}

// --- Kernel 3: Fused BatchNorm2d + ReLU + Global Average Pooling ---
// For the final classification head. Reduces the feature map in-place.
__global__ void bn_relu_global_avg_pool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output, // Shape (N, C)
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float eps,
    const int N, const int C, const int H, const int W) {

    const int nc_idx = blockIdx.x;
    const int c = nc_idx % C;

    extern __shared__ float sdata[];
    float my_sum = 0.0f;
    const int HW = H * W;

    const float scale = weight[c] * rsqrtf(var[c] + eps);
    const float shift = bias[c] - mean[c] * scale;

    const int n = nc_idx / C;
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        const int input_idx = n * C * HW + c * HW + i;
        my_sum += fmaxf(0.f, input[input_idx] * scale + shift);
    }
    sdata[threadIdx.x] = my_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) output[nc_idx] = sdata[0] / (float)HW;
}

// --- C++ Wrappers ---

// Wrapper for Kernel 1
torch::Tensor bn_relu_cuda_vectorized(
    torch::Tensor input, torch::Tensor mean, torch::Tensor var,
    torch::Tensor weight, torch::Tensor bias, double eps) {
    
    auto input_cont = input.contiguous();
    TORCH_CHECK(input_cont.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input_cont.numel() % 4 == 0, "Input numel must be divisible by 4 for vectorization");

    const auto C = input_cont.size(1);
    const int HW = input_cont.size(2) * input_cont.size(3);
    auto output = torch::empty_like(input_cont);
    if (input_cont.numel() == 0) return output;

    const int num_vec_elements = input_cont.numel() / 4;
    const int block_size = 256;
    int num_blocks = std::min((num_vec_elements + block_size - 1) / block_size, 4096);

    bn_relu_inference_kernel_vectorized_gridstride<<<num_blocks, block_size>>>(
        (const float4*)input_cont.data_ptr<float>(), (float4*)output.data_ptr<float>(),
        mean.data_ptr<float>(), var.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        static_cast<float>(eps), num_vec_elements, C, HW);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return output;
}

// Wrapper for Kernel 2
torch::Tensor bn_relu_avgpool_cuda(
    torch::Tensor input, torch::Tensor mean, torch::Tensor var,
    torch::Tensor weight, torch::Tensor bias, double eps) {

    auto input_cont = input.contiguous();
    TORCH_CHECK(input_cont.is_cuda(), "Input must be a contiguous CUDA tensor");
    
    const auto N = input_cont.size(0); const auto C = input_cont.size(1);
    const auto H_in = input_cont.size(2); const auto W_in = input_cont.size(3);
    TORCH_CHECK(H_in % 2 == 0 && W_in % 2 == 0, "Input dimensions must be even for 2x2 pooling");

    const int H_out = H_in / 2; const int W_out = W_in / 2;
    auto output = torch::empty({N, C, H_out, W_out}, input_cont.options());
    const int total_out_elements = output.numel();
    if (total_out_elements == 0) return output;
    
    const int block_size = 256;
    int num_blocks = std::min((total_out_elements + block_size - 1) / block_size, 4096);

    bn_relu_avgpool_kernel<<<num_blocks, block_size>>>(
        input_cont.data_ptr<float>(), output.data_ptr<float>(),
        mean.data_ptr<float>(), var.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        static_cast<float>(eps), total_out_elements, C, H_in, W_in, H_out, W_out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return output;
}

// Wrapper for Kernel 3
torch::Tensor bn_relu_global_avg_pool_cuda(
    torch::Tensor input, torch::Tensor mean, torch::Tensor var,
    torch::Tensor weight, torch::Tensor bias, double eps) {

    auto input_cont = input.contiguous();
    TORCH_CHECK(input_cont.is_cuda(), "Input must be a contiguous CUDA tensor");

    const auto N = input_cont.size(0); const auto C = input_cont.size(1);
    const auto H = input_cont.size(2); const auto W = input_cont.size(3);
    const int HW = H * W;
    auto output = torch::empty({N, C}, input_cont.options());
    if (input_cont.numel() == 0) return output;

    int block_size = 1;
    while(block_size < HW && block_size < 1024) block_size *= 2;
    
    const int num_feature_maps = N * C;
    bn_relu_global_avg_pool_kernel<<<num_feature_maps, block_size, block_size * sizeof(float)>>>(
        input_cont.data_ptr<float>(), output.data_ptr<float>(),
        mean.data_ptr<float>(), var.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        static_cast<float>(eps), N, C, H, W);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return output;
}
"""

fused_kernels_cpp_source = """
torch::Tensor bn_relu_cuda_vectorized(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
torch::Tensor bn_relu_avgpool_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
torch::Tensor bn_relu_global_avg_pool_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
"""

fused_ops = load_inline(
    name="densenet_fused_kernels_v_master",
    cpp_sources=fused_kernels_cpp_source,
    cuda_sources=fused_kernels_source,
    functions=["bn_relu_cuda_vectorized", "bn_relu_avgpool_cuda", "bn_relu_global_avg_pool_cuda"],
    verbose=False,
)

class _DenseLayer(nn.Module):
    def __init__(self, in_features: int, growth_rate: int):
        super(_DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # C++ wrapper handles the .contiguous() call.
        bn_out = fused_ops.bn_relu_cuda_vectorized(
            x, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, self.bn.eps
        )
        conv_out = self.conv(bn_out)
        return self.dropout(conv_out)

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            _DenseLayer(num_input_features + i * growth_rate, growth_rate) for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            new_feature = layer(x)
            x = torch.cat([x, new_feature], 1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        # Algorithmic reordering: (BN -> ReLU -> Pool) are fused, then Conv.
        self.bn = nn.BatchNorm2d(num_input_features)
        # The 1x1 conv now operates on the spatially smaller, downsampled feature map.
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)

    def forward(self, x):
        # Apply fused BN + ReLU + AvgPool. Wrapper handles .contiguous().
        x = fused_ops.bn_relu_avgpool_cuda(
            x, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, self.bn.eps
        )
        # Apply the 1x1 convolution on the smaller feature map.
        x = self.conv(x)
        return x

class Model(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(Model, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_features = 64
        block_layers = [6, 12, 48, 32]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features //= 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        # Use general-purpose BN+ReLU fusion. Output of Conv is contiguous.
        x = fused_ops.bn_relu_cuda_vectorized(
            x, self.bn0.running_mean, self.bn0.running_var, self.bn0.weight, self.bn0.bias, self.bn0.eps
        )
        x = self.pool0(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.dense_blocks) - 1:
                # Use the triple-fused BN+ReLU+AvgPool kernel in the transition layer.
                x = self.transition_layers[i](x)

        # Use the highly optimized fused kernel for the final classification head.
        x = fused_ops.bn_relu_global_avg_pool_cuda(
            x, self.final_bn.running_mean, self.final_bn.running_var, 
            self.final_bn.weight, self.final_bn.bias, self.final_bn.eps
        )
        # Output is already (N, C), so no view is needed.
        x = self.classifier(x)
        return x

batch_size = 10
num_classes = 10
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, 3, height, width, device='cuda')]

def get_init_inputs():
    return [32, num_classes]

# EVOLVE-BLOCK-END