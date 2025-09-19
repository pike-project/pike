# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Combined CUDA source for two highly optimized fused kernels.
# This version synthesizes the best features from all prior attempts.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm> // For std::min

// --- Kernel 1: Fused BatchNorm + ReLU with Grid-Stride Loop ---
// This kernel is applied to the numerous intermediate layers in DenseNet.
// The grid-stride loop makes the kernel robust and can improve performance
// by increasing arithmetic intensity and hiding memory latency.
__global__ void batch_norm_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float eps,
    const int total_elements,
    const int C,
    const int spatial_dim) { // H * W

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x) {
        const int c = (idx / spatial_dim) % C;
        const float mean = running_mean[c];
        const float var = running_var[c];
        const float gamma = weight[c];
        const float beta = bias[c];
        const float inv_std = rsqrtf(var + eps);
        const float normalized_val = gamma * (input[idx] - mean) * inv_std + beta;
        output[idx] = fmaxf(0.0f, normalized_val);
    }
}

torch::Tensor batch_norm_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps) {

    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const int spatial_dim = H * W;
    auto output = torch::empty_like(input);
    const int total_elements = input.numel();

    if (total_elements == 0) return output;

    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    batch_norm_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        static_cast<float>(eps), total_elements, C, spatial_dim);

    return output;
}

// --- Kernel 2: Fused BatchNorm + ReLU + AdaptiveAvgPool with Tuned Block Size & In-Kernel Division ---
// This kernel targets the final stage of the network. It combines the best micro-optimizations:
// 1. Tuned block size (64) for the small final spatial dimension (7x7=49).
// 2. In-kernel division to avoid a separate kernel launch for the averaging step.
__global__ void bn_relu_adaptive_avg_pool_kernel(
    const float* input, float* output, const float* weight, const float* bias,
    const float* running_mean, const float* running_var, float eps,
    int N, int C, int H, int W) {

    // Each block computes the average for one channel of one batch item.
    const int n = blockIdx.y;
    const int c = blockIdx.x;
    const int output_idx = n * C + c;
    const int spatial_size = H * W;

    extern __shared__ float sdata[];
    float thread_sum = 0.0f;

    const float* input_ptr = input + n * C * spatial_size + c * spatial_size;

    // Pre-calculate BN parameters for this channel
    const float mean = running_mean[c];
    const float var = running_var[c];
    const float w = weight[c];
    const float b = bias[c];
    const float inv_std = rsqrtf(var + eps);

    // Each thread processes multiple elements
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = input_ptr[i];
        val = (val - mean) * inv_std * w + b; // BatchNorm
        val = fmaxf(0.0f, val);              // ReLU
        thread_sum += val;                   // Accumulate
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // In-block reduction using shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // First thread writes the final result, including division for the average.
    if (threadIdx.x == 0) {
        if (spatial_size > 0) {
            output[output_idx] = sdata[0] / static_cast<float>(spatial_size);
        } else {
            output[output_idx] = 0.0f;
        }
    }
}

torch::Tensor bn_relu_adaptive_avg_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps) {

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    auto output = torch::empty({N, C}, input.options());

    const int block_size = 64; // Tuned for small 7x7 reduction
    dim3 blockDim(block_size);
    dim3 gridDim(C, N);
    size_t shared_mem_size = block_size * sizeof(float);

    bn_relu_adaptive_avg_pool_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), weight.data_ptr<float>(),
        bias.data_ptr<float>(), running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(), (float)eps, N, C, H, W);
    
    // The final division is now done inside the kernel, so we just return.
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor batch_norm_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps);

torch::Tensor bn_relu_adaptive_avg_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps);
"""

# JIT compile the two fused kernels into a single extension module
fused_ops = load_inline(
    name="fused_densenet_ops_v_best",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["batch_norm_relu_cuda", "bn_relu_adaptive_avg_pool_cuda"],
    verbose=True,
)

class FusedBatchNormReLU(nn.Module):
    """A module that wraps a BatchNorm2d layer to fuse its operation with a ReLU for inference."""
    def __init__(self, bn_module: nn.BatchNorm2d):
        super().__init__()
        self.bn = bn_module

    def forward(self, x):
        if self.training:
            return F.relu(self.bn(x), inplace=True)
        else:
            return fused_ops.batch_norm_relu_cuda(
                x, self.bn.weight, self.bn.bias, self.bn.running_mean,
                self.bn.running_var, self.bn.eps)

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            FusedBatchNormReLU(nn.BatchNorm2d(in_features)),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0))

    def forward(self, x):
        # ALGORITHMIC OPTIMIZATION: More efficient concatenation logic.
        # This avoids redundant data movement by only concatenating the new feature
        # with the accumulated features from previous layers.
        for layer in self.layers:
            new_feature = layer(x)
            x = torch.cat([x, new_feature], 1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            FusedBatchNormReLU(nn.BatchNorm2d(num_input_features)),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.transition(x)

class Model(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.fused_bn_relu1 = FusedBatchNormReLU(self.bn1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        num_features = 64
        block_layers = [6, 12, 24, 16]
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.fused_bn_relu1(x)
        x = self.pool1(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)
        
        # Use the triple-fused kernel for the final stage, which replaces BN -> ReLU -> Pool
        x = fused_ops.bn_relu_adaptive_avg_pool_cuda(
            x.contiguous(), self.final_bn.weight, self.final_bn.bias,
            self.final_bn.running_mean, self.final_bn.running_var, self.final_bn.eps)
        
        # The output of the fused kernel is already flattened (N, C), ready for the classifier
        x = self.classifier(x)
        return x

# Testing the DenseNet121 model
batch_size = 10
num_classes = 10
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]

# EVOLVE-BLOCK-END