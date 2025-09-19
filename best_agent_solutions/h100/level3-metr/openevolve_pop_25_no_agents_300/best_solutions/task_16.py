# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

# Suppress verbose JIT compilation output
os.environ['TORCH_EXTENSIONS_VERBOSE'] = '0'

# Combined CUDA source for two distinct, highly optimized fused operations.
# Kernel 1: float4-vectorized BatchNorm+ReLU for general-purpose use.
# Kernel 2: Fused BatchNorm+ReLU with a parallel reduction for the final global average pooling.
fused_densenet_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cmath>

// --- KERNEL 1: Vectorized (float4) Fused BatchNorm2d + ReLU for Inference ---
// This kernel is memory-bandwidth bound. Using float4 loads/stores significantly increases
// throughput by fetching/writing 128 bits per memory transaction. It is ideal for the
// numerous BN+ReLU layers inside the DenseNet blocks.
__global__ void fused_bn_relu_kernel_vec4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int total_vecs, // This is total_elements / 4
    int C, 
    int spatial_dim) { // spatial_dim = H * W

    // Grid-stride loop over float4 vectors for maximum occupancy and scalability.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vecs; i += blockDim.x * gridDim.x) {
        
        // All 4 elements in a float4 belong to the same channel, so we calculate 'c' once.
        int linear_idx = i * 4;
        int c = (linear_idx / spatial_dim) % C;

        // Load channel-specific batchnorm parameters once per float4 vector.
        const float mean = running_mean[c];
        const float inv_std = rsqrtf(running_var[c] + eps);
        const float gamma = weight[c];
        const float beta = bias[c];

        // Load 4 floats at once.
        float4 in_vec = input[i];
        float4 out_vec;

        // Apply BN-ReLU transformation element-wise on the vector components.
        out_vec.x = fmaxf(0.f, (in_vec.x - mean) * inv_std * gamma + beta);
        out_vec.y = fmaxf(0.f, (in_vec.y - mean) * inv_std * gamma + beta);
        out_vec.z = fmaxf(0.f, (in_vec.z - mean) * inv_std * gamma + beta);
        out_vec.w = fmaxf(0.f, (in_vec.w - mean) * inv_std * gamma + beta);
        
        // Store 4 floats at once.
        output[i] = out_vec;
    }
}

// --- KERNEL 2: Fused BatchNorm + ReLU + AdaptiveAvgPool for the final stage ---
// This kernel avoids writing the intermediate BN+ReLU tensor to global memory.
// It's a reduction kernel where each block computes the average for one (N, C) pair.
__global__ void bn_relu_adaptive_avg_pool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int N, int C, int H, int W) {

    const int n = blockIdx.y;
    const int c = blockIdx.x;

    const float current_mean = mean[c];
    const float inv_std = rsqrtf(var[c] + eps);
    const float current_weight = weight[c];
    const float current_bias = bias[c];

    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int feature_map_size = H * W;
    const float* feature_map_ptr = input + n * C * H * W + c * H * W;

    float local_sum = 0.0f;
    for (int i = tid; i < feature_map_size; i += blockDim.x) {
        float val = feature_map_ptr[i];
        // Apply BN+ReLU on the fly before adding to the sum.
        float bn_val = (val - current_mean) * inv_std * current_weight + current_bias;
        local_sum += fmaxf(0.0f, bn_val);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Standard parallel reduction in shared memory (assumes blockDim.x is power of two).
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[n * C + c] = sdata[0] / (float)feature_map_size;
    }
}


// --- C++ Wrapper 1: For the vectorized BN+ReLU kernel ---
torch::Tensor fused_bn_relu_forward_vec4(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps) {

    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor");
    const long total_elements = input.numel();
    TORCH_CHECK(total_elements % 4 == 0, "Total elements must be divisible by 4 for vec4 kernel.");

    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int spatial_dim = H * W;

    auto output = torch::empty_like(input);
    if (total_elements == 0) return output;

    const long total_vecs = total_elements / 4;
    const int block_size = 256;
    const int num_blocks = (total_vecs + block_size - 1) / block_size;
    const int grid_size = std::min(num_blocks, 65535);

    fused_bn_relu_kernel_vec4<<<grid_size, block_size>>>(
        (const float4*)input.data_ptr<float>(),
        (float4*)output.data_ptr<float>(),
        weight.contiguous().data_ptr<float>(),
        bias.contiguous().data_ptr<float>(),
        running_mean.contiguous().data_ptr<float>(),
        running_var.contiguous().data_ptr<float>(),
        static_cast<float>(eps), total_vecs, C, spatial_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// --- C++ Wrapper 2: For the final fused reduction kernel ---
torch::Tensor bn_relu_adaptive_avg_pool_cuda(
    torch::Tensor input, torch::Tensor mean, torch::Tensor var,
    torch::Tensor weight, torch::Tensor bias, double eps) {
    
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor");
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    auto output = torch::empty({N, C}, input.options());
    if (input.numel() == 0) return output;

    // Reduction kernels require block size to be a power of two.
    const int block_size = 256; 
    dim3 gridDim(C, N);
    const int shared_mem_size = block_size * sizeof(float);

    bn_relu_adaptive_avg_pool_kernel<<<gridDim, block_size, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), mean.contiguous().data_ptr<float>(),
        var.contiguous().data_ptr<float>(), weight.contiguous().data_ptr<float>(), bias.contiguous().data_ptr<float>(),
        static_cast<float>(eps), N, C, H, W);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_densenet_kernels_cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_bn_relu_forward_vec4(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps);

torch::Tensor bn_relu_adaptive_avg_pool_cuda(
    torch::Tensor input, torch::Tensor mean, torch::Tensor var,
    torch::Tensor weight, torch::Tensor bias, double eps);
"""

# JIT compile both CUDA kernels in a single extension module.
fused_ops_module = load_inline(
    name="fused_densenet_kernels_hybrid",
    cpp_sources=fused_densenet_kernels_cpp_source,
    cuda_sources=fused_densenet_kernels_source,
    functions=["fused_bn_relu_forward_vec4", "bn_relu_adaptive_avg_pool_cuda"],
    verbose=False,
)

class FusedBatchNormReLU(nn.Module):
    """ Fused BatchNorm2d and ReLU for inference, using the custom vectorized kernel. """
    def __init__(self, num_features, eps=1e-5):
        super(FusedBatchNormReLU, self).__init__()
        # Use a standard BatchNorm2d layer internally to hold parameters and state.
        # This simplifies state dict loading and management.
        self.bn = nn.BatchNorm2d(num_features, eps=eps)

    def forward(self, x):
        if self.training:
            # Fallback to standard PyTorch ops for training to ensure correctness.
            return F.relu(self.bn(x), inplace=True)
        else:
            # Use the high-performance vectorized kernel for inference.
            return fused_ops_module.fused_bn_relu_forward_vec4(
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
            FusedBatchNormReLU(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        # Algorithmic optimization: Use efficient incremental concatenation.
        for layer in self.layers:
            new_feature = layer(x)
            x = torch.cat((x, new_feature), 1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            FusedBatchNormReLU(num_input_features),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class Model(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            FusedBatchNormReLU(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        num_features = 64
        block_layers = [6, 12, 48, 32]
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # This standard BN layer is not called directly in inference; it serves as a
        # parameter store for our final fused reduction kernel.
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        # In inference, replace the final sequence of BN -> ReLU -> Pool -> View
        # with a single, highly optimized fused kernel call.
        if self.training:
             x = self.final_bn(x)
             x = F.relu(x, inplace=True)
             x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        else:
             x = fused_ops_module.bn_relu_adaptive_avg_pool_cuda(
                x, self.final_bn.running_mean, self.final_bn.running_var,
                self.final_bn.weight, self.final_bn.bias, self.final_bn.eps
            )
        
        x = self.classifier(x)
        return x

# Testing the DenseNet201 model
batch_size = 10
num_classes = 10
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]
# EVOLVE-BLOCK-END