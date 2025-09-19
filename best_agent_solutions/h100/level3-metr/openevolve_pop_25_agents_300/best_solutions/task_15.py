# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Combined CUDA source for a general-purpose Fused BatchNorm+ReLU and a specialized
# Fused BatchNorm+ReLU+AdaptiveAvgPool. This combination targets the most frequent
# operator patterns in the DenseNet architecture.
fused_kernels_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

// --- Kernel 1a: Fused BatchNorm + ReLU (Scalar Fallback) ---
// A generic, templated kernel that works for any floating-point data type.
// It uses a grid-stride loop for maximum hardware utilization.
template <typename T>
__global__ void batch_norm_relu_kernel_scalar(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int total_elements,
    int C,
    int spatial_dim) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += gridDim.x * blockDim.x) {
        int c = (i / spatial_dim) % C;

        // Load BN parameters for the current channel
        float mean = running_mean[c];
        float var = running_var[c];
        float w = weight ? weight[c] : 1.0f;
        float b = bias ? bias[c] : 0.0f;
        float inv_std = rsqrtf(var + eps);

        // Perform fused operation: BN -> ReLU
        T val = input[i];
        float normalized = w * (static_cast<float>(val) - mean) * inv_std + b;
        output[i] = static_cast<T>(fmaxf(0.f, normalized));
    }
}

// --- Kernel 1b: Fused BatchNorm + ReLU (Vectorized with float4) ---
// An optimized kernel for float32 data that processes 4 elements at a time,
// maximizing memory bandwidth.
__global__ void batch_norm_relu_kernel_float4(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int total_vec_elements,
    int C,
    int spatial_dim) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vec_elements; i += gridDim.x * blockDim.x) {
        const int linear_idx = i * 4;
        // Since spatial_dim (H*W) is a multiple of 4, all 4 elements
        // of a float4 vector belong to the same channel.
        const int c = (linear_idx / spatial_dim) % C;

        const float4 in_val = *reinterpret_cast<const float4*>(&input[linear_idx]);
        
        const float mean = running_mean[c];
        const float var = running_var[c];
        const float w = weight ? weight[c] : 1.0f;
        const float b = bias ? bias[c] : 0.0f;
        const float inv_std = rsqrtf(var + eps);
        
        float4 out_val;
        out_val.x = fmaxf(0.f, w * (in_val.x - mean) * inv_std + b);
        out_val.y = fmaxf(0.f, w * (in_val.y - mean) * inv_std + b);
        out_val.z = fmaxf(0.f, w * (in_val.z - mean) * inv_std + b);
        out_val.w = fmaxf(0.f, w * (in_val.w - mean) * inv_std + b);

        *reinterpret_cast<float4*>(&output[linear_idx]) = out_val;
    }
}

// C++ Wrapper with Dynamic Dispatch for BatchNorm + ReLU
torch::Tensor batch_norm_relu_forward(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {

    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    
    auto output = torch::empty_like(input);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const long total_elements = input.numel();
    const int spatial_dim = H * W;

    const int block_size = 256;

    // Dispatch to the faster vectorized kernel if possible (float32, aligned data).
    if (input.scalar_type() == torch::kFloat32 && (spatial_dim % 4 == 0)) {
        const int total_vec_elements = total_elements / 4;
        const int num_blocks = std::min((total_vec_elements + block_size - 1) / block_size, 65535);
        
        batch_norm_relu_kernel_float4<<<num_blocks, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            weight.defined() ? weight.data_ptr<float>() : nullptr,
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            static_cast<float>(eps),
            total_vec_elements,
            C,
            spatial_dim
        );
    } else {
        // Fallback to the generic scalar kernel for other data types or incompatible shapes.
        const int num_blocks = std::min((int)(total_elements + block_size - 1) / block_size, 65535);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "batch_norm_relu_forward_scalar", ([&] {
            batch_norm_relu_kernel_scalar<scalar_t><<<num_blocks, block_size>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.defined() ? weight.data_ptr<float>() : nullptr,
                bias.defined() ? bias.data_ptr<float>() : nullptr,
                running_mean.data_ptr<float>(),
                running_var.data_ptr<float>(),
                static_cast<float>(eps),
                total_elements,
                C,
                spatial_dim
            );
        }));
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// --- Kernel 2: Fused BatchNorm + ReLU + AdaptiveAvgPool ---
// This specialized kernel handles the final block of the network. Each CUDA block
// processes one channel of the input tensor, performing BN, ReLU, and reduction.
__global__ void bn_relu_adaptive_avg_pool_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float eps,
    const int B,
    const int C,
    const int H,
    const int W) {

    const int b = blockIdx.x;
    const int c = blockIdx.y;
    extern __shared__ float sdata[];

    // Load channel-specific parameters once per block
    const float mean = running_mean[c];
    const float var = running_var[c];
    const float gamma = weight[c];
    const float beta = bias[c];
    const float inv_std = rsqrtf(var + eps);
    
    float my_sum = 0.0f;
    const int plane_size = H * W;
    const int thread_idx = threadIdx.x;
    const int block_dim = blockDim.x;
    const float* x_plane_ptr = x + b * C * H * W + c * H * W;

    // Each thread computes a partial sum over the spatial dimensions
    for (int i = thread_idx; i < plane_size; i += block_dim) {
        const float val = x_plane_ptr[i];
        const float normalized_val = (val - mean) * inv_std * gamma + beta;
        const float relu_val = fmaxf(0.0f, normalized_val);
        my_sum += relu_val;
    }
    sdata[thread_idx] = my_sum;
    __syncthreads();

    // Perform parallel reduction within the block using shared memory
    for (int s = block_dim / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            sdata[thread_idx] += sdata[thread_idx + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final average to global memory
    if (thread_idx == 0) {
        out[b * C + c] = sdata[0] / (float)plane_size;
    }
}

torch::Tensor bn_relu_pool_cuda(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {

    const auto B = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    auto out = torch::empty({B, C}, x.options());
    const int block_dim = 256;
    const dim3 block_size(block_dim);
    const dim3 grid_size(B, C);
    const size_t shared_mem_size = block_dim * sizeof(float);

    bn_relu_adaptive_avg_pool_kernel<<<grid_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        (float)eps,
        B, C, H, W
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

fused_kernels_cpp_source = """
torch::Tensor batch_norm_relu_forward(
    torch::Tensor input, torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight, torch::Tensor bias, double eps);

torch::Tensor bn_relu_pool_cuda(
    torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight, torch::Tensor bias, double eps);
"""

# JIT compile the CUDA/C++ code with optimization flags
fused_ops = load_inline(
    name="densenet_fused_kernels_final",
    cpp_sources=fused_kernels_cpp_source,
    cuda_sources=fused_kernels_source,
    functions=["batch_norm_relu_forward", "bn_relu_pool_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class FusedBatchNormReLU(nn.BatchNorm2d):
    """
    Custom module that fuses BatchNorm2d and ReLU for efficient inference.
    It inherits from nn.BatchNorm2d to seamlessly handle parameters and falls back
    to the standard PyTorch implementation during training.
    """
    def forward(self, x):
        # Use custom kernel only for inference with tracked running stats
        if not self.training and self.track_running_stats:
            return fused_ops.batch_norm_relu_forward(
                x, self.running_mean, self.running_var, self.weight, self.bias, self.eps
            )
        else:
            # Fallback to PyTorch's native implementation for training
            out = F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps
            )
            return F.relu(out, inplace=True)

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        # Replace the standard BN+ReLU with our fused version
        return nn.Sequential(
            FusedBatchNormReLU(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        # Slightly cleaner forward pass that avoids creating a list
        features = x
        for layer in self.layers:
            new_feature = layer(features)
            features = torch.cat([features, new_feature], 1)
        return features

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        # Replace the standard BN+ReLU with our fused version
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

        # Initial convolution block with the fused BN+ReLU module
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            FusedBatchNormReLU(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = 64
        block_layers = [6, 12, 24, 16]

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

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        # Apply the highly specialized fused kernel for the final stage during inference
        if not self.training and x.is_cuda:
            # This single call replaces BN, ReLU, AdaptiveAvgPool, and view
            x = fused_ops.bn_relu_pool_cuda(
                x, self.final_bn.running_mean, self.final_bn.running_var,
                self.final_bn.weight, self.final_bn.bias, self.final_bn.eps
            )
        else:
            # Standard PyTorch path for training or non-CUDA execution
            x = self.final_bn(x)
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        x = self.classifier(x)
        return x

batch_size = 10
num_classes = 10
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]

# EVOLVE-BLOCK-END