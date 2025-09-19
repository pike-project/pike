# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os
import tempfile

# Use a temporary directory for Torch C++ extensions to avoid permission issues.
os.environ['TORCH_EXTENSIONS_DIR'] = os.path.join(tempfile.gettempdir(), 'torch_extensions')

# --- Kernel: Fused & Vectorized BatchNorm + ReLU ---
# This kernel combines the best strategies from prior attempts:
# 1. Fusion: BatchNorm and ReLU are combined into a single kernel, saving memory
#    bandwidth by eliminating an intermediate tensor and reducing kernel launch overhead.
# 2. Direct Computation: Following the top-performing solution, all BN parameters are passed
#    directly to the kernel. This avoids Python-side pre-computation ("folding") which
#    was shown to add overhead and slow down the model.
# 3. Vectorization: Data is loaded, processed, and stored using `float4`, which reads/writes
#    4 floating-point numbers in a single transaction. This is a powerful optimization for
#    memory-bound kernels, maximizing GPU memory bandwidth.
fused_bn_relu_vectorized_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h> // For C10_CUDA_KERNEL_LAUNCH_CHECK

__global__ void bn_relu_fused_vectorized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int num_vec_elements, // total_elements / 4
    int C,
    int inner_size) { // inner_size is H*W

    // Grid-stride loop over vectorized elements (float4) for robustness.
    for (int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
         vec_idx < num_vec_elements;
         vec_idx += blockDim.x * gridDim.x) {

        // All 4 elements in a float4 vector share the same channel index.
        int c = ((vec_idx * 4) / inner_size) % C;

        // Load per-channel BN parameters ONCE for the 4 elements.
        const float mean = running_mean[c];
        const float var = running_var[c];
        const float w = weight[c];
        const float b = bias[c];
        const float inv_std = rsqrtf(var + eps);

        // Load a vector of 4 floats in a single memory transaction.
        const float4 in_val = reinterpret_cast<const float4*>(input)[vec_idx];
        float4 out_val;

        // Apply the fused operation element-wise to each component of the vector.
        out_val.x = fmaxf(0.0f, (in_val.x - mean) * inv_std * w + b);
        out_val.y = fmaxf(0.0f, (in_val.y - mean) * inv_std * w + b);
        out_val.z = fmaxf(0.0f, (in_val.z - mean) * inv_std * w + b);
        out_val.w = fmaxf(0.0f, (in_val.w - mean) * inv_std * w + b);

        // Store the resulting vector in a single memory transaction.
        reinterpret_cast<float4*>(output)[vec_idx] = out_val;
    }
}

// C++ wrapper to launch the kernel from PyTorch.
torch::Tensor bn_relu_cuda_inference_vectorized(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    // Input validation is critical for custom CUDA ops.
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor (NCHW)");
    const int total_elements = input.numel();
    // Vectorization requires that the memory can be interpreted as a sequence of float4s.
    // For MobileNetV1 feature maps, H*W is always divisible by 4, so this holds.
    TORCH_CHECK(total_elements % 4 == 0, "Vectorized kernel requires numel to be divisible by 4");

    auto output = torch::empty_like(input);
    if (total_elements == 0) return output;

    const int C = input.size(1);
    const int inner_size = input.size(2) * input.size(3);
    const int num_vec_elements = total_elements / 4;

    // Standard kernel launch configuration.
    const int block_size = 256;
    const int num_blocks = std::min((num_vec_elements + block_size - 1) / block_size, 4096);
    
    bn_relu_fused_vectorized_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        num_vec_elements,
        C,
        inner_size);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK(); // Error checking for safer execution.
    return output;
}
"""

fused_bn_relu_vectorized_cpp_source = """
torch::Tensor bn_relu_cuda_inference_vectorized(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps);
"""

# JIT compile the custom CUDA kernel. Use a unique name to avoid build cache conflicts.
fused_ops = load_inline(
    name="fused_bn_relu_vectorized_v_SOTA",
    cpp_sources=fused_bn_relu_vectorized_cpp_source,
    cuda_sources=fused_bn_relu_vectorized_source,
    functions=["bn_relu_cuda_inference_vectorized"],
    verbose=False,
)

class FusedBatchNormReLU(nn.BatchNorm2d):
    """
    Custom module that fuses BatchNorm2d and ReLU for inference using a vectorized kernel.
    It inherits from nn.BatchNorm2d, making it a seamless drop-in replacement.
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # During training, we must use the standard PyTorch implementation to ensure
        # correct gradient calculation and updates to running_mean/running_var.
        if self.training:
            bn_output = super(FusedBatchNormReLU, self).forward(input)
            return F.relu(bn_output, inplace=True)
        
        # During inference (model.eval()), we use our highly optimized custom kernel.
        else:
            return fused_ops.bn_relu_cuda_inference_vectorized(
                input, self.weight, self.bias, self.running_mean, self.running_var, self.eps
            )


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        """
        MobileNetV1 architecture with a highly optimized fused and vectorized BatchNorm+ReLU kernel.
        """
        super(Model, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                FusedBatchNormReLU(oup)
            )
        
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                FusedBatchNormReLU(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                FusedBatchNormReLU(oup),
            )
        
        self.model = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
    
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Boilerplate for testing
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000
alpha = 1.0

def get_inputs():
    # Input tensor must be on the GPU for the custom CUDA kernel.
    return [torch.randn(batch_size, input_channels, height, width).cuda()]

def get_init_inputs():
    return [num_classes, input_channels, alpha]
# EVOLVE-BLOCK-END