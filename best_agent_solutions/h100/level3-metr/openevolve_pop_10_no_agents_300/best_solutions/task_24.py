# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Fused BatchNorm + ReLU using pre-folded weights
# This is the most efficient approach for inference, as it minimizes calculations within the kernel.
fused_bn_relu_folded_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Kernel for float32 precision
__global__ void fused_bn_relu_folded_kernel_float(const float* __restrict__ input,
                                     const float* __restrict__ scale,
                                     const float* __restrict__ bias,
                                     float* __restrict__ output,
                                     const int C,
                                     const int inner_size,
                                     const long long total_elements) {
    // Grid-stride loop for maximum efficiency and flexibility
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_elements;
         i += gridDim.x * blockDim.x) {
        
        // Calculate channel index
        const int c = (i / inner_size) % C;
        // Fused operation: y = max(0, x * scale + bias)
        const float val = input[i] * scale[c] + bias[c];
        output[i] = fmaxf(val, 0.0f);
    }
}

// Specialized kernel for float16 (half) precision
__global__ void fused_bn_relu_folded_kernel_half(const __half* __restrict__ input,
                                          const __half* __restrict__ scale,
                                          const __half* __restrict__ bias,
                                          __half* __restrict__ output,
                                          const int C,
                                          const int inner_size,
                                          const long long total_elements) {
    const __half zero = __float2half(0.0f);
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_elements;
         i += gridDim.x * blockDim.x) {

        const int c = (i / inner_size) % C;
        // Use CUDA's fused multiply-add and max intrinsics for half precision
        const __half val = __hadd(__hmul(input[i], scale[c]), bias[c]);
        output[i] = __hmax(val, zero);
    }
}


torch::Tensor fused_bn_relu_folded_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias) {
    // --- Input validation ---
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(torch::MemoryFormat::Contiguous), "Input must be contiguous");
    TORCH_CHECK(scale.is_cuda(), "Scale must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(input.scalar_type() == scale.scalar_type() && input.scalar_type() == bias.scalar_type(), "All tensors must have the same dtype");

    auto output = torch::empty_like(input, torch::MemoryFormat::Contiguous);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const int inner_size = H * W; // Spatial dimensions
    const long long total_elements = input.numel();

    if (total_elements == 0) {
        return output;
    }

    // --- Kernel launch configuration ---
    // Use 1024 threads per block for memory-bound kernels to maximize occupancy
    const int block_size = 1024;
    // Heuristic for grid size: ensure enough blocks to saturate the GPU, but not too many to cause launch overhead.
    const int grid_size = std::min((int)((total_elements + block_size - 1) / block_size), 4096);

    // --- Dispatch to appropriate kernel based on data type ---
    if (input.scalar_type() == torch::kFloat32) {
        fused_bn_relu_folded_kernel_float<<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            scale.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            C, inner_size, total_elements);
    } else if (input.scalar_type() == torch::kFloat16) {
        fused_bn_relu_folded_kernel_half<<<grid_size, block_size>>>(
            (const __half*)input.data_ptr(),
            (const __half*)scale.data_ptr(),
            (const __half*)bias.data_ptr(),
            (__half*)output.data_ptr(),
            C, inner_size, total_elements);
    } else {
        TORCH_CHECK(false, "Unsupported dtype for fused_bn_relu_folded. Only float32 and float16 are supported.");
    }
    
    // Check for errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}
"""

fused_bn_relu_folded_cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_bn_relu_folded_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias);
"""

# JIT compile the custom CUDA kernel with optimizations
fused_op_module = load_inline(
    name="fused_op_module_efficientnet",
    cpp_sources=fused_bn_relu_folded_cpp_source,
    cuda_sources=fused_bn_relu_folded_source,
    functions=["fused_bn_relu_folded_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class FusedBatchNormReLU(nn.Module):
    """
    Fuses a BatchNorm2d and ReLU into a single operation for efficient inference.
    The BatchNorm parameters are "folded" into a single scale and bias during initialization.
    """
    def __init__(self, bn_module: nn.BatchNorm2d):
        super().__init__()
        # Ensure the bn_module is in eval mode to use running_mean and running_var
        bn_module.eval()
            
        # Pre-compute the folded parameters: y = (x - mean) / sqrt(var + eps) * gamma + beta
        # Becomes y = x * (gamma / std) + (beta - mean * gamma / std)
        gamma = bn_module.weight
        beta = bn_module.bias
        running_mean = bn_module.running_mean
        running_var = bn_module.running_var
        eps = bn_module.eps
        
        std = torch.sqrt(running_var + eps)
        scale = gamma / std
        bias = beta - running_mean * scale
        
        # Register scale and bias as buffers. They are part of the model's state
        # but are not parameters to be trained.
        self.register_buffer('scale', scale)
        self.register_buffer('bias', bias)

    def forward(self, x):
        # Call the custom CUDA kernel
        return fused_op_module.fused_bn_relu_folded_cuda(x, self.scale, self.bias)

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Manually fuse the first BatchNorm and ReLU
        self.fused_bn_relu1 = FusedBatchNormReLU(nn.BatchNorm2d(32))
        
        # Define and fuse the MBConv blocks
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)
        
        # Final layers with fusion
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.fused_bn_relu_final = FusedBatchNormReLU(nn.BatchNorm2d(1408))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1408, num_classes)
    
    def _fuse_bn_relu_in_list(self, module_list):
        """Iterates a list of modules, replacing BatchNorm2d -> ReLU sequences with our fused module."""
        fused_list = []
        i = 0
        while i < len(module_list):
            # Check for the pattern: BatchNorm2d followed by ReLU
            if (i + 1 < len(module_list) and
                isinstance(module_list[i], nn.BatchNorm2d) and
                isinstance(module_list[i+1], nn.ReLU)):
                
                bn = module_list[i]
                fused_module = FusedBatchNormReLU(bn)
                fused_list.append(fused_module)
                i += 2  # Skip the next module (ReLU) as it has been fused
            else:
                fused_list.append(module_list[i])
                i += 1
        return fused_list

    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        layers = []
        expanded_channels = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(expanded_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # Depthwise convolution
        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(expanded_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Squeeze and Excitation
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(expanded_channels // 4, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())
        
        # Output phase
        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        # Apply fusion to the generated list of layers
        fused_layers = self._fuse_bn_relu_in_list(layers)
        
        return nn.Sequential(*fused_layers)
    
    def forward(self, x):
        x = self.fused_bn_relu1(self.conv1(x))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.fused_bn_relu_final(self.conv_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Test code
batch_size = 2
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END