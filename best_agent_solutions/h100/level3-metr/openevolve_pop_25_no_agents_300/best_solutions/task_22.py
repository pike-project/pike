# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for fused operations.
# This version uses float4 vectorization for the bulk of the computation
# and a scalar loop for the remainder, making it robust to any tensor size.
# It also uses __ldg for potentially faster reads from global memory.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <c10/cuda/CUDAException.h>

// Enum to specify the activation function
enum class ActivationType {
    NONE = 0,
    RELU = 1,
    RELU6 = 2
};

// Device function to apply the chosen activation
__device__ __forceinline__ float apply_activation(float x, ActivationType act_type) {
    switch (act_type) {
        case ActivationType::RELU:
            return fmaxf(0.0f, x);
        case ActivationType::RELU6:
            return fminf(fmaxf(0.0f, x), 6.0f);
        case ActivationType::NONE:
        default:
            return x;
    }
}

// Fused kernel for BatchNorm -> Activation (Vectorized with float4 + scalar remainder)
__global__ void fused_bn_activation_kernel_hybrid(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    ActivationType act_type,
    int N, int C, int HW,
    int total_elements) {

    const int total_float4 = total_elements / 4;
    const int grid_stride = blockDim.x * gridDim.x;

    // Vectorized part
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_float4; i += grid_stride) {
        const int idx = i * 4;
        const int c = (idx / HW) % C;

        // Assuming HW is a multiple of 4, the channel won't change within a float4.
        const float inv_std = rsqrtf(running_var[c] + eps);
        const float scale = weight[c] * inv_std;
        const float shift = bias[c] - running_mean[c] * scale;

        const float4 in_vec = __ldg(&((const float4*)input)[i]);
        float4 out_vec;
        out_vec.x = in_vec.x * scale + shift;
        out_vec.y = in_vec.y * scale + shift;
        out_vec.z = in_vec.z * scale + shift;
        out_vec.w = in_vec.w * scale + shift;
        
        out_vec.x = apply_activation(out_vec.x, act_type);
        out_vec.y = apply_activation(out_vec.y, act_type);
        out_vec.z = apply_activation(out_vec.z, act_type);
        out_vec.w = apply_activation(out_vec.w, act_type);
        
        ((float4*)output)[i] = out_vec;
    }

    // Scalar remainder part
    const int remainder_start = total_float4 * 4;
    for (int idx = remainder_start + blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += grid_stride) {
        const int c = (idx / HW) % C;

        const float inv_std = rsqrtf(running_var[c] + eps);
        const float scale = weight[c] * inv_std;
        const float shift = bias[c] - running_mean[c] * scale;

        float bn_val = __ldg(&input[idx]) * scale + shift;
        output[idx] = apply_activation(bn_val, act_type);
    }
}

// Fused kernel for BatchNorm -> Add (Vectorized with float4 + scalar remainder)
__global__ void fused_bn_add_kernel_hybrid(
    const float* __restrict__ input,
    const float* __restrict__ identity,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int HW,
    int total_elements) {

    const int total_float4 = total_elements / 4;
    const int grid_stride = blockDim.x * gridDim.x;

    // Vectorized part
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_float4; i += grid_stride) {
        const int idx = i * 4;
        const int c = (idx / HW) % C;

        const float inv_std = rsqrtf(running_var[c] + eps);
        const float scale = weight[c] * inv_std;
        const float shift = bias[c] - running_mean[c] * scale;

        const float4 in_vec = __ldg(&((const float4*)input)[i]);
        const float4 identity_vec = __ldg(&((const float4*)identity)[i]);
        float4 out_vec;

        out_vec.x = (in_vec.x * scale + shift) + identity_vec.x;
        out_vec.y = (in_vec.y * scale + shift) + identity_vec.y;
        out_vec.z = (in_vec.z * scale + shift) + identity_vec.z;
        out_vec.w = (in_vec.w * scale + shift) + identity_vec.w;
        
        ((float4*)output)[i] = out_vec;
    }

    // Scalar remainder part
    const int remainder_start = total_float4 * 4;
    for (int idx = remainder_start + blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += grid_stride) {
        const int c = (idx / HW) % C;

        const float inv_std = rsqrtf(running_var[c] + eps);
        const float scale = weight[c] * inv_std;
        const float shift = bias[c] - running_mean[c] * scale;

        float bn_val = __ldg(&input[idx]) * scale + shift;
        output[idx] = bn_val + __ldg(&identity[idx]);
    }
}

// C++ wrapper for fused_bn_activation
torch::Tensor fused_bn_activation_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var,
    double eps, int activation_type) {

    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor");
    
    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto HW = input.size(2) * input.size(3);
    const auto total_elements = input.numel();
    auto output = torch::empty_like(input);

    if (total_elements == 0) return output;

    const int block_size = 256;
    const int num_blocks = std::min((int)((total_elements + block_size - 1) / block_size), 4096);
    
    fused_bn_activation_kernel_hybrid<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        static_cast<float>(eps), static_cast<ActivationType>(activation_type),
        N, C, HW, total_elements
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

// C++ wrapper for fused_bn_add
torch::Tensor fused_bn_add_cuda(
    torch::Tensor input, torch::Tensor identity, torch::Tensor weight,
    torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, double eps) {

    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor");
    TORCH_CHECK(identity.is_cuda() && identity.is_contiguous(), "Identity must be a contiguous CUDA tensor");
    TORCH_CHECK(input.sizes() == identity.sizes(), "Input and identity tensors must have the same shape");

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto HW = input.size(2) * input.size(3);
    const auto total_elements = input.numel();
    auto output = torch::empty_like(input);

    if (total_elements == 0) return output;

    const int block_size = 256;
    const int num_blocks = std::min((int)((total_elements + block_size - 1) / block_size), 4096);
    
    fused_bn_add_kernel_hybrid<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), identity.data_ptr<float>(), output.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        static_cast<float>(eps), N, C, HW, total_elements
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_bn_activation_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int);
torch::Tensor fused_bn_add_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
"""

# JIT compile the CUDA code
fused_ops = load_inline(
    name="fused_ops_vec4_hybrid",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_bn_activation_cuda", "fused_bn_add_cuda"],
    verbose=False,
)


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.has_expand = expand_ratio != 1
        hidden_dim = in_channels * expand_ratio
        
        if self.has_expand:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.expand_bn = nn.BatchNorm2d(hidden_dim)
        
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = x
        
        if self.has_expand:
            # Expansion phase with fused BN-ReLU6
            x_conv = self.expand_conv(x)
            x = fused_ops.fused_bn_activation_cuda(
                x_conv, self.expand_bn.weight, self.expand_bn.bias,
                self.expand_bn.running_mean, self.expand_bn.running_var,
                self.expand_bn.eps, 2  # 2 for ReLU6
            )
        
        # Depthwise phase with fused BN-ReLU6
        x_conv = self.depthwise_conv(x)
        x = fused_ops.fused_bn_activation_cuda(
            x_conv, self.depthwise_bn.weight, self.depthwise_bn.bias,
            self.depthwise_bn.running_mean, self.depthwise_bn.running_var,
            self.depthwise_bn.eps, 2  # 2 for ReLU6
        )
        
        # Projection phase
        x_proj = self.project_conv(x)
        
        if self.use_residual:
            # Fused BN + residual add
            x = fused_ops.fused_bn_add_cuda(
                x_proj, identity, self.project_bn.weight, self.project_bn.bias,
                self.project_bn.running_mean, self.project_bn.running_var,
                self.project_bn.eps
            )
        else:
            # Fused BN with no activation
            x = fused_ops.fused_bn_activation_cuda(
                x_proj, self.project_bn.weight, self.project_bn.bias,
                self.project_bn.running_mean, self.project_bn.running_var,
                self.project_bn.eps, 0 # 0 for NONE
            )
        
        return x

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB0 architecture implementation in PyTorch.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(Model, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            # MBConv1 (32, 16, 1, 1)
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the EfficientNetB0 model.

        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # Initial block with fused BN-ReLU
        x_conv = self.conv1(x)
        x = fused_ops.fused_bn_activation_cuda(
            x_conv, self.bn1.weight, self.bn1.bias,
            self.bn1.running_mean, self.bn1.running_var,
            self.bn1.eps, 1 # 1 for ReLU
        )

        x = self.blocks(x)
        
        # Final block with fused BN-ReLU
        x_conv = self.conv2(x)
        x = fused_ops.fused_bn_activation_cuda(
            x_conv, self.bn2.weight, self.bn2.bias,
            self.bn2.running_mean, self.bn2.running_var,
            self.bn2.eps, 1 # 1 for ReLU
        )

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    # Ensure input tensor is on CUDA and has a contiguous memory layout
    return [torch.randn(batch_size, 3, 224, 224).cuda().contiguous()]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END