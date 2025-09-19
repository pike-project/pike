# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused BatchNorm2d and ReLU
# This version combines the best features of prior attempts:
# 1. 2D Grid Launch (inspired by Program 2): Assigns one block per (N, C) pair for better data locality.
# 2. Vectorized Memory Access (inspired by Program 3): Uses float4/half2 to maximize memory bandwidth.
# 3. Pre-computation of scale/shift (common best practice): Folds the BN math into a single FMA.
# 4. Templated for float/half support.
fused_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

template <typename T>
__global__ void fused_bn_relu_kernel_vectorized_2d(
    const T* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    const int C, const int H, const int W) {

    // Each block processes one plane (n, c).
    const int n = blockIdx.y;
    const int c = blockIdx.x;

    // Pre-fetch scale and shift for the current channel.
    const float s = scale[c];
    const float sh = shift[c];

    const int hw = H * W;
    const long long plane_offset = (long long)(n * C + c) * hw;
    const T* input_ptr = input + plane_offset;
    T* output_ptr = output + plane_offset;
    
    // Vectorization requires alignment. We check W % 4 == 0 in the host code.
    const int vec_size = 4;
    const int total_vectors_in_plane = hw / vec_size;

    // Use a grid-stride loop over the vectors in the plane.
    for (int i = threadIdx.x; i < total_vectors_in_plane; i += blockDim.x) {
        if constexpr (std::is_same_v<T, float>) {
            // Load 4 floats at once
            float4 loaded_val = reinterpret_cast<const float4*>(input_ptr)[i];
            float4 result;
            // Apply BN folding (FMA) and ReLU (fmaxf) element-wise
            result.x = fmaxf(0.f, loaded_val.x * s + sh);
            result.y = fmaxf(0.f, loaded_val.y * s + sh);
            result.z = fmaxf(0.f, loaded_val.z * s + sh);
            result.w = fmaxf(0.f, loaded_val.w * s + sh);
            // Store 4 floats at once
            reinterpret_cast<float4*>(output_ptr)[i] = result;
        } else if constexpr (std::is_same_v<T, half>) {
            // For half precision, process 2 half2s per loop iteration (4 halfs total)
            const __half2 scale_h2 = __float2half2_rn(s);
            const __half2 shift_h2 = __float2half2_rn(sh);
            const __half2 zero_h2 = __float2half2_rn(0.f);
            
            const int h2_idx = i * 2;
            const half2 val1 = reinterpret_cast<const half2*>(input_ptr)[h2_idx];
            const half2 val2 = reinterpret_cast<const half2*>(input_ptr)[h2_idx + 1];

            // Use efficient half2 intrinsics for FMA and max operations
            half2 res1 = __hfma2(val1, scale_h2, shift_h2);
            res1 = __hmax2(res1, zero_h2);

            half2 res2 = __hfma2(val2, scale_h2, shift_h2);
            res2 = __hmax2(res2, zero_h2);

            reinterpret_cast<half2*>(output_ptr)[h2_idx] = res1;
            reinterpret_cast<half2*>(output_ptr)[h2_idx+1] = res2;
        }
    }
}

torch::Tensor fused_bn_relu_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps)
{
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4-dimensional (NCHW)");
    TORCH_CHECK(x.size(3) % 4 == 0, "Input tensor's width must be a multiple of 4 for vectorization");
    
    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    // Pre-compute scale and shift tensors in float32 for precision.
    const auto scale = (weight.to(torch::kFloat32) / torch::sqrt(running_var.to(torch::kFloat32) + eps));
    const auto shift = (bias.to(torch::kFloat32) - running_mean.to(torch::kFloat32) * scale);

    auto output = torch::empty_like(x);

    const int block_size = 256; // A generally good block size for this kind of workload
    const dim3 grid_dim(C, N); // Launch a 2D grid: one block per (N, C) plane
    const dim3 block_dim(block_size);

    // Dispatch to the templated kernel based on the input tensor's data type.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_bn_relu_launcher_vec2d", ([&] {
        fused_bn_relu_kernel_vectorized_2d<scalar_t><<<grid_dim, block_dim>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            scale.data_ptr<float>(),
            shift.data_ptr<float>(),
            C, H, W);
    }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_bn_relu_cpp_source = """
torch::Tensor fused_bn_relu_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps);
"""

# Compile the inline CUDA code for the fused operation
fused_bn_relu = load_inline(
    name="fused_bn_relu_vec2d", # Use a unique name to avoid caching conflicts
    cpp_sources=fused_bn_relu_cpp_source,
    cuda_sources=fused_bn_relu_source,
    functions=["fused_bn_relu_forward"],
    verbose=False,
)

class FusedLayer(nn.Module):
    """
    A custom layer that replaces the sequence:
    1. BatchNorm2d
    2. ReLU
    with a single custom CUDA kernel, followed by the Conv2d.
    This structure cleanly reuses standard nn.Modules for parameter management,
    following the successful pattern of Program 1.
    """
    def __init__(self, in_features: int, growth_rate: int):
        super(FusedLayer, self).__init__()
        # We instantiate the original layers to inherit their parameters and initialization
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom fused kernel for BatchNorm + ReLU.
        # This implementation is for inference, using the running mean and variance.
        bn_relu_out = fused_bn_relu.fused_bn_relu_forward(
            x, self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps
        )
        # Apply the subsequent convolution
        return self.conv(bn_relu_out)

class Model(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        The model structure is identical to Program 1, which was the top performer.
        The performance gain comes from the improved CUDA kernel.
        """
        super(Model, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        """ Creates a single layer with our custom fused implementation. """
        return FusedLayer(in_features, growth_rate)
    
    def forward(self, x):
        """
        This forward pass uses the list-based torch.cat method, which empirically
        outperformed pre-allocation strategies in previous attempts for this specific
        model configuration.
        """
        features = [x]
        for layer in self.layers:
            # The input to each layer is the concatenation of all previous features.
            # The torch.cat operation in the previous iteration ensures x is contiguous.
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)
        return x

# Use the same parameters as the original and top-performing solutions.
batch_size = 10
num_layers = 6
num_input_features = 32
growth_rate = 32
height, width = 224, 224

def get_inputs():
    # Use float32 on CUDA, matching the top-performing setup.
    return [torch.randn(batch_size, num_input_features, height, width).cuda()]

def get_init_inputs():
    return [num_layers, num_input_features , growth_rate]
# EVOLVE-BLOCK-END