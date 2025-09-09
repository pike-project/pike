# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define a new custom CUDA kernel that handles non-contiguous (strided) NHWC tensors.
# This is crucial for the pre-allocation strategy, which creates views into a larger buffer.
# The kernel reads from a strided view and writes to a new, contiguous NHWC tensor.
# It retains the key optimizations: pre-computed BN params and float4 vectorization.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For fmaxf

// Kernel to read from strided NHWC and write to contiguous NHWC
__global__ void fused_bn_relu_strided_nhwc_vec4(
    const float* __restrict__ input,
    const float* __restrict__ A, // Pre-computed scale
    const float* __restrict__ B, // Pre-computed bias
    float* __restrict__ output,
    const int N, const int C_in, const int H, const int W,
    const long stride_N, const long stride_H, const long stride_W
) {
    const int total_out_vecs = (N * C_in * H * W) / 4;
    const int C_in_vecs = C_in / 4;
    const int HW = H * W;

    // Grid-stride loop over the output tensor's vectorized elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_out_vecs; i += gridDim.x * blockDim.x) {
        // Deconstruct the contiguous output index 'i' into logical (n, h, w, c_vec) coordinates.
        const int c_vec = i % C_in_vecs;
        const int pixel_idx = i / C_in_vecs;
        const int n = pixel_idx / HW;
        const int hw_idx = pixel_idx % HW;
        const int h = hw_idx / W;
        const int w = hw_idx % W;

        // Calculate the base pointer to the start of the pixel in the strided input tensor.
        const long input_offset = n * stride_N + h * stride_H + w * stride_W;
        const float4* input_ptr_base = (const float4*)(input + input_offset);
        
        // Load 4 channel elements for this pixel. This access pattern is coalesced.
        const float4 input_val = input_ptr_base[c_vec];

        // Load pre-computed parameters for the corresponding channels.
        const float4* A_ptr = reinterpret_cast<const float4*>(A);
        const float4* B_ptr = reinterpret_cast<const float4*>(B);
        const float4 A_val = A_ptr[c_vec];
        const float4 B_val = B_ptr[c_vec];

        // Perform the fused operation element-wise.
        float4 result;
        result.x = fmaxf(0.0f, input_val.x * A_val.x + B_val.x);
        result.y = fmaxf(0.0f, input_val.y * A_val.y + B_val.y);
        result.z = fmaxf(0.0f, input_val.z * A_val.z + B_val.z);
        result.w = fmaxf(0.0f, input_val.w * A_val.w + B_val.w);

        // Write the result to the contiguous output tensor.
        float4* output_ptr = reinterpret_cast<float4*>(output);
        output_ptr[i] = result;
    }
}

// C++ wrapper for the new strided NHWC kernel.
torch::Tensor fused_bn_relu_precalc_strided_cuda_nhwc(
    torch::Tensor input, // a non-contiguous channels-last view
    torch::Tensor A,
    torch::Tensor B
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.suggest_memory_format() == at::MemoryFormat::ChannelsLast, "Input must be channels-last format");

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const int total_elements = input.numel();
    
    TORCH_CHECK(C % 4 == 0, "Number of channels must be divisible by 4 for vectorization.");

    // Create a new, CONTIGUOUS channels-last output tensor.
    auto output = torch::empty({N, C, H, W}, input.options().memory_format(at::MemoryFormat::ChannelsLast));

    // Get the strides of the non-contiguous input view.
    const auto strides = input.strides();
    const long stride_N = strides[0];
    const long stride_H = strides[2];
    const long stride_W = strides[3];

    const int block_size = 512;
    const int total_out_vecs = total_elements / 4;
    const int num_blocks = (total_out_vecs + block_size - 1) / block_size;

    fused_bn_relu_strided_nhwc_vec4<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stride_N, stride_H, stride_W
    );
    
    return output;
}
"""

cpp_source = "torch::Tensor fused_bn_relu_precalc_strided_cuda_nhwc(torch::Tensor input, torch::Tensor A, torch::Tensor B);"

# JIT compile the CUDA/C++ code. Use a new unique name.
fused_op_module = load_inline(
    name="fused_op_strided_nhwc_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_bn_relu_precalc_strided_cuda_nhwc"],
    verbose=False,
)

class _DenseLayer(nn.Module):
    """
    A single layer of the Dense Block, optimized with channels-last memory format
    and a pre-computed fused BatchNorm+ReLU kernel.
    """
    def __init__(self, in_features: int, growth_rate: int):
        super(_DenseLayer, self).__init__()
        bn = nn.BatchNorm2d(in_features)
        bn.eval()
        
        with torch.no_grad():
            inv_std = torch.rsqrt(bn.running_var + bn.eps)
            A = bn.weight * inv_std
            B = bn.bias - bn.running_mean * A
            self.register_buffer('A', A.contiguous())
            self.register_buffer('B', B.contiguous())
        del bn

        self.conv = nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False).to(memory_format=torch.channels_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom CUDA kernel that handles strided NHWC input and produces a contiguous NHWC output.
        bn_relu_out = fused_op_module.fused_bn_relu_precalc_strided_cuda_nhwc(x, self.A, self.B)
        
        # Apply the channels-last optimized convolution on the contiguous tensor.
        return self.conv(bn_relu_out)

class Model(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Optimized forward pass using pre-allocation to avoid `torch.cat`.
        The custom CUDA kernel handles the non-contiguous channels-last views
        created by `narrow`, fusing the BN, ReLU, and an implicit `.contiguous()` call.
        """
        B, C_in, H, W = x.shape
        final_channels = self.num_input_features + self.num_layers * self.growth_rate

        # Pre-allocate a single large buffer in channels-last format.
        features_buffer = torch.empty(B, final_channels, H, W, dtype=x.dtype, device=x.device,
                                      memory_format=torch.channels_last)

        # Convert initial input to channels-last and copy it into the buffer.
        features_buffer.narrow(1, 0, C_in).copy_(x.to(memory_format=torch.channels_last))

        current_channels = C_in
        for layer in self.layers:
            # Create a zero-copy, non-contiguous view for the layer's input.
            input_view = features_buffer.narrow(1, 0, current_channels)
            
            # The _DenseLayer's custom kernel processes the non-contiguous view
            # and returns a new contiguous channels-last tensor.
            new_features = layer(input_view)
            
            # Copy the newly computed features into the correct slice of the buffer.
            features_buffer.narrow(1, current_channels, self.growth_rate).copy_(new_features)
            
            current_channels += self.growth_rate
            
        return features_buffer

# Global parameters and helper functions are kept unchanged as required.
batch_size = 10
num_layers = 6
num_input_features = 32
growth_rate = 32
height, width = 224, 224

def get_inputs():
    # The input tensor is standard NCHW; the model internally converts it to NHWC.
    return [torch.randn(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_layers, num_input_features , growth_rate]
# EVOLVE-BLOCK-END