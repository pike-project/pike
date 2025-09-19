# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This solution combines the best features from prior top-performing programs to achieve maximum speedup.
# The core strategy is twofold:
# 1. Macro-optimization: Overlap the execution of the two independent convolution branches (`expand1x1` and `expand3x3`)
#    using CUDA streams. This maximizes GPU utilization by running compute-bound tasks in parallel.
# 2. Micro-optimization: Fuse the subsequent memory-bound operations (two ReLUs and a concatenation) into a single,
#    highly-optimized CUDA kernel. This kernel leverages several advanced techniques:
#    - Vectorization (`float4`): Each thread processes four float values at once, quadrupling memory bandwidth.
#    - Read-Only Cache (`__ldg`): Uses the `__ldg()` intrinsic to hint the GPU to load input data through the
#      more efficient texture/read-only cache, which is ideal for this kernel's access pattern.
#    - Tuned Launch Parameters: A large block size of 1024 is used to maximize occupancy and hide memory latency,
#      which is crucial for memory-bound kernels.
#    - Compiler Optimizations: The kernel is compiled with `-O3` and `--use_fast_math` for maximum performance.
fused_expand_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h> // For C10_CUDA_KERNEL_LAUNCH_CHECK

__global__ void fused_expand_relu_cat_kernel(
    const float4* __restrict__ in1,
    const float4* __restrict__ in2,
    float4* __restrict__ out,
    const long long total_elements_vec4,
    const int batch_stride_out_vec4,
    const int batch_stride_in1_vec4,
    const int batch_stride_in2_vec4,
    const int C1_spatial_size_vec4
) {
    // Grid-stride loop over float4 elements for scalability.
    for (long long idx_vec4 = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx_vec4 < total_elements_vec4;
         idx_vec4 += (long long)gridDim.x * blockDim.x) {

        // Decompose the linear float4 index to find its location within the batch.
        const int n_out = idx_vec4 / batch_stride_out_vec4;
        const int remainder_in_batch = idx_vec4 % batch_stride_out_vec4;

        float4 val;
        // Determine which input tensor this element comes from and load using the read-only cache.
        if (remainder_in_batch < C1_spatial_size_vec4) {
            // Element belongs to the first input tensor.
            const int in1_idx_vec4 = n_out * batch_stride_in1_vec4 + remainder_in_batch;
            val = __ldg(&in1[in1_idx_vec4]);
        } else {
            // Element belongs to the second input tensor.
            const int remainder_after_in1 = remainder_in_batch - C1_spatial_size_vec4;
            const int in2_idx_vec4 = n_out * batch_stride_in2_vec4 + remainder_after_in1;
            val = __ldg(&in2[in2_idx_vec4]);
        }

        // Apply ReLU activation to each component of the float4 vector.
        // This is accelerated by the --use_fast_math compilation flag.
        val.x = fmaxf(0.f, val.x);
        val.y = fmaxf(0.f, val.y);
        val.z = fmaxf(0.f, val.z);
        val.w = fmaxf(0.f, val.w);

        // Write the result to the output tensor in a single wide transaction.
        out[idx_vec4] = val;
    }
}

torch::Tensor fused_expand_relu_cat_cuda(torch::Tensor in1, torch::Tensor in2) {
    // Input validation checks
    TORCH_CHECK(in1.is_cuda() && in2.is_cuda(), "Input tensors must be on CUDA");
    TORCH_CHECK(in1.is_contiguous() && in2.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(in1.dtype() == torch::kFloat32 && in2.dtype() == torch::kFloat32, "Input tensors must be float32");
    TORCH_CHECK(in1.dim() == 4 && in2.dim() == 4, "Input tensors must be 4D");
    TORCH_CHECK(in1.size(0) == in2.size(0) && in1.size(2) == in2.size(2) && in1.size(3) == in2.size(3), "N, H, W dimensions must match");

    // Get tensor dimensions
    const auto N = in1.size(0);
    const auto C1 = in1.size(1);
    const auto H = in1.size(2);
    const auto W = in1.size(3);
    const auto C2 = in2.size(1);

    // Check for vectorization compatibility.
    TORCH_CHECK(W % 4 == 0, "Width must be a multiple of 4 for vectorized kernel");

    // Prepare the output tensor
    auto out = torch::empty({N, C1 + C2, H, W}, in1.options());
    const long long total_elements = out.numel();

    if (total_elements == 0) {
        return out;
    }

    const long long total_elements_vec4 = total_elements / 4;

    // Pre-calculate strides in float units
    const int spatial_size = H * W;
    const int batch_stride_in1 = C1 * spatial_size;
    const int batch_stride_in2 = C2 * spatial_size;
    const int batch_stride_out = (C1 + C2) * spatial_size;
    const int C1_spatial_size = C1 * spatial_size;

    // Convert strides to float4 units for the kernel
    const int batch_stride_out_vec4 = batch_stride_out / 4;
    const int batch_stride_in1_vec4 = batch_stride_in1 / 4;
    const int batch_stride_in2_vec4 = batch_stride_in2 / 4;
    const int C1_spatial_size_vec4 = C1_spatial_size / 4;

    // Kernel launch configuration: Use a large block size to maximize occupancy.
    const int block_size = 1024;
    const int grid_size = std::min((int)((total_elements_vec4 + block_size - 1) / block_size), 65535);

    fused_expand_relu_cat_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const float4*>(in1.data_ptr<float>()),
        reinterpret_cast<const float4*>(in2.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        total_elements_vec4,
        batch_stride_out_vec4,
        batch_stride_in1_vec4,
        batch_stride_in2_vec4,
        C1_spatial_size_vec4
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

fused_expand_cpp_source = "torch::Tensor fused_expand_relu_cat_cuda(torch::Tensor in1, torch::Tensor in2);"

# JIT compile the custom CUDA kernel with a unique name and aggressive optimization flags.
fused_expand_op = load_inline(
    name="fused_expand_op_vmax",
    cpp_sources=fused_expand_cpp_source,
    cuda_sources=fused_expand_source,
    functions=["fused_expand_relu_cat_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class Model(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Model, self).__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        # Activations are removed from the expand layers; they will be handled by our fused kernel.
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)

        # Create CUDA streams to run the two independent expand convolutions concurrently.
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
    
    def forward(self, x):
        # Squeeze part is unchanged
        x = self.squeeze_activation(self.squeeze(x))
        
        # Launch the two expand convolutions on separate streams to enable parallel execution.
        with torch.cuda.stream(self.stream1):
            expand1x1_out = self.expand1x1(x)
        
        with torch.cuda.stream(self.stream2):
            expand3x3_out = self.expand3x3(x)
            
        # Synchronize the main stream to wait for both convolutions to complete.
        torch.cuda.current_stream().wait_stream(self.stream1)
        torch.cuda.current_stream().wait_stream(self.stream2)
        
        # Call our single fused kernel to perform both ReLUs and the concatenation.
        return fused_expand_op.fused_expand_relu_cat_cuda(expand1x1_out, expand3x3_out)

# Test code
batch_size = 10
num_input_features = 3
num_output_features = 64
height, width = 224, 224
squeeze_channels = 6
expand1x1_channels = 64
expand3x3_channels = 64

def get_inputs():
    # Ensure input tensor is on the correct device (CUDA) for the model
    return [torch.randn(batch_size, num_input_features, height, width).cuda()]

def get_init_inputs():
    return [num_input_features, squeeze_channels, expand1x1_channels, expand3x3_channels]
# EVOLVE-BLOCK-END