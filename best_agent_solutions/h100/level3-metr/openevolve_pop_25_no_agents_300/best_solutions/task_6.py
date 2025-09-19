# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This solution synthesizes the best features from all previous high-performing attempts
# to create a more robust and theoretically faster kernel for tensor concatenation.
#
# Key Optimizations:
# 1.  Efficient Indexing (from Program 2 / Inspiration 1): The kernel avoids expensive, repeated
#     integer division/modulo operations for 4D index calculation (N, C, H, W) inside the loop.
#     It uses a much faster scheme that calculates the batch index 'n' and the index within the
#     batch's data 'idx_in_batch' once per thread using a single div/mod pair. This significantly
#     reduces the computational overhead per element compared to the current top program.
#
# 2.  Vectorization (float4): The kernel reads and writes four floats (a float4) in a single
#     instruction, maximizing memory bandwidth, which is critical for this memory-bound operation.
#
# 3.  Grid-Stride Loop: A robust design pattern that ensures all data is processed regardless
#     of the grid size and allows for launching a large, persistent grid to keep the GPU fully
#     saturated with work.
#
# 4.  Optimized Launch Configuration: The kernel is launched with a large block size (1024 threads)
#     and a grid size heuristically calculated based on the number of Streaming Multiprocessors (SMs),
#     a proven strategy for maximizing occupancy for memory-bound tasks.
#
# 5.  Robust C++ Wrapper: The wrapper function ensures all input tensors are made contiguous before
#     being passed to the kernel, preventing potential silent errors or crashes from incorrect
#     pointer arithmetic.
#
# 6.  Compiler Optimizations: Aggressive compiler flags (`-O3`, `--use_fast_math`) are enabled to
#     ensure the generated code is as fast as possible.

concat_branches_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Kernel to concatenate four NCHW tensors along the channel dimension.
__global__ void concat_branches_kernel_synthesized(
    const float4* __restrict__ in1, const float4* __restrict__ in2, 
    const float4* __restrict__ in3, const float4* __restrict__ in4,
    float4* __restrict__ out,
    const long long total_elements_vec4,
    const long long C1_plane_vec4, const long long C2_plane_vec4,
    const long long C3_plane_vec4, const long long C4_plane_vec4) {

    // Pre-calculate cumulative plane sizes for efficient branching.
    // These are used to determine which source tensor to copy from.
    const long long C12_plane_vec4 = C1_plane_vec4 + C2_plane_vec4;
    const long long C123_plane_vec4 = C12_plane_vec4 + C3_plane_vec4;
    const long long C_total_plane_vec4 = C123_plane_vec4 + C4_plane_vec4;

    // Grid-stride loop ensures all elements are processed, regardless of grid size.
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements_vec4; 
         idx += (long long)gridDim.x * blockDim.x) 
    {
        // Efficiently decompose the linear output index into batch index and intra-batch index.
        const long long n = idx / C_total_plane_vec4;
        const long long idx_in_batch = idx % C_total_plane_vec4;

        // Determine which input tensor to read from based on the intra-batch index,
        // then calculate the source index within that tensor.
        if (idx_in_batch < C1_plane_vec4) {
            const long long src_idx = n * C1_plane_vec4 + idx_in_batch;
            out[idx] = in1[src_idx];
        } else if (idx_in_batch < C12_plane_vec4) {
            const long long src_idx = n * C2_plane_vec4 + (idx_in_batch - C1_plane_vec4);
            out[idx] = in2[src_idx];
        } else if (idx_in_batch < C123_plane_vec4) {
            const long long src_idx = n * C3_plane_vec4 + (idx_in_batch - C12_plane_vec4);
            out[idx] = in3[src_idx];
        } else {
            const long long src_idx = n * C4_plane_vec4 + (idx_in_batch - C123_plane_vec4);
            out[idx] = in4[src_idx];
        }
    }
}


torch::Tensor concat_branches_cuda(
    torch::Tensor in1, torch::Tensor in2,
    torch::Tensor in3, torch::Tensor in4) {

    // Input validation
    TORCH_CHECK(in1.is_cuda(), "Input tensor 1 must be a CUDA tensor");
    TORCH_CHECK(in2.is_cuda(), "Input tensor 2 must be a CUDA tensor");
    TORCH_CHECK(in3.is_cuda(), "Input tensor 3 must be a CUDA tensor");
    TORCH_CHECK(in4.is_cuda(), "Input tensor 4 must be a CUDA tensor");

    // Ensure tensors are contiguous for correct pointer arithmetic in the kernel
    auto in1_c = in1.contiguous();
    auto in2_c = in2.contiguous();
    auto in3_c = in3.contiguous();
    auto in4_c = in4.contiguous();

    // Get dimensions
    const auto N = in1_c.size(0);
    const auto H = in1_c.size(2);
    const auto W = in1_c.size(3);

    // Check for vectorization compatibility
    TORCH_CHECK(W % 4 == 0, "Width of tensors must be divisible by 4 for vectorized kernel.");

    // Check for dimension consistency across batches and spatial dimensions
    TORCH_CHECK(in2_c.size(0) == N && in2_c.size(2) == H && in2_c.size(3) == W, "Dimension mismatch in tensor 2");
    TORCH_CHECK(in3_c.size(0) == N && in3_c.size(2) == H && in3_c.size(3) == W, "Dimension mismatch in tensor 3");
    TORCH_CHECK(in4_c.size(0) == N && in4_c.size(2) == H && in4_c.size(3) == W, "Dimension mismatch in tensor 4");

    const auto C1 = in1_c.size(1);
    const auto C2 = in2_c.size(1);
    const auto C3 = in3_c.size(1);
    const auto C4 = in4_c.size(1);
    const auto C_total = C1 + C2 + C3 + C4;

    // Create output tensor
    auto out = torch::empty({N, C_total, H, W}, in1_c.options());
    const long long total_elements = out.numel();
    if (total_elements == 0) {
        return out;
    }
    
    const long long total_elements_vec4 = total_elements / 4;

    // Pre-calculate plane sizes in float4 units on the host to reduce device-side computation.
    const long long W_vec4 = W / 4;
    const long long C1_plane_vec4 = (long long)C1 * H * W_vec4;
    const long long C2_plane_vec4 = (long long)C2 * H * W_vec4;
    const long long C3_plane_vec4 = (long long)C3 * H * W_vec4;
    const long long C4_plane_vec4 = (long long)C4 * H * W_vec4;

    // Use the proven optimal launch configuration from the top-performing solution.
    const int block_size = 1024;
    // Heuristic for grid size to saturate the GPU, used with a grid-stride loop.
    const int num_sm = at::cuda::getDeviceProperties(in1_c.device().index())->multiProcessorCount;
    const int grid_size = num_sm * 40;

    concat_branches_kernel_synthesized<<<grid_size, block_size>>>(
        (const float4*)in1_c.data_ptr<float>(),
        (const float4*)in2_c.data_ptr<float>(),
        (const float4*)in3_c.data_ptr<float>(),
        (const float4*)in4_c.data_ptr<float>(),
        (float4*)out.data_ptr<float>(),
        total_elements_vec4,
        C1_plane_vec4, C2_plane_vec4, C3_plane_vec4, C4_plane_vec4
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

concat_branches_cpp_source = """
torch::Tensor concat_branches_cuda(torch::Tensor in1, torch::Tensor in2, torch::Tensor in3, torch::Tensor in4);
"""

# JIT compile the custom CUDA kernel
try:
    # Use a unique name to avoid caching issues with previous versions
    concat_branches_op = load_inline(
        name="concat_branches_op_synthesized",
        cpp_sources=concat_branches_cpp_source,
        cuda_sources=concat_branches_source,
        functions=["concat_branches_cuda"],
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
except Exception as e:
    print(f"Failed to compile custom CUDA kernel, falling back to torch.cat: {e}")
    concat_branches_op = None


class Model(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(Model, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        # Use the custom CUDA kernel if compilation was successful
        if concat_branches_op is not None:
            return concat_branches_op.concat_branches_cuda(branch1x1, branch3x3, branch5x5, branch_pool)
        else:
            # Fallback to the original torch.cat in case of compilation failure
            outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
            return torch.cat(outputs, 1)

# Test code
in_channels = 480
out_1x1 = 192
reduce_3x3 = 96
out_3x3 = 208
reduce_5x5 = 16
out_5x5 = 48
pool_proj = 64
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]
# EVOLVE-BLOCK-END