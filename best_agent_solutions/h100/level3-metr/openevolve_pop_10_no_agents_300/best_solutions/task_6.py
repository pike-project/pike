# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for concatenating 4 tensors along dim=1
# This version is vectorized to use float4 for higher memory bandwidth.
fused_cat4_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void fused_cat4_kernel(
    const float* t1, const float* t2, const float* t3, const float* t4,
    float* out,
    int B, int C1, int C2, int C3, int C4, int H, int W)
{
    const int C_out = C1 + C2 + C3 + C4;
    const int W_vec = W / 4;
    const long long N_vec = (long long)B * C_out * H * W_vec;
    
    const int plane_size = H * W;
    const int C1_offset = C1;
    const int C2_offset = C1 + C2;
    const int C3_offset = C1 + C2 + C3;

    for (long long i_vec = (long long)blockIdx.x * blockDim.x + threadIdx.x; i_vec < N_vec; i_vec += (long long)blockDim.x * gridDim.x) {
        // Decompose linear vectorized output index 'i_vec' into (b, c_out, h, w_vec)
        long long temp = i_vec;
        const int w_vec = temp % W_vec;
        temp /= W_vec;
        const int h = temp % H;
        temp /= H;
        const int c_out = temp % C_out;
        temp /= C_out;
        const int b = temp;

        const float* src_ptr;
        int c_in;
        int C_in;

        // Determine which input tensor to read from based on the output channel index 'c_out'
        if (c_out < C1_offset) {
            src_ptr = t1; c_in = c_out; C_in = C1;
        } else if (c_out < C2_offset) {
            src_ptr = t2; c_in = c_out - C1_offset; C_in = C2;
        } else if (c_out < C3_offset) {
            src_ptr = t3; c_in = c_out - C2_offset; C_in = C3;
        } else {
            src_ptr = t4; c_in = c_out - C3_offset; C_in = C4;
        }

        // Calculate source index for the start of the float4 vector
        const long long src_idx_scalar = (long long)b * C_in * plane_size + (long long)c_in * plane_size + (long long)h * W + w_vec * 4;
        
        // Create float4 pointers for vectorized access
        float4* out_ptr_vec = reinterpret_cast<float4*>(out);
        const float4* src_ptr_vec = reinterpret_cast<const float4*>(src_ptr);

        // Calculate vector indices
        const long long out_idx_vec = i_vec;
        const long long src_idx_vec = src_idx_scalar / 4;

        // Perform vectorized copy
        out_ptr_vec[out_idx_vec] = src_ptr_vec[src_idx_vec];
    }
}

torch::Tensor fused_cat4_cuda(torch::Tensor t1, torch::Tensor t2, torch::Tensor t3, torch::Tensor t4) {
    // Input validation checks
    TORCH_CHECK(t1.dim() == 4, "Input tensor 1 must be 4D");
    TORCH_CHECK(t2.dim() == 4, "Input tensor 2 must be 4D");
    TORCH_CHECK(t3.dim() == 4, "Input tensor 3 must be 4D");
    TORCH_CHECK(t4.dim() == 4, "Input tensor 4 must be 4D");
    TORCH_CHECK(t1.scalar_type() == torch::kFloat32, "Input tensors must be float32");
    TORCH_CHECK(t1.device().is_cuda(), "Input tensors must be on a CUDA device");

    const auto B = t1.size(0);
    const auto H = t1.size(2);
    const auto W = t1.size(3);

    TORCH_CHECK(t2.size(0) == B && t2.size(2) == H && t2.size(3) == W, "Tensor 2 shape mismatch");
    TORCH_CHECK(t3.size(0) == B && t3.size(2) == H && t3.size(3) == W, "Tensor 3 shape mismatch");
    TORCH_CHECK(t4.size(0) == B && t4.size(2) == H && t4.size(3) == W, "Tensor 4 shape mismatch");
    TORCH_CHECK(W > 0 && W % 4 == 0, "Tensor width must be positive and divisible by 4 for vectorized kernel");

    const auto C1 = t1.size(1);
    const auto C2 = t2.size(1);
    const auto C3 = t3.size(1);
    const auto C4 = t4.size(1);
    const auto C_out = C1 + C2 + C3 + C4;

    // Allocate output tensor
    auto out = torch::empty({B, C_out, H, W}, t1.options());
    const int64_t total_elements = out.numel();
    if (total_elements == 0) {
        return out;
    }

    // Kernel launch configuration for vectorized kernel
    const int64_t total_elements_vec = total_elements / 4;
    const int block_size = 256;
    const int num_blocks = std::min((int)((total_elements_vec + block_size - 1) / block_size), 65535);

    fused_cat4_kernel<<<num_blocks, block_size>>>(
        t1.data_ptr<float>(), t2.data_ptr<float>(), t3.data_ptr<float>(), t4.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C1, C2, C3, C4, H, W
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_cat4_cpp_source = (
    "torch::Tensor fused_cat4_cuda(torch::Tensor t1, torch::Tensor t2, torch::Tensor t3, torch::Tensor t4);"
)

# Compile the inline CUDA code for the fused concatenation
fused_cat4 = load_inline(
    name="fused_cat4",
    cpp_sources=fused_cat4_cpp_source,
    cuda_sources=fused_cat4_source,
    functions=["fused_cat4_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


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
        
        # Create CUDA streams for parallel execution of branches
        self.s1 = torch.cuda.Stream()
        self.s2 = torch.cuda.Stream()
        self.s3 = torch.cuda.Stream()
        self.s4 = torch.cuda.Stream()
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        # Enqueue computations for each branch on a separate CUDA stream
        with torch.cuda.stream(self.s1):
            branch1x1 = self.branch1x1(x)
        
        with torch.cuda.stream(self.s2):
            branch3x3 = self.branch3x3(x)

        with torch.cuda.stream(self.s3):
            branch5x5 = self.branch5x5(x)

        with torch.cuda.stream(self.s4):
            branch_pool = self.branch_pool(x)
        
        # Synchronize the default stream to wait for all branch streams to complete
        torch.cuda.current_stream().wait_stream(self.s1)
        torch.cuda.current_stream().wait_stream(self.s2)
        torch.cuda.current_stream().wait_stream(self.s3)
        torch.cuda.current_stream().wait_stream(self.s4)
        
        # Replace torch.cat with the custom fused and vectorized CUDA kernel
        return fused_cat4.fused_cat4_cuda(branch1x1, branch3x3, branch5x5, branch_pool)

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
    # Ensure inputs are on CUDA and contiguous for safety with custom kernels
    return [torch.randn(batch_size, in_channels, height, width, device='cuda').contiguous()]

def get_init_inputs():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]
# EVOLVE-BLOCK-END