# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for both fusions in a single source string.
# This approach is cleaner and reduces compilation overhead.
kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// --- KERNEL 1: Fused BatchNorm2d + ReLU (Optimized Version) ---
// This version assigns each thread block to a specific (N, C) pair.
// This avoids redundant parameter calculations and complex indexing inside the loop.
__global__ void fused_bn_relu_kernel(const float* __restrict__ input, float* __restrict__ output,
                                     const float* __restrict__ gamma, const float* __restrict__ beta,
                                     const float* __restrict__ running_mean, const float* __restrict__ running_var,
                                     float eps, int C, int HW) {
    // Each block is responsible for one channel 'c' and one batch item 'n'.
    const int c = blockIdx.x;
    const int n = blockIdx.y;

    // Calculate scale and bias once per block. These are stored in registers
    // and reused by all threads in the block, reducing redundant calculations.
    const float scale = gamma[c] * rsqrtf(running_var[c] + eps);
    const float bias = beta[c] - running_mean[c] * scale;

    // Calculate the base pointer for this (n, c) pair.
    const int offset = (n * C + c) * HW;
    const float* input_ptr = input + offset;
    float* output_ptr = output + offset;

    // Threads in the block stride over the spatial dimension (HW), ensuring
    // perfectly coalesced memory accesses for maximum bandwidth.
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        output_ptr[i] = fmaxf(0.0f, input_ptr[i] * scale + bias);
    }
}

torch::Tensor fused_bn_relu_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta,
                                 torch::Tensor running_mean, torch::Tensor running_var, double eps) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor (NCHW)");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    const auto sizes = input.sizes();
    const int N = sizes[0];
    const int C = sizes[1];
    const int H = sizes[2];
    const int W = sizes[3];
    const int HW = H * W;

    auto output = torch::empty_like(input);

    // Launch one block per (Channel, Batch Item) pair. This mapping is more
    // efficient for this specific operation.
    const dim3 grid_size(C, N);
    const int block_size = 256; // A common, effective block size.

    fused_bn_relu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        (float)eps, C, HW);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(err)); }

    return output;
}

// --- KERNEL 2: Fused Global Average Pooling + Flatten ---

// A device function for performing an efficient reduction within a warp using shuffle instructions.
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// A device function for performing a block-wide reduction by first reducing within
// warps, and then reducing the results from each warp. This is faster than a pure
// shared memory approach on modern GPUs.
__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32]; // Max 32 warps per block (1024 threads)
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce_sum(val); // Each warp reduces its own sum

    if (lane == 0) shared[wid] = val; // Warp leaders write to shared memory
    __syncthreads();

    // The first warp reduces the partial sums from shared memory
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}


__global__ void fused_global_avg_pool_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int N, int C, int H, int W) {
    const int c = blockIdx.x;
    const int n = blockIdx.y;
    
    if (c >= C || n >= N) return;

    const float* input_ptr = input + (n * C + c) * H * W;
    const int spatial_size = H * W;

    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        thread_sum += input_ptr[i];
    }

    // Reduce sums across the entire block
    thread_sum = block_reduce_sum(thread_sum);

    // Thread 0 writes the final averaged result
    if (threadIdx.x == 0) {
        output[n * C + c] = thread_sum / spatial_size;
    }
}

torch::Tensor fused_global_avg_pool_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    const auto sizes = input.sizes();
    const int N = sizes[0];
    const int C = sizes[1];
    const int H = sizes[2];
    const int W = sizes[3];

    // The output is flattened directly to (N, C).
    auto output = torch::empty({N, C}, input.options());

    // Larger block size is often better for reductions.
    const int block_size = 512;
    const dim3 grid_size(C, N);
    
    fused_global_avg_pool_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H, W);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(err)); }

    return output;
}
"""

# C++ source for function declarations
kernels_cpp_source = """
torch::Tensor fused_bn_relu_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta,
                                 torch::Tensor running_mean, torch::Tensor running_var, double eps);
torch::Tensor fused_global_avg_pool_cuda(torch::Tensor input);
"""

# JIT compile the kernels
fused_ops = load_inline(
    name="efficientnet_fused_ops_combined",
    cpp_sources=kernels_cpp_source,
    cuda_sources=kernels_source,
    functions=["fused_bn_relu_cuda", "fused_global_avg_pool_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class FusedBatchNormReLU(nn.Module):
    """Wraps a BatchNorm2d layer to replace its forward pass with our custom fused kernel."""
    def __init__(self, bn_layer):
        super().__init__()
        # Copy parameters and buffers from the original BatchNorm layer
        self.register_parameter('weight', bn_layer.weight)
        self.register_parameter('bias', bn_layer.bias)
        self.register_buffer('running_mean', bn_layer.running_mean)
        self.register_buffer('running_var', bn_layer.running_var)
        self.eps = bn_layer.eps

    def forward(self, x):
        # The model must be in eval mode for the custom kernel to be correct
        return fused_ops.fused_bn_relu_cuda(
            x.contiguous(), self.weight, self.bias,
            self.running_mean, self.running_var, self.eps
        )

class FusedGlobalAvgPool(nn.Module):
    """Replaces nn.AdaptiveAvgPool2d and torch.flatten with a single fused kernel."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return fused_ops.fused_global_avg_pool_cuda(x.contiguous())


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_relu1 = FusedBatchNormReLU(nn.BatchNorm2d(32))
        
        self.mbconv1 = self._make_mbconv_block(32, 96, 1, 3)
        self.mbconv2 = self._make_mbconv_block(96, 144, 2, 6)
        self.mbconv3 = self._make_mbconv_block(144, 192, 2, 6)
        self.mbconv4 = self._make_mbconv_block(192, 288, 2, 6)
        self.mbconv5 = self._make_mbconv_block(288, 384, 1, 6)
        
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_relu_final = FusedBatchNormReLU(nn.BatchNorm2d(1408))
        
        # Replace the original avgpool and subsequent flatten with our new fused module
        self.fused_pool = FusedGlobalAvgPool()
        self.fc = nn.Linear(1408, num_classes)
    
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        """Helper function to create an MBConv block using our FusedBatchNormReLU module."""
        layers = []
        expanded_channels = in_channels * expand_ratio
        
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(FusedBatchNormReLU(nn.BatchNorm2d(expanded_channels)))
        
        layers.append(nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False))
        layers.append(FusedBatchNormReLU(nn.BatchNorm2d(expanded_channels)))
        
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(expanded_channels // 4, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.Sigmoid())
        
        layers.append(nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass utilizing all fused and optimized operations."""
        x = self.bn_relu1(self.conv1(x))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.bn_relu_final(self.conv_final(x))
        # A single call to our fused kernel replaces two separate operations
        x = self.fused_pool(x)
        x = self.fc(x)
        return x

batch_size = 2
num_classes = 1000

def get_inputs():
    # Ensure input tensor is on the correct device for the CUDA kernels
    return [torch.randn(batch_size, 3, 224, 224).cuda()]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END