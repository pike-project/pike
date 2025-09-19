# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- Fused Kernel Source Code ---
# This solution is based on the top-performing prior attempt, which employs two
# highly-effective and distinct fusion strategies for different parts of the network.
#
# 1. 3-Way Fused Kernel (Bias-Add + ReLU + MaxPool):
#    For the convolutional blocks, we fuse three sequential operations into a single
#    kernel. This is achieved by performing the convolution without a bias in PyTorch
#    (which is highly optimized in cuDNN) and then passing the result to a custom kernel
#    that adds the bias, applies ReLU, and performs 2x2 max-pooling in one pass. This
#    minimizes kernel launch overhead and, crucially, reduces memory bandwidth by
#    eliminating two intermediate tensors (conv output and relu output).
#
# 2. Warp-Optimized Fused Kernel (Linear + ReLU):
#    For the fully-connected layers (which are matrix-vector multiplications for batch_size=1),
#    we use a specialized warp-level GEMV kernel. Each warp computes a single output element,
#    using efficient __shfl_down_sync instructions for parallel reduction. This avoids
#    the overhead of shared memory and is significantly faster than a generic GEMM call
#    for this specific problem size.
fused_lenet_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

#define WARP_SIZE 32

// --- Kernel 1: Fused Bias-Add + ReLU + MaxPool2d(2x2, stride=2) ---
// Each thread computes one output element of the max-pooled feature map.
__global__ void fused_bias_relu_maxpool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in
) {
    const int W_out = W_in / 2;
    const int H_out = H_in / 2;
    const int total_outputs = N * C * H_out * W_out;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_outputs) {
        // Map linear output index to 4D coordinates
        const int w_out = idx % W_out;
        const int h_out = (idx / W_out) % H_out;
        const int c = (idx / (W_out * H_out)) % C;
        const int n = idx / (W_out * H_out * C);

        // Find top-left corner of 2x2 window in the input tensor
        const int h_in_start = h_out * 2;
        const int w_in_start = w_out * 2;

        const int base_idx_in = n * C * H_in * W_in + c * H_in * W_in;
        const float channel_bias = bias[c];
        
        // Initialize max_val to 0.0f. This handles the lower bound of ReLU implicitly
        // when using fmaxf.
        float max_val = 0.0f;

        // Process 2x2 window: load, add bias, apply ReLU, and find max in one go.
        // This logic correctly computes max(pool(relu(input + bias))).
        float v00 = input[base_idx_in + h_in_start * W_in + w_in_start];
        max_val = fmaxf(max_val, fmaxf(0.0f, v00 + channel_bias));

        float v01 = input[base_idx_in + h_in_start * W_in + (w_in_start + 1)];
        max_val = fmaxf(max_val, fmaxf(0.0f, v01 + channel_bias));

        float v10 = input[base_idx_in + (h_in_start + 1) * W_in + w_in_start];
        max_val = fmaxf(max_val, fmaxf(0.0f, v10 + channel_bias));
        
        float v11 = input[base_idx_in + (h_in_start + 1) * W_in + (w_in_start + 1)];
        max_val = fmaxf(max_val, fmaxf(0.0f, v11 + channel_bias));

        output[idx] = max_val;
    }
}


// --- Kernel 2: Warp-based Fused Linear + ReLU for batch_size=1 (GEMV) ---
// Each warp is responsible for computing one output element. This design
// ensures coalesced memory reads from the weight matrix and uses fast
// warp-shuffle instructions for the reduction step.
__global__ void warp_gemv_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W,
    const float* __restrict__ b,
    float* __restrict__ out,
    const int N_out, // output features
    const int K_in  // input features
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (warp_id >= N_out) return;

    const int lane_id = threadIdx.x % WARP_SIZE;
    float partial_sum = 0.0f;

    // Each thread in the warp computes a partial sum.
    // The loop strides by WARP_SIZE, leading to coalesced memory access to W.
    for (int k = lane_id; k < K_in; k += WARP_SIZE) {
        partial_sum += x[k] * W[warp_id * K_in + k];
    }
    
    // In-warp reduction using shuffle instructions. This is much faster
    // than using shared memory for reductions within a single warp.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, offset);
    }

    // Lane 0 of each warp now holds the final dot product.
    // It adds the bias, applies ReLU, and writes the final result.
    if (lane_id == 0) {
        float total_sum = partial_sum + b[warp_id];
        out[warp_id] = fmaxf(0.0f, total_sum);
    }
}


// --- C++ Wrapper Functions for PyTorch ---

torch::Tensor fused_bias_relu_maxpool_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(input.size(2) % 2 == 0 && input.size(3) % 2 == 0, "Input H/W must be even");
    TORCH_CHECK(input.size(1) == bias.size(0), "Input channels must match bias size");

    const int N = input.size(0), C = input.size(1), H_in = input.size(2), W_in = input.size(3);
    const int H_out = H_in / 2, W_out = W_in / 2;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    const int total_outputs = N * C * H_out * W_out;
    if (total_outputs == 0) return output;

    const int block_size = 256;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;

    fused_bias_relu_maxpool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H_in, W_in
    );
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor fused_linear_relu_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b) {
    TORCH_CHECK(x.is_cuda() && W.is_cuda() && b.is_cuda(), "All inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "Input must be 2D with batch size 1");
    
    const int K_in = x.size(1);
    const int N_out = W.size(0);

    auto out = torch::empty({1, N_out}, x.options());
    const auto x_vec = x.squeeze(0);

    const int block_size = 256;
    const int warps_per_block = block_size / WARP_SIZE;
    const int num_blocks = (N_out + warps_per_block - 1) / warps_per_block;

    warp_gemv_relu_kernel<<<num_blocks, block_size>>>(
        x_vec.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(),
        out.data_ptr<float>(), N_out, K_in
    );
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}
"""

# C++ source for function declarations
cpp_source = """
torch::Tensor fused_bias_relu_maxpool_cuda(torch::Tensor input, torch::Tensor bias);
torch::Tensor fused_linear_relu_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b);
"""

# JIT compile the CUDA/C++ code into a single module.
# Using a unique name to ensure a fresh compilation.
fused_ops = load_inline(
    name="fused_lenet_ops_final_opt",
    cpp_sources=cpp_source,
    cuda_sources=fused_lenet_kernels_source,
    functions=["fused_bias_relu_maxpool_cuda", "fused_linear_relu_cuda"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture with maximally fused CUDA kernels.
        - Conv blocks use a 3-way fused kernel (BiasAdd + ReLU + MaxPool).
        - FC blocks use a highly-optimized warp-level fused kernel (Linear + ReLU).
        """
        super(Model, self).__init__()
        
        # Define layers as usual; PyTorch manages weights/biases.
        # We will bypass the default forward implementations for most layers.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        # --- First Convolutional Block ---
        # 1. Perform convolution (without bias) using efficient cuDNN via PyTorch.
        conv1_out = F.conv2d(x, self.conv1.weight, None, self.conv1.stride, self.conv1.padding)
        # 2. Apply the custom 3-way fused kernel for bias-add, ReLU, and max-pooling.
        x = fused_ops.fused_bias_relu_maxpool_cuda(conv1_out, self.conv1.bias)
        
        # --- Second Convolutional Block ---
        conv2_out = F.conv2d(x, self.conv2.weight, None, self.conv2.stride, self.conv2.padding)
        x = fused_ops.fused_bias_relu_maxpool_cuda(conv2_out, self.conv2.bias)
        
        # --- Flattening ---
        x = x.view(-1, 16*5*5)
        
        # --- Fused Fully Connected Layers ---
        # Use the custom warp-based kernel for Linear + ReLU on fc1 and fc2.
        x = fused_ops.fused_linear_relu_cuda(x, self.fc1.weight, self.fc1.bias)
        x = fused_ops.fused_linear_relu_cuda(x, self.fc2.weight, self.fc2.bias)
        
        # --- Final Layer (Standard PyTorch) ---
        # No ReLU follows this layer, so it's executed by the highly optimized PyTorch backend.
        x = self.fc3(x)
        
        return x

# Boilerplate for model evaluation
batch_size = 1
num_classes = 10

def get_inputs():
    # Input tensor must be on CUDA for the custom kernels.
    return [torch.randn(batch_size, 1, 32, 32).cuda()]

def get_init_inputs():
    return [num_classes]

# EVOLVE-BLOCK-END