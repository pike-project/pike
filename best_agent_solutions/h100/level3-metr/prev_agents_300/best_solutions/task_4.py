import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels with warp-level optimizations.
fused_lenet_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For fmaxf

// --- Warp-Level Reduction Helper ---
// Uses shuffle-down instructions to sum a float value across all 32 threads in a warp.
// The final sum is present in lane 0 of the warp. This is much faster than
// using shared memory and __syncthreads() for reductions within a single warp.
__inline__ __device__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// --- Optimized Fused Kernel: Conv2d + Bias + ReLU + MaxPool2d ---
// This version assigns one warp to compute each intermediate (pre-pooling) output pixel,
// using warp-level primitives for the reduction part of the convolution.
// Grid: (C_out, H_pool_out, W_pool_out). Each block computes one final output pixel.
__global__ void fused_conv_bias_relu_pool_kernel(
    const float* __restrict__ input,    // Input Tensor (1, C_in, H_in, W_in)
    const float* __restrict__ weight,   // Conv weights (C_out, C_in, KH, KW)
    const float* __restrict__ bias,     // Conv bias (C_out)
    float* __restrict__ output,       // Output Tensor (1, C_out, H_pool_out, W_pool_out)
    int C_in, int H_in, int W_in,
    int C_out, int KH, int KW,
    int H_pool_out, int W_pool_out,
    int pool_size, int pool_stride
) {
    const int c_out = blockIdx.x;
    const int h_out = blockIdx.y;
    const int w_out = blockIdx.z;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // A block has pool_size*pool_size warps. If a warp is outside this, it's inactive.
    if (warp_id >= pool_size * pool_size) return;

    extern __shared__ float s_pool_values[];

    // Map each warp to one of the intermediate pixels in the pooling window
    const int ph = warp_id / pool_size;
    const int pw = warp_id % pool_size;

    const int h_conv = h_out * pool_stride + ph;
    const int w_conv = w_out * pool_stride + pw;

    // --- Convolution for one intermediate output pixel (performed by one warp) ---
    float conv_sum = 0.0f;
    const int total_dot_elements = C_in * KH * KW;

    // Parallelize the dot product over the lanes of the warp
    for (int i = lane_id; i < total_dot_elements; i += 32) {
        const int kw = i % KW;
        const int kh = (i / KW) % KH;
        const int c_in = i / (KW * KH);
        
        const int h_in_coord = h_conv + kh;
        const int w_in_coord = w_conv + kw;
        
        const int input_idx = c_in * (H_in * W_in) + h_in_coord * W_in + w_in_coord;
        const int weight_idx = c_out * (C_in * KH * KW) + i;
        
        conv_sum += input[input_idx] * weight[weight_idx];
    }

    // --- Warp-level reduction for the convolution sum ---
    conv_sum = warp_reduce_sum(conv_sum);

    // --- Bias, ReLU, and Store for Pooling ---
    // Lane 0 of each warp now holds the complete sum for its intermediate pixel.
    if (lane_id == 0) {
        float final_conv_val = conv_sum + bias[c_out];
        s_pool_values[warp_id] = fmaxf(0.0f, final_conv_val);
    }
    __syncthreads(); // Sync warps to ensure all s_pool_values are written

    // --- Max Pooling ---
    // Thread 0 of the entire block performs the final max-pool over the intermediate values.
    if (tid == 0) {
        float max_val = -1.0e20f;
        for (int i = 0; i < pool_size * pool_size; ++i) {
            max_val = fmaxf(max_val, s_pool_values[i]);
        }
        const int output_idx = c_out * (H_pool_out * W_pool_out) + h_out * W_pool_out + w_out;
        output[output_idx] = max_val;
    }
}

// --- Optimized Fused GEMV (Matrix-Vector) + Bias + ReLU Kernel ---
// Uses a two-stage reduction: warp-level shuffles first, then a single
// warp reduces the results from each warp's leader via shared memory.
__global__ void gemv_bias_relu_kernel(
    const float* __restrict__ x, const float* __restrict__ W, const float* __restrict__ b,
    float* __restrict__ out, int in_features, int out_features)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= out_features) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = (blockDim.x + 31) / 32;

    extern __shared__ float sdata[]; // Shared memory for warp-level partial sums

    // Each thread computes its partial sum
    float partial_sum = 0.0f;
    for (int i = tid; i < in_features; i += blockDim.x) {
        partial_sum += x[i] * W[row_idx * in_features + i];
    }

    // Stage 1: Intra-warp reduction
    float warp_sum = warp_reduce_sum(partial_sum);

    // Lane 0 of each warp writes its result to shared memory
    if (lane_id == 0) {
        sdata[warp_id] = warp_sum;
    }
    __syncthreads();

    // Stage 2: The first warp reduces the results from all warps
    if (warp_id == 0) {
        float final_sum = (lane_id < num_warps) ? sdata[lane_id] : 0.0f;
        final_sum = warp_reduce_sum(final_sum);

        // Lane 0 (thread 0) writes the final result
        if (lane_id == 0) {
            float biased_sum = final_sum + b[row_idx];
            out[row_idx] = fmaxf(0.0f, biased_sum); 
        }
    }
}

// --- Optimized Fused GEMV (Matrix-Vector) + Bias Kernel (no ReLU) ---
// Same warp-level optimization as the ReLU version.
__global__ void gemv_bias_kernel(
    const float* __restrict__ x, const float* __restrict__ W, const float* __restrict__ b,
    float* __restrict__ out, int in_features, int out_features)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= out_features) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = (blockDim.x + 31) / 32;

    extern __shared__ float sdata[];

    float partial_sum = 0.0f;
    for (int i = tid; i < in_features; i += blockDim.x) {
        partial_sum += x[i] * W[row_idx * in_features + i];
    }

    float warp_sum = warp_reduce_sum(partial_sum);

    if (lane_id == 0) {
        sdata[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float final_sum = (lane_id < num_warps) ? sdata[lane_id] : 0.0f;
        final_sum = warp_reduce_sum(final_sum);
        
        if (lane_id == 0) {
            out[row_idx] = final_sum + b[row_idx];
        }
    }
}


// --- C++ Wrapper Functions ---

torch::Tensor fused_conv_layer_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int pool_size, int pool_stride
) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(input.dim() == 4 && input.size(0) == 1, "Input must be of shape (1, C, H, W)");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && bias.is_contiguous(), "All tensors must be contiguous");

    const int C_in = input.size(1), H_in = input.size(2), W_in = input.size(3);
    const int C_out = weight.size(0), KH = weight.size(2), KW = weight.size(3);

    const int H_conv_out = H_in - KH + 1;
    const int W_conv_out = W_in - KW + 1;
    const int H_pool_out = (H_conv_out - pool_size) / pool_stride + 1;
    const int W_pool_out = (W_conv_out - pool_size) / pool_stride + 1;

    auto output = torch::empty({1, C_out, H_pool_out, W_pool_out}, input.options());
    
    const dim3 grid(C_out, H_pool_out, W_pool_out);
    // Block dim must be multiple of 32. One warp per intermediate value.
    const int block_size = pool_size * pool_size * 32; 
    const dim3 block(block_size);
    // Shared mem to store intermediate values for pooling.
    const size_t shared_mem_size = pool_size * pool_size * sizeof(float);
    
    fused_conv_bias_relu_pool_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), C_in, H_in, W_in, C_out, KH, KW,
        H_pool_out, W_pool_out, pool_size, pool_stride
    );
    return output;
}

torch::Tensor gemv_bias_relu_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b) {
    TORCH_CHECK(x.is_cuda() && W.is_cuda() && b.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must be of shape (1, N)");
    TORCH_CHECK(x.is_contiguous() && W.is_contiguous() && b.is_contiguous(), "All tensors must be contiguous");
    
    const int in_features = x.size(1);
    const int out_features = W.size(0);
    auto out = torch::empty({1, out_features}, x.options());
    
    const int block_size = 256;
    const int num_warps = (block_size + 31) / 32;
    const size_t shared_mem_size = num_warps * sizeof(float);
    
    gemv_bias_relu_kernel<<<out_features, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(),
        out.data_ptr<float>(), in_features, out_features);
    return out;
}

torch::Tensor gemv_bias_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b) {
    TORCH_CHECK(x.is_cuda() && W.is_cuda() && b.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must be of shape (1, N)");
    TORCH_CHECK(x.is_contiguous() && W.is_contiguous() && b.is_contiguous(), "All tensors must be contiguous");

    const int in_features = x.size(1);
    const int out_features = W.size(0);
    auto out = torch::empty({1, out_features}, x.options());

    const int block_size = 256;
    const int num_warps = (block_size + 31) / 32;
    const size_t shared_mem_size = num_warps * sizeof(float);

    gemv_bias_kernel<<<out_features, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), W.data_ptr<float>(), b.data_ptr<float>(),
        out.data_ptr<float>(), in_features, out_features);
    return out;
}
"""

# C++ source for function signatures
fused_lenet_cpp_source = """
torch::Tensor fused_conv_layer_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int pool_size, int pool_stride);
torch::Tensor gemv_bias_relu_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b);
torch::Tensor gemv_bias_cuda(torch::Tensor x, torch::Tensor W, torch::Tensor b);
"""

# JIT compile the kernels
fused_lenet_kernels = load_inline(
    name="fused_lenet_kernels_v2",
    cpp_sources=fused_lenet_cpp_source,
    cuda_sources=fused_lenet_source,
    functions=["fused_conv_layer_cuda", "gemv_bias_relu_cuda", "gemv_bias_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture with fully fused and warp-optimized custom kernels 
        for both convolutional and fully-connected layers, optimized for batch_size=1.
        """
        super(ModelNew, self).__init__()
        
        # Create temporary PyTorch layers to leverage their default weight initialization.
        conv1_temp = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        conv2_temp = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        fc1_temp = nn.Linear(in_features=16*5*5, out_features=120)
        fc2_temp = nn.Linear(in_features=120, out_features=84)
        fc3_temp = nn.Linear(in_features=84, out_features=num_classes)
        
        # Store weights and biases as nn.Parameter to ensure they are part of the model's state,
        # are tracked by optimizers, and are moved to the correct device.
        self.conv1_weight = nn.Parameter(conv1_temp.weight)
        self.conv1_bias = nn.Parameter(conv1_temp.bias)
        
        self.conv2_weight = nn.Parameter(conv2_temp.weight)
        self.conv2_bias = nn.Parameter(conv2_temp.bias)

        self.fc1_weight = nn.Parameter(fc1_temp.weight)
        self.fc1_bias = nn.Parameter(fc1_temp.bias)
        
        self.fc2_weight = nn.Parameter(fc2_temp.weight)
        self.fc2_bias = nn.Parameter(fc2_temp.bias)

        self.fc3_weight = nn.Parameter(fc3_temp.weight)
        self.fc3_bias = nn.Parameter(fc3_temp.bias)

    def forward(self, x):
        """
        Forward pass using the custom fused CUDA kernels.

        :param x: The input tensor, shape (1, 1, 32, 32)
        :return: The output tensor, shape (1, num_classes)
        """
        # First fused block: Conv2d(5x5) + Bias + ReLU + MaxPool2d(2x2, stride=2)
        x = fused_lenet_kernels.fused_conv_layer_cuda(x, self.conv1_weight, self.conv1_bias, 2, 2)
        
        # Second fused block: Conv2d(5x5) + Bias + ReLU + MaxPool2d(2x2, stride=2)
        x = fused_lenet_kernels.fused_conv_layer_cuda(x, self.conv2_weight, self.conv2_bias, 2, 2)
        
        # Flatten the output for the fully connected layers.
        # .contiguous() is crucial to ensure the tensor has a compatible memory layout for the custom kernel.
        x = x.view(1, -1).contiguous()
        
        # Custom GEMV+Bias+ReLU for fc1
        x = fused_lenet_kernels.gemv_bias_relu_cuda(x, self.fc1_weight, self.fc1_bias)
        
        # Custom GEMV+Bias+ReLU for fc2
        x = fused_lenet_kernels.gemv_bias_relu_cuda(x, self.fc2_weight, self.fc2_bias)
        
        # Custom GEMV+Bias for fc3 (final layer has no ReLU)
        x = fused_lenet_kernels.gemv_bias_cuda(x, self.fc3_weight, self.fc3_bias)
        
        return x

# Test code for the LeNet-5 model
batch_size = 1
num_classes = 10

def get_inputs():
    # ModelNew requires inputs to be on the correct device (CUDA)
    return [torch.randn(batch_size, 1, 32, 32).cuda()]

def get_init_inputs():
    return [num_classes]