# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This solution synthesizes the best features from all previous top-performing models
# and introduces new optimizations to push performance further.
#
# High-Level Strategy:
# 1.  (From Program 1) Adopt the aggressive "mega-kernel" that fuses ReLU, MaxPool, Flatten,
#     Linear, and ReLU for the conv2 -> fc1 transition. This is the biggest performance win.
# 2.  (New) Enhance all custom kernels with memory vectorization (float2/float4) to
#     maximize memory bandwidth, a key bottleneck.
# 3.  (From Program 1/3) Use highly efficient warp-shuffle-based reductions for all
#     linear layer dot products.
# 4.  (From Program 2) Use an optimized 3D grid for the first pooling kernel to
#     eliminate complex index calculations.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>

// --- Warp-level Reduction Helper ---
// Reduces a value across all 32 threads in a warp using efficient shuffle instructions.
// The final sum is available in lane 0.
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// --- Fused Linear + ReLU Kernel (for fc2) ---
// Combines float4 vectorization with warp-shuffle reduction for maximum efficiency.
__global__ void linear_relu_vectorized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int out_features, int in_features) {

    int j = blockIdx.x; // Each block computes one output feature j.

    if (j < out_features) {
        float thread_sum = 0.0f;
        const float4* input_f4 = reinterpret_cast<const float4*>(input);
        const float4* weight_f4 = reinterpret_cast<const float4*>(weight + j * in_features);
        const int in_features_f4 = in_features / 4;

        // Each thread computes a partial sum using vectorized loads.
        for (int i = threadIdx.x; i < in_features_f4; i += blockDim.x) {
            float4 in_val = input_f4[i];
            float4 wt_val = weight_f4[i];
            thread_sum += in_val.x * wt_val.x + in_val.y * wt_val.y + in_val.z * wt_val.z + in_val.w * wt_val.w;
        }

        // Two-stage reduction: fast warp-level reduction, then shared memory for inter-warp.
        float warp_sum = warp_reduce_sum(thread_sum);

        extern __shared__ float sdata[];
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;

        if (lane_id == 0) sdata[warp_id] = warp_sum;
        __syncthreads();

        thread_sum = (threadIdx.x < (blockDim.x / 32)) ? sdata[lane_id] : 0.0f;
        if (warp_id == 0) {
            float block_sum = warp_reduce_sum(thread_sum);
            if (lane_id == 0) {
                float final_val = block_sum + bias[j];
                out[j] = fmaxf(0.0f, final_val);
            }
        }
    }
}


// --- Fused ReLU + MaxPool2D Kernel (for conv1) ---
// Combines 3D grid mapping, float2 vectorized loads, and post-max ReLU.
__global__ void relu_maxpool2d_kernel_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_h, int in_w,
    int out_h, int out_w) {

    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_out = blockIdx.z;

    if (w_out < out_w && h_out < out_h) {
        const int h_in_start = h_out * 2;
        const int w_in_start = w_out * 2;
        
        const float* p_in_row0 = input + c_out * (in_h * in_w) + h_in_start * in_w + w_in_start;
        const float* p_in_row1 = p_in_row0 + in_w;

        // Use float2 to load two adjacent floats at once
        float2 v0 = *(reinterpret_cast<const float2*>(p_in_row0));
        float2 v1 = *(reinterpret_cast<const float2*>(p_in_row1));
        
        float max_val = fmaxf(fmaxf(v0.x, v0.y), fmaxf(v1.x, v1.y));
        
        const int out_idx = c_out * out_h * out_w + h_out * out_w + w_out;
        output[out_idx] = fmaxf(0.0f, max_val);
    }
}

// --- NEW: Vectorized Mega-Kernel (ReLU+MaxPool+Flatten+Linear+ReLU for conv2 -> fc1) ---
// This kernel enhances the mega-kernel from Program 1 with float4 vectorization on the
// weight matrix, further improving memory bandwidth.
__global__ void fused_pool_linear_relu_vectorized_kernel(
    const float* __restrict__ conv_out,      // Input from conv2, e.g., [16, 10, 10]
    const float* __restrict__ weight,        // fc1 weight, e.g., [120, 400]
    const float* __restrict__ bias,          // fc1 bias, e.g., [120]
    float* __restrict__ out,                 // fc1 output, e.g., [120]
    int C, int H, int W,                     // Dimensions of conv_out (16, 10, 10)
    int out_features, int in_features) {     // Dimensions of linear layer (120, 400)

    int j = blockIdx.x; // Each block computes one output feature j.

    if (j < out_features) {
        float thread_sum = 0.0f;
        const int pool_h_out = H / 2;
        const int pool_w_out = W / 2;
        const int plane_size_pool = pool_h_out * pool_w_out; // 25

        const float4* weight_f4 = reinterpret_cast<const float4*>(weight + j * in_features);
        const int in_features_v4 = in_features / 4;

        for (int i_v4 = threadIdx.x; i_v4 < in_features_v4; i_v4 += blockDim.x) {
            float4 pooled_relu_vals;

            // Unroll the loop to compute 4 pooled values on-the-fly
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                int i = i_v4 * 4 + k;
                int c_pool = i / plane_size_pool;
                int i_plane = i % plane_size_pool;
                int h_pool = i_plane / pool_w_out;
                int w_pool = i_plane % pool_w_out;

                int h_conv_start = h_pool * 2;
                int w_conv_start = w_pool * 2;

                const float* p_conv = conv_out + c_pool * (H * W) + h_conv_start * W + w_conv_start;
                float v00 = p_conv[0], v01 = p_conv[1], v10 = p_conv[W], v11 = p_conv[W + 1];
                float max_val = fmaxf(fmaxf(v00, v01), fmaxf(v10, v11));
                
                // Store into the float4 components
                ((float*)&pooled_relu_vals)[k] = fmaxf(0.0f, max_val);
            }

            float4 wt_val = weight_f4[i_v4];
            thread_sum += pooled_relu_vals.x * wt_val.x + pooled_relu_vals.y * wt_val.y + 
                          pooled_relu_vals.z * wt_val.z + pooled_relu_vals.w * wt_val.w;
        }

        // --- In-Block Reduction (identical to linear_relu_vectorized_kernel) ---
        float warp_sum = warp_reduce_sum(thread_sum);
        extern __shared__ float sdata[];
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        if (lane_id == 0) sdata[warp_id] = warp_sum;
        __syncthreads();
        thread_sum = (threadIdx.x < (blockDim.x / 32)) ? sdata[lane_id] : 0.0f;
        if (warp_id == 0) {
            float block_sum = warp_reduce_sum(thread_sum);
            if (lane_id == 0) {
                float final_val = block_sum + bias[j];
                out[j] = fmaxf(0.0f, final_val);
            }
        }
    }
}

// --- C++ Launcher Functions ---
torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    auto input_vec = input.squeeze(0);
    const int out_features = weight.size(0);
    const int in_features = weight.size(1);
    TORCH_CHECK(in_features % 4 == 0, "in_features must be divisible by 4");
    auto out = torch::empty({out_features}, input.options());
    const int block_size = 256;
    const int num_blocks = out_features;
    const int shared_mem_size = (block_size / 32) * sizeof(float);
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(input));
    linear_relu_vectorized_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        input_vec.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), out_features, in_features);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.unsqueeze(0);
}

torch::Tensor relu_maxpool2d_cuda(torch::Tensor input) {
    const auto c = input.size(1), h = input.size(2), w = input.size(3);
    TORCH_CHECK(w % 2 == 0, "Input width must be even for float2 vectorization");
    const int out_h = h / 2, out_w = w / 2;
    auto out = torch::empty({1, c, out_h, out_w}, input.options());
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(input));
    dim3 block_dim(16, 16, 1);
    dim3 grid_dim((out_w + block_dim.x - 1) / block_dim.x, (out_h + block_dim.y - 1) / block_dim.y, c);
    relu_maxpool2d_kernel_optimized<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), out.data_ptr<float>(), h, w, out_h, out_w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_pool_linear_relu_cuda(torch::Tensor conv_out, torch::Tensor weight, torch::Tensor bias) {
    const int C = conv_out.size(1), H = conv_out.size(2), W = conv_out.size(3);
    const int out_features = weight.size(0), in_features = weight.size(1);
    TORCH_CHECK(in_features == C * (H/2) * (W/2), "in_features mismatch");
    TORCH_CHECK(in_features % 4 == 0, "in_features must be divisible by 4");
    auto out = torch::empty({out_features}, conv_out.options());
    const int block_size = 256;
    const int num_blocks = out_features;
    const int shared_mem_size = (block_size / 32) * sizeof(float);
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(conv_out));
    fused_pool_linear_relu_vectorized_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        conv_out.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), C, H, W, out_features, in_features);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out.unsqueeze(0);
}
"""

fused_ops_cpp_source = """
torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor relu_maxpool2d_cuda(torch::Tensor input);
torch::Tensor fused_pool_linear_relu_cuda(torch::Tensor conv_out, torch::Tensor weight, torch::Tensor bias);
"""

# JIT compile the CUDA kernels with a unique name to avoid caching issues.
fused_ops = load_inline(
    name="fused_lenet_ops_v_ultimate",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["linear_relu_cuda", "relu_maxpool2d_cuda", "fused_pool_linear_relu_cuda"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture with maximally fused and optimized CUDA kernels.
        """
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        """
        Forward pass using the custom fused kernels for maximum performance.
        """
        # Block 1: cuDNN Conv -> Optimized Fused ReLU+MaxPool
        x = self.conv1(x)
        x = fused_ops.relu_maxpool2d_cuda(x)
        
        # Block 2: cuDNN Conv -> Vectorized Mega-Kernel (Pool->Flatten->Linear->ReLU)
        x = self.conv2(x)
        x = fused_ops.fused_pool_linear_relu_cuda(x, self.fc1.weight, self.fc1.bias)
        
        # Block 3: Optimized Fused Linear+ReLU
        x = fused_ops.linear_relu_cuda(x, self.fc2.weight, self.fc2.bias)
        
        # Final Layer: Standard Linear
        x = self.fc3(x)
        
        return x

# Configuration for the LeNet-5 model test.
batch_size = 1
num_classes = 10

def get_inputs():
    return [torch.randn(batch_size, 1, 32, 32).cuda()]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END