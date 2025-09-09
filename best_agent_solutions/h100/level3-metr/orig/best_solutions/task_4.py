import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- Optimized Custom CUDA Kernels Source Code (V2) ---
# This version introduces a more efficient convolution kernel design.
# Each thread computes a 2x2 output patch and performs pooling in registers,
# which improves thread utilization and reduces shared memory traffic.

custom_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <cmath> // for fmaxf

// ============================================================================
// Common Definitions for All Kernels
// ============================================================================
#define KERNEL_W 5
#define KERNEL_H 5
#define KERNEL_SIZE (KERNEL_W * KERNEL_H)

// ============================================================================
// Definitions for V2 Convolution Kernels (Work-per-Thread Strategy)
// ============================================================================
#define CONV_BLOCK_W 8
#define CONV_BLOCK_H 8
#define CONV_BLOCK_SIZE (CONV_BLOCK_W * CONV_BLOCK_H)

// An 8x8 block computes an 8x8 tile of pooled output.
// This requires a 16x16 tile of convolution output (2x2 per thread).
#define CONV_OUT_TILE_W (CONV_BLOCK_W * 2)
#define CONV_OUT_TILE_H (CONV_BLOCK_H * 2)

// Input tile size needed for a 16x16 conv output tile with 5x5 kernel
#define S_IN_TILE_W (CONV_OUT_TILE_W + KERNEL_W - 1)
#define S_IN_TILE_H (CONV_OUT_TILE_H + KERNEL_H - 1)
#define S_IN_TILE_SIZE (S_IN_TILE_W * S_IN_TILE_H)

// ============================================================================
// Kernel 1: V2 SPECIALIZED Fused Conv2D(C_in=1) + ReLU + MaxPool
// ============================================================================
__global__ void fused_conv1_relu_maxpool_v2_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* out,
    const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out)
{
    extern __shared__ float s_mem[];
    float (*s_in)[S_IN_TILE_H][S_IN_TILE_W] = (float (*)[S_IN_TILE_H][S_IN_TILE_W])s_mem;
    float *s_weights = s_mem + S_IN_TILE_SIZE;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * CONV_BLOCK_W + tx;

    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int out_c = blockIdx.z;

    const int g_in_tile_start_y = block_y * CONV_OUT_TILE_H;
    const int g_in_tile_start_x = block_x * CONV_OUT_TILE_W;

    for (int i = tid; i < S_IN_TILE_SIZE; i += CONV_BLOCK_SIZE) {
        int sm_y = i / S_IN_TILE_W;
        int sm_x = i % S_IN_TILE_W;
        int g_y = g_in_tile_start_y + sm_y;
        int g_x = g_in_tile_start_x + sm_x;
        if (g_y >= 0 && g_y < H_in && g_x >= 0 && g_x < W_in) {
            (*s_in)[sm_y][sm_x] = input[g_y * W_in + g_x];
        } else {
            (*s_in)[sm_y][sm_x] = 0.0f;
        }
    }

    if (tid < KERNEL_SIZE) {
         s_weights[tid] = weight[out_c * KERNEL_SIZE + tid];
    }
    __syncthreads();

    const int sm_conv_y_base = ty * 2;
    const int sm_conv_x_base = tx * 2;
    float psum00 = 0.0f, psum01 = 0.0f, psum10 = 0.0f, psum11 = 0.0f;
    
    #pragma unroll
    for (int ky = 0; ky < KERNEL_H; ++ky) {
        #pragma unroll
        for (int kx = 0; kx < KERNEL_W; ++kx) {
            float w = s_weights[ky * KERNEL_W + kx];
            psum00 += (*s_in)[sm_conv_y_base + ky][sm_conv_x_base + kx] * w;
            psum01 += (*s_in)[sm_conv_y_base + ky][sm_conv_x_base + kx + 1] * w;
            psum10 += (*s_in)[sm_conv_y_base + ky + 1][sm_conv_x_base + kx] * w;
            psum11 += (*s_in)[sm_conv_y_base + ky + 1][sm_conv_x_base + kx + 1] * w;
        }
    }
    
    const float bias_val = bias[out_c];
    psum00 = fmaxf(0.0f, psum00 + bias_val);
    psum01 = fmaxf(0.0f, psum01 + bias_val);
    psum10 = fmaxf(0.0f, psum10 + bias_val);
    psum11 = fmaxf(0.0f, psum11 + bias_val);

    float max_val = fmaxf(fmaxf(psum00, psum01), fmaxf(psum10, psum11));

    const int out_y = block_y * CONV_BLOCK_H + ty;
    const int out_x = block_x * CONV_BLOCK_W + tx;
    if (out_y < H_out && out_x < W_out) {
        out[out_c * H_out * W_out + out_y * W_out + out_x] = max_val;
    }
}

// ============================================================================
// Kernel 2: V2 GENERAL Fused Conv2D(C_in>1) + ReLU + MaxPool
// ============================================================================
__global__ void fused_convN_relu_maxpool_v2_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* out,
    const int C_in, const int H_in, const int W_in,
    const int C_out, const int H_out, const int W_out)
{
    extern __shared__ float s_mem[];
    float (*s_in_buffs)[S_IN_TILE_H][S_IN_TILE_W] = (float (*)[S_IN_TILE_H][S_IN_TILE_W])s_mem;
    float *s_weights = s_mem + 2 * S_IN_TILE_SIZE;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * CONV_BLOCK_W + tx;

    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int out_c = blockIdx.z;

    const int g_in_tile_start_y = block_y * CONV_OUT_TILE_H;
    const int g_in_tile_start_x = block_x * CONV_OUT_TILE_W;
    
    auto load_tile = [&](const float* g_input_base, float (*s_target)[S_IN_TILE_W]) {
        for (int i = tid; i < S_IN_TILE_SIZE; i += CONV_BLOCK_SIZE) {
            int sm_y = i / S_IN_TILE_W;
            int sm_x = i % S_IN_TILE_W;
            int g_y = g_in_tile_start_y + sm_y;
            int g_x = g_in_tile_start_x + sm_x;
            if (g_y >= 0 && g_y < H_in && g_x >= 0 && g_x < W_in) {
                s_target[sm_y][sm_x] = g_input_base[g_y * W_in + g_x];
            } else {
                s_target[sm_y][sm_x] = 0.0f;
            }
        }
    };

    float psum00 = 0.0f, psum01 = 0.0f, psum10 = 0.0f, psum11 = 0.0f;
    const float* weight_for_out_c = weight + out_c * C_in * KERNEL_SIZE;

    load_tile(input, s_in_buffs[0]);

    for (int c = 0; c < C_in; ++c) {
        __syncthreads();
        float (*s_input_compute)[S_IN_TILE_W] = s_in_buffs[c % 2];

        if (c + 1 < C_in) {
            load_tile(input + (c + 1) * H_in * W_in, s_in_buffs[(c + 1) % 2]);
        }
        if (tid < KERNEL_SIZE) {
             s_weights[tid] = weight_for_out_c[c * KERNEL_SIZE + tid];
        }
        __syncthreads();

        const int sm_conv_y_base = ty * 2;
        const int sm_conv_x_base = tx * 2;
        #pragma unroll
        for (int ky = 0; ky < KERNEL_H; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < KERNEL_W; ++kx) {
                float w = s_weights[ky * KERNEL_W + kx];
                psum00 += s_input_compute[sm_conv_y_base + ky][sm_conv_x_base + kx] * w;
                psum01 += s_input_compute[sm_conv_y_base + ky][sm_conv_x_base + kx + 1] * w;
                psum10 += s_input_compute[sm_conv_y_base + ky + 1][sm_conv_x_base + kx] * w;
                psum11 += s_input_compute[sm_conv_y_base + ky + 1][sm_conv_x_base + kx + 1] * w;
            }
        }
    }
    
    const float bias_val = bias[out_c];
    psum00 = fmaxf(0.0f, psum00 + bias_val);
    psum01 = fmaxf(0.0f, psum01 + bias_val);
    psum10 = fmaxf(0.0f, psum10 + bias_val);
    psum11 = fmaxf(0.0f, psum11 + bias_val);

    float max_val = fmaxf(fmaxf(psum00, psum01), fmaxf(psum10, psum11));

    const int out_y = block_y * CONV_BLOCK_H + ty;
    const int out_x = block_x * CONV_BLOCK_W + tx;
    if (out_y < H_out && out_x < W_out) {
        out[out_c * H_out * W_out + out_y * W_out + out_x] = max_val;
    }
}

// Host-side wrappers for the V2 convolution kernels
torch::Tensor conv1_relu_maxpool_v2_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.dim() == 4 && input.size(0) == 1 && input.size(1) == 1, "Input must be 4D NCHW with N=1, C=1 for conv1 kernel");
    input = input.squeeze(0).squeeze(0).contiguous();
    const int H_in = input.size(0);
    const int W_in = input.size(1);
    const int C_out = weight.size(0);
    const int H_conv = H_in - KERNEL_H + 1;
    const int W_conv = W_in - KERNEL_W + 1;
    const int H_out = H_conv / 2;
    const int W_out = W_conv / 2;
    TORCH_CHECK(H_conv % 2 == 0 && W_conv % 2 == 0, "Conv output dimensions must be even for 2x2 pooling");

    auto out = torch::empty({C_out, H_out, W_out}, input.options());
    const dim3 block_dim(CONV_BLOCK_W, CONV_BLOCK_H);
    const dim3 grid_dim( (W_out + CONV_BLOCK_W - 1) / CONV_BLOCK_W, (H_out + CONV_BLOCK_H - 1) / CONV_BLOCK_H, C_out);
    const size_t shared_mem_size = (S_IN_TILE_SIZE + KERNEL_SIZE) * sizeof(float);

    fused_conv1_relu_maxpool_v2_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), H_in, W_in, C_out, H_out, W_out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { throw std::runtime_error(std::string("CUDA error in conv1_relu_maxpool_v2_cuda: ") + cudaGetErrorString(err)); }
    return out.unsqueeze(0);
}

torch::Tensor convN_relu_maxpool_v2_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.dim() == 4 && input.size(0) == 1, "Input must be 4D NCHW with N=1 for this custom kernel");
    input = input.squeeze(0).contiguous();
    const int C_in = input.size(0);
    const int H_in = input.size(1);
    const int W_in = input.size(2);
    const int C_out = weight.size(0);
    const int H_conv = H_in - KERNEL_H + 1;
    const int W_conv = W_in - KERNEL_W + 1;
    const int H_out = H_conv / 2;
    const int W_out = W_conv / 2;
    TORCH_CHECK(H_conv % 2 == 0 && W_conv % 2 == 0, "Conv output dimensions must be even for 2x2 pooling");

    auto out = torch::empty({C_out, H_out, W_out}, input.options());
    const dim3 block_dim(CONV_BLOCK_W, CONV_BLOCK_H);
    const dim3 grid_dim( (W_out + CONV_BLOCK_W - 1) / CONV_BLOCK_W, (H_out + CONV_BLOCK_H - 1) / CONV_BLOCK_H, C_out);
    const size_t shared_mem_size = (2 * S_IN_TILE_SIZE + KERNEL_SIZE) * sizeof(float);

    fused_convN_relu_maxpool_v2_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), C_in, H_in, W_in, C_out, H_out, W_out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { throw std::runtime_error(std::string("CUDA error in convN_relu_maxpool_v2_cuda: ") + cudaGetErrorString(err)); }
    return out.unsqueeze(0);
}

// ============================================================================
// Kernel 3: ADVANCED Fused Linear + ReLU (Warp-per-Row GEMV)
// ============================================================================
#define WARP_SIZE 32
#define FC_TILE_N 8 

__global__ void gemv_relu_kernel(const float* __restrict__ input,
                                  const float* __restrict__ weight,
                                  const float* __restrict__ bias,
                                  float* out,
                                  const int N, const int K,
                                  const bool apply_relu) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row_idx = blockIdx.x * FC_TILE_N + warp_id;

    if (row_idx >= N) return;

    const float4* input4 = reinterpret_cast<const float4*>(input);
    const float4* weight4_row = reinterpret_cast<const float4*>(weight + row_idx * K);

    float psum = 0.0f;
    for (int k4_idx = lane_id; k4_idx < K / 4; k4_idx += WARP_SIZE) {
        float4 i_vec = input4[k4_idx];
        float4 w_vec = weight4_row[k4_idx];
        psum += i_vec.x * w_vec.x + i_vec.y * w_vec.y + i_vec.z * w_vec.z + i_vec.w * w_vec.w;
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        psum += __shfl_down_sync(0xFFFFFFFF, psum, offset);
    }

    if (lane_id == 0) {
        float result = psum + bias[row_idx];
        if (apply_relu) {
            result = fmaxf(0.0f, result);
        }
        out[row_idx] = result;
    }
}

torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, bool apply_relu) {
    TORCH_CHECK(input.dim() == 2 && input.size(0) == 1, "Input tensor must have shape [1, K]");
    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    const int N = weight.size(0);
    const int K = weight.size(1);
    TORCH_CHECK(K % 4 == 0, "Vectorized kernel requires inner dimension (K) to be divisible by 4.");

    auto out = torch::empty({1, N}, input.options());
    const int block_size = FC_TILE_N * WARP_SIZE;
    const dim3 grid_dim((N + FC_TILE_N - 1) / FC_TILE_N);
    const dim3 block_dim(block_size);

    gemv_relu_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), N, K, apply_relu);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { throw std::runtime_error(std::string("CUDA error in linear_relu_cuda: ") + cudaGetErrorString(err)); }
    return out;
}
"""

custom_kernels_cpp_source = """
torch::Tensor conv1_relu_maxpool_v2_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor convN_relu_maxpool_v2_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor linear_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, bool apply_relu);
"""

custom_kernels = load_inline(
    name="custom_lenet_kernels_v8_work_per_thread",
    cpp_sources=custom_kernels_cpp_source,
    cuda_sources=custom_kernels_source,
    functions=["conv1_relu_maxpool_v2_cuda", "convN_relu_maxpool_v2_cuda", "linear_relu_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_70"]
)


class ModelNew(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 optimized with highly efficient custom CUDA kernels and CUDA Graphs.
        - Kernels: 
          - Convolutions use a "work-per-thread" fused kernel where each thread computes a
            2x2 output patch and performs pooling in registers, maximizing efficiency.
          - A specialized version is used for the first layer (C_in=1).
          - A double-buffered version is used for the second layer (C_in>1).
          - FC layers use a high-performance warp-per-row GEMV kernel.
        - Execution: Uses CUDA Graphs to capture the entire forward pass, eliminating
          kernel launch overhead on subsequent runs for maximum speed.
        
        :param num_classes: The number of output classes.
        """
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
        
        self.custom_conv1 = custom_kernels.conv1_relu_maxpool_v2_cuda
        self.custom_convN = custom_kernels.convN_relu_maxpool_v2_cuda
        self.custom_linear_relu = custom_kernels.linear_relu_cuda

        self.graph = None
        self.static_input = None
        self.static_output = None

    def _capture_graph(self, x):
        """Records the forward pass into a CUDA graph."""
        self.static_input = x.clone()
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            y = self.custom_conv1(self.static_input, self.conv1.weight, self.conv1.bias)
            y = self.custom_convN(y, self.conv2.weight, self.conv2.bias)
            y = y.view(-1, 16*5*5)
            y = self.custom_linear_relu(y, self.fc1.weight, self.fc1.bias, True)
            y = self.custom_linear_relu(y, self.fc2.weight, self.fc2.bias, True)
            self.static_output = self.custom_linear_relu(y, self.fc3.weight, self.fc3.bias, False)

    def forward(self, x):
        """
        Forward pass using CUDA Graph execution.
        The first call will be slower as it captures the graph. Subsequent calls will be much faster.
        """
        if self.graph is None or not x.shape == self.static_input.shape:
            self._capture_graph(x)

        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

# Helper functions to generate inputs for the model
batch_size = 1
num_classes = 10

def get_inputs():
    # fc1 in_features = 16*5*5 = 400, which is divisible by 4.
    # fc2 in_features = 120, which is divisible by 4.
    # fc3 in_features = 84, which is divisible by 4.
    # All FC layers satisfy the vectorized linear kernel's requirement.
    return [torch.randn(batch_size, 1, 32, 32).cuda()]

def get_init_inputs():
    return [num_classes]