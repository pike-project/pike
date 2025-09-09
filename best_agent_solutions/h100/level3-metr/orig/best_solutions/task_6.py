import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- Original Model Definition (Required for Correct Initialization) ---
class Model(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(Model, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

# --- Fused CUDA Kernels ---
# This single source string contains all the custom CUDA kernels for efficiency.
inception_v2_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath> // For fmaxf
#include <ATen/cuda/CUDAContext.h>

#define TILE_W 32
#define TILE_H 8

// --- Kernel 1: Simple 1x1 Convolution ---
// Writes into the full output tensor using an offset.
__global__ void branch_1x1_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
    float* __restrict__ out,
    const int N, const int C_in, const int H, const int W,
    const int C_out_branch, const int C_out_total, const int C_offset)
{
    const int total_threads = gridDim.x * blockDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW = H * W;

    // Grid-stride loop over the output elements of THIS BRANCH
    for (int idx_branch = thread_id; idx_branch < N * C_out_branch * HW; idx_branch += total_threads) {
        const int w_out = idx_branch % W;
        const int h_out = (idx_branch / W) % H;
        const int c_out_local = (idx_branch / HW) % C_out_branch;
        const int n = idx_branch / (C_out_branch * HW);

        float acc = b[c_out_local];
        const int x_offset = n * C_in * HW + h_out * W + w_out;
        const float* w_ptr = w + c_out_local * C_in;

        for (int c_in = 0; c_in < C_in; ++c_in) {
            acc += x[x_offset + c_in * HW] * w_ptr[c_in];
        }

        // Calculate index into the FULL output tensor
        const int c_out_global = c_out_local + C_offset;
        const int out_idx = n * C_out_total * HW + c_out_global * HW + h_out * W + w_out;
        out[out_idx] = acc;
    }
}


// --- Kernel 2: Tiled and Fused (Conv1x1 -> Conv3x3) ---
__global__ void branch_3x3_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w1, const float* __restrict__ b1,
    const float* __restrict__ w2, const float* __restrict__ b2,
    float* __restrict__ out,
    const int N, const int C_in, const int H, const int W,
    const int C_reduce, const int C_out_branch, const int C_out_total, const int C_offset,
    const int grid_w)
{
    const int n = blockIdx.z;
    const int c_out_local = blockIdx.y; // Channel index within this branch
    const int tile_idx_y = blockIdx.x / grid_w;
    const int tile_idx_x = blockIdx.x % grid_w;
    const int h_base = tile_idx_y * TILE_H;
    const int w_base = tile_idx_x * TILE_W;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    constexpr int PADDED_TILE_H = TILE_H + 2;
    constexpr int PADDED_TILE_W = TILE_W + 2;
    __shared__ float smem_inter[PADDED_TILE_H][PADDED_TILE_W];

    float acc = 0.0f;

    for (int c_red = 0; c_red < C_reduce; ++c_red) {
        for (int i = ty; i < PADDED_TILE_H; i += TILE_H) {
            for (int j = tx; j < PADDED_TILE_W; j += TILE_W) {
                const int h_in = h_base + i - 1;
                const int w_in = w_base + j - 1;

                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    float inter_val = b1[c_red];
                    const int x_offset = n * C_in * H * W + h_in * W + w_in;
                    const float* w1_ptr = w1 + c_red * C_in;
                    for (int c_in_idx = 0; c_in_idx < C_in; ++c_in_idx) {
                        inter_val += x[x_offset + c_in_idx * H * W] * w1_ptr[c_in_idx];
                    }
                    smem_inter[i][j] = inter_val;
                } else {
                    smem_inter[i][j] = 0.0f;
                }
            }
        }
        __syncthreads();

        const float* w2_ptr = w2 + (c_out_local * C_reduce + c_red) * 9;
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                acc += smem_inter[ty + kh][tx + kw] * w2_ptr[kh * 3 + kw];
            }
        }
        __syncthreads();
    }

    const int h_out = h_base + ty;
    const int w_out = w_base + tx;
    if (ty < TILE_H && tx < TILE_W && h_out < H && w_out < W) {
        const int c_out_global = c_out_local + C_offset;
        out[n * C_out_total * H * W + c_out_global * H * W + h_out * W + w_out] = acc + b2[c_out_local];
    }
}


// --- Kernel 3: Tiled and Fused (Conv1x1 -> Conv5x5) ---
__global__ void branch_5x5_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w1, const float* __restrict__ b1,
    const float* __restrict__ w2, const float* __restrict__ b2,
    float* __restrict__ out,
    const int N, const int C_in, const int H, const int W,
    const int C_reduce, const int C_out_branch, const int C_out_total, const int C_offset,
    const int grid_w)
{
    const int n = blockIdx.z;
    const int c_out_local = blockIdx.y;
    const int tile_idx_y = blockIdx.x / grid_w;
    const int tile_idx_x = blockIdx.x % grid_w;
    const int h_base = tile_idx_y * TILE_H;
    const int w_base = tile_idx_x * TILE_W;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    constexpr int PADDED_TILE_H = TILE_H + 4;
    constexpr int PADDED_TILE_W = TILE_W + 4;
    __shared__ float smem_inter[PADDED_TILE_H][PADDED_TILE_W];

    float acc = 0.0f;

    for (int c_red = 0; c_red < C_reduce; ++c_red) {
        for (int i = ty; i < PADDED_TILE_H; i += TILE_H) {
            for (int j = tx; j < PADDED_TILE_W; j += TILE_W) {
                const int h_in = h_base + i - 2;
                const int w_in = w_base + j - 2;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    float inter_val = b1[c_red];
                    const int x_offset = n * C_in * H * W + h_in * W + w_in;
                    const float* w1_ptr = w1 + c_red * C_in;
                    for (int c_in_idx = 0; c_in_idx < C_in; ++c_in_idx) {
                        inter_val += x[x_offset + c_in_idx * H * W] * w1_ptr[c_in_idx];
                    }
                    smem_inter[i][j] = inter_val;
                } else {
                    smem_inter[i][j] = 0.0f;
                }
            }
        }
        __syncthreads();

        const float* w2_ptr = w2 + (c_out_local * C_reduce + c_red) * 25;
        for (int kh = 0; kh < 5; ++kh) {
            for (int kw = 0; kw < 5; ++kw) {
                acc += smem_inter[ty + kh][tx + kw] * w2_ptr[kh * 5 + kw];
            }
        }
        __syncthreads();
    }

    const int h_out = h_base + ty;
    const int w_out = w_base + tx;
    if (ty < TILE_H && tx < TILE_W && h_out < H && w_out < W) {
        const int c_out_global = c_out_local + C_offset;
        out[n * C_out_total * H * W + c_out_global * H * W + h_out * W + w_out] = acc + b2[c_out_local];
    }
}


// --- Kernel 4: Tiled and Fused (MaxPool3x3 -> Conv1x1) ---
__global__ void branch_pool_fused_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w, const float* __restrict__ b,
    float* __restrict__ out,
    const int N, const int C_in, const int H, const int W,
    const int C_out_branch, const int C_out_total, const int C_offset,
    const int grid_w)
{
    const int n = blockIdx.z;
    const int c_out_local = blockIdx.y;
    const int tile_idx_y = blockIdx.x / grid_w;
    const int tile_idx_x = blockIdx.x % grid_w;
    const int h_base = tile_idx_y * TILE_H;
    const int w_base = tile_idx_x * TILE_W;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int h_out = h_base + ty;
    const int w_out = w_base + tx;

    if (h_out < H && w_out < W) {
        float acc = b[c_out_local];
        const float* w_ptr = w + c_out_local * C_in;

        for (int c_in_idx = 0; c_in_idx < C_in; ++c_in_idx) {
            float max_val = -1.0e20f;
            for (int kh = -1; kh <= 1; ++kh) {
                for (int kw = -1; kw <= 1; ++kw) {
                    const int h_in = h_out + kh;
                    const int w_in = w_out + kw;
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        max_val = fmaxf(max_val, x[n * C_in * H * W + c_in_idx * H * W + h_in * W + w_in]);
                    }
                }
            }
            acc += max_val * w_ptr[c_in_idx];
        }
        const int c_out_global = c_out_local + C_offset;
        out[n * C_out_total * H * W + c_out_global * H * W + h_out * W + w_out] = acc;
    }
}


// --- C++ Wrappers for PyTorch ---
void branch_1x1_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out, int C_offset) {
    const int N = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
    const int C_out_branch = w.size(0);
    const int C_out_total = out.size(1);
    const int block_size = 256;
    const int num_blocks = (N * C_out_branch * H * W + block_size - 1) / block_size;
    branch_1x1_kernel<<<std::min(num_blocks, 65535), block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
        N, C_in, H, W, C_out_branch, C_out_total, C_offset);
}

void branch_3x3_fused_cuda(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2, torch::Tensor out, int C_offset) {
    const int N = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
    const int C_reduce = w1.size(0), C_out_branch = w2.size(0);
    const int C_out_total = out.size(1);
    const dim3 block(TILE_W, TILE_H);
    const int grid_w = (W + TILE_W - 1) / TILE_W;
    const dim3 grid((H + TILE_H - 1) / TILE_H * grid_w, C_out_branch, N);
    branch_3x3_fused_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), w1.data_ptr<float>(), b1.data_ptr<float>(), w2.data_ptr<float>(), b2.data_ptr<float>(),
        out.data_ptr<float>(), N, C_in, H, W, C_reduce, C_out_branch, C_out_total, C_offset, grid_w);
}

void branch_5x5_fused_cuda(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2, torch::Tensor out, int C_offset) {
    const int N = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
    const int C_reduce = w1.size(0), C_out_branch = w2.size(0);
    const int C_out_total = out.size(1);
    const dim3 block(TILE_W, TILE_H);
    const int grid_w = (W + TILE_W - 1) / TILE_W;
    const dim3 grid((H + TILE_H - 1) / TILE_H * grid_w, C_out_branch, N);
    branch_5x5_fused_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), w1.data_ptr<float>(), b1.data_ptr<float>(), w2.data_ptr<float>(), b2.data_ptr<float>(),
        out.data_ptr<float>(), N, C_in, H, W, C_reduce, C_out_branch, C_out_total, C_offset, grid_w);
}

void branch_pool_fused_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out, int C_offset) {
    const int N = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
    const int C_out_branch = w.size(0);
    const int C_out_total = out.size(1);
    const dim3 block(TILE_W, TILE_H);
    const int grid_w = (W + TILE_W - 1) / TILE_W;
    const dim3 grid((H + TILE_H - 1) / TILE_H * grid_w, C_out_branch, N);
    branch_pool_fused_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        out.data_ptr<float>(), N, C_in, H, W, C_out_branch, C_out_total, C_offset, grid_w);
}
"""

inception_v2_kernels_cpp_source = """
void branch_1x1_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out, int C_offset);
void branch_3x3_fused_cuda(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2, torch::Tensor out, int C_offset);
void branch_5x5_fused_cuda(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2, torch::Tensor out, int C_offset);
void branch_pool_fused_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor out, int C_offset);
"""

# Compile all kernels in one go
inception_v2_kernels = load_inline(
    name="inception_v2_kernels",
    cpp_sources=inception_v2_kernels_cpp_source,
    cuda_sources=inception_v2_kernels_source,
    functions=[
        "branch_1x1_cuda",
        "branch_3x3_fused_cuda",
        "branch_5x5_fused_cuda",
        "branch_pool_fused_cuda"
    ],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(ModelNew, self).__init__()
        original_model = Model(in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj)

        self.out_1x1 = out_1x1
        self.out_3x3 = out_3x3
        self.out_5x5 = out_5x5
        self.pool_proj = pool_proj
        self.total_out_channels = out_1x1 + out_3x3 + out_5x5 + pool_proj

        self.b1_w = nn.Parameter(original_model.branch1x1.weight)
        self.b1_b = nn.Parameter(original_model.branch1x1.bias)

        self.b2_w1 = nn.Parameter(original_model.branch3x3[0].weight)
        self.b2_b1 = nn.Parameter(original_model.branch3x3[0].bias)
        self.b2_w2 = nn.Parameter(original_model.branch3x3[1].weight)
        self.b2_b2 = nn.Parameter(original_model.branch3x3[1].bias)

        self.b3_w1 = nn.Parameter(original_model.branch5x5[0].weight)
        self.b3_b1 = nn.Parameter(original_model.branch5x5[0].bias)
        self.b3_w2 = nn.Parameter(original_model.branch5x5[1].weight)
        self.b3_b2 = nn.Parameter(original_model.branch5x5[1].bias)

        self.b4_w = nn.Parameter(original_model.branch_pool[1].weight)
        self.b4_b = nn.Parameter(original_model.branch_pool[1].bias)

        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
        self.stream3 = torch.cuda.Stream()
        self.stream4 = torch.cuda.Stream()

    def forward(self, x):
        x_contig = x.contiguous()
        
        out = torch.empty(x.size(0), self.total_out_channels, x.size(2), x.size(3),
                          device=x.device, dtype=x.dtype)

        # Calculate channel offsets for writing into the single output tensor
        c_offset = 0
        with torch.cuda.stream(self.stream1):
            inception_v2_kernels.branch_1x1_cuda(x_contig, self.b1_w, self.b1_b, out, c_offset)
        c_offset += self.out_1x1

        with torch.cuda.stream(self.stream2):
            inception_v2_kernels.branch_3x3_fused_cuda(x_contig, self.b2_w1, self.b2_b1, self.b2_w2, self.b2_b2, out, c_offset)
        c_offset += self.out_3x3

        with torch.cuda.stream(self.stream3):
            inception_v2_kernels.branch_5x5_fused_cuda(x_contig, self.b3_w1, self.b3_b1, self.b3_w2, self.b3_b2, out, c_offset)
        c_offset += self.out_5x5

        with torch.cuda.stream(self.stream4):
            inception_v2_kernels.branch_pool_fused_cuda(x_contig, self.b4_w, self.b4_b, out, c_offset)

        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.stream1)
        current_stream.wait_stream(self.stream2)
        current_stream.wait_stream(self.stream3)
        current_stream.wait_stream(self.stream4)

        return out