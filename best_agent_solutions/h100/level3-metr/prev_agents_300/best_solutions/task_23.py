import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

# ==============================================================================
# 1. CUDA C++ Kernel for Fused Initial Conv (3x3, s=2) + BN + ReLU (NHWC, FP16)
# ==============================================================================
# This kernel replaces the Triton initial_conv kernel. It uses a classic tiled
# algorithm with shared memory to maximize data reuse for the input patch.
# This is much more efficient for convolutions than the previous Triton kernel.
initial_conv_cuda_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tiling dimensions for the kernel
constexpr int BLOCK_H = 8;
constexpr int BLOCK_W = 8;
constexpr int BLOCK_C = 32; // Must match C_out of initial conv
constexpr int THREADS_X = 8; // Must match BLOCK_W
constexpr int THREADS_Y = 8; // Must match BLOCK_H

// Shared memory padding to avoid bank conflicts
#define PADDED_W(W) (W + 8)

__global__ void initial_conv_bn_relu_nhwc_cuda_kernel(
    const half* __restrict__ x,
    const half* __restrict__ w,
    const half* __restrict__ bn_bias, // bn_scale is fused into weights
    half* __restrict__ y,
    int H_in, int W_in, int C_out, int H_out, int W_out
) {
    // --- Shared memory allocation ---
    // Input patch size: ((BLOCK_H - 1) * stride_h + K_h) x ((BLOCK_W - 1) * stride_w + K_w)
    // For stride=2, K=3: (7 * 2 + 3) x (7 * 2 + 3) = 17x17
    constexpr int SHMEM_H = (BLOCK_H - 1) * 2 + 3;
    constexpr int SHMEM_W = (BLOCK_W - 1) * 2 + 3;
    __shared__ half s_input[SHMEM_H][PADDED_W(SHMEM_W)][3];

    // --- Block and Thread Indexing ---
    const int n = blockIdx.z;
    const int block_h = blockIdx.y;
    const int block_w = blockIdx.x;

    const int thread_h = threadIdx.y;
    const int thread_w = threadIdx.x;

    // --- Cooperative Loading into Shared Memory ---
    const int h_in_start = block_h * BLOCK_H * 2 - 1; // padding=1
    const int w_in_start = block_w * BLOCK_W * 2 - 1; // padding=1

    // Each thread loads multiple elements to fill the shared memory tile
    // Number of threads is THREADS_Y * THREADS_X, which is smaller than SHMEM_H * SHMEM_W
    // So we need a loop.
    int thread_id = thread_h * THREADS_X + thread_w;
    int total_threads = THREADS_X * THREADS_Y;
    for (int i = thread_id; i < SHMEM_H * SHMEM_W; i += total_threads) {
        int sh_h = i / SHMEM_W;
        int sh_w = i % SHMEM_W;

        int cur_h = h_in_start + sh_h;
        int cur_w = w_in_start + sh_w;
        bool is_valid = (cur_h >= 0 && cur_h < H_in && cur_w >= 0 && cur_w < W_in);
        
        // Pointer to global memory for loading
        const half* x_ptr = x + n * H_in * W_in * 3 + cur_h * W_in * 3 + cur_w * 3;

        if (is_valid) {
            s_input[sh_h][sh_w][0] = x_ptr[0];
            s_input[sh_h][sh_w][1] = x_ptr[1];
            s_input[sh_h][sh_w][2] = x_ptr[2];
        } else {
            s_input[sh_h][sh_w][0] = __float2half(0.0f);
            s_input[sh_h][sh_w][1] = __float2half(0.0f);
            s_input[sh_h][sh_w][2] = __float2half(0.0f);
        }
    }
    __syncthreads();

    // --- Compute Convolution ---
    // Each thread computes 1 output pixel for all BLOCK_C channels.
    float acc[BLOCK_C] = {0.0f};
    
    // Iterate over the 3x3 kernel
    for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
            // Read input from shared memory
            const half s_in_r = s_input[thread_h * 2 + kh][thread_w * 2 + kw][0];
            const half s_in_g = s_input[thread_h * 2 + kh][thread_w * 2 + kw][1];
            const half s_in_b = s_input[thread_h * 2 + kh][thread_w * 2 + kw][2];
            
            #pragma unroll
            for (int c_out = 0; c_out < BLOCK_C; ++c_out) {
                // Weight layout is (C_out, KH, KW, C_in=3)
                // The original code had a bug here, assuming C_in=3, KH=3, KW=3, so 27 values per output channel
                const int w_base_idx = c_out * 9 + kh * 3 + kw;
                const half w_r = w[w_base_idx * 3 + 0];
                const half w_g = w[w_base_idx * 3 + 1];
                const half w_b = w[w_base_idx * 3 + 2];
                acc[c_out] += __half2float(s_in_r) * __half2float(w_r) +
                              __half2float(s_in_g) * __half2float(w_g) +
                              __half2float(s_in_b) * __half2float(w_b);
            }
        }
    }

    // --- Store Output with Fused BN+ReLU ---
    const int h_out = block_h * BLOCK_H + thread_h;
    const int w_out = block_w * BLOCK_W + thread_w;

    if (h_out < H_out && w_out < W_out) {
        half* y_ptr = y + n * H_out * W_out * C_out + h_out * W_out * C_out + w_out * C_out;
        #pragma unroll
        for (int c_out = 0; c_out < BLOCK_C; ++c_out) {
            float val = acc[c_out] + __half2float(bn_bias[c_out]);
            val = fmaxf(0.f, val);
            y_ptr[c_out] = __float2half(val);
        }
    }
}

torch::Tensor initial_conv_cuda_forward(
    torch::Tensor x, torch::Tensor w, torch::Tensor bn_bias
) {
    CHECK_INPUT(x); CHECK_INPUT(w); CHECK_INPUT(bn_bias);
    TORCH_CHECK(x.dtype() == torch::kFloat16, "Input x must be FP16");

    const int N = x.size(0);
    const int H_in = x.size(1);
    const int W_in = x.size(2);
    const int C_out = w.size(0);

    const int H_out = (H_in + 2 * 1 - 3) / 2 + 1;
    const int W_out = (W_in + 2 * 1 - 3) / 2 + 1;

    auto y = torch::empty({N, H_out, W_out, C_out}, x.options());

    dim3 threads(THREADS_X, THREADS_Y);
    dim3 blocks((W_out + BLOCK_W - 1) / BLOCK_W, (H_out + BLOCK_H - 1) / BLOCK_H, N);
    
    initial_conv_bn_relu_nhwc_cuda_kernel<<<blocks, threads>>>(
        (const half*)x.data_ptr<at::Half>(), (const half*)w.data_ptr<at::Half>(), (const half*)bn_bias.data_ptr<at::Half>(), (half*)y.data_ptr<at::Half>(),
        H_in, W_in, C_out, H_out, W_out
    );
    return y;
}
"""

# ==============================================================================
# 2. CUDA C++ Kernel for Fused 3x3 Depthwise Conv + BN + ReLU6 (NHWC, FP16)
# ==============================================================================
# This kernel replaces the Triton DW-conv kernel. It also uses a tiled algorithm
# with shared memory to avoid redundant global memory loads, which is the main
# bottleneck for depthwise convolutions.
dwconv_cuda_source = """
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tiling configuration for DW conv
constexpr int DW_BLOCK_H = 16;
constexpr int DW_BLOCK_W = 16;
constexpr int DW_CHANNELS_PER_THREAD = 4; // Each thread handles 4 channels
constexpr int DW_THREADS_X = 4;           // Reduced from 16 to limit shared memory
constexpr int DW_THREADS_Y = 16;          // Increased from 4 to maintain 64 threads/block

// Shared memory padding
#define PADDED_DW(W) (W + 8)

template <int stride>
__global__ void dwconv3x3_bn_relu6_nhwc_cuda_kernel(
    const half* __restrict__ x,
    const half* __restrict__ w,
    const half* __restrict__ bn_scale,
    const half* __restrict__ bn_bias,
    half* __restrict__ y,
    int H_in, int W_in, int C, int H_out, int W_out
) {
    // --- Shared memory allocation ---
    constexpr int SHMEM_H = (DW_BLOCK_H - 1) * stride + 3;
    constexpr int SHMEM_W = (DW_BLOCK_W - 1) * stride + 3;
    extern __shared__ half s_data[]; // Dynamic shared memory
    half (*s_input)[PADDED_DW(SHMEM_W)] = (half (*)[PADDED_DW(SHMEM_W)])s_data;
    
    // --- Block and Thread Indexing ---
    const int n = blockIdx.z;
    const int c_group = blockIdx.y;
    const int hw_group = blockIdx.x;
    
    const int num_w_blocks = (W_out + DW_BLOCK_W - 1) / DW_BLOCK_W;
    const int block_h = hw_group / num_w_blocks;
    const int block_w = hw_group % num_w_blocks;

    const int thread_h = threadIdx.y;
    const int thread_w_base = threadIdx.x;
    const int c_start = c_group * DW_THREADS_X * DW_CHANNELS_PER_THREAD;

    // --- Cooperative Loading into Shared Memory ---
    const int h_in_start = block_h * DW_BLOCK_H * stride - 1; // padding=1
    const int w_in_start = block_w * DW_BLOCK_W * stride - 1; // padding=1
    
    const int load_c_base = c_start + thread_w_base * DW_CHANNELS_PER_THREAD;
    if (load_c_base < C) {
      for (int i = thread_h; i < SHMEM_H; i += DW_THREADS_Y) {
          int cur_h = h_in_start + i;
          for (int j = 0; j < SHMEM_W; ++j) { // No parallelism here, simple loop
              int cur_w = w_in_start + j;
              bool is_valid = (cur_h >= 0 && cur_h < H_in && cur_w >= 0 && cur_w < W_in);
              
              #pragma unroll
              for(int c_off = 0; c_off < DW_CHANNELS_PER_THREAD; ++c_off) {
                  int current_c = load_c_base + c_off;
                  if (current_c < C) {
                    const half* x_ptr = x + n*H_in*W_in*C + cur_h*W_in*C + cur_w*C + current_c;
                    int shmem_idx = current_c - c_start;
                    if (is_valid) {
                        s_input[i * (DW_THREADS_X * DW_CHANNELS_PER_THREAD) + shmem_idx][j] = *x_ptr;
                    } else {
                        s_input[i * (DW_THREADS_X * DW_CHANNELS_PER_THREAD) + shmem_idx][j] = __float2half(0.0f);
                    }
                  }
              }
          }
      }
    }
    __syncthreads();

    // --- Compute Convolution ---
    // Each thread now computes a 1x4 tile of output pixels.
    if (thread_h < DW_BLOCK_H) {
      for (int w_offset = 0; w_offset < 4; ++w_offset) {
          const int thread_w = thread_w_base * 4 + w_offset;
          if (thread_w < DW_BLOCK_W) {
              const int h_out = block_h * DW_BLOCK_H + thread_h;
              const int w_out = block_w * DW_BLOCK_W + thread_w;
              
              if (h_out < H_out && w_out < W_out) {
                  const int c_compute = c_start;
                  if(c_compute < C) {
                      half* y_ptr = y + n*H_out*W_out*C + h_out*W_out*C + w_out*C + c_compute;
                      #pragma unroll
                      for (int c_off_thread = 0; c_off_thread < DW_THREADS_X * DW_CHANNELS_PER_THREAD; c_off_thread += DW_CHANNELS_PER_THREAD) {
                          float acc[DW_CHANNELS_PER_THREAD] = {0.0f};
                          #pragma unroll
                          for (int kh = 0; kh < 3; ++kh) {
                              #pragma unroll
                              for (int kw = 0; kw < 3; ++kw) {
                                  #pragma unroll
                                  for (int c_off = 0; c_off < DW_CHANNELS_PER_THREAD; ++c_off) {
                                      int current_c = c_compute + c_off_thread + c_off;
                                      if (current_c < C) {
                                        int shmem_idx = c_off_thread + c_off;
                                        half s_val = s_input[(thread_h * stride + kh) * (DW_THREADS_X * DW_CHANNELS_PER_THREAD) + shmem_idx][thread_w * stride + kw];
                                        half w_val = w[(kh * 3 + kw) * C + current_c];
                                        acc[c_off] += __half2float(s_val) * __half2float(w_val);
                                      }
                                  }
                              }
                          }
                          
                          #pragma unroll
                          for (int c_off = 0; c_off < DW_CHANNELS_PER_THREAD; ++c_off) {
                              int current_c = c_compute + c_off_thread + c_off;
                              if (current_c < C) {
                                float val = acc[c_off];
                                val = val * __half2float(bn_scale[current_c]) + __half2float(bn_bias[current_c]);
                                val = fmaxf(0.f, val);
                                val = fminf(6.f, val);
                                y_ptr[c_off_thread + c_off] = __float2half(val);
                              }
                          }
                      }
                  }
              }
          }
      }
    }
}


torch::Tensor dwconv_cuda_forward(
    torch::Tensor x, torch::Tensor w, torch::Tensor bn_scale, torch::Tensor bn_bias, int stride
) {
    CHECK_INPUT(x); CHECK_INPUT(w); CHECK_INPUT(bn_scale); CHECK_INPUT(bn_bias);
    TORCH_CHECK(x.dtype() == torch::kFloat16, "Input x must be FP16");

    const int N = x.size(0);
    const int H_in = x.size(1);
    const int W_in = x.size(2);
    const int C = x.size(3);

    const int H_out = (H_in + 2 * 1 - 3) / stride + 1;
    const int W_out = (W_in + 2 * 1 - 3) / stride + 1;
    auto y = torch::empty({N, H_out, W_out, C}, x.options());
    
    const int channels_per_block = DW_THREADS_X * DW_CHANNELS_PER_THREAD;
    dim3 threads(DW_THREADS_X, DW_THREADS_Y);
    dim3 blocks(
        ((H_out + DW_BLOCK_H - 1) / DW_BLOCK_H) * ((W_out + DW_BLOCK_W - 1) / DW_BLOCK_W),
        (C + channels_per_block - 1) / channels_per_block,
        N
    );
    
    constexpr int SHMEM_H = (DW_BLOCK_H - 1) * 2 + 3; // Max stride is 2
    constexpr int SHMEM_W = (DW_BLOCK_W - 1) * 2 + 3;
    size_t shmem_size = channels_per_block * SHMEM_H * PADDED_DW(SHMEM_W) * sizeof(half);

    if (stride == 1) {
        dwconv3x3_bn_relu6_nhwc_cuda_kernel<1><<<blocks, threads, shmem_size>>>(
            (const half*)x.data_ptr<at::Half>(), (const half*)w.data_ptr<at::Half>(), (const half*)bn_scale.data_ptr<at::Half>(), (const half*)bn_bias.data_ptr<at::Half>(), (half*)y.data_ptr<at::Half>(),
            H_in, W_in, C, H_out, W_out);
    } else { // stride == 2
        dwconv3x3_bn_relu6_nhwc_cuda_kernel<2><<<blocks, threads, shmem_size>>>(
            (const half*)x.data_ptr<at::Half>(), (const half*)w.data_ptr<at::Half>(), (const half*)bn_scale.data_ptr<at::Half>(), (const half*)bn_bias.data_ptr<at::Half>(), (half*)y.data_ptr<at::Half>(),
            H_in, W_in, C, H_out, W_out);
    }
    return y;
}
"""

cuda_kernels = load_inline(
    name="effnet_cuda_kernels",
    cpp_sources=[
        "torch::Tensor initial_conv_cuda_forward(torch::Tensor, torch::Tensor, torch::Tensor);",
        "torch::Tensor dwconv_cuda_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int);"
    ],
    cuda_sources=[initial_conv_cuda_source, dwconv_cuda_source],
    functions=["initial_conv_cuda_forward", "dwconv_cuda_forward"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math"],
    verbose=False
)

# ==============================================================================
# 3. Triton Kernels for GEMM and Head (Unchanged from previous solution)
# ==============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bn_relu6_nhwc_kernel(
    A, W, B, C, M, N, K,
    stride_am, stride_ak, stride_wk, stride_wn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Simplified PID calculation to fix potential out-of-bounds access from complex grouped scheduling
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = W + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_wk
    bias = tl.load(B + offs_bn, mask=offs_bn < N, other=0.0)
    result = accumulator + bias[None, :]
    result = tl.maximum(0, result)
    result = tl.minimum(6, result)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, result.to(C.dtype.element_ty), mask=c_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_HW': 32}, num_warps=4),
    ],
    key=['C_out', 'C_in', 'H', 'W'],
)
@triton.jit
def fused_head_nhwc_kernel(
    x_ptr, w_ptr, bias_ptr, output_ptr,
    N, H, W, C_in, C_out,
    x_stride_n, x_stride_h, x_stride_w, x_stride_c,
    w_stride_k, w_stride_n,
    out_stride_n, out_stride_c,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_HW: tl.constexpr
):
    pid_batch = tl.program_id(axis=0)
    pid_n_group = tl.program_id(axis=1)
    x_ptr += pid_batch * x_stride_n
    output_ptr += pid_batch * out_stride_n
    offs_n = pid_n_group * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < C_out
    pool_accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for hw_start in range(0, H * W, BLOCK_SIZE_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_SIZE_HW)
        h = offs_hw // W
        w = offs_hw % W
        hw_mask = offs_hw < H * W
        tile_acc = tl.zeros((BLOCK_SIZE_HW, BLOCK_SIZE_N), dtype=tl.float32)
        for k_start in range(0, C_in, BLOCK_SIZE_K):
            offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = offs_k < C_in
            a_ptrs = x_ptr + (h[:, None] * x_stride_h + w[:, None] * x_stride_w + offs_k[None, :] * x_stride_c)
            a = tl.load(a_ptrs, mask=hw_mask[:, None] & k_mask[None, :], other=0.0)
            b_ptrs = w_ptr + (offs_k[:, None] * w_stride_k + offs_n[None, :] * w_stride_n)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            tile_acc += tl.dot(a, b)
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        tile_result = tl.maximum(tile_acc + bias[None, :], 0)
        tile_result = tl.where(hw_mask[:, None], tile_result, 0.0)
        pool_accumulator += tl.sum(tile_result, axis=0)
    avg_accumulator = pool_accumulator / (H * W)
    output_ptrs = output_ptr + offs_n * out_stride_c
    tl.store(output_ptrs, avg_accumulator.to(output_ptr.dtype.element_ty), mask=n_mask)

# ==============================================================================
# 4. Fused PyTorch Modules and Wrappers
# ==============================================================================
class FusedInitialConvBNReLU_CUDA(nn.Module):
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        super().__init__()
        bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bn_bias = bn.bias - bn.running_mean * bn_scale
        fused_weight = conv.weight * bn_scale.view(-1, 1, 1, 1)
        # (C_out, C_in, KH, KW) -> (C_out, KH, KW, C_in) for easier indexing in CUDA
        self.register_buffer('weight', fused_weight.permute(0, 2, 3, 1).contiguous())
        self.register_buffer('bias', bn_bias.contiguous())
    
    def forward(self, x):
        return cuda_kernels.initial_conv_cuda_forward(x, self.weight, self.bias)

class FusedDepthwiseConvBNReLU_CUDA(nn.Module):
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        super().__init__()
        self.stride = conv.stride[0]
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bias = bn.bias - bn.running_mean * scale
        # (C, 1, KH, KW) -> (KH, KW, C)
        self.register_buffer('conv_weight', conv.weight.squeeze(1).permute(1, 2, 0).contiguous())
        self.register_buffer('bn_scale', scale.contiguous())
        self.register_buffer('bn_bias', bias.contiguous())

    def forward(self, x):
        return cuda_kernels.dwconv_cuda_forward(x, self.conv_weight, self.bn_scale, self.bn_bias, self.stride)

class FusedGemmBNReLU6_Triton(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.register_buffer('weight', weight.contiguous())
        self.register_buffer('bias', bias.contiguous())
    
    def forward(self, x):
        N, H, W, C_in = x.shape
        x_reshaped = x.view(-1, C_in) # Use view for efficiency
        M, K = x_reshaped.shape
        C_out = self.weight.shape[1]
        Y = torch.empty((M, C_out), device=x.device, dtype=x.dtype)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(C_out, META['BLOCK_SIZE_N']),)
        conv1x1_bn_relu6_nhwc_kernel[grid](
            x_reshaped, self.weight, self.bias, Y, M, C_out, K,
            x_reshaped.stride(0), x_reshaped.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            Y.stride(0), Y.stride(1)
        )
        return Y.view(N, H, W, C_out)

class FusedSuperHead_Triton(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.register_buffer('weight', weight.contiguous())
        self.register_buffer('bias', bias.contiguous())
    
    def forward(self, x):
        N, H, W, C_in = x.shape
        C_out = self.weight.shape[1]
        output = torch.empty((N, C_out), device=x.device, dtype=x.dtype)
        grid = lambda META: (N, triton.cdiv(C_out, META['BLOCK_SIZE_N']))
        fused_head_nhwc_kernel[grid](
            x, self.weight, self.bias, output,
            N, H, W, C_in, C_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(0), self.weight.stride(1),
            output.stride(0), output.stride(1))
        return output

# ==============================================================================
# 5. Final Optimized Model with Hybrid Kernels, FP16, and CUDA Graphs
# ==============================================================================
class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        # Use a dummy model to get correctly initialized weights
        original_model = self._get_original_model(num_classes)
        
        def _fuse_gemm_bn(conv, bn):
            bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            bn_bias = bn.bias - bn.running_mean * bn_scale
            fused_weight = conv.weight.squeeze() * bn_scale.view(-1, 1)
            return fused_weight, bn_bias

        # Build fused model using the new CUDA kernels for convs
        self.initial_block = FusedInitialConvBNReLU_CUDA(original_model.conv1, original_model.bn1)
        
        mbconv1 = original_model.mbconv1
        w_exp1, b_exp1 = _fuse_gemm_bn(mbconv1[0], mbconv1[1])
        self.mbconv1_exp = FusedGemmBNReLU6_Triton(w_exp1.T, b_exp1)
        self.mbconv1_dw = FusedDepthwiseConvBNReLU_CUDA(mbconv1[3], mbconv1[4])
        
        all_mbconvs = [
            original_model.mbconv1, original_model.mbconv2, original_model.mbconv3,
            original_model.mbconv4, original_model.mbconv5, original_model.mbconv6,
            original_model.mbconv7
        ]
        
        self.fused_blocks = nn.ModuleList()
        for i in range(len(all_mbconvs) - 1):
            # Project from block i, expand for block i+1
            proj_conv = all_mbconvs[i][6]
            proj_bn = all_mbconvs[i][7]
            exp_conv = all_mbconvs[i+1][0]
            exp_bn = all_mbconvs[i+1][1]
            dw_conv = all_mbconvs[i+1][3]
            dw_bn = all_mbconvs[i+1][4]

            w_proj, b_proj = _fuse_gemm_bn(proj_conv, proj_bn)
            w_exp, b_exp = _fuse_gemm_bn(exp_conv, exp_bn)

            w_final = torch.mm(w_exp, w_proj)
            b_final = torch.mv(w_exp, b_proj) + b_exp

            fused_layer = FusedGemmBNReLU6_Triton(w_final.T, b_final)
            dw_layer = FusedDepthwiseConvBNReLU_CUDA(dw_conv, dw_bn)
            self.fused_blocks.append(nn.Sequential(fused_layer, dw_layer))

        w_proj_last, b_proj_last = _fuse_gemm_bn(all_mbconvs[-1][6], all_mbconvs[-1][7])
        w_head, b_head = _fuse_gemm_bn(original_model.conv2, original_model.bn2)
        w_final_head = torch.mm(w_head, w_proj_last)
        b_final_head = torch.mv(w_head, b_proj_last) + b_head
        self.super_head = FusedSuperHead_Triton(w_final_head.T, b_final_head)
        
        self.fc = original_model.fc
        
        self.half()
        self.eval() # Set to eval mode for optimizations
        self.graph = None
        self.static_input = None
        self.static_output = None

    def _get_original_model(self, num_classes):
        # Helper to create an instance of the original architecture
        class OriginalModel(nn.Module):
            def __init__(self, num_classes=1000):
                super(OriginalModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(32)
                self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
                self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
                self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
                self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
                self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
                self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
                self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)
                self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
                self.bn2 = nn.BatchNorm2d(1280)
                self.fc = nn.Linear(1280, num_classes)
            def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
                hidden_dim = round(in_channels * expand_ratio)
                return nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
                    nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels))
        return OriginalModel(num_classes)

    def _forward_impl(self, x):
        x = self.initial_block(x)
        x = self.mbconv1_dw(self.mbconv1_exp(x))
        
        for block in self.fused_blocks:
            x = block(x)
        
        x = self.super_head(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        if x.dtype != torch.float16:
            x = x.half()
        # NHWC input format
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.training:
            return self._forward_impl(x).float()
        
        # CUDA Graph capture for inference
        if self.graph is None or x.shape != self.static_input.shape:
            self.static_input = x.clone()
            
            # Warmup
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._forward_impl(self.static_input)
            torch.cuda.current_stream().wait_stream(s)
            
            # Capture
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self._forward_impl(self.static_input)
        
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone().float()

# ==============================================================================
# 6. Boilerplate for Testing
# ==============================================================================
batch_size = 10
input_shape = (3, 240, 240)
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, *input_shape)] 

def get_init_inputs():
    return [num_classes]