# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for a fused Fire module expand operation.
# This solution synthesizes the best features from prior top-performing attempts.
#
# --- Key Optimizations ---
# 1. Full Kernel Fusion (Retained): Fuses the two expand convolutions, ReLUs, and concat operation.
# 2. Tiling & Shared Memory Caching (Retained): Caches input tiles and all weights/biases in shared memory.
# 3. Vectorized `float4` Computation (Retained): Processes 4 output channels simultaneously.
# 4. Transposed Weights for Coalesced Access (Retained from Program 1): Weights are pre-transposed in
#    PyTorch, enabling fully coalesced `float4` loads from shared memory, which is the most critical optimization.
# 5. Loop Unrolling (NEW SYNTHESIS): Unrolls the main output channel loop by a factor of 2. Each thread
#    computes two `float4` vectors (8 output channels total) per iteration, increasing ILP.
# 6. Fused Multiply-Add (FMA) (NEW SYNTHESIS): Uses `fmaf` for accumulation, mapping to faster hardware instructions.
fused_fire_expand_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_H 8
#define TILE_W 32
#define PADDED_H (TILE_H + 2) // +1 halo on each side for 3x3 kernel
#define PADDED_W (TILE_W + 2)

__global__ void fused_fire_expand_kernel_coalesced_unrolled(
    // Inputs
    const float* __restrict__ squeeze_out,
    // Note: Weights are pre-transposed for coalesced access
    const float* __restrict__ expand1x1_weight_T, const float* __restrict__ expand1x1_bias,
    const float* __restrict__ expand3x3_weight_T, const float* __restrict__ expand3x3_bias,
    // Output
    float* __restrict__ out,
    // Tensor dimensions
    int N, int SQUEEZE_C, int H, int W,
    int EXPAND1x1_C, int EXPAND3x3_C
) {
    // Use dynamic shared memory to cache both the input tile and all weights/biases.
    extern __shared__ float s_mem[];
    
    // --- Shared Memory Partitioning ---
    const int s_input_size = SQUEEZE_C * PADDED_H * PADDED_W;
    const int s_1x1_w_size = EXPAND1x1_C * SQUEEZE_C;
    const int s_1x1_b_size = EXPAND1x1_C;
    const int s_3x3_w_size = EXPAND3x3_C * SQUEEZE_C * 9;

    float* s_squeeze_in = s_mem;
    float* s_expand1x1_w = s_squeeze_in + s_input_size;
    float* s_expand1x1_b = s_expand1x1_w + s_1x1_w_size;
    float* s_expand3x3_w = s_expand1x1_b + s_1x1_b_size;
    float* s_expand3x3_b = s_expand3x3_w + s_3x3_w_size;

    // --- Cooperative Loading into Shared Memory ---
    const int tid = threadIdx.x + threadIdx.y * TILE_W;
    const int num_threads = TILE_H * TILE_W;
    const int block_base_h = blockIdx.y * TILE_H;
    const int block_base_w = blockIdx.x * TILE_W;
    const int n = blockIdx.z;

    // 1. Load weights and biases
    for (int i = tid; i < s_1x1_w_size; i += num_threads) s_expand1x1_w[i] = expand1x1_weight_T[i];
    for (int i = tid; i < s_1x1_b_size; i += num_threads) s_expand1x1_b[i] = expand1x1_bias[i];
    for (int i = tid; i < s_3x3_w_size; i += num_threads) s_expand3x3_w[i] = expand3x3_weight_T[i];
    for (int i = tid; i < EXPAND3x3_C; i += num_threads) s_expand3x3_b[i] = expand3x3_bias[i];

    // 2. Load input tile from global memory
    for (int i = tid; i < s_input_size; i += num_threads) {
        const int c = i / (PADDED_H * PADDED_W);
        const int remainder = i % (PADDED_H * PADDED_W);
        const int ph = remainder / PADDED_W;
        const int pw = remainder % PADDED_W;
        const int h_in = block_base_h + ph - 1;
        const int w_in = block_base_w + pw - 1;

        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            s_squeeze_in[i] = squeeze_out[n * SQUEEZE_C * H * W + c * H * W + h_in * W + w_in];
        } else {
            s_squeeze_in[i] = 0.0f;
        }
    }
    
    __syncthreads();

    // --- Computation ---
    const int w_out = block_base_w + threadIdx.x;
    const int h_out = block_base_h + threadIdx.y;

    if (w_out >= W || h_out >= H) {
        return;
    }

    const int TOTAL_EXPAND_C = EXPAND1x1_C + EXPAND3x3_C;
    const int sh = threadIdx.y + 1;
    const int sw = threadIdx.x + 1;
    const int channel_stride = H * W;
    const int out_base_idx = n * TOTAL_EXPAND_C * H * W + h_out * W + w_out;

    // --- 1x1 Expansion (Vectorized by 4, Unrolled by 2 -> 8 output channels per iter) ---
    for (int out_c_base = 0; out_c_base < EXPAND1x1_C; out_c_base += 8) {
        float4 acc0 = *reinterpret_cast<const float4*>(&s_expand1x1_b[out_c_base + 0]);
        float4 acc1 = *reinterpret_cast<const float4*>(&s_expand1x1_b[out_c_base + 4]);

        for (int in_c = 0; in_c < SQUEEZE_C; ++in_c) {
            const float pixel_val = s_squeeze_in[in_c * PADDED_H * PADDED_W + sh * PADDED_W + sw];
            const float4 w0 = *reinterpret_cast<const float4*>(&s_expand1x1_w[in_c * EXPAND1x1_C + out_c_base + 0]);
            const float4 w1 = *reinterpret_cast<const float4*>(&s_expand1x1_w[in_c * EXPAND1x1_C + out_c_base + 4]);
            
            acc0.x = fmaf(pixel_val, w0.x, acc0.x); acc0.y = fmaf(pixel_val, w0.y, acc0.y);
            acc0.z = fmaf(pixel_val, w0.z, acc0.z); acc0.w = fmaf(pixel_val, w0.w, acc0.w);
            acc1.x = fmaf(pixel_val, w1.x, acc1.x); acc1.y = fmaf(pixel_val, w1.y, acc1.y);
            acc1.z = fmaf(pixel_val, w1.z, acc1.z); acc1.w = fmaf(pixel_val, w1.w, acc1.w);
        }
        acc0.x = fmaxf(0.0f, acc0.x); acc0.y = fmaxf(0.0f, acc0.y); acc0.z = fmaxf(0.0f, acc0.z); acc0.w = fmaxf(0.0f, acc0.w);
        acc1.x = fmaxf(0.0f, acc1.x); acc1.y = fmaxf(0.0f, acc1.y); acc1.z = fmaxf(0.0f, acc1.z); acc1.w = fmaxf(0.0f, acc1.w);

        out[out_base_idx + (out_c_base + 0) * channel_stride] = acc0.x; out[out_base_idx + (out_c_base + 1) * channel_stride] = acc0.y;
        out[out_base_idx + (out_c_base + 2) * channel_stride] = acc0.z; out[out_base_idx + (out_c_base + 3) * channel_stride] = acc0.w;
        out[out_base_idx + (out_c_base + 4) * channel_stride] = acc1.x; out[out_base_idx + (out_c_base + 5) * channel_stride] = acc1.y;
        out[out_base_idx + (out_c_base + 6) * channel_stride] = acc1.z; out[out_base_idx + (out_c_base + 7) * channel_stride] = acc1.w;
    }

    // --- 3x3 Expansion (Vectorized by 4, Unrolled by 2 -> 8 output channels per iter) ---
    for (int out_c_base_3x3 = 0; out_c_base_3x3 < EXPAND3x3_C; out_c_base_3x3 += 8) {
        float4 acc0 = *reinterpret_cast<const float4*>(&s_expand3x3_b[out_c_base_3x3 + 0]);
        float4 acc1 = *reinterpret_cast<const float4*>(&s_expand3x3_b[out_c_base_3x3 + 4]);
        
        for (int in_c = 0; in_c < SQUEEZE_C; ++in_c) {
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < 3; ++kw) {
                    const float pixel_val = s_squeeze_in[in_c * PADDED_H * PADDED_W + (sh - 1 + kh) * PADDED_W + (sw - 1 + kw)];
                    const int kernel_offset = kh * 3 + kw;
                    const float4 w0 = *reinterpret_cast<const float4*>(&s_expand3x3_w[(in_c * 9 + kernel_offset) * EXPAND3x3_C + out_c_base_3x3 + 0]);
                    const float4 w1 = *reinterpret_cast<const float4*>(&s_expand3x3_w[(in_c * 9 + kernel_offset) * EXPAND3x3_C + out_c_base_3x3 + 4]);

                    acc0.x = fmaf(pixel_val, w0.x, acc0.x); acc0.y = fmaf(pixel_val, w0.y, acc0.y);
                    acc0.z = fmaf(pixel_val, w0.z, acc0.z); acc0.w = fmaf(pixel_val, w0.w, acc0.w);
                    acc1.x = fmaf(pixel_val, w1.x, acc1.x); acc1.y = fmaf(pixel_val, w1.y, acc1.y);
                    acc1.z = fmaf(pixel_val, w1.z, acc1.z); acc1.w = fmaf(pixel_val, w1.w, acc1.w);
                }
            }
        }
        acc0.x = fmaxf(0.0f, acc0.x); acc0.y = fmaxf(0.0f, acc0.y); acc0.z = fmaxf(0.0f, acc0.z); acc0.w = fmaxf(0.0f, acc0.w);
        acc1.x = fmaxf(0.0f, acc1.x); acc1.y = fmaxf(0.0f, acc1.y); acc1.z = fmaxf(0.0f, acc1.z); acc1.w = fmaxf(0.0f, acc1.w);

        const int out_c_offset = EXPAND1x1_C + out_c_base_3x3;
        out[out_base_idx + (out_c_offset + 0) * channel_stride] = acc0.x; out[out_base_idx + (out_c_offset + 1) * channel_stride] = acc0.y;
        out[out_base_idx + (out_c_offset + 2) * channel_stride] = acc0.z; out[out_base_idx + (out_c_offset + 3) * channel_stride] = acc0.w;
        out[out_base_idx + (out_c_offset + 4) * channel_stride] = acc1.x; out[out_base_idx + (out_c_offset + 5) * channel_stride] = acc1.y;
        out[out_base_idx + (out_c_offset + 6) * channel_stride] = acc1.z; out[out_base_idx + (out_c_offset + 7) * channel_stride] = acc1.w;
    }
}

torch::Tensor fused_fire_expand_coalesced_unrolled_forward(
    torch::Tensor squeeze_out,
    torch::Tensor expand1x1_weight_T, torch::Tensor expand1x1_bias,
    torch::Tensor expand3x3_weight_T, torch::Tensor expand3x3_bias
) {
    const auto N = squeeze_out.size(0);
    const auto SQUEEZE_C = squeeze_out.size(1);
    const auto H = squeeze_out.size(2);
    const auto W = squeeze_out.size(3);

    const auto EXPAND1x1_C = expand1x1_bias.size(0);
    const auto EXPAND3x3_C = expand3x3_bias.size(0);
    
    TORCH_CHECK(EXPAND1x1_C % 8 == 0, "expand1x1_channels must be divisible by 8 for the coalesced+unrolled kernel.");
    TORCH_CHECK(EXPAND3x3_C % 8 == 0, "expand3x3_channels must be divisible by 8 for the coalesced+unrolled kernel.");

    const auto TOTAL_EXPAND_C = EXPAND1x1_C + EXPAND3x3_C;
    auto out = torch::empty({N, TOTAL_EXPAND_C, H, W}, squeeze_out.options());
    
    const size_t input_tile_smem_elems = SQUEEZE_C * PADDED_H * PADDED_W;
    const size_t weights_smem_elems = expand1x1_weight_T.numel() + expand1x1_bias.numel() +
                                      expand3x3_weight_T.numel() + expand3x3_bias.numel();
    const size_t shared_mem_size = (input_tile_smem_elems + weights_smem_elems) * sizeof(float);

    dim3 block_size(TILE_W, TILE_H, 1);
    dim3 grid_size((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H, N);

    fused_fire_expand_kernel_coalesced_unrolled<<<grid_size, block_size, shared_mem_size>>>(
        squeeze_out.data_ptr<float>(),
        expand1x1_weight_T.data_ptr<float>(), expand1x1_bias.data_ptr<float>(),
        expand3x3_weight_T.data_ptr<float>(), expand3x3_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, SQUEEZE_C, H, W,
        EXPAND1x1_C, EXPAND3x3_C
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return out;
}
"""

fused_fire_expand_cpp_source = """
torch::Tensor fused_fire_expand_coalesced_unrolled_forward(
    torch::Tensor squeeze_out,
    torch::Tensor expand1x1_weight_T, torch::Tensor expand1x1_bias,
    torch::Tensor expand3x3_weight_T, torch::Tensor expand3x3_bias
);
"""

# JIT compile the CUDA kernel
fused_fire_op = load_inline(
    name="fused_fire_op_coalesced_unrolled",
    cpp_sources=fused_fire_expand_cpp_source,
    cuda_sources=fused_fire_expand_source,
    functions=["fused_fire_expand_coalesced_unrolled_forward"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Model, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        squeeze_out = self.squeeze_activation(self.squeeze(x))
        
        # Transpose weights for coalesced memory access in the CUDA kernel
        # Original shape: (C_out, C_in, K, K)
        # Target shape for 1x1: (C_in, C_out)
        # Target shape for 3x3: (C_in, K*K, C_out)
        
        expand1x1_w = self.expand1x1.weight.squeeze() # (C_out, C_in)
        expand1x1_w_t = expand1x1_w.permute(1, 0).contiguous()

        e3x3_c_out, e3x3_c_in, _, _ = self.expand3x3.weight.shape
        expand3x3_w = self.expand3x3.weight.view(e3x3_c_out, e3x3_c_in, 9) # (C_out, C_in, 9)
        expand3x3_w_t = expand3x3_w.permute(1, 2, 0).contiguous() # (C_in, 9, C_out)
        
        return fused_fire_op.fused_fire_expand_coalesced_unrolled_forward(
            squeeze_out.contiguous(),
            expand1x1_w_t,
            self.expand1x1.bias.contiguous(),
            expand3x3_w_t,
            self.expand3x3.bias.contiguous()
        )

# Test code
batch_size = 10
num_input_features = 3
num_output_features = 64
height, width = 224, 224
squeeze_channels = 6
expand1x1_channels = 64
expand3x3_channels = 64

def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width).cuda()]

def get_init_inputs():
    return [num_input_features, squeeze_channels, expand1x1_channels, expand3x3_channels]
# EVOLVE-BLOCK-END