import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA source for a fully fused Fire module.
# This improved version uses pre-transposed weights. The weights are transposed once
# on the CPU during model initialization and stored in a kernel-friendly layout.
# This allows the kernel to use simple, coalesced memory copies to load weights
# into shared memory, which is significantly faster than the previous on-the-fly
# transposition.
fused_fire_pretransposed_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <math.h>
#include <c10/cuda/CUDAException.h>

// --- Kernel Configuration ---
#define TILE_H 16
#define TILE_W 16
#define PADDED_TILE_H (TILE_H + 2)
#define PADDED_TILE_W (TILE_W + 2)
#define BLOCK_H PADDED_TILE_H
#define BLOCK_W PADDED_TILE_W
#define VEC_SIZE 4          // Using float4 for vectorization
#define UNROLL_FACTOR 4     // Unroll factor for expand loops

// A device function for ReLU on a float4 vector
__device__ inline float4 relu4(float4 v) {
    v.x = fmaxf(0.0f, v.x);
    v.y = fmaxf(0.0f, v.y);
    v.z = fmaxf(0.0f, v.z);
    v.w = fmaxf(0.0f, v.w);
    return v;
}
// Scalar ReLU
__device__ inline float relu(float x) {
    return fmaxf(0.0f, x);
}

// Helper function to write a float4 to non-contiguous channel locations in NCHW layout.
__device__ inline void write_output_vectorized(float* base_pixel_ptr, int stride, int channel_offset, float4 value) {
    base_pixel_ptr[(channel_offset + 0) * stride] = value.x;
    base_pixel_ptr[(channel_offset + 1) * stride] = value.y;
    base_pixel_ptr[(channel_offset + 2) * stride] = value.z;
    base_pixel_ptr[(channel_offset + 3) * stride] = value.w;
}

__global__ void fused_fire_pretransposed_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ squeeze_w_global, const float* __restrict__ squeeze_b_global,
    const float* __restrict__ expand1x1_w_T_global, const float* __restrict__ expand1x1_b_global,
    const float* __restrict__ expand3x3_w_T_global, const float* __restrict__ expand3x3_b_global,
    int N, int C_in, int H, int W,
    int C_squeeze, int C_expand1x1, int C_expand3x3) {

    // --- Shared Memory Declaration ---
    extern __shared__ char s_mem_char[];
    float* s_mem = (float*)s_mem_char;

    // Pointers to different sections of shared memory
    float* s_squeeze_w = s_mem;
    float* s_squeeze_b = s_squeeze_w + C_squeeze * C_in;
    float* s_expand1x1_w = s_squeeze_b + C_squeeze;
    float* s_expand1x1_b = s_expand1x1_w + C_squeeze * C_expand1x1;
    float* s_expand3x3_w = s_expand1x1_b + C_expand1x1;
    float* s_expand3x3_b = s_expand3x3_w + C_squeeze * 9 * C_expand3x3;
    float* s_squeeze_result = s_expand3x3_b + C_expand3x3;

    // --- Thread and Block Indexing ---
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;
    const int n = blockIdx.z;

    // --- Part 1: Load ALL weights and biases into Shared Memory ---
    // This is now done with simple, coalesced copies from pre-transposed global memory.

    // Load squeeze weights & biases
    for (int i = tid; i < C_squeeze * C_in; i += num_threads) s_squeeze_w[i] = squeeze_w_global[i];
    for (int i = tid; i < C_squeeze; i += num_threads) s_squeeze_b[i] = squeeze_b_global[i];

    // Load expand1x1 weights (pre-transposed) & biases
    for (int i = tid; i < C_squeeze * C_expand1x1; i += num_threads) s_expand1x1_w[i] = expand1x1_w_T_global[i];
    for (int i = tid; i < C_expand1x1; i += num_threads) s_expand1x1_b[i] = expand1x1_b_global[i];

    // Load expand3x3 weights (pre-transposed) & biases
    const int C_sq_k = C_squeeze * 9;
    for (int i = tid; i < C_sq_k * C_expand3x3; i += num_threads) s_expand3x3_w[i] = expand3x3_w_T_global[i];
    for (int i = tid; i < C_expand3x3; i += num_threads) s_expand3x3_b[i] = expand3x3_b_global[i];

    __syncthreads(); // Ensure all weights/biases are loaded

    // --- Part 2: Squeeze Convolution (1x1) + ReLU ---
    const int block_h_start = blockIdx.y * TILE_H;
    const int block_w_start = blockIdx.x * TILE_W;
    int current_h = block_h_start + ty - 1;
    int current_w = block_w_start + tx - 1;
    const float* x_n = x + n * C_in * H * W;

    if (current_h >= 0 && current_h < H && current_w >= 0 && current_w < W) {
        const float* x_pixel_start = x_n + current_h * W + current_w;
        for (int c_sq = 0; c_sq < C_squeeze; ++c_sq) {
            float acc = s_squeeze_b[c_sq];
            for (int c_in = 0; c_in < C_in; ++c_in) {
                acc += x_pixel_start[c_in * H * W] * s_squeeze_w[c_sq * C_in + c_in];
            }
            s_squeeze_result[c_sq * PADDED_TILE_H * PADDED_TILE_W + ty * PADDED_TILE_W + tx] = relu(acc);
        }
    } else {
        for (int c_sq = 0; c_sq < C_squeeze; ++c_sq) {
            s_squeeze_result[c_sq * PADDED_TILE_H * PADDED_TILE_W + ty * PADDED_TILE_W + tx] = 0.0f;
        }
    }

    __syncthreads(); // Ensure squeeze result is written before expand phase

    // --- Part 3: VECTORIZED Expand Convolutions from Shared Memory ---
    if (ty < TILE_H && tx < TILE_W) {
        int global_h = block_h_start + ty;
        int global_w = block_w_start + tx;

        if (global_h < H && global_w < W) {
            float* out_pixel_base = out + n * (C_expand1x1 + C_expand3x3) * H * W + global_h * W + global_w;
            int out_stride = H * W;
            
            constexpr int unroll_channels = VEC_SIZE * UNROLL_FACTOR;

            // --- 3a: Expand 1x1 Branch ---
            for (int i = 0; i < C_expand1x1 / unroll_channels; ++i) {
                const int c_ex1_offset = i * unroll_channels;
                
                float4 acc[UNROLL_FACTOR];
                #pragma unroll
                for(int j=0; j<UNROLL_FACTOR; ++j) acc[j] = *(reinterpret_cast<const float4*>(s_expand1x1_b + c_ex1_offset + j*VEC_SIZE));

                for (int c_sq = 0; c_sq < C_squeeze; ++c_sq) {
                    float s_val = s_squeeze_result[c_sq * PADDED_TILE_H * PADDED_TILE_W + (ty + 1) * PADDED_TILE_W + (tx + 1)];
                    const float4* w_vec_ptr = reinterpret_cast<const float4*>(&s_expand1x1_w[c_sq * C_expand1x1 + c_ex1_offset]);
                    
                    #pragma unroll
                    for(int j=0; j<UNROLL_FACTOR; ++j) {
                        float4 w = w_vec_ptr[j];
                        acc[j].x += s_val * w.x; acc[j].y += s_val * w.y; acc[j].z += s_val * w.z; acc[j].w += s_val * w.w;
                    }
                }

                #pragma unroll
                for(int j=0; j<UNROLL_FACTOR; ++j) write_output_vectorized(out_pixel_base, out_stride, c_ex1_offset + j*VEC_SIZE, relu4(acc[j]));
            }

            // --- 3b: Expand 3x3 Branch ---
            for (int i = 0; i < C_expand3x3 / unroll_channels; ++i) {
                const int c_ex3_offset = i * unroll_channels;
                
                float4 acc[UNROLL_FACTOR];
                #pragma unroll
                for(int j=0; j<UNROLL_FACTOR; ++j) acc[j] = *(reinterpret_cast<const float4*>(s_expand3x3_b + c_ex3_offset + j*VEC_SIZE));
                
                for (int c_sq = 0; c_sq < C_squeeze; ++c_sq) {
                    const float* s_sq_tile_ptr = &s_squeeze_result[c_sq * PADDED_TILE_H * PADDED_TILE_W];
                    const float* w_base_ptr = &s_expand3x3_w[(c_sq * 9) * C_expand3x3];

                    #pragma unroll
                    for (int ky = 0; ky < 3; ++ky) {
                        #pragma unroll
                        for (int kx = 0; kx < 3; ++kx) {
                            float s_val = s_sq_tile_ptr[(ty + ky) * PADDED_TILE_W + (tx + kx)];
                            const float4* w_vec_ptr = reinterpret_cast<const float4*>(&w_base_ptr[(ky*3+kx)*C_expand3x3 + c_ex3_offset]);

                            #pragma unroll
                            for(int j=0; j<UNROLL_FACTOR; ++j) {
                                float4 w = w_vec_ptr[j];
                                acc[j].x += s_val * w.x; acc[j].y += s_val * w.y; acc[j].z += s_val * w.z; acc[j].w += s_val * w.w;
                            }
                        }
                    }
                }

                const int out_offset = C_expand1x1 + c_ex3_offset;
                #pragma unroll
                for(int j=0; j<UNROLL_FACTOR; ++j) write_output_vectorized(out_pixel_base, out_stride, out_offset + j*VEC_SIZE, relu4(acc[j]));
            }
        }
    }
}


torch::Tensor fused_fire_pretransposed_cuda(
    torch::Tensor x,
    torch::Tensor squeeze_w, torch::Tensor squeeze_b,
    torch::Tensor expand1x1_w_T, torch::Tensor expand1x1_b,
    torch::Tensor expand3x3_w_T, torch::Tensor expand3x3_b) {
    
    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    const int C_squeeze = squeeze_b.size(0);
    const int C_expand1x1 = expand1x1_b.size(0);
    const int C_expand3x3 = expand3x3_b.size(0);
    
    constexpr int unroll_channels = VEC_SIZE * UNROLL_FACTOR;
    TORCH_CHECK(C_expand1x1 % unroll_channels == 0, "C_expand1x1 must be divisible by VEC_SIZE * UNROLL_FACTOR (16)");
    TORCH_CHECK(C_expand3x3 % unroll_channels == 0, "C_expand3x3 must be divisible by VEC_SIZE * UNROLL_FACTOR (16)");

    const int C_out = C_expand1x1 + C_expand3x3;
    auto out = torch::empty({N, C_out, H, W}, x.options());
    
    if (out.numel() == 0) return out;

    dim3 block_dim(BLOCK_W, BLOCK_H); 
    dim3 grid_dim((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H, N);
    
    size_t s_squeeze_w_size = C_squeeze * C_in * sizeof(float);
    size_t s_squeeze_b_size = C_squeeze * sizeof(float);
    size_t s_expand1x1_w_size = C_squeeze * C_expand1x1 * sizeof(float);
    size_t s_expand1x1_b_size = C_expand1x1 * sizeof(float);
    size_t s_expand3x3_w_size = C_squeeze * 9 * C_expand3x3 * sizeof(float);
    size_t s_expand3x3_b_size = C_expand3x3 * sizeof(float);
    size_t s_squeeze_result_size = C_squeeze * PADDED_TILE_H * PADDED_TILE_W * sizeof(float);
    
    size_t shared_mem_size = s_squeeze_w_size + s_squeeze_b_size + 
                             s_expand1x1_w_size + s_expand1x1_b_size +
                             s_expand3x3_w_size + s_expand3x3_b_size +
                             s_squeeze_result_size;
    
    // TORCH_CHECK(shared_mem_size < 48 * 1024, "Exceeded shared memory capacity");

    fused_fire_pretransposed_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        squeeze_w.data_ptr<float>(), squeeze_b.data_ptr<float>(),
        expand1x1_w_T.data_ptr<float>(), expand1x1_b.data_ptr<float>(),
        expand3x3_w_T.data_ptr<float>(), expand3x3_b.data_ptr<float>(),
        N, C_in, H, W,
        C_squeeze, C_expand1x1, C_expand3x3
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

fused_fire_pretransposed_cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_fire_pretransposed_cuda(
    torch::Tensor x,
    torch::Tensor squeeze_w, torch::Tensor squeeze_b,
    torch::Tensor expand1x1_w_T, torch::Tensor expand1x1_b,
    torch::Tensor expand3x3_w_T, torch::Tensor expand3x3_b);
"""

# JIT Compile the new "pre-transposed weights" custom CUDA kernel
fused_fire_pretransposed = load_inline(
    name="fused_fire_module_pretransposed",
    cpp_sources=fused_fire_pretransposed_cpp_source,
    cuda_sources=fused_fire_pretransposed_source,
    functions=["fused_fire_pretransposed_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"]
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(ModelNew, self).__init__()
        
        # Standard PyTorch layers to store the original learnable parameters
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        
        # The compiled custom operator
        self.fused_op = fused_fire_pretransposed

        # --- Pre-process and buffer the weights for the custom kernel ---
        # This is done once at initialization.

        # 1. Squeeze weights don't need transposition for the kernel.
        # We just need to flatten them. Shape: (C_squeeze, C_in)
        sq_w = self.squeeze.weight.data.view(squeeze_channels, in_channels)
        self.register_buffer('squeeze_w_flat', sq_w)

        # 2. Transpose expand1x1 weights for coalesced access.
        # Original: (C_ex1, C_sq, 1, 1) -> Target: (C_sq, C_ex1)
        ex1_w = self.expand1x1.weight.data.view(expand1x1_channels, squeeze_channels)
        ex1_w_T = ex1_w.transpose(0, 1).contiguous()
        self.register_buffer('expand1x1_w_T', ex1_w_T)

        # 3. Transpose and reorder expand3x3 weights for coalesced access.
        # Original: (C_ex3, C_sq, 3, 3) -> Target: (C_sq * 9, C_ex3)
        ex3_w = self.expand3x3.weight.data.view(expand3x3_channels, squeeze_channels, 9)
        ex3_w_T = ex3_w.permute(1, 2, 0).contiguous().view(squeeze_channels * 9, expand3x3_channels)
        self.register_buffer('expand3x3_w_T', ex3_w_T)


    def forward(self, x):
        """
        Calls the custom CUDA kernel with pre-transposed weights.
        The weights are accessed from the buffers created during __init__.
        """
        return self.fused_op.fused_fire_pretransposed_cuda(
            x.contiguous(memory_format=torch.contiguous_format),
            self.squeeze_w_flat,          # Use pre-processed buffer
            self.squeeze.bias,
            self.expand1x1_w_T,           # Use pre-processed buffer
            self.expand1x1.bias,
            self.expand3x3_w_T,           # Use pre-processed buffer
            self.expand3x3.bias
        )