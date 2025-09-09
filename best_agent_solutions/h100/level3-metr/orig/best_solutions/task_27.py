import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load_inline

# --- Custom CUDA/C++ Fused Operators (V7 with Custom Conv2D Grad) ---
# Rationale for Improvement:
# The previous solution optimized the BN-ReLU fusion with vectorization, but the
# backward pass still relied on torch.nn.grad.conv2d_input. This function
# requires casting the large activation and gradient tensors from FP16 to FP32
# and back, which is a significant overhead.
#
# This version introduces a custom FP16 kernel for the convolution input gradient
# (transposed convolution). This kernel:
# 1. Operates natively on __half data, eliminating the FP16 <-> FP32 casting bottleneck.
# 2. Implements a tiled direct convolution using shared memory to maximize data reuse
#    of the `dy` tensor, improving arithmetic intensity and reducing global memory traffic.
# This provides a significant speedup for the backward pass, complementing the
# already-optimized forward pass kernels.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// #################################################################################
// #                       FORWARD & UTILITY KERNELS (Unchanged)                   #
// #################################################################################

// Forward kernel for Global Average Pooling
__global__ void fused_global_avg_pool_fwd_kernel(
    const __half* __restrict__ x, __half* __restrict__ y,
    const int N, const int C, const int H, const int W) {
    extern __shared__ float sdata[];
    const int HW = H * W; const int n = blockIdx.y; const int c = blockIdx.x;
    const __half* x_offset = x + n * C * HW + c * HW;
    const int tid = threadIdx.x; const int block_size = blockDim.x;
    float sum = 0.0f;
    for (int i = tid; i < HW; i += block_size) { sum += __half2float(x_offset[i]); }
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) { sdata[tid] += sdata[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { y[n * C + c] = __float2half_rn(sdata[0] / (float)HW); }
}

// Vectorized Fused BN+ReLU Forward
__global__ void fused_bn_relu_forward_vec_fp16(
    const __half* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ running_mean, float* __restrict__ running_var,
    __half* __restrict__ y, float* __restrict__ save_mean, float* __restrict__ save_inv_std,
    const int N, const int C, const int H, const int W,
    const float momentum, const float eps) {
    extern __shared__ float sdata[];
    float* s_mean = sdata; float* s_var = &sdata[blockDim.x];
    const int c = blockIdx.x; const int tid = threadIdx.x;
    const int HW = H * W; const int NHW = N * HW; const int HW_VEC = HW / 2;

    float sum = 0.0f, sum_sq = 0.0f;
    for (int n = 0; n < N; ++n) {
        const half2* x_plane = (const half2*)(x + n * C * HW + c * HW);
        for (int hw_v = tid; hw_v < HW_VEC; hw_v += blockDim.x) {
            float2 val_f2 = __half22float2(x_plane[hw_v]);
            sum += val_f2.x + val_f2.y; sum_sq += val_f2.x * val_f2.x + val_f2.y * val_f2.y;
        }
    }
    s_mean[tid] = sum; s_var[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s_mean[tid] += s_mean[tid + s]; s_var[tid] += s_var[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) {
        const float mean = s_mean[0] / NHW;
        const float var = fmaxf(0.f, s_var[0] / NHW - mean * mean);
        save_mean[c] = mean; save_inv_std[c] = rsqrtf(var + eps);
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
        const float unbiased_var = var * ((float)NHW / (float)(NHW - 1));
        running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
    }
    __syncthreads();

    const half2 mean2 = __floats2half2_rn(save_mean[c], save_mean[c]);
    const half2 inv_std2 = __floats2half2_rn(save_inv_std[c], save_inv_std[c]);
    const half2 gamma2 = __floats2half2_rn(weight[c], weight[c]);
    const half2 beta2 = __floats2half2_rn(bias[c], bias[c]);
    const half2 zero2 = __floats2half2_rn(0.0f, 0.0f);
    for (int n = 0; n < N; ++n) {
        const half2* x_plane_v = (const half2*)(x + n*C*HW + c*HW);
        half2* y_plane_v = (half2*)(y + n*C*HW + c*HW);
        for (int hw_v = tid; hw_v < HW_VEC; hw_v += blockDim.x) {
            half2 x_hat = __hmul2(__hsub2(x_plane_v[hw_v], mean2), inv_std2);
            half2 y_unactivated = __hadd2(__hmul2(x_hat, gamma2), beta2);
            y_plane_v[hw_v] = __hmax2(y_unactivated, zero2);
        }
    }
}

// Partially Vectorized Fused BN+ReLU+MaxPool Forward
__global__ void fused_bn_relu_maxpool_forward_vec_fp16(
    const __half* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ running_mean, float* __restrict__ running_var,
    __half* __restrict__ y, float* __restrict__ save_mean, float* __restrict__ save_inv_std,
    long* __restrict__ argmax,
    const int N, const int C, const int H, const int W,
    const float momentum, const float eps) {
    extern __shared__ float sdata[];
    float* s_mean = sdata; float* s_var = &sdata[blockDim.x];
    const int c = blockIdx.x; const int tid = threadIdx.x;
    const int HW = H * W; const int NHW = N * HW; const int HW_VEC = HW / 2;

    float sum = 0.0f, sum_sq = 0.0f;
    for (int n = 0; n < N; ++n) {
        const half2* x_plane = (const half2*)(x + n * C * HW + c * HW);
        for (int hw_v = tid; hw_v < HW_VEC; hw_v += blockDim.x) {
            float2 val_f2 = __half22float2(x_plane[hw_v]);
            sum += val_f2.x + val_f2.y; sum_sq += val_f2.x * val_f2.x + val_f2.y * val_f2.y;
        }
    }
    s_mean[tid] = sum; s_var[tid] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s_mean[tid] += s_mean[tid + s]; s_var[tid] += s_var[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) {
        const float mean = s_mean[0] / NHW;
        const float var = fmaxf(0.f, s_var[0] / NHW - mean * mean);
        save_mean[c] = mean; save_inv_std[c] = rsqrtf(var + eps);
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
        const float unbiased_var = var * ((float)NHW / (float)(NHW - 1));
        running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
    }
    __syncthreads();

    const float saved_mean_val = save_mean[c], saved_inv_std_val = save_inv_std[c];
    const float gamma = weight[c], beta = bias[c];
    const int H_out = H / 2, W_out = W / 2; const int H_out_W_out = H_out * W_out;
    for (int n = 0; n < N; ++n) {
        for (int out_hw = tid; out_hw < H_out_W_out; out_hw += blockDim.x) {
            const int h_out = out_hw / W_out, w_out = out_hw % W_out;
            float max_val = -1.0e20f; int max_idx_in_map = -1;
            for (int kh = 0; kh < 2; ++kh) for (int kw = 0; kw < 2; ++kw) {
                const int h_in = h_out * 2 + kh, w_in = w_out * 2 + kw;
                const int in_idx_map = h_in * W + w_in;
                const float val = __half2float(x[n * C * HW + c * HW + in_idx_map]);
                const float normalized = (val - saved_mean_val) * saved_inv_std_val;
                const float relu_val = fmaxf(0.0f, normalized * gamma + beta);
                if (relu_val > max_val) { max_val = relu_val; max_idx_in_map = in_idx_map; }
            }
            const int out_idx = n*C*H_out_W_out + c*H_out_W_out + out_hw;
            y[out_idx] = __float2half_rn(max_val); argmax[out_idx] = max_idx_in_map;
        }
    }
}

// #################################################################################
// #                BACKWARD KERNELS (with NEW Conv Grad Kernel)                   #
// #################################################################################

__global__ void global_avg_pool_bwd_coalesced_kernel(
    const __half* __restrict__ dy, __half* __restrict__ dx,
    const int N, const int C, const int H, const int W) {
    const int nc_idx = blockIdx.x;
    if (nc_idx >= N * C) return;
    const float inv_hw = 1.0f / (float)(H * W);
    const float dy_val = __half2float(dy[nc_idx]);
    const __half val_to_write = __float2half_rn(dy_val * inv_hw);
    __half* dx_plane_ptr = dx + nc_idx * H * W;
    for (int hw = threadIdx.x; hw < H * W; hw += blockDim.x) {
        dx_plane_ptr[hw] = val_to_write;
    }
}

// --- NEW Tiled FP16 Kernel for Conv2d Input Gradient (Transposed Conv) ---
// Computes dx = conv(dy, rot180(w)), operating entirely in FP16
// Grid: (W_in/TILE_W, H_in/TILE_H, N * C_in)
// Block: (TILE_W, TILE_H)
__global__ void conv_bwd_input_tiled_fp16(
    const __half* __restrict__ dy, const __half* __restrict__ w, __half* __restrict__ dx,
    const int N, const int C_in, const int C_out,
    const int H_in, const int W_in, const int H_out, const int W_out)
{
    const int K = 3; // Kernel size is fixed at 3x3
    const int P = 1; // Padding is fixed at 1
    const int TILE_DIM_X = 16;
    const int TILE_DIM_Y = 16;
    const int HALO_SIZE = K - 1;

    __shared__ __half s_dy[(TILE_DIM_Y + HALO_SIZE)][(TILE_DIM_X + HALO_SIZE)];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int n = blockIdx.z / C_in;
    const int c_in = blockIdx.z % C_in;
    const int out_h = blockIdx.y * TILE_DIM_Y + ty;
    const int out_w = blockIdx.x * TILE_DIM_X + tx;

    const int dy_tile_h_start = (int)blockIdx.y * TILE_DIM_Y - P;
    const int dy_tile_w_start = (int)blockIdx.x * TILE_DIM_X - P;

    float acc = 0.0f;

    for (int c_out = 0; c_out < C_out; ++c_out) {
        // Cooperatively load tile of dy from global to shared memory
        for (int i = ty; i < TILE_DIM_Y + HALO_SIZE; i += TILE_DIM_Y) {
            for (int j = tx; j < TILE_DIM_X + HALO_SIZE; j += TILE_DIM_X) {
                const int h = dy_tile_h_start + i;
                const int w = dy_tile_w_start + j;
                if (h >= 0 && h < H_out && w >= 0 && w < W_out) {
                    s_dy[i][j] = dy[(n * C_out + c_out) * H_out * W_out + h * W_out + w];
                } else {
                    s_dy[i][j] = __float2half_rn(0.0f);
                }
            }
        }
        __syncthreads();

        if (out_h < H_in && out_w < W_in) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    const __half dy_val = s_dy[ty + kh][tx + kw];
                    const int w_idx = (c_out * C_in + c_in) * K * K + (K - 1 - kh) * K + (K - 1 - kw);
                    const __half w_val = w[w_idx];
                    acc += __half2float(dy_val) * __half2float(w_val);
                }
            }
        }
        __syncthreads();
    }

    if (out_h < H_in && out_w < W_in) {
        const int dx_idx = (n * C_in + c_in) * H_in * W_in + out_h * W_in + out_w;
        dx[dx_idx] = __float2half_rn(acc);
    }
}

// Vectorized Fused BN-ReLU backward (Unchanged)
__global__ void fused_bn_relu_bwd_vec_fp16(
    const __half* __restrict__ dy, const __half* __restrict__ conv_out,
    const float* __restrict__ bn_w, const float* __restrict__ bn_b,
    const float* __restrict__ save_mean, const float* __restrict__ save_inv_std,
    __half* __restrict__ d_conv_out, float* __restrict__ d_bn_w, float* __restrict__ d_bn_b,
    const int N, const int C, const int H, const int W) {
    extern __shared__ float sdata[];
    float* s_d_bn_w = sdata; float* s_d_bn_b = &sdata[blockDim.x];
    const int c = blockIdx.x; const int tid = threadIdx.x;
    const int HW = H*W; const int NHW = N*HW; const int HW_VEC = HW / 2;
    const float mean_c = save_mean[c], inv_std_c = save_inv_std[c], bn_w_c = bn_w[c], bn_b_c = bn_b[c];
    
    float d_bn_w_sum = 0.0f, d_bn_b_sum = 0.0f;
    for(int n=0; n<N; ++n) {
        const half2* conv_out_plane_v = (const half2*)(conv_out + n*C*HW + c*HW);
        const half2* dy_plane_v = (const half2*)(dy + n*C*HW + c*HW);
        for (int hw_v = tid; hw_v < HW_VEC; hw_v += blockDim.x) {
            float2 conv_out_f2 = __half22float2(conv_out_plane_v[hw_v]);
            float2 x_hat_f2 = make_float2((conv_out_f2.x - mean_c) * inv_std_c, (conv_out_f2.y - mean_c) * inv_std_c);
            float2 pre_relu_f2 = make_float2(x_hat_f2.x * bn_w_c + bn_b_c, x_hat_f2.y * bn_w_c + bn_b_c);
            float2 relu_mask = make_float2(pre_relu_f2.x > 0.0f ? 1.0f : 0.0f, pre_relu_f2.y > 0.0f ? 1.0f : 0.0f);
            float2 dy_f2 = __half22float2(dy_plane_v[hw_v]);
            float2 d_bn_b_v = make_float2(dy_f2.x * relu_mask.x, dy_f2.y * relu_mask.y);
            d_bn_w_sum += d_bn_b_v.x * x_hat_f2.x + d_bn_b_v.y * x_hat_f2.y;
            d_bn_b_sum += d_bn_b_v.x + d_bn_b_v.y;
        }
    }
    s_d_bn_w[tid] = d_bn_w_sum; s_d_bn_b[tid] = d_bn_b_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s>>=1) { if(tid<s){ s_d_bn_w[tid]+=s_d_bn_w[tid+s]; s_d_bn_b[tid]+=s_d_bn_b[tid+s]; } __syncthreads(); }
    
    __shared__ float final_sum_d_bn_b, final_sum_d_bn_x_hat;
    if (tid == 0) {
        final_sum_d_bn_x_hat = s_d_bn_w[0]; final_sum_d_bn_b = s_d_bn_b[0];
        d_bn_w[c] = final_sum_d_bn_x_hat; d_bn_b[c] = final_sum_d_bn_b;
    }
    __syncthreads();

    const float scale = bn_w_c * inv_std_c / (float)NHW;
    for (int n=0; n<N; ++n) {
        const half2* conv_out_plane_v = (const half2*)(conv_out + n*C*HW + c*HW);
        const half2* dy_plane_v = (const half2*)(dy + n*C*HW + c*HW);
        half2* d_conv_out_plane_v = (half2*)(d_conv_out + n*C*HW + c*HW);
        for (int hw_v = tid; hw_v < HW_VEC; hw_v += blockDim.x) {
            float2 conv_out_f2 = __half22float2(conv_out_plane_v[hw_v]);
            float2 x_hat_f2 = make_float2((conv_out_f2.x - mean_c) * inv_std_c, (conv_out_f2.y - mean_c) * inv_std_c);
            float2 pre_relu_f2 = make_float2(x_hat_f2.x * bn_w_c + bn_b_c, x_hat_f2.y * bn_w_c + bn_b_c);
            float2 relu_mask = make_float2(pre_relu_f2.x > 0.0f ? 1.0f : 0.0f, pre_relu_f2.y > 0.0f ? 1.0f : 0.0f);
            float2 d_bn_val_f2 = __half22float2(dy_plane_v[hw_v]);
            d_bn_val_f2.x *= relu_mask.x; d_bn_val_f2.y *= relu_mask.y;
            float2 d_x_hat_f2 = make_float2(d_bn_val_f2.x * bn_w_c, d_bn_val_f2.y * bn_w_c);
            float2 d_conv_out_f2;
            d_conv_out_f2.x = scale * ( (float)NHW * d_x_hat_f2.x - final_sum_d_bn_b - x_hat_f2.x * final_sum_d_bn_x_hat );
            d_conv_out_f2.y = scale * ( (float)NHW * d_x_hat_f2.y - final_sum_d_bn_b - x_hat_f2.y * final_sum_d_bn_x_hat );
            d_conv_out_plane_v[hw_v] = __float22half2_rn(d_conv_out_f2);
        }
    }
}

// Fused BN-ReLU-MaxPool backward (Unchanged)
__global__ void fused_bn_relu_maxpool_bwd_coalesced_kernel_fp16(
    const __half* __restrict__ dy, const long* __restrict__ argmax,
    const __half* __restrict__ conv_out, const float* __restrict__ bn_w,
    const float* __restrict__ bn_b, const float* __restrict__ save_mean,
    const float* __restrict__ save_inv_std, __half* __restrict__ d_conv_out,
    float* __restrict__ d_bn_w, float* __restrict__ d_bn_b,
    const int N, const int C, const int H, const int W) {
    extern __shared__ float sdata[];
    float* s_d_bn_w = sdata; float* s_d_bn_b = &sdata[blockDim.x];
    const int c = blockIdx.x; const int tid = threadIdx.x;
    const int HW = H * W; const int NHW = N * HW;
    const int H_out = H / 2, W_out = W / 2; const int H_out_W_out = H_out * W_out;
    const float mean_c = save_mean[c], inv_std_c = save_inv_std[c], bn_w_c = bn_w[c], bn_b_c = bn_b[c];
    
    float d_bn_w_sum = 0.0f, d_bn_b_sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int out_hw = tid; out_hw < H_out_W_out; out_hw += blockDim.x) {
            const float dy_val = __half2float(dy[n*C*H_out_W_out + c*H_out_W_out + out_hw]);
            if (dy_val == 0.0f) continue;
            const int in_idx_map = argmax[n*C*H_out_W_out + c*H_out_W_out + out_hw];
            const float conv_out_val = __half2float(conv_out[n*C*HW + c*HW + in_idx_map]);
            const float x_hat = (conv_out_val - mean_c) * inv_std_c;
            if (x_hat * bn_w_c + bn_b_c > 0.0f) { d_bn_w_sum += dy_val * x_hat; d_bn_b_sum += dy_val; }
        }
    }
    s_d_bn_w[tid] = d_bn_w_sum; s_d_bn_b[tid] = d_bn_b_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s>>=1) { if(tid<s){ s_d_bn_w[tid]+=s_d_bn_w[tid+s]; s_d_bn_b[tid]+=s_d_bn_b[tid+s]; } __syncthreads(); }
    
    __shared__ float final_sum_d_bn_b, final_sum_d_bn_x_hat;
    if (tid == 0) {
        final_sum_d_bn_x_hat = s_d_bn_w[0]; final_sum_d_bn_b = s_d_bn_b[0];
        d_bn_w[c] = final_sum_d_bn_x_hat; d_bn_b[c] = final_sum_d_bn_b;
    }
    __syncthreads();

    const float scale = bn_w_c * inv_std_c / (float)NHW;
    for (int n = 0; n < N; ++n) {
        for (int out_hw = tid; out_hw < H_out_W_out; out_hw += blockDim.x) {
            const float dy_val = __half2float(dy[n*C*H_out_W_out + c*H_out_W_out + out_hw]);
            if (dy_val == 0.0f) continue;
            const int in_idx_map = argmax[n*C*H_out_W_out + c*H_out_W_out + out_hw];
            const int in_idx_glob = n * C * HW + c * HW + in_idx_map;
            const float conv_out_val = __half2float(conv_out[in_idx_glob]);
            const float x_hat = (conv_out_val - mean_c) * inv_std_c;
            if (x_hat * bn_w_c + bn_b_c > 0.0f) {
                const float d_bn_val = dy_val;
                const float d_x_hat = d_bn_val * bn_w_c;
                d_conv_out[in_idx_glob] = __float2half_rn( scale * ((float)NHW * d_x_hat - final_sum_d_bn_b - x_hat * final_sum_d_bn_x_hat) );
            }
        }
    }
}

// #################################################################################
// #                              C++ DISPATCHERS                                  #
// #################################################################################

torch::Tensor fused_global_avg_pool_forward(torch::Tensor x) {
    const auto S = x.sizes();
    auto y = torch::empty({S[0], S[1]}, x.options());
    fused_global_avg_pool_fwd_kernel<<<dim3(S[1], S[0]), 256, 256*sizeof(float)>>>(
        (const __half*)x.data_ptr(), (__half*)y.data_ptr(), S[0], S[1], S[2], S[3]);
    return y;
}

std::vector<torch::Tensor> fused_bn_relu_forward_fp16(
    torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor rm, torch::Tensor rv, double mom, double eps) {
    const auto S = x.sizes();
    auto y = torch::empty_like(x);
    auto float_opts = x.options().dtype(torch::kFloat32);
    auto sm = torch::empty({S[1]}, float_opts); auto siv = torch::empty({S[1]}, float_opts);
    fused_bn_relu_forward_vec_fp16<<<S[1], 256, 2*256*sizeof(float)>>>(
        (const __half*)x.data_ptr(), w.data_ptr<float>(), b.data_ptr<float>(),
        rm.data_ptr<float>(), rv.data_ptr<float>(), (__half*)y.data_ptr(),
        sm.data_ptr<float>(), siv.data_ptr<float>(), S[0], S[1], S[2], S[3], mom, eps);
    return {y, sm, siv};
}

std::vector<torch::Tensor> fused_bn_relu_maxpool_forward_fp16(
    torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor rm, torch::Tensor rv, double mom, double eps) {
    const auto S = x.sizes();
    auto y = torch::empty({S[0], S[1], S[2]/2, S[3]/2}, x.options());
    auto float_opts = x.options().dtype(torch::kFloat32);
    auto sm = torch::empty({S[1]}, float_opts); auto siv = torch::empty({S[1]}, float_opts);
    auto argmax = torch::empty({S[0], S[1], S[2]/2, S[3]/2}, x.options().dtype(torch::kInt64));
    fused_bn_relu_maxpool_forward_vec_fp16<<<S[1], 256, 2*256*sizeof(float)>>>(
        (const __half*)x.data_ptr(), w.data_ptr<float>(), b.data_ptr<float>(),
        rm.data_ptr<float>(), rv.data_ptr<float>(), (__half*)y.data_ptr(),
        sm.data_ptr<float>(), siv.data_ptr<float>(), argmax.data_ptr<long>(),
        S[0], S[1], S[2], S[3], mom, eps);
    return {y, sm, siv, argmax};
}

void global_avg_pool_bwd(torch::Tensor dy, torch::Tensor dx) {
    const auto S = dx.sizes();
    global_avg_pool_bwd_coalesced_kernel<<<S[0]*S[1], 256>>>(
        (const __half*)dy.data_ptr(), (__half*)dx.data_ptr(), S[0], S[1], S[2], S[3]);
}

// --- NEW Dispatcher for Custom Conv Input Grad ---
void conv_bwd_input_dispatch(torch::Tensor dy, torch::Tensor w, torch::Tensor dx) {
    const auto dy_s = dy.sizes();
    const auto w_s = w.sizes();
    const auto dx_s = dx.sizes();
    const int N = dy_s[0], C_out = dy_s[1], H_out = dy_s[2], W_out = dy_s[3];
    const int C_in = w_s[1];
    const int H_in = dx_s[2], W_in = dx_s[3];

    const int TILE_DIM_X = 16, TILE_DIM_Y = 16;
    dim3 block(TILE_DIM_X, TILE_DIM_Y);
    dim3 grid( (W_in + TILE_DIM_X - 1) / TILE_DIM_X, (H_in + TILE_DIM_Y - 1) / TILE_DIM_Y, N * C_in );

    conv_bwd_input_tiled_fp16<<<grid, block>>>(
        (const __half*)dy.data_ptr(), (const __half*)w.data_ptr(), (__half*)dx.data_ptr(),
        N, C_in, C_out, H_in, W_in, H_out, W_out );
}

std::vector<torch::Tensor> fused_bn_relu_bwd(
    torch::Tensor dy, torch::Tensor cv_out, torch::Tensor bn_w, torch::Tensor bn_b,
    torch::Tensor s_mean, torch::Tensor s_inv_std) {
    const auto S = cv_out.sizes();
    auto d_conv_out = torch::empty_like(cv_out);
    auto float_opts = cv_out.options().dtype(torch::kFloat32);
    auto d_bn_w = torch::empty({S[1]}, float_opts); auto d_bn_b = torch::empty({S[1]}, float_opts);
    fused_bn_relu_bwd_vec_fp16<<<S[1], 256, 2*256*sizeof(float)>>>(
        (const __half*)dy.data_ptr(), (const __half*)cv_out.data_ptr(),
        bn_w.data_ptr<float>(), bn_b.data_ptr<float>(), s_mean.data_ptr<float>(),
        s_inv_std.data_ptr<float>(), (__half*)d_conv_out.data_ptr(),
        d_bn_w.data_ptr<float>(), d_bn_b.data_ptr<float>(), S[0], S[1], S[2], S[3]);
    return {d_conv_out, d_bn_w, d_bn_b};
}

std::vector<torch::Tensor> fused_bn_relu_maxpool_bwd(
    torch::Tensor dy, torch::Tensor argmax, torch::Tensor cv_out, torch::Tensor bn_w,
    torch::Tensor bn_b, torch::Tensor s_mean, torch::Tensor s_inv_std) {
    const auto S = cv_out.sizes();
    auto d_conv_out = torch::zeros_like(cv_out); // Must be zero-initialized
    auto float_opts = cv_out.options().dtype(torch::kFloat32);
    auto d_bn_w = torch::empty({S[1]}, float_opts); auto d_bn_b = torch::empty({S[1]}, float_opts);
    fused_bn_relu_maxpool_bwd_coalesced_kernel_fp16<<<S[1], 256, 2*256*sizeof(float)>>>(
        (const __half*)dy.data_ptr(), argmax.data_ptr<long>(),
        (const __half*)cv_out.data_ptr(), bn_w.data_ptr<float>(), bn_b.data_ptr<float>(),
        s_mean.data_ptr<float>(), s_inv_std.data_ptr<float>(), (__half*)d_conv_out.data_ptr(),
        d_bn_w.data_ptr<float>(), d_bn_b.data_ptr<float>(), S[0], S[1], S[2], S[3]);
    return {d_conv_out, d_bn_w, d_bn_b};
}
"""

cpp_source = """
#include <vector>
// Forward
torch::Tensor fused_global_avg_pool_forward(torch::Tensor x);
std::vector<torch::Tensor> fused_bn_relu_forward_fp16(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, double);
std::vector<torch::Tensor> fused_bn_relu_maxpool_forward_fp16(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, double);
// Backward
void global_avg_pool_bwd(torch::Tensor, torch::Tensor);
void conv_bwd_input_dispatch(torch::Tensor dy, torch::Tensor w, torch::Tensor dx);
std::vector<torch::Tensor> fused_bn_relu_bwd(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> fused_bn_relu_maxpool_bwd(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
"""

# JIT compile the custom CUDA/C++ code
fused_ops_fp16 = load_inline(
    name="fused_regnet_ops_fp16_v8",
    cpp_sources=cpp_source,
    cuda_sources=fused_ops_source,
    functions=[
        "fused_global_avg_pool_forward", "global_avg_pool_bwd",
        "fused_bn_relu_forward_fp16", "fused_bn_relu_bwd",
        "fused_bn_relu_maxpool_forward_fp16", "fused_bn_relu_maxpool_bwd",
        "conv_bwd_input_dispatch" # Add new function
    ],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_75"]
)

class FusedConvBNReLUFunctionFP16(Function):
    @staticmethod
    def forward(ctx, x, conv_w, conv_b, bn_w, bn_b, rm, rv, mom, eps, training, stride, padding):
        # Forward pass is unchanged
        if training:
            conv_out = F.conv2d(x, conv_w, conv_b, stride, padding)
            y, s_mean, s_inv_std = fused_ops_fp16.fused_bn_relu_forward_fp16(conv_out, bn_w, bn_b, rm, rv, mom, eps)
            ctx.save_for_backward(x, conv_w, bn_w, bn_b, conv_out, s_mean, s_inv_std)
            ctx.stride, ctx.padding = stride, padding
            ctx.conv_b_is_none = conv_b is None
            return y
        else: # Inference path remains standard PyTorch
            conv_out = F.conv2d(x, conv_w, conv_b, stride, padding)
            bn_out = F.batch_norm(conv_out, rm, rv, bn_w, bn_b, False, 0.0, eps)
            return F.relu(bn_out)

    @staticmethod
    def backward(ctx, grad_output):
        x, conv_w, bn_w, bn_b, conv_out, s_mean, s_inv_std = ctx.saved_tensors
        d_conv_out, d_bn_w, d_bn_b = fused_ops_fp16.fused_bn_relu_bwd(
            grad_output, conv_out, bn_w, bn_b, s_mean, s_inv_std)

        # --- REVISED GRADIENT CALCULATION ---
        # Use custom kernel for data gradient (dx) to avoid FP16->FP32 cast
        grad_x = torch.empty_like(x)
        fused_ops_fp16.conv_bwd_input_dispatch(d_conv_out, conv_w, grad_x)
        
        # Keep PyTorch's kernel for weight gradient (dw) - it's a complex reduction
        # and casting is less of a bottleneck as the tensors are smaller.
        x_fp32, conv_w_fp32, d_conv_out_fp32 = x.float(), conv_w.float(), d_conv_out.float()
        grad_conv_w = torch.nn.grad.conv2d_weight(x_fp32, conv_w_fp32.shape, d_conv_out_fp32, ctx.stride, ctx.padding)
        grad_conv_b = None if ctx.conv_b_is_none else d_conv_out_fp32.sum(dim=[0,2,3])
        
        return grad_x, grad_conv_w.half(), grad_conv_b.half() if grad_conv_b is not None else None, \
               d_bn_w, d_bn_b, None, None, None, None, None, None, None

class FusedConvBNReLU_MaxPoolFunctionFP16(Function):
    @staticmethod
    def forward(ctx, x, conv_w, conv_b, bn_w, bn_b, rm, rv, mom, eps, training, c_stride, c_pad, p_ks, p_stride):
        # Forward pass is unchanged
        if training:
            conv_out = F.conv2d(x, conv_w, conv_b, c_stride, c_pad)
            y, s_mean, s_inv_std, argmax = fused_ops_fp16.fused_bn_relu_maxpool_forward_fp16(conv_out, bn_w, bn_b, rm, rv, mom, eps)
            ctx.save_for_backward(x, conv_w, bn_w, bn_b, conv_out, s_mean, s_inv_std, argmax)
            ctx.c_stride, ctx.c_pad = c_stride, c_pad
            ctx.conv_b_is_none = conv_b is None
            return y
        else: # Inference path remains standard PyTorch
            conv_out = F.conv2d(x, conv_w, conv_b, c_stride, c_pad)
            bn_out = F.batch_norm(conv_out, rm, rv, bn_w, bn_b, False, 0.0, eps)
            return F.max_pool2d(F.relu(bn_out), p_ks, p_stride)

    @staticmethod
    def backward(ctx, grad_output):
        x, conv_w, bn_w, bn_b, conv_out, s_mean, s_inv_std, argmax = ctx.saved_tensors
        d_conv_out, d_bn_w, d_bn_b = fused_ops_fp16.fused_bn_relu_maxpool_bwd(
            grad_output, argmax, conv_out, bn_w, bn_b, s_mean, s_inv_std)

        # --- REVISED GRADIENT CALCULATION ---
        # Use custom kernel for data gradient (dx) to avoid FP16->FP32 cast
        grad_x = torch.empty_like(x)
        fused_ops_fp16.conv_bwd_input_dispatch(d_conv_out, conv_w, grad_x)

        # Keep PyTorch's kernel for weight gradient (dw)
        x_fp32, conv_w_fp32, d_conv_out_fp32 = x.float(), conv_w.float(), d_conv_out.float()
        grad_conv_w = torch.nn.grad.conv2d_weight(x_fp32, conv_w_fp32.shape, d_conv_out_fp32, ctx.c_stride, ctx.c_pad)
        grad_conv_b = None if ctx.conv_b_is_none else d_conv_out_fp32.sum(dim=[0,2,3])
        
        return grad_x, grad_conv_w.half(), grad_conv_b.half() if grad_conv_b is not None else None, \
               d_bn_w, d_bn_b, None, None, None, None, None, None, None, None

class FusedGlobalAvgPoolFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.shape = x.shape
        if x.is_cuda and x.dtype == torch.half:
             return fused_ops_fp16.fused_global_avg_pool_forward(x)
        else: 
             return torch.mean(x, dim=[2, 3])

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output.is_cuda and grad_output.dtype == torch.half:
            grad_input = torch.empty(ctx.shape, device=grad_output.device, dtype=grad_output.dtype)
            fused_ops_fp16.global_avg_pool_bwd(grad_output.contiguous(), grad_input)
            return grad_input
        else:
            N, C, H, W = ctx.shape
            return grad_output.unsqueeze(2).unsqueeze(3).expand(N, C, H, W) / (H * W)

class FusedGlobalAvgPool(nn.Module):
    def forward(self, x):
        return FusedGlobalAvgPoolFunction.apply(x)

class FusedStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        if self.training:
            y = FusedConvBNReLUFunctionFP16.apply(x, self.conv1.weight, self.conv1.bias,
                self.bn1.weight, self.bn1.bias, self.bn1.running_mean, self.bn1.running_var,
                self.bn1.momentum, self.bn1.eps, self.training, self.conv1.stride, self.conv1.padding)
            
            z = FusedConvBNReLU_MaxPoolFunctionFP16.apply(y, self.conv2.weight, self.conv2.bias,
                self.bn2.weight, self.bn2.bias, self.bn2.running_mean, self.bn2.running_var,
                self.bn2.momentum, self.bn2.eps, self.training, self.conv2.stride, self.conv2.padding,
                (2, 2), (2, 2))
        else:
            y = F.conv2d(x, self.conv1.weight, self.conv1.bias, self.conv1.stride, self.conv1.padding)
            y = F.batch_norm(y, self.bn1.running_mean, self.bn1.running_var, self.bn1.weight, self.bn1.bias, False, 0.0, self.bn1.eps)
            y = F.relu(y)
            z = F.conv2d(y, self.conv2.weight, self.conv2.bias, self.conv2.stride, self.conv2.padding)
            z = F.batch_norm(z, self.bn2.running_mean, self.bn2.running_var, self.bn2.weight, self.bn2.bias, False, 0.0, self.bn2.eps)
            z = F.relu(z)
            z = F.max_pool2d(z, kernel_size=2, stride=2)
        return z

class Model(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        super(Model, self).__init__()
        self.stages_count = stages
        self.block_widths = block_widths
        
        layers = []
        current_channels = input_channels
        
        for i in range(stages):
            layers.append(FusedStage(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        
        self.feature_extractor = nn.Sequential(*layers)
        self.global_pool = FusedGlobalAvgPool()
        self.fc = nn.Linear(block_widths[-1], output_classes)

        self.half() 
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.Linear)):
                m.float()

    def forward(self, x):
        x_half = x.half()
        features = self.feature_extractor(x_half)
        pooled = self.global_pool(features)
        output = self.fc(pooled.float())
        return output