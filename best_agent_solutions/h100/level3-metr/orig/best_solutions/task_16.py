import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

# --- JIT Compilation for Fused CUDA Kernels ---

# Set a flag to control kernel usage
# This allows falling back to PyTorch default if compilation fails or for debugging
USE_CUSTOM_KERNELS = False
try:
    # --- CUDA Source (Kernels and Launchers) ---
    cuda_source = """
    #include <torch/extension.h>
    #include <c10/cuda/CUDAException.h>
    #include <cuda_fp16.h> // For __half and half2
    #include <cmath> // For rsqrtf

    // --- Vector Type Helper ---
    template <typename T> struct Vec2 { struct type { T x, y; }; };
    template <> struct Vec2<float> { using type = float2; };
    template <> struct Vec2<__half> { using type = half2; };
    
    // --- Tiling and Kernel Dimension Defines ---
    #define DENSE_OUT_TILE_DIM 16
    #define DENSE_KERNEL_DIM 3
    #define DENSE_PADDING 1
    #define DENSE_IN_TILE_DIM (DENSE_OUT_TILE_DIM + DENSE_KERNEL_DIM - 1)
    #define DENSE_IN_TILE_DIM_VEC (DENSE_IN_TILE_DIM / 2)

    #define TRANS_OUT_TILE_H 8
    #define TRANS_OUT_TILE_W 16
    #define TRANS_IN_TILE_H (TRANS_OUT_TILE_H * 2)
    #define TRANS_IN_TILE_W (TRANS_OUT_TILE_W * 2)

    // --- Utility for Block-wide Reduction (with Warp Shuffle) ---
    __device__ __forceinline__ float warp_reduce_sum(float val) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }

    // --- Kernel 1 (IMPROVED V2): Vectorized, Tiled Fused BN->ReLU->Conv3x3 (for DenseBlock) ---
    // Changes from V1:
    // 1. Added early exit for threads whose output pixel is in the padding region.
    // 2. Corrected and simplified boundary handling logic for vectorized memory loads.
    template <typename T, typename T2>
    __global__ void fused_dense_layer_kernel_v2(
        const T* __restrict__ x,
        const T* __restrict__ bn_w, const T* __restrict__ bn_b,
        const T* __restrict__ mean, const T* __restrict__ var,
        const T* __restrict__ conv_w,
        T* __restrict__ y_map,
        const int write_offset_c,
        const int C_total,
        const float eps, const int N, const int C_in, const int H, const int W, const int C_out) {

        __shared__ T2 s_x[DENSE_IN_TILE_DIM][DENSE_IN_TILE_DIM_VEC];

        const int tx = threadIdx.x; // 0..15
        const int ty = threadIdx.y; // 0..15

        const int n_idx = blockIdx.z / C_out;
        const int c_out_idx = blockIdx.z % C_out;
        const int h_tile_idx = blockIdx.y;
        const int w_tile_idx = blockIdx.x;

        const int h_out = h_tile_idx * DENSE_OUT_TILE_DIM + ty;
        const int w_out = w_tile_idx * DENSE_OUT_TILE_DIM + tx;

        // SIGNIFICANT IMPROVEMENT: Early exit for threads that write to padding.
        // This avoids running the entire heavy computation loop for nothing.
        if (h_out >= H || w_out >= W) {
            return;
        }

        float acc = 0.0f;

        const int h_in_tile_start = h_tile_idx * DENSE_OUT_TILE_DIM - DENSE_PADDING;
        const int w_in_tile_start = w_tile_idx * DENSE_OUT_TILE_DIM - DENSE_PADDING;

        for (int c_in = 0; c_in < C_in; ++c_in) {
            // Load a tile of input data for the current channel into shared memory.
            #pragma unroll
            for (int i = ty; i < DENSE_IN_TILE_DIM; i += blockDim.y) {
                const int h_in = h_in_tile_start + i;

                // If the entire row is outside the image (top/bottom padding), write zeros.
                if (h_in < 0 || h_in >= H) {
                    #pragma unroll
                    for (int j = tx; j < DENSE_IN_TILE_DIM_VEC; j += blockDim.x) {
                        s_x[i][j] = T2{static_cast<T>(0.0f), static_cast<T>(0.0f)};
                    }
                } else {
                    // Row is valid, load vector by vector, handling horizontal boundaries.
                    const T* x_s_ptr = x + (int64_t)n_idx * C_in * H * W + (int64_t)c_in * H * W + (int64_t)h_in * W;

                    #pragma unroll
                    for (int j = tx; j < DENSE_IN_TILE_DIM_VEC; j += blockDim.x) {
                        const int w_in_start = w_in_tile_start + j * 2;
                        
                        // Fast path: vector is fully inside the image width.
                        if (w_in_start >= 0 && w_in_start + 1 < W) {
                            s_x[i][j] = *(reinterpret_cast<const T2*>(x_s_ptr + w_in_start));
                        } else {
                            // Slow path: vector straddles a boundary or is fully outside.
                            T vals[2];
                            vals[0] = (w_in_start >= 0 && w_in_start < W) ? x_s_ptr[w_in_start] : static_cast<T>(0.0f);
                            vals[1] = (w_in_start + 1 >= 0 && w_in_start + 1 < W) ? x_s_ptr[w_in_start + 1] : static_cast<T>(0.0f);
                            s_x[i][j] = *reinterpret_cast<T2*>(vals);
                        }
                    }
                }
            }
            __syncthreads();

            // Perform BN -> ReLU -> Conv using data from shared memory.
            const T inv_std = rsqrtf(var[c_in] + static_cast<T>(eps));
            const T bn_weight_c = bn_w[c_in];
            const T bn_bias_c = bn_b[c_in];
            const T mean_c = mean[c_in];

            #pragma unroll
            for (int kh = 0; kh < DENSE_KERNEL_DIM; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < DENSE_KERNEL_DIM; ++kw) {
                    const int w_s_idx = tx + kw;
                    const T2 s_x_vec = s_x[ty + kh][w_s_idx / 2];
                    const T x_val = (w_s_idx % 2 == 0) ? s_x_vec.x : s_x_vec.y;
                    
                    const T bn_val = (x_val - mean_c) * inv_std * bn_weight_c + bn_bias_c;
                    const T relu_val = bn_val > static_cast<T>(0.0f) ? bn_val : static_cast<T>(0.0f);
                    const int64_t conv_w_idx = (int64_t)c_out_idx * C_in * 9 + (int64_t)c_in * 9 + kh * 3 + kw;
                    acc += static_cast<float>(relu_val) * static_cast<float>(conv_w[conv_w_idx]);
                }
            }
            __syncthreads();
        }

        const int c_out_global_idx = c_out_idx + write_offset_c;
        const int64_t y_idx = (int64_t)n_idx * C_total * H * W + (int64_t)c_out_global_idx * H * W + (int64_t)h_out * W + w_out;
        y_map[y_idx] = static_cast<T>(acc);
    }

    // --- Kernel 2: Vectorized, Tiled Fused BN->ReLU->Conv1x1->AvgPool2x2 (for TransitionLayer) - Unchanged ---
    template <typename T, typename T2>
    __global__ void tiled_fused_transition_kernel_vec(
        const T* __restrict__ x,
        const T* __restrict__ bn_w, const T* __restrict__ bn_b,
        const T* __restrict__ mean, const T* __restrict__ var,
        const T* __restrict__ conv_w,
        T* __restrict__ y,
        const float eps, const int N, const int C_in, const int H, const int W, const int C_out) {

        __shared__ T2 s_x[TRANS_IN_TILE_H][TRANS_IN_TILE_W / 2];
        const int tx = threadIdx.x, ty = threadIdx.y;
        const int n_idx = blockIdx.z / C_out, c_out_idx = blockIdx.z % C_out;
        const int h_tile_idx = blockIdx.y, w_tile_idx = blockIdx.x;
        const int H_out = H / 2, W_out = W / 2;
        const int h_out = h_tile_idx * TRANS_OUT_TILE_H + ty, w_out = w_tile_idx * TRANS_OUT_TILE_W + tx;

        if (h_out >= H_out || w_out >= W_out) return;

        float acc = 0.0f;
        const int h_in_tile_start = h_tile_idx * TRANS_IN_TILE_H;
        const int w_in_tile_start = w_tile_idx * TRANS_IN_TILE_W;

        for (int c_in = 0; c_in < C_in; ++c_in) {
            const T2* x_vec_ptr = reinterpret_cast<const T2*>(x + (int64_t)n_idx * C_in * H * W + (int64_t)c_in * H * W);
            for (int i = ty; i < TRANS_IN_TILE_H; i += blockDim.y) {
                for (int j = tx; j < TRANS_IN_TILE_W / 2; j += blockDim.x) {
                    s_x[i][j] = x_vec_ptr[ (int64_t)(h_in_tile_start + i) * (W/2) + (w_in_tile_start/2 + j) ];
                }
            }
            __syncthreads();

            const T inv_std = rsqrtf(var[c_in] + static_cast<T>(eps));
            const T bn_weight_c = bn_w[c_in], bn_bias_c = bn_b[c_in], mean_c = mean[c_in];
            const T weight_val = conv_w[(int64_t)c_out_idx * C_in + c_in];

            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    const int h_idx_s = ty * 2 + dy;
                    const int w_idx_s = tx * 2 + dx;
                    const T2 val_vec = s_x[h_idx_s][w_idx_s / 2];
                    const T x_val = (w_idx_s % 2 == 0) ? val_vec.x : val_vec.y;
                    const T bn_val = (x_val - mean_c) * inv_std * bn_weight_c + bn_bias_c;
                    const T relu_val = bn_val > static_cast<T>(0.0f) ? bn_val : static_cast<T>(0.0f);
                    acc += static_cast<float>(relu_val) * static_cast<float>(weight_val);
                }
            }
            __syncthreads();
        }

        const int64_t y_idx = (int64_t)n_idx * C_out * H_out * W_out + (int64_t)c_out_idx * H_out * W_out + (int64_t)h_out * W_out + w_out;
        y[y_idx] = static_cast<T>(acc / 4.0f);
    }
    
    // --- Kernel 3: Vectorized Fused Final BN->ReLU->GlobalAvgPool - Unchanged ---
    template <typename T, typename T2>
    __global__ void fused_final_op_kernel_vec(
        const T* __restrict__ x,
        const T* __restrict__ bn_w, const T* __restrict__ bn_b,
        const T* __restrict__ mean, const T* __restrict__ var,
        T* __restrict__ y,
        float eps, int N, int C, int H, int W) {

        const int n = blockIdx.x;
        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x % 32;
        const int warps_per_block = blockDim.x / 32;
        const int c = blockIdx.y * warps_per_block + warp_id;
        if (c >= C) return;

        const int64_t HW = H * W;
        const int64_t HW_vec = HW / 2;
        const T inv_std = rsqrtf(var[c] + static_cast<T>(eps));
        const T bn_weight_c = bn_w[c], bn_bias_c = bn_b[c], mean_c = mean[c];
        const T2* x_c_ptr = reinterpret_cast<const T2*>(x + (int64_t)n * C * HW + (int64_t)c * HW);

        float my_sum = 0.0f;
        for (int i = lane_id; i < HW_vec; i += 32) {
            const T2 x_vec = x_c_ptr[i];
            const T bn_val1 = (x_vec.x - mean_c) * inv_std * bn_weight_c + bn_bias_c;
            const T relu_val1 = bn_val1 > static_cast<T>(0.0f) ? bn_val1 : static_cast<T>(0.0f);
            my_sum += static_cast<float>(relu_val1);
            const T bn_val2 = (x_vec.y - mean_c) * inv_std * bn_weight_c + bn_bias_c;
            const T relu_val2 = bn_val2 > static_cast<T>(0.0f) ? bn_val2 : static_cast<T>(0.0f);
            my_sum += static_cast<float>(relu_val2);
        }
        my_sum = warp_reduce_sum(my_sum);
        if (lane_id == 0) {
            atomicAdd((float*)(y + (int64_t)n * C + c), my_sum / HW);
        }
    }

    // --- LAUNCHER IMPLEMENTATIONS ---
    void launch_fused_dense_layer_v2(at::Tensor x, at::Tensor bn_w, at::Tensor bn_b, at::Tensor mean, at::Tensor var, at::Tensor conv_w, at::Tensor y_map, int write_offset_c, float eps) {
        const int N = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
        const int C_out = conv_w.size(0);
        const int C_total = y_map.size(1);

        const dim3 block_dim(DENSE_OUT_TILE_DIM, DENSE_OUT_TILE_DIM);
        const dim3 grid_dim((W + block_dim.x - 1) / block_dim.x, (H + block_dim.y - 1) / block_dim.y, (long long)N * C_out);
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_dense_layer_v2_launcher", ([&] {
            using vec_t = typename Vec2<scalar_t>::type;
            fused_dense_layer_kernel_v2<scalar_t, vec_t><<<grid_dim, block_dim>>>(
                x.data_ptr<scalar_t>(), bn_w.data_ptr<scalar_t>(), bn_b.data_ptr<scalar_t>(),
                mean.data_ptr<scalar_t>(), var.data_ptr<scalar_t>(), conv_w.data_ptr<scalar_t>(),
                y_map.data_ptr<scalar_t>(), write_offset_c, C_total,
                eps, N, C_in, H, W, C_out);
        }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    void launch_fused_transition_vec(at::Tensor x, at::Tensor bn_w, at::Tensor bn_b, at::Tensor mean, at::Tensor var, at::Tensor conv_w, at::Tensor y, float eps) {
        const int N = x.size(0), C_in = x.size(1), H = x.size(2), W = x.size(3);
        const int C_out = y.size(1), H_out = y.size(2), W_out = y.size(3);
        const dim3 block_dim(TRANS_OUT_TILE_W, TRANS_OUT_TILE_H);
        const dim3 grid_dim((W_out + block_dim.x - 1) / block_dim.x, (H_out + block_dim.y - 1) / block_dim.y, (long long)N * C_out);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_transition_vec_launcher", ([&] {
            using vec_t = typename Vec2<scalar_t>::type;
            tiled_fused_transition_kernel_vec<scalar_t, vec_t><<<grid_dim, block_dim>>>(
                x.data_ptr<scalar_t>(), bn_w.data_ptr<scalar_t>(), bn_b.data_ptr<scalar_t>(),
                mean.data_ptr<scalar_t>(), var.data_ptr<scalar_t>(), conv_w.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(), eps, N, C_in, H, W, C_out);
        }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    void launch_fused_final_op_vec(at::Tensor x, at::Tensor bn_w, at::Tensor bn_b, at::Tensor mean, at::Tensor var, at::Tensor y, float eps) {
        const int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
        const int threads = 256;
        const int warps_per_block = threads / 32;
        const dim3 blocks(N, (C + warps_per_block - 1) / warps_per_block);
        
        y.zero_(); // Zero out y before atomic add

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "fused_final_op_vec_launcher", ([&] {
            using vec_t = typename Vec2<scalar_t>::type;
            fused_final_op_kernel_vec<scalar_t, vec_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(), bn_w.data_ptr<scalar_t>(), bn_b.data_ptr<scalar_t>(),
                mean.data_ptr<scalar_t>(), var.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), 
                eps, N, C, H, W);
        }));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    """

    # --- C++ Source (Python Interface) ---
    cpp_source = """
    #include <torch/extension.h>

    // Forward declarations
    void launch_fused_dense_layer_v2(at::Tensor x, at::Tensor bn_w, at::Tensor bn_b, at::Tensor mean, at::Tensor var, at::Tensor conv_w, at::Tensor y_map, int write_offset_c, float eps);
    void launch_fused_transition_vec(at::Tensor x, at::Tensor bn_w, at::Tensor bn_b, at::Tensor mean, at::Tensor var, at::Tensor conv_w, at::Tensor y, float eps);
    void launch_fused_final_op_vec(at::Tensor x, at::Tensor bn_w, at::Tensor bn_b, at::Tensor mean, at::Tensor var, at::Tensor y, float eps);

    void fused_dense_layer_inplace_forward_v2(
            torch::Tensor x, torch::Tensor bn_w, torch::Tensor bn_b,
            torch::Tensor mean, torch::Tensor var, torch::Tensor conv_w,
            torch::Tensor y_map, int write_offset_c, double eps) {
        TORCH_CHECK(x.size(3) % 2 == 0, "Vectorized dense kernel requires input width to be even.");
        launch_fused_dense_layer_v2(x, bn_w, bn_b, mean, var, conv_w, y_map, write_offset_c, static_cast<float>(eps));
    }

    torch::Tensor fused_transition_forward_vec(
            torch::Tensor x, torch::Tensor bn_w, torch::Tensor bn_b,
            torch::Tensor mean, torch::Tensor var, torch::Tensor conv_w, double eps) {
        const int N = x.size(0), H = x.size(2), W = x.size(3), C_out = conv_w.size(0);
        TORCH_CHECK(W % 2 == 0, "Vectorized transition kernel requires input width to be even.");
        auto y = torch::empty({N, C_out, H / 2, W / 2}, x.options());
        launch_fused_transition_vec(x, bn_w, bn_b, mean, var, conv_w, y, static_cast<float>(eps));
        return y;
    }

    torch::Tensor fused_final_op_forward_vec(
            torch::Tensor x, torch::Tensor bn_w, torch::Tensor bn_b,
            torch::Tensor mean, torch::Tensor var, double eps) {
        const auto N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
        TORCH_CHECK((H * W) % 2 == 0, "Vectorized final op kernel requires feature map area to be even.");
        auto y = torch::empty({N, C}, x.options());
        launch_fused_final_op_vec(x, bn_w, bn_b, mean, var, y, static_cast<float>(eps));
        return y;
    }
    """

    fused_ops = load_inline(
        name="densenet_fused_ops_v2",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=[
            'fused_dense_layer_inplace_forward_v2',
            'fused_transition_forward_vec',
            'fused_final_op_forward_vec'
        ],
        verbose=False,
        extra_cuda_cflags=['-O3', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']
    )
    USE_CUSTOM_KERNELS = True
except Exception as e:
    print(f"Skipping custom CUDA kernels due to compilation error: {e}")
    fused_ops = None
    USE_CUSTOM_KERNELS = False


# --- Autograd and Module Wrappers ---

class FusedDenseLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_slice, out_map, write_offset, bn_module, conv_module, bn_weight, bn_bias, conv_weight):
        if bn_module.training:
            with torch.cuda.amp.autocast(enabled=False):
                F.batch_norm(x_slice.float(), bn_module.running_mean, bn_module.running_var, bn_weight.float(), bn_bias.float(),
                            True, bn_module.momentum, bn_module.eps)
        
        conv_weight_cont = conv_weight.contiguous()
        fused_ops.fused_dense_layer_inplace_forward_v2(
            x_slice, bn_weight, bn_bias, bn_module.running_mean,
            bn_module.running_var, conv_weight_cont, out_map, write_offset, bn_module.eps
        )
        
        ctx.save_for_backward(x_slice, bn_weight, bn_bias, bn_module.running_mean, 
                              bn_module.running_var, conv_weight_cont)
        ctx.bn_eps = bn_module.eps
        ctx.padding = conv_module.padding[0]
        return out_map.narrow(1, write_offset, conv_module.out_channels)

    @staticmethod
    def backward(ctx, grad_output):
        x, bn_w, bn_b, mean, var, conv_w, = ctx.saved_tensors
        with torch.enable_grad():
            x_recomp = x.detach().requires_grad_(True)
            bn_w_recomp = bn_w.detach().requires_grad_(True)
            bn_b_recomp = bn_b.detach().requires_grad_(True)
            conv_w_recomp = conv_w.detach().requires_grad_(True)
            
            bn_out = F.batch_norm(x_recomp, mean, var, bn_w_recomp, bn_b_recomp, False, 0, ctx.bn_eps)
            relu_out = F.relu(bn_out, inplace=True)
            conv_out = F.conv2d(relu_out, conv_w_recomp, padding=ctx.padding)
        
        grad_x, grad_bn_w, grad_bn_b, grad_conv_w = torch.autograd.grad(
            outputs=conv_out, inputs=[x_recomp, bn_w_recomp, bn_b_recomp, conv_w_recomp], grad_outputs=grad_output)
        return grad_x, None, None, None, None, grad_bn_w, grad_bn_b, grad_conv_w

class FusedDenseLayer(nn.Module):
    def __init__(self, in_features, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False)
    def forward(self, x_slice, out_map, write_offset):
        return FusedDenseLayerFunction.apply(x_slice, out_map, write_offset, self.bn, self.conv, self.bn.weight, self.bn.bias, self.conv.weight)

class FusedTransitionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bn_module, conv_module, bn_weight, bn_bias, conv_weight):
        if bn_module.training:
            with torch.cuda.amp.autocast(enabled=False):
                F.batch_norm(x.float(), bn_module.running_mean, bn_module.running_var, bn_weight.float(), bn_bias.float(), True, bn_module.momentum, bn_module.eps)
        output = fused_ops.fused_transition_forward_vec(x, bn_weight, bn_bias, bn_module.running_mean, bn_module.running_var, conv_weight.contiguous().squeeze(), bn_module.eps)
        ctx.save_for_backward(x, bn_weight, bn_bias, bn_module.running_mean, bn_module.running_var, conv_weight.contiguous())
        ctx.bn_eps = bn_module.eps
        return output
    @staticmethod
    def backward(ctx, grad_output):
        x, bn_w, bn_b, mean, var, conv_w = ctx.saved_tensors
        with torch.enable_grad():
            x_r, bn_w_r, bn_b_r, conv_w_r = x.detach().requires_grad_(True), bn_w.detach().requires_grad_(True), bn_b.detach().requires_grad_(True), conv_w.detach().requires_grad_(True)
            bn_out = F.batch_norm(x_r, mean, var, bn_w_r, bn_b_r, False, 0, ctx.bn_eps)
            relu_out = F.relu(bn_out, inplace=True)
            conv_out = F.conv2d(relu_out, conv_w_r, bias=None, kernel_size=1)
            pool_out = F.avg_pool2d(conv_out, kernel_size=2, stride=2)
        grad_x, grad_bn_w, grad_bn_b, grad_conv_w = torch.autograd.grad(outputs=pool_out, inputs=[x_r, bn_w_r, bn_b_r, conv_w_r], grad_outputs=grad_output)
        return grad_x, None, None, grad_bn_w, grad_bn_b, grad_conv_w

class FusedTransition(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
    def forward(self, x):
        return FusedTransitionFunction.apply(x, self.bn, self.conv, self.bn.weight, self.bn.bias, self.conv.weight)

class FusedFinalOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bn_module, bn_weight, bn_bias):
        if bn_module.training:
            with torch.cuda.amp.autocast(enabled=False):
                F.batch_norm(x.float(), bn_module.running_mean, bn_module.running_var, bn_weight.float(), bn_bias.float(), True, bn_module.momentum, bn_module.eps)
        output = fused_ops.fused_final_op_forward_vec(x, bn_weight, bn_bias, bn_module.running_mean, bn_module.running_var, bn_module.eps)
        ctx.save_for_backward(x, bn_weight, bn_bias, bn_module.running_mean, bn_module.running_var)
        ctx.bn_eps = bn_module.eps
        return output
    @staticmethod
    def backward(ctx, grad_output):
        x, bn_w, bn_b, mean, var = ctx.saved_tensors
        with torch.enable_grad():
            x_r, bn_w_r, bn_b_r = x.detach().requires_grad_(True), bn_w.detach().requires_grad_(True), bn_b.detach().requires_grad_(True)
            bn_out = F.batch_norm(x_r, mean, var, bn_w_r, bn_b_r, False, 0, ctx.bn_eps)
            relu_out = F.relu(bn_out, inplace=True)
            pool_out = F.adaptive_avg_pool2d(relu_out, (1, 1))
        grad_x, grad_bn_w, grad_bn_b = torch.autograd.grad(outputs=pool_out, inputs=[x_r, bn_w_r, bn_b_r], grad_outputs=grad_output.unsqueeze(-1).unsqueeze(-1))
        return grad_x, None, grad_bn_w, grad_bn_b

# --- Modified Model Architecture ---

class DenseBlockNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlockNew, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            if USE_CUSTOM_KERNELS:
                self.layers.append(FusedDenseLayer(in_features, growth_rate))
            else:
                self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(in_features), nn.ReLU(inplace=True),
                    nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_custom_path = USE_CUSTOM_KERNELS and x.is_cuda and x.shape[3] % 2 == 0
        
        if not use_custom_path:
            features = [x]
            for layer in self.layers:
                layer_input = torch.cat(features, 1)
                new_feature = layer(layer_input)
                features.append(new_feature)
            return torch.cat(features, 1)

        num_input_features = x.shape[1]
        final_num_features = num_input_features + self.num_layers * self.growth_rate
        B, _, H, W = x.shape
        
        # Pre-allocate the full output tensor to avoid torch.cat
        feature_map = x.new_empty((B, final_num_features, H, W))
        feature_map.narrow(1, 0, num_input_features).copy_(x)
        
        write_offset = num_input_features
        for i, layer in enumerate(self.layers):
            current_features_count = num_input_features + i * self.growth_rate
            # Create a view of the feature map to be used as input for the current layer
            layer_input = feature_map.narrow(1, 0, current_features_count)
            # The custom kernel writes its output directly into the correct slice of the feature map
            layer(layer_input, feature_map, write_offset)
            write_offset += self.growth_rate
            
        return feature_map

class TransitionLayerNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayerNew, self).__init__()
        self.use_custom_kernel = USE_CUSTOM_KERNELS
        if self.use_custom_kernel:
            self.fused_transition = FusedTransition(num_input_features, num_output_features)
        self.fallback_transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features), nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_custom_kernel and x.is_cuda and x.shape[3] % 2 == 0:
            return self.fused_transition(x)
        return self.fallback_transition(x)

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        num_features = 64
        block_layers = [6, 12, 48, 32] # DenseNet-201

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlockNew(num_layers, num_features, growth_rate)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayerNew(num_features, num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        self.half() # Convert model parameters to FP16
        # Classifier should remain in FP32 for stability
        self.classifier.float()
        # BatchNorm running stats should be FP32
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input should be converted to FP16 for performance
        x = x.half()
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)
        
        use_final_kernel = USE_CUSTOM_KERNELS and x.is_cuda and (x.shape[2]*x.shape[3]) % 2 == 0
        if use_final_kernel:
            # Custom kernel handles BN -> ReLU -> Pool
            x = FusedFinalOpFunction.apply(x, self.final_bn, self.final_bn.weight.half(), self.final_bn.bias.half())
        else:
            # Fallback path
            x = self.final_bn(x)
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)

        # Final classification is done in FP32
        x = self.classifier(x.float())
        return x