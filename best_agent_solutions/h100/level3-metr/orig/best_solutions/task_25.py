import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- Custom CUDA Kernel Source ---
# This version introduces:
# 1. CUDA Streams to parallelize the group convolutions in the pointwise layer.
# 2. A redesigned depthwise kernel that processes multiple channels simultaneously
#    within a thread block to reduce synchronization and improve data locality.
# This version fixes compilation errors related to cuBLAS calls and stream management.
fused_shufflenet_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h> // For at::cuda::getCurrentCUDAStream
#include <vector>

// --- cuBLAS Handle Management ---
static cublasHandle_t get_cublas_handle() {
    static bool initialized = false;
    static cublasHandle_t handle;
    if (!initialized) {
        TORCH_CHECK(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cuBLAS handle creation failed");
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        initialized = true;
    }
    return handle;
}

// --- Stream Manager for Parallel Group Convolutions ---
// This manager provides a pool of CUDA streams to run group convolutions concurrently.
// It is not thread-safe but is suitable for typical single-threaded PyTorch models.
static std::vector<cudaStream_t>& get_cuda_streams(int num_streams) {
    static std::vector<cudaStream_t> streams;
    if (streams.size() < (size_t)num_streams) {
        for(size_t i = streams.size(); i < (size_t)num_streams; ++i) {
            cudaStream_t stream;
            C10_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            streams.push_back(stream);
        }
    }
    return streams;
}

// --- Fused Pointwise Post-Processing Kernels (Unchanged) ---
__global__ void pointwise_postprocess_kernel_vec4_shm(
    const half* __restrict__ gemm_out,
    const half* __restrict__ bn_scale, const half* __restrict__ bn_bias,
    const half* __restrict__ residual, half* __restrict__ final_out,
    const bool with_relu,
    const int N, const int C_out, const int H, const int W
) {
    extern __shared__ half shm_bn_params[];
    half* shm_bn_scale = shm_bn_params;
    half* shm_bn_bias = shm_bn_params + C_out;

    for(int i = threadIdx.x; i < C_out; i += blockDim.x) {
        shm_bn_scale[i] = bn_scale[i];
        shm_bn_bias[i] = bn_bias[i];
    }
    __syncthreads();

    const size_t total_elements = (size_t)N * C_out * H * W;
    const __half2* gemm_out_h2 = reinterpret_cast<const __half2*>(gemm_out);
    const __half2* residual_h2 = residual ? reinterpret_cast<const __half2*>(residual) : nullptr;
    __half2* final_out_h2 = reinterpret_cast<__half2*>(final_out);

    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements / 4; idx += (size_t)gridDim.x * blockDim.x) {
        const size_t linear_start_idx = idx * 4;
        const int c = (linear_start_idx / (size_t)(H * W)) % C_out;

        const __half2 gemm_h2_a = gemm_out_h2[idx * 2];
        const __half2 gemm_h2_b = gemm_out_h2[idx * 2 + 1];
        const float2 gemm_f2_a = __half22float2(gemm_h2_a);
        const float2 gemm_f2_b = __half22float2(gemm_h2_b);

        const float bn_s = __half2float(shm_bn_scale[c]);
        const float bn_b = __half2float(shm_bn_bias[c]);

        float4 processed_vals;
        processed_vals.x = fmaf(gemm_f2_a.x, bn_s, bn_b);
        processed_vals.y = fmaf(gemm_f2_a.y, bn_s, bn_b);
        processed_vals.z = fmaf(gemm_f2_b.x, bn_s, bn_b);
        processed_vals.w = fmaf(gemm_f2_b.y, bn_s, bn_b);

        if (with_relu) {
            processed_vals.x = fmaxf(processed_vals.x, 0.0f);
            processed_vals.y = fmaxf(processed_vals.y, 0.0f);
            processed_vals.z = fmaxf(processed_vals.z, 0.0f);
            processed_vals.w = fmaxf(processed_vals.w, 0.0f);
        }

        if (residual) {
            const __half2 res_h2_a = residual_h2[idx * 2];
            const __half2 res_h2_b = residual_h2[idx * 2 + 1];
            processed_vals.x += __half2float(res_h2_a.x);
            processed_vals.y += __half2float(res_h2_a.y);
            processed_vals.z += __half2float(res_h2_b.x);
            processed_vals.w += __half2float(res_h2_b.y);
        }

        final_out_h2[idx * 2]     = __floats2half2_rn(processed_vals.x, processed_vals.y);
        final_out_h2[idx * 2 + 1] = __floats2half2_rn(processed_vals.z, processed_vals.w);
    }
}


// --- C++ Dispatcher for Pointwise Conv (Optimized with CUDA Streams) ---
torch::Tensor cublas_fused_pointwise_cuda(
    torch::Tensor input, torch::Tensor conv_w,
    torch::Tensor bn_scale, torch::Tensor bn_bias,
    c10::optional<torch::Tensor> residual, bool with_relu, int groups)
{
    const auto N = input.size(0), C_in = input.size(1), H = input.size(2), W = input.size(3);
    const auto C_out = conv_w.size(0);
    TORCH_CHECK(input.is_contiguous() && conv_w.is_contiguous(), "Inputs must be contiguous");
    auto gemm_out = torch::empty({N, C_out, H, W}, input.options());

    const int m = C_out / groups;
    const int n = H * W;
    const int k = C_in / groups;
    const int batch_count = N;
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    cublasHandle_t handle = get_cublas_handle();

    // Use CUDA streams to launch group convolutions in parallel
    auto& streams = get_cuda_streams(groups);
    for (int g = 0; g < groups; ++g) {
        const half* weight_g = (const half*)conv_w.data_ptr<at::Half>() + g * m * k;
        const half* input_g = (const half*)input.data_ptr<at::Half>() + g * k * n;
        half* gemm_out_g = (half*)gemm_out.data_ptr<at::Half>() + g * m * n;
        const long long stride_input = (long long)C_in * n;
        const long long stride_weight = 0;
        const long long stride_output = (long long)C_out * n;

        // FIX: Use TORCH_CHECK for cublasStatus_t, not C10_CUDA_CHECK
        TORCH_CHECK(cublasSetStream(handle, streams[g]) == CUBLAS_STATUS_SUCCESS, "cublasSetStream failed for group " + std::to_string(g));
        cublasStatus_t status = cublasHgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
            input_g, n, stride_input,
            weight_g, k, stride_weight,
            &beta, gemm_out_g, n, stride_output,
            batch_count
        );
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasHgemmStridedBatched failed for group " + std::to_string(g));
    }
    
    // Synchronize all streams to ensure GEMMs are complete before post-processing
    for (int g = 0; g < groups; ++g) {
        C10_CUDA_CHECK(cudaStreamSynchronize(streams[g]));
    }
    // Reset cuBLAS to use the default PyTorch stream for subsequent operations
    // FIX: Use TORCH_CHECK, and at::cuda::getCurrentCUDAStream instead of c10::cuda
    TORCH_CHECK(cublasSetStream(handle, at::cuda::getCurrentCUDAStream()) == CUBLAS_STATUS_SUCCESS, "cublasSetStream failed to reset to default stream");

    auto output = torch::empty_like(gemm_out);
    const int threads = 256;
    const half* residual_ptr = residual.has_value() ? (const half*)residual->data_ptr<at::Half>() : nullptr;
    
    const size_t total_elements = gemm_out.numel();
    const int num_blocks = (total_elements / 4 + threads - 1) / threads;
    const size_t shm_size = (size_t)C_out * 2 * sizeof(half); // scale + bias
    pointwise_postprocess_kernel_vec4_shm<<<num_blocks, threads, shm_size, at::cuda::getCurrentCUDAStream()>>>(
        (const half*)gemm_out.data_ptr<at::Half>(), (const half*)bn_scale.data_ptr<at::Half>(),
        (const half*)bn_bias.data_ptr<at::Half>(), residual_ptr,
        (half*)output.data_ptr<at::Half>(), with_relu, N, C_out, H, W);
    
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

// --- Kernel 2: Multi-Channel Fused Depthwise Conv + BN + Channel Shuffle ---
#define KERNEL_SIZE 3
#define KERNEL_DIM (KERNEL_SIZE * KERNEL_SIZE)
#define CHANNELS_PER_BLOCK 4
#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 16
#define VEC_SIZE 4
#define TILE_W_PIXELS (BLOCK_DIM_X * VEC_SIZE)
#define TILE_H_PIXELS BLOCK_DIM_Y
#define SHM_W (TILE_W_PIXELS + KERNEL_SIZE - 1)
#define SHM_H (TILE_H_PIXELS + KERNEL_SIZE - 1)

__global__ void fused_depthwise_multichan_half_kernel(
    const half* __restrict__ input, const half* __restrict__ conv_w,
    const half* __restrict__ bn_scale, const half* __restrict__ bn_bias,
    half* __restrict__ output, const int N, const int C, const int H, const int W,
    const int groups)
{
    __shared__ half shm_input[CHANNELS_PER_BLOCK][SHM_H][SHM_W];

    const int num_channel_groups_in_grid = (C + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    const int n = blockIdx.z / num_channel_groups_in_grid;
    const int c_start = (blockIdx.z % num_channel_groups_in_grid) * CHANNELS_PER_BLOCK;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base_in_x = blockIdx.x * TILE_W_PIXELS - (KERNEL_SIZE / 2);
    const int base_in_y = blockIdx.y * TILE_H_PIXELS - (KERNEL_SIZE / 2);
    const int tid = ty * blockDim.x + tx;
    const int num_threads = blockDim.x * blockDim.y;
    const half zero_h = __float2half(0.0f);

    // --- Load Phase: Collaboratively load input tiles for 4 channels ---
    for (int i = tid; i < CHANNELS_PER_BLOCK * SHM_H * SHM_W; i += num_threads) {
        const int c_offset = i / (SHM_H * SHM_W);
        const int plane_idx = i % (SHM_H * SHM_W);
        const int shm_y = plane_idx / SHM_W, shm_x = plane_idx % SHM_W;
        const int c = c_start + c_offset;
        if (c < C) {
            const int in_y = base_in_y + shm_y, in_x = base_in_x + shm_x;
            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                shm_input[c_offset][shm_y][shm_x] = input[(size_t)n * C * H * W + (size_t)c * H * W + (size_t)in_y * W + in_x];
            } else {
                shm_input[c_offset][shm_y][shm_x] = zero_h;
            }
        } else {
            shm_input[c_offset][shm_y][shm_x] = zero_h;
        }
    }
    __syncthreads();

    // --- Compute and Write Phase ---
    const int out_y = blockIdx.y * TILE_H_PIXELS + ty;
    const int out_x_base = blockIdx.x * TILE_W_PIXELS + tx * VEC_SIZE;
    if (out_y < H && out_x_base < W) {
        float4 accs[CHANNELS_PER_BLOCK];
        #pragma unroll
        for(int i = 0; i < CHANNELS_PER_BLOCK; ++i) accs[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const int shm_read_y_base = ty + (KERNEL_SIZE / 2), shm_read_x_base = tx * VEC_SIZE + (KERNEL_SIZE / 2);

        #pragma unroll
        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                const int shm_y_idx = shm_read_y_base - 1 + kh, shm_x_idx = shm_read_x_base - 1 + kw;
                #pragma unroll
                for (int c_offset = 0; c_offset < CHANNELS_PER_BLOCK; ++c_offset) {
                    const int c = c_start + c_offset;
                    if (c < C) {
                        const float weight_val = __half2float(conv_w[c * KERNEL_DIM + kh * KERNEL_SIZE + kw]);
                        const float p0 = __half2float(shm_input[c_offset][shm_y_idx][shm_x_idx + 0]);
                        const float p1 = __half2float(shm_input[c_offset][shm_y_idx][shm_x_idx + 1]);
                        const float p2 = __half2float(shm_input[c_offset][shm_y_idx][shm_x_idx + 2]);
                        const float p3 = __half2float(shm_input[c_offset][shm_y_idx][shm_x_idx + 3]);
                        accs[c_offset].x = fmaf(p0, weight_val, accs[c_offset].x);
                        accs[c_offset].y = fmaf(p1, weight_val, accs[c_offset].y);
                        accs[c_offset].z = fmaf(p2, weight_val, accs[c_offset].z);
                        accs[c_offset].w = fmaf(p3, weight_val, accs[c_offset].w);
                    }
                }
            }
        }

        #pragma unroll
        for (int c_offset = 0; c_offset < CHANNELS_PER_BLOCK; ++c_offset) {
            const int c = c_start + c_offset;
            if (c < C) {
                float4 val = accs[c_offset];
                const float bn_s = __half2float(bn_scale[c]), bn_b = __half2float(bn_bias[c]);
                val.x = fmaf(val.x, bn_s, bn_b); val.y = fmaf(val.y, bn_s, bn_b);
                val.z = fmaf(val.z, bn_s, bn_b); val.w = fmaf(val.w, bn_s, bn_b);

                const int channels_per_group = C / groups, g = c / channels_per_group, c_in_group = c % channels_per_group;
                const int c_new = c_in_group * groups + g;
                const size_t out_base_addr = (size_t)n * C * H * W + (size_t)c_new * H * W + (size_t)out_y * W + out_x_base;
                
                if (out_x_base + VEC_SIZE <= W) {
                    reinterpret_cast<__half2*>(&output[out_base_addr])[0] = __floats2half2_rn(val.x, val.y);
                    reinterpret_cast<__half2*>(&output[out_base_addr])[1] = __floats2half2_rn(val.z, val.w);
                }
            }
        }
    }
}

torch::Tensor fused_depthwise_conv_bn_shuffle_cuda(
    torch::Tensor input, torch::Tensor conv_w, torch::Tensor bn_scale, torch::Tensor bn_bias, int groups)
{
    const auto N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    TORCH_CHECK(W % VEC_SIZE == 0, "Vectorized depthwise kernel requires width to be a multiple of 4.");
    auto output = torch::empty_like(input);
    
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
    const int num_channel_groups_in_grid = (C + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK;
    dim3 grid_dim( (W + TILE_W_PIXELS - 1) / TILE_W_PIXELS, (H + TILE_H_PIXELS - 1) / TILE_H_PIXELS, (unsigned int)(N * num_channel_groups_in_grid) );
    
    fused_depthwise_multichan_half_kernel<<<grid_dim, block_dim, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*)input.data_ptr<at::Half>(), (const half*)conv_w.data_ptr<at::Half>(),
        (const half*)bn_scale.data_ptr<at::Half>(), (const half*)bn_bias.data_ptr<at::Half>(),
        (half*)output.data_ptr<at::Half>(), N, C, H, W, groups);
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

# --- C++ Source for Signatures ---
fused_shufflenet_cpp_source = """
torch::Tensor cublas_fused_pointwise_cuda(
    torch::Tensor input, torch::Tensor conv_w,
    torch::Tensor bn_scale, torch::Tensor bn_bias,
    c10::optional<torch::Tensor> residual, bool with_relu, int groups);

torch::Tensor fused_depthwise_conv_bn_shuffle_cuda(
    torch::Tensor input, torch::Tensor conv_w,
    torch::Tensor bn_scale, torch::Tensor bn_bias, int groups);
"""

# --- JIT Compilation ---
fused_ops = load_inline(
    name="fused_shufflenet_kernels_v8_fixed",
    cpp_sources=fused_shufflenet_cpp_source,
    cuda_sources=fused_shufflenet_kernels_source,
    functions=["cublas_fused_pointwise_cuda", "fused_depthwise_conv_bn_shuffle_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_70"],
    extra_ldflags=["-lcublas"],
)

# --- Python Module Wrappers ---
class FusedPointwise(nn.Module):
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        super().__init__()
        bn.eval()
        self.groups = conv.groups
        gamma, beta, mean, var, eps = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
        std = torch.sqrt(var + eps)
        bn_scale_fp32 = gamma / std
        bn_bias_fp32 = beta - (mean * bn_scale_fp32)
        conv_weight_flat_fp32 = conv.weight.detach().view(conv.out_channels, -1)
        self.register_buffer('bn_scale', bn_scale_fp32.to(torch.float16))
        self.register_buffer('bn_bias', bn_bias_fp32.to(torch.float16))
        self.register_buffer('conv_weight_flat', conv_weight_flat_fp32.to(torch.float16))

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None, with_relu: bool = False):
        return fused_ops.cublas_fused_pointwise_cuda(
            x, self.conv_weight_flat, self.bn_scale, self.bn_bias, residual, with_relu, self.groups)

class FusedDepthwiseShuffle(nn.Module):
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d, shuffle_groups: int):
        super().__init__()
        bn.eval()
        self.shuffle_groups = shuffle_groups
        gamma, beta, mean, var, eps = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
        std = torch.sqrt(var + eps)
        bn_scale_fp32 = gamma / std
        bn_bias_fp32 = beta - (mean * bn_scale_fp32)
        conv_weight_flat_fp32 = conv.weight.detach().view(conv.out_channels, -1)
        self.register_buffer('bn_scale', bn_scale_fp32.to(torch.float16))
        self.register_buffer('bn_bias', bn_bias_fp32.to(torch.float16))
        self.register_buffer('conv_weight_flat', conv_weight_flat_fp32.to(torch.float16))

    def forward(self, x):
        return fused_ops.fused_depthwise_conv_bn_shuffle_cuda(
            x, self.conv_weight_flat, self.bn_scale, self.bn_bias, self.shuffle_groups)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ModelNew, self).__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, groups=groups, bias=False)
        bn1 = nn.BatchNorm2d(mid_channels)
        conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False)
        bn2 = nn.BatchNorm2d(mid_channels)
        conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1, 0, groups=groups, bias=False)
        bn3 = nn.BatchNorm2d(out_channels)
        
        self.block1 = FusedPointwise(conv1, bn1)
        self.block2 = FusedDepthwiseShuffle(conv2, bn2, groups)
        self.block3 = FusedPointwise(conv3, bn3)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
            shortcut_bn = nn.BatchNorm2d(out_channels)
            self.shortcut = FusedPointwise(shortcut_conv, shortcut_bn)
        
        self.half()
        self.eval()
    
    def forward(self, x):
        x_half = x.half()

        if isinstance(self.shortcut, FusedPointwise):
            shortcut_out = self.shortcut(x_half, with_relu=False)
        else:
            shortcut_out = self.shortcut(x_half)

        main_out = self.block1(x_half, with_relu=True)
        main_out = self.block2(main_out)
        final_out = self.block3(main_out, residual=shortcut_out, with_relu=True)
        
        return final_out.float()

# --- Helper Functions ---
batch_size = 10
input_channels = 240
out_channels = 480
groups = 3
height = 224
width = 224

def get_inputs():
    assert width % 4 == 0, "Input width must be a multiple of 4 for the optimized kernel"
    return [torch.randn(batch_size, input_channels, height, width).cuda()]

def get_init_inputs():
    return [input_channels, out_channels, groups]

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x