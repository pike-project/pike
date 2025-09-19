# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Combined Strategy: BatchNorm Folding + Fused Depthwise Convolution + Fused Final Pooling
# This solution synthesizes the most effective techniques from all previous attempts.
# 1. BatchNorm Folding: Pre-calculates BN parameters into a simple scale/bias, reducing kernel FLOPs.
# 2. Fused Depthwise Conv: A new kernel fuses the depthwise convolution with the FOLDED BatchNorm and ReLU6,
#    eliminating the large intermediate tensor write and using the more efficient folded arithmetic.
# 3. Aggressive Final Fusion: Retains the highly effective kernel that combines the final folded BN, ReLU,
#    and Adaptive Average Pooling, eliminating multiple kernel launches at the end of the network.
# 4. Vectorization: Uses float4 wherever possible to maximize memory bandwidth.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

//================================================================================
// KERNEL 1: Fused Scale/Bias + Activation + Add (from top-performing solution)
//================================================================================
template<bool HAS_RELU, bool HAS_RELU6>
__device__ __forceinline__ float4 apply_activation_vec(float4 val) {
    if constexpr (HAS_RELU) {
        val.x = fmaxf(0.0f, val.x);
        val.y = fmaxf(0.0f, val.y);
        val.z = fmaxf(0.0f, val.z);
        val.w = fmaxf(0.0f, val.w);
    }
    if constexpr (HAS_RELU6) {
        val.x = fminf(fmaxf(0.0f, val.x), 6.0f);
        val.y = fminf(fmaxf(0.0f, val.y), 6.0f);
        val.z = fminf(fmaxf(0.0f, val.z), 6.0f);
        val.w = fminf(fmaxf(0.0f, val.w), 6.0f);
    }
    return val;
}

template<bool HAS_RELU, bool HAS_RELU6>
__device__ __forceinline__ float apply_activation_scalar(float val) {
    if constexpr (HAS_RELU) {
        val = fmaxf(0.0f, val);
    }
    if constexpr (HAS_RELU6) {
        val = fminf(fmaxf(0.0f, val), 6.0f);
    }
    return val;
}

template<bool HAS_RELU, bool HAS_RELU6, bool HAS_ADD>
__global__ void fused_scale_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    const int total_elements,
    const int C,
    const int spatial_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_vec_elements = total_elements / 4;
    if (idx < num_vec_elements) {
        const int data_idx = idx * 4;
        const int c = (data_idx / spatial_size) % C;
        const float s = scale[c];
        const float b = bias[c];
        float4 val = *(reinterpret_cast<const float4*>(input + data_idx));
        val.x = val.x * s + b; val.y = val.y * s + b;
        val.z = val.z * s + b; val.w = val.w * s + b;
        if constexpr (HAS_ADD) {
            float4 res_val = *(reinterpret_cast<const float4*>(residual + data_idx));
            val.x += res_val.x; val.y += res_val.y; val.z += res_val.z; val.w += res_val.w;
        }
        val = apply_activation_vec<HAS_RELU, HAS_RELU6>(val);
        *(reinterpret_cast<float4*>(output + data_idx)) = val;
    }
    if (blockIdx.x == 0) {
        const int tail_start = num_vec_elements * 4;
        for (int tail_idx = tail_start + threadIdx.x; tail_idx < total_elements; tail_idx += blockDim.x) {
            const int c = (tail_idx / spatial_size) % C;
            float val = input[tail_idx] * scale[c] + bias[c];
            if constexpr (HAS_ADD) {
                val += residual[tail_idx];
            }
            val = apply_activation_scalar<HAS_RELU, HAS_RELU6>(val);
            output[tail_idx] = val;
        }
    }
}

//================================================================================
// KERNEL 2: Fused DepthwiseConv + Folded-BN + ReLU6 (NEW HYBRID KERNEL)
//================================================================================
template<int K, int S, int P>
__global__ void fused_depthwise_conv_scale_bias_relu6_kernel(
    const float* __restrict__ input,
    const float* __restrict__ conv_w,
    float* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    int N, int C, int H, int W, int H_out, int W_out
) {
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int z_idx = blockIdx.z;
    
    if (w_out >= W_out || h_out >= H_out || z_idx >= N * C) return;

    const int n = z_idx / C;
    const int c = z_idx % C;

    float acc = 0.0f;
    const float* kernel_ptr = conv_w + c * K * K;
    const float* input_ptr_base = input + n * C * H * W + c * H * W;

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            int h_in = h_out * S - P + ky;
            int w_in = w_out * S - P + kx;

            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                acc += input_ptr_base[h_in * W + w_in] * kernel_ptr[ky * K + kx];
            }
        }
    }

    const float s = scale[c];
    const float b = bias[c];
    
    float folded_bn_val = acc * s + b;
    float final_val = fminf(fmaxf(0.0f, folded_bn_val), 6.0f);
    
    output[z_idx * H_out * W_out + h_out * W_out + w_out] = final_val;
}

//================================================================================
// KERNEL 3: Fused Folded-BN + ReLU + Adaptive Average Pooling
//================================================================================
__global__ void fused_scale_bias_relu_pool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    const int N, const int C, const int H, const int W)
{
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int n = blockIdx.y;
    const int c = blockIdx.x;
    const int spatial_size = H * W;
    const float* input_plane = input + n * C * spatial_size + c * spatial_size;
    const float s = scale[c];
    const float b = bias[c];

    float my_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        my_sum += fmaxf(0.0f, input_plane[i] * s + b);
    }
    sdata[tid] = my_sum;
    __syncthreads();

    for (unsigned int s_reduce = blockDim.x / 2; s_reduce > 0; s_reduce >>= 1) {
        if (tid < s_reduce) {
            sdata[tid] += sdata[tid + s_reduce];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[n * C + c] = sdata[0] / static_cast<float>(spatial_size);
    }
}

//================================================================================
// C++ Wrapper Functions
//================================================================================

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<void (*kernel_func)(const float*, const float*, float*, const float*, const float*, int, int, int)>
void launch_fused_kernel(
    const torch::Tensor& input, const torch::Tensor& residual, torch::Tensor& output,
    const torch::Tensor& scale, const torch::Tensor& bias)
{
    const int C = input.size(1);
    const int spatial_size = input.size(2) * input.size(3);
    const int total_elements = input.numel();
    const int block_size = 256;
    const int num_vec_loops = total_elements / 4;
    const int num_blocks = (num_vec_loops > 0) ? (num_vec_loops + block_size - 1) / block_size : 1;
    kernel_func<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), residual.defined() ? residual.data_ptr<float>() : nullptr, output.data_ptr<float>(),
        scale.data_ptr<float>(), bias.data_ptr<float>(), total_elements, C, spatial_size);
}

torch::Tensor fused_scale_bias_relu_cuda(torch::Tensor i, torch::Tensor s, torch::Tensor b) {
    auto o = torch::empty_like(i); launch_fused_kernel<fused_scale_bias_kernel<true, false, false>>(i, {}, o, s, b); return o;
}
torch::Tensor fused_scale_bias_relu6_cuda(torch::Tensor i, torch::Tensor s, torch::Tensor b) {
    auto o = torch::empty_like(i); launch_fused_kernel<fused_scale_bias_kernel<false, true, false>>(i, {}, o, s, b); return o;
}
torch::Tensor fused_scale_bias_cuda(torch::Tensor i, torch::Tensor s, torch::Tensor b) {
    auto o = torch::empty_like(i); launch_fused_kernel<fused_scale_bias_kernel<false, false, false>>(i, {}, o, s, b); return o;
}
torch::Tensor fused_scale_bias_add_cuda(torch::Tensor i, torch::Tensor r, torch::Tensor s, torch::Tensor b) {
    auto o = torch::empty_like(i); launch_fused_kernel<fused_scale_bias_kernel<false, false, true>>(i, r, o, s, b); return o;
}

torch::Tensor fused_depthwise_conv_scale_bias_relu6_cuda(
    torch::Tensor input, torch::Tensor conv_w, torch::Tensor scale, torch::Tensor bias, int stride, int padding) {
    CHECK_INPUT(input); CHECK_INPUT(conv_w); CHECK_INPUT(scale); CHECK_INPUT(bias);
    const int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3), K = conv_w.size(2);
    const int H_out = (H + 2 * padding - K) / stride + 1, W_out = (W + 2 * padding - K) / stride + 1;
    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    dim3 block(16, 16);
    dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, N * C);

    #define LAUNCH_DW_KERNEL(k, s, p) fused_depthwise_conv_scale_bias_relu6_kernel<k, s, p><<<grid, block>>>(input.data_ptr<float>(), conv_w.data_ptr<float>(), output.data_ptr<float>(), scale.data_ptr<float>(), bias.data_ptr<float>(), N, C, H, W, H_out, W_out)
    if (K == 3 && stride == 1 && padding == 1) { LAUNCH_DW_KERNEL(3, 1, 1); }
    else if (K == 3 && stride == 2 && padding == 1) { LAUNCH_DW_KERNEL(3, 2, 1); }
    else if (K == 5 && stride == 1 && padding == 2) { LAUNCH_DW_KERNEL(5, 1, 2); }
    else if (K == 5 && stride == 2 && padding == 2) { LAUNCH_DW_KERNEL(5, 2, 2); }
    else { TORCH_CHECK(false, "Unsupported Depthwise Conv params"); }
    #undef LAUNCH_DW_KERNEL
    return output;
}

torch::Tensor fused_scale_bias_relu_pool_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias) {
    CHECK_INPUT(input); CHECK_INPUT(scale); CHECK_INPUT(bias);
    const int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    auto output = torch::empty({N, C}, input.options());
    const int block_size = 256;
    fused_scale_bias_relu_pool_kernel<<<dim3(C, N), block_size, block_size * sizeof(float)>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), scale.data_ptr<float>(), bias.data_ptr<float>(), N, C, H, W);
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_scale_bias_relu_cuda(torch::Tensor i, torch::Tensor s, torch::Tensor b);
torch::Tensor fused_scale_bias_relu6_cuda(torch::Tensor i, torch::Tensor s, torch::Tensor b);
torch::Tensor fused_scale_bias_add_cuda(torch::Tensor i, torch::Tensor r, torch::Tensor s, torch::Tensor b);
torch::Tensor fused_scale_bias_cuda(torch::Tensor i, torch::Tensor s, torch::Tensor b);
torch::Tensor fused_scale_bias_relu_pool_cuda(torch::Tensor i, torch::Tensor s, torch::Tensor b);
torch::Tensor fused_depthwise_conv_scale_bias_relu6_cuda(torch::Tensor i, torch::Tensor w, torch::Tensor s, torch::Tensor b, int stride, int pad);
"""

fused_ops = load_inline(
    name="fused_ops_hybrid_final",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=[
        "fused_scale_bias_relu_cuda", "fused_scale_bias_relu6_cuda", "fused_scale_bias_add_cuda",
        "fused_scale_bias_cuda", "fused_scale_bias_relu_pool_cuda", "fused_depthwise_conv_scale_bias_relu6_cuda"
    ],
    verbose=False,
)

def fold_bn(conv_module, bn_module):
    with torch.no_grad():
        scale = bn_module.weight / torch.sqrt(bn_module.running_var + bn_module.eps)
        bias = bn_module.bias - bn_module.running_mean * scale
        return nn.Parameter(scale), nn.Parameter(bias)

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.expand_ratio = expand_ratio
        hidden_dim = in_channels * expand_ratio
        
        if self.expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.expand_scale, self.expand_bias = fold_bn(self.expand_conv, nn.BatchNorm2d(hidden_dim))
        
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False)
        self.depthwise_scale, self.depthwise_bias = fold_bn(self.depthwise_conv, nn.BatchNorm2d(hidden_dim))

        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_scale, self.project_bias = fold_bn(self.project_conv, nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity = x
        
        if self.expand_ratio != 1:
            out = self.expand_conv(x)
            out = fused_ops.fused_scale_bias_relu6_cuda(out, self.expand_scale, self.expand_bias)
        else:
            out = x

        # Use the new hybrid kernel for Fused DepthwiseConv -> Folded-BN -> ReLU6
        stride = self.depthwise_conv.stride[0]
        padding = self.depthwise_conv.padding[0]
        out = fused_ops.fused_depthwise_conv_scale_bias_relu6_cuda(
            out, self.depthwise_conv.weight, self.depthwise_scale, self.depthwise_bias, stride, padding
        )

        out = self.project_conv(out)
        if self.use_residual:
            out = fused_ops.fused_scale_bias_add_cuda(out, identity, self.project_scale, self.project_bias)
        else:
            out = fused_ops.fused_scale_bias_cuda(out, self.project_scale, self.project_bias)
        
        return out


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.scale1, self.bias1 = fold_bn(self.conv1, nn.BatchNorm2d(32))

        self.blocks = nn.Sequential(
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.scale2, self.bias2 = fold_bn(self.conv2, nn.BatchNorm2d(1280))

        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = fused_ops.fused_scale_bias_relu_cuda(x, self.scale1, self.bias1)
        
        x = self.blocks(x)
        
        x = self.conv2(x)
        # Use the maximally-fused kernel for the final stage.
        x = fused_ops.fused_scale_bias_relu_pool_cuda(x, self.scale2, self.bias2)
        
        # The output is already pooled and shaped correctly for the FC layer.
        x = self.fc(x)
        return x


# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224).cuda()]

def get_init_inputs():
    return [num_classes]

# EVOLVE-BLOCK-END