# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define and compile custom CUDA kernels for aggressive fusion.
# - fused_bias_act_kernel: Fuses bias addition with an activation (ReLU/Sigmoid).
# - fused_bias_relu_avgpool_kernel: Fuses bias addition, ReLU, and global average pooling.
# - fused_linear_act_kernel: Fuses a linear (GEMM) operation with an optional bias and activation.
#   This is tailored for the small, awkwardly-sized matrices in the SE block, aiming to
#   reduce kernel launch overhead compared to calling cuBLAS + separate activation kernels.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused Bias + Activation Kernel (ReLU or Sigmoid)
template <int ACT_MODE>
__global__ void fused_bias_act_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int C,
    const int spatial_size) {

    const int c = blockIdx.x; // Channel index
    const int n = blockIdx.y; // Batch index

    const float bias_val = __ldg(bias + c);
    const int base_idx = (n * C + c) * spatial_size;
    const float* input_ptr = input + base_idx;
    float* output_ptr = output + base_idx;

    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = __ldg(input_ptr + i) + bias_val;
        if (ACT_MODE == 1) { // ReLU
            val = fmaxf(val, 0.0f);
        } else if (ACT_MODE == 2) { // Sigmoid
            val = 1.0f / (1.0f + expf(-val));
        }
        output_ptr[i] = val;
    }
}

// Fused Bias + ReLU + Average Pooling Kernel
__global__ void fused_bias_relu_avgpool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int spatial_size) {

    extern __shared__ float sdata[];
    const int n = blockIdx.y;
    const int c = blockIdx.x;
    const int C = gridDim.x;

    const float bias_val = __ldg(bias + c);
    const float* input_ptr = input + (n * C + c) * spatial_size;

    float partial_sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = __ldg(input_ptr + i) + bias_val;
        partial_sum += fmaxf(val, 0.0f);
    }
    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[n * C + c] = sdata[0] / static_cast<float>(spatial_size);
    }
}

// Fused Linear + Bias + Activation Kernel
template<int ACT_MODE>
__global__ void fused_linear_act_kernel(
    const float* __restrict__ input,      // Shape: (N, C_in)
    const float* __restrict__ weight,     // Shape: (C_out, C_in)
    const float* __restrict__ bias,       // Shape: (C_out), can be nullptr
    float* __restrict__ output,         // Shape: (N, C_out)
    int N, int C_in, int C_out) {

    const int n_idx = blockIdx.y;     // Batch index (0 to N-1)
    const int c_out_idx = blockIdx.x; // Output channel index (0 to C_out-1)

    if (n_idx >= N || c_out_idx >= C_out) return;

    float accumulator = 0.0f;
    const float* input_row = input + n_idx * C_in;
    const float* weight_row = weight + c_out_idx * C_in;

    // Each thread computes a partial dot product
    for (int k = threadIdx.x; k < C_in; k += blockDim.x) {
        accumulator += __ldg(input_row + k) * __ldg(weight_row + k);
    }

    // Intra-block reduction to get the final dot product
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = accumulator;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Thread 0 adds bias, applies activation, and writes to output
    if (threadIdx.x == 0) {
        float val = sdata[0];
        if (bias != nullptr) {
            val += __ldg(bias + c_out_idx);
        }

        if (ACT_MODE == 1) { // ReLU
            val = fmaxf(val, 0.0f);
        } else if (ACT_MODE == 2) { // Sigmoid
            val = 1.0f / (1.0f + expf(-val));
        }

        output[n_idx * C_out + c_out_idx] = val;
    }
}


// --- C++ Wrappers for CUDA Kernels ---

torch::Tensor bias_act(
    const torch::Tensor& input, const torch::Tensor& bias, int act_mode) {
    
    const int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    auto output = torch::empty_like(input);
    const int spatial_size = H * W;
    const int threads_per_block = 512;
    dim3 grid(C, N);
    dim3 block(threads_per_block);

    switch(act_mode) {
        case 1: fused_bias_act_kernel<1><<<grid, block>>>(input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), C, spatial_size); break;
        case 2: fused_bias_act_kernel<2><<<grid, block>>>(input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), C, spatial_size); break;
        default: fused_bias_act_kernel<0><<<grid, block>>>(input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), C, spatial_size); break;
    }
    return output;
}

torch::Tensor bias_relu_avgpool(
    const torch::Tensor& input, const torch::Tensor& bias) {
    
    const int N = input.size(0), C = input.size(1);
    const int spatial_size = input.size(2) * input.size(3);
    auto output = torch::empty({N, C}, input.options());
    const int threads_per_block = 256;
    const size_t shared_mem_size = threads_per_block * sizeof(float);
    dim3 grid(C, N);
    dim3 block(threads_per_block);

    fused_bias_relu_avgpool_kernel<<<grid, block, shared_mem_size>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(), spatial_size);
    return output;
}

torch::Tensor linear_act(
    const torch::Tensor& input, const torch::Tensor& weight, 
    const c10::optional<torch::Tensor>& bias_opt, int act_mode) {
    
    const int N = input.size(0), C_in = input.size(1), C_out = weight.size(0);
    auto output = torch::empty({N, C_out}, input.options());

    // Dynamically select block size based on inner dimension (C_in) to reduce thread waste.
    int threads_per_block = 32;
    if (C_in > 32) threads_per_block = 64;
    if (C_in > 64) threads_per_block = 128;
    if (C_in > 128) threads_per_block = 256;

    const size_t shared_mem_size = threads_per_block * sizeof(float);
    dim3 grid(C_out, N);
    dim3 block(threads_per_block);
    
    const float* bias_ptr = bias_opt.has_value() ? bias_opt.value().data_ptr<float>() : nullptr;

    switch(act_mode) {
        case 1: fused_linear_act_kernel<1><<<grid, block, shared_mem_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(), N, C_in, C_out); break;
        case 2: fused_linear_act_kernel<2><<<grid, block, shared_mem_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(), N, C_in, C_out); break;
        default: fused_linear_act_kernel<0><<<grid, block, shared_mem_size>>>(input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(), N, C_in, C_out); break;
    }
    return output;
}

// Exported Python-callable functions
torch::Tensor bias_add_relu_cuda(const torch::Tensor& input, const torch::Tensor& bias) { return bias_act(input, bias, 1); }
torch::Tensor linear_relu(const torch::Tensor& input, const torch::Tensor& weight) { return linear_act(input, weight, c10::nullopt, 1); }
torch::Tensor linear_sigmoid(const torch::Tensor& input, const torch::Tensor& weight) { return linear_act(input, weight, c10::nullopt, 2); }
torch::Tensor linear_bias_add(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias) { return linear_act(input, weight, bias, 0); }
"""

cpp_source = """
torch::Tensor bias_add_relu_cuda(const torch::Tensor& input, const torch::Tensor& bias);
torch::Tensor bias_relu_avgpool(const torch::Tensor& input, const torch::Tensor& bias);
torch::Tensor linear_relu(const torch::Tensor& input, const torch::Tensor& weight);
torch::Tensor linear_sigmoid(const torch::Tensor& input, const torch::Tensor& weight);
torch::Tensor linear_bias_add(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias);
"""

fused_ops = load_inline(
    name="fused_ops_v_aggressive_tuned",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["bias_add_relu_cuda", "bias_relu_avgpool", "linear_relu", "linear_sigmoid", "linear_bias_add"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

def _fuse_conv_bn(conv, bn):
    """Fuses a Conv2d and BatchNorm2d layer, returning new weight and bias Parameters."""
    conv.eval(); bn.eval()
    gamma, beta, mean, var = bn.weight, bn.bias, bn.running_mean, bn.running_var
    w_conv = conv.weight.clone().detach()
    b_conv = conv.bias.clone().detach() if conv.bias is not None else torch.zeros_like(mean)
    
    inv_std = torch.rsqrt(var + bn.eps)
    scale = gamma * inv_std
    
    w_fused = w_conv * scale.view(-1, 1, 1, 1)
    b_fused = (b_conv - mean) * scale + beta
    
    return nn.Parameter(w_fused, requires_grad=False), nn.Parameter(b_fused, requires_grad=False)

class MBConvBlock(nn.Module):
    """MBConvBlock with pre-fused Conv-BN and a highly optimized SE-Projection path."""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        self.has_expansion = expand_ratio != 1

        if self.has_expansion:
            self.w_exp, self.b_exp = _fuse_conv_bn(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels)
            )
        
        self.w_dw, self.b_dw = _fuse_conv_bn(
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels)
        )
        self.dw_params = {'stride': stride, 'padding': 1, 'groups': expanded_channels}

        se_conv1 = nn.Conv2d(expanded_channels, expanded_channels // 4, 1, bias=False)
        self.w_se1 = nn.Parameter(se_conv1.weight.squeeze(), requires_grad=False)
        se_conv2 = nn.Conv2d(expanded_channels // 4, expanded_channels, 1, bias=False)
        self.w_se2 = nn.Parameter(se_conv2.weight.squeeze(), requires_grad=False)

        w_proj, b_proj = _fuse_conv_bn(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.w_proj_flat = nn.Parameter(w_proj.squeeze(), requires_grad=False)
        self.b_proj = nn.Parameter(b_proj, requires_grad=False)

    def forward(self, x):
        if self.has_expansion:
            x = F.conv2d(x, self.w_exp, bias=None, stride=1, padding=0)
            x = fused_ops.bias_add_relu_cuda(x, self.b_exp)
        
        x_dw = F.conv2d(x, self.w_dw, bias=None, **self.dw_params)

        # Fused SE block operations
        # 1. Fused BiasAdd + ReLU + AvgPool
        x_se = fused_ops.bias_relu_avgpool(x_dw, self.b_dw)
        
        # 2. Fused Linear + ReLU
        x_se = fused_ops.linear_relu(x_se, self.w_se1)
        
        # 3. Fused Linear + Sigmoid
        x_se = fused_ops.linear_sigmoid(x_se, self.w_se2)
        
        # 4. Fused Linear + BiasAdd (Projection)
        x_out = fused_ops.linear_bias_add(x_se, self.w_proj_flat, self.b_proj)
        
        return x_out.view(-1, self.b_proj.size(0), 1, 1)

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        self.w1, self.b1 = _fuse_conv_bn(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.conv1_params = {'stride': 2, 'padding': 1}
        
        self.mbconv1 = MBConvBlock(32, 96, 1, 3)
        self.mbconv2 = MBConvBlock(96, 144, 2, 6)
        self.mbconv3 = MBConvBlock(144, 192, 2, 6)
        self.mbconv4 = MBConvBlock(192, 288, 2, 6)
        self.mbconv5 = MBConvBlock(288, 384, 1, 6)
        
        self.w_final, self.b_final = _fuse_conv_bn(
            nn.Conv2d(384, 1408, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1408)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1408, num_classes)
    
    def forward(self, x):
        x = F.conv2d(x, self.w1, bias=None, **self.conv1_params)
        x = fused_ops.bias_add_relu_cuda(x, self.b1)
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        
        x = F.conv2d(x, self.w_final, bias=None, stride=1, padding=0)
        x = fused_ops.bias_add_relu_cuda(x, self.b_final)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

batch_size = 2
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224).cuda()]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END