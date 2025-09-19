# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This version combines the best optimizations from all prior high-performing attempts:
# 1. Fused BN -> ReLU -> MaxPool for the initial block.
# 2. Aggressively fused BN_main -> BN_shortcut -> Add -> ReLU for downsampling blocks.
# 3. Vectorized (float4) BN -> ReLU and BN -> Add -> ReLU for standard residual blocks.
# 4. Fused AdaptiveAvgPool -> Flatten using shared memory reduction for the final pooling.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm> // For std::min

// --- KERNEL 1: Fused BatchNorm -> ReLU -> MaxPool (for initial block) ---
__global__ void fused_bn_relu_maxpool_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight, const float* __restrict__ bias,
    const float* __restrict__ mean, const float* __restrict__ var, const float eps,
    float* out, const int N, const int C,
    const int H_in, const int W_in, const int H_out, const int W_out) {

    const int KERNEL_SIZE = 3, STRIDE = 2, PADDING = 1;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N * C * H_out * W_out; idx += gridDim.x * blockDim.x) {
        const int w_out = idx % W_out;
        const int h_out = (idx / W_out) % H_out;
        const int c = (idx / (W_out * H_out)) % C;
        const int n = idx / (C * W_out * H_out);

        const int h_start = h_out * STRIDE - PADDING;
        const int w_start = w_out * STRIDE - PADDING;

        float max_val = -1.0f/0.0f; // Negative infinity

        const float inv_std = rsqrtf(var[c] + eps);
        const float bn_w = weight[c];
        const float bn_b = bias[c];
        const float bn_m = mean[c];

        for (int kh = 0; kh < KERNEL_SIZE; ++kh) {
            const int h_in = h_start + kh;
            if (h_in >= 0 && h_in < H_in) {
                for (int kw = 0; kw < KERNEL_SIZE; ++kw) {
                    const int w_in = w_start + kw;
                    if (w_in >= 0 && w_in < W_in) {
                        const int in_idx = n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in;
                        float val = x[in_idx];
                        val = (val - bn_m) * inv_std * bn_w + bn_b; // BatchNorm
                        val = fmaxf(0.0f, val);                     // ReLU
                        max_val = fmaxf(max_val, val);              // MaxPool
                    }
                }
            }
        }
        out[idx] = max_val;
    }
}

// --- KERNEL 2 & 3: Vectorized Fused BN->ReLU and BN->Add->ReLU ---
__global__ void vectorized_fused_bn_relu_kernel(
    const float4* __restrict__ x, float4* out, const float* __restrict__ weight,
    const float* __restrict__ bias, const float* __restrict__ mean, const float* __restrict__ var,
    const float eps, const int total_vec_elements, const int C, const int HW_div_4) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vec_elements; i += gridDim.x * blockDim.x) {
        const int c = (i / HW_div_4) % C;
        const float inv_std = rsqrtf(var[c] + eps);
        const float w = weight[c], b = bias[c], m = mean[c];
        const float4 x_val = x[i];
        float4 out_val;
        out_val.x = fmaxf(0.0f, (x_val.x - m) * inv_std * w + b);
        out_val.y = fmaxf(0.0f, (x_val.y - m) * inv_std * w + b);
        out_val.z = fmaxf(0.0f, (x_val.z - m) * inv_std * w + b);
        out_val.w = fmaxf(0.0f, (x_val.w - m) * inv_std * w + b);
        out[i] = out_val;
    }
}

__global__ void vectorized_fused_bn_add_relu_kernel(
    const float4* __restrict__ x, const float4* __restrict__ identity, float4* out,
    const float* __restrict__ weight, const float* __restrict__ bias, const float* __restrict__ mean,
    const float* __restrict__ var, const float eps, const int total_vec_elements, const int C, const int HW_div_4) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vec_elements; i += gridDim.x * blockDim.x) {
        const int c = (i / HW_div_4) % C;
        const float inv_std = rsqrtf(var[c] + eps);
        const float w = weight[c], b = bias[c], m = mean[c];
        const float4 x_val = x[i];
        const float4 identity_val = identity[i];
        float4 out_val;
        out_val.x = fmaxf(0.0f, ((x_val.x - m) * inv_std * w + b) + identity_val.x);
        out_val.y = fmaxf(0.0f, ((x_val.y - m) * inv_std * w + b) + identity_val.y);
        out_val.z = fmaxf(0.0f, ((x_val.z - m) * inv_std * w + b) + identity_val.z);
        out_val.w = fmaxf(0.0f, ((x_val.w - m) * inv_std * w + b) + identity_val.w);
        out[i] = out_val;
    }
}

// --- KERNEL 4: Vectorized Fused (BN_main + BN_shortcut + Add + ReLU) ---
__global__ void vectorized_fused_two_bns_add_relu_kernel(
    const float4* __restrict__ main_x, const float4* __restrict__ identity_x,
    const float* __restrict__ w1, const float* __restrict__ b1, const float* __restrict__ m1, const float* __restrict__ v1, const float eps1,
    const float* __restrict__ w2, const float* __restrict__ b2, const float* __restrict__ m2, const float* __restrict__ v2, const float eps2,
    float4* out, const int total_vec_elements, const int C, const int HW_div_4) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_vec_elements; i += gridDim.x * blockDim.x) {
        const int c = (i / HW_div_4) % C;
        const float inv_std1 = rsqrtf(v1[c] + eps1), w1_c = w1[c], b1_c = b1[c], m1_c = m1[c];
        const float inv_std2 = rsqrtf(v2[c] + eps2), w2_c = w2[c], b2_c = b2[c], m2_c = m2[c];
        const float4 main_val = main_x[i];
        const float4 identity_val = identity_x[i];
        float4 out_val;
        out_val.x = fmaxf(0.0f, ((main_val.x - m1_c) * inv_std1 * w1_c + b1_c) + ((identity_val.x - m2_c) * inv_std2 * w2_c + b2_c));
        out_val.y = fmaxf(0.0f, ((main_val.y - m1_c) * inv_std1 * w1_c + b1_c) + ((identity_val.y - m2_c) * inv_std2 * w2_c + b2_c));
        out_val.z = fmaxf(0.0f, ((main_val.z - m1_c) * inv_std1 * w1_c + b1_c) + ((identity_val.z - m2_c) * inv_std2 * w2_c + b2_c));
        out_val.w = fmaxf(0.0f, ((main_val.w - m1_c) * inv_std1 * w1_c + b1_c) + ((identity_val.w - m2_c) * inv_std2 * w2_c + b2_c));
        out[i] = out_val;
    }
}

// --- KERNEL 5: Fused AdaptiveAvgPool2d -> Flatten ---
__global__ void fused_avgpool_flatten_kernel(
    const float* __restrict__ x, float* __restrict__ out,
    const int N, const int C, const int HW) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int c = blockIdx.x;
    const int n = blockIdx.y;
    const int base_idx = (n * C + c) * HW;

    float partial_sum = 0.0f;
    for (int i = tid; i < HW; i += blockDim.x) {
        partial_sum += x[base_idx + i];
    }
    sdata[tid] = partial_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sdata[tid] += sdata[tid + s]; }
        __syncthreads();
    }

    if (tid == 0) { out[n * C + c] = sdata[0] / static_cast<float>(HW); }
}

// --- C++ WRAPPERS ---
torch::Tensor fused_bn_relu_maxpool_cuda(
    torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor m, torch::Tensor v, double eps) {
    const auto N = x.size(0), C = x.size(1), H_in = x.size(2), W_in = x.size(3);
    const int K = 3, S = 2, P = 1;
    const int H_out = (H_in + 2 * P - K) / S + 1;
    const int W_out = (W_in + 2 * P - K) / S + 1;
    auto out = torch::empty({N, C, H_out, W_out}, x.options());
    const auto total_elements = out.numel();
    if (total_elements == 0) return out;
    const int block_size = 1024;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    fused_bn_relu_maxpool_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), m.data_ptr<float>(),
        v.data_ptr<float>(), static_cast<float>(eps), out.data_ptr<float>(),
        N, C, H_in, W_in, H_out, W_out);
    return out;
}

torch::Tensor fused_bn_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor m, torch::Tensor v, double eps) {
    const auto C = x.size(1), H = x.size(2), W = x.size(3);
    const auto total = x.numel();
    auto out = torch::empty_like(x);
    if (total == 0) return out;
    const int block_size = 256;
    const int num_vec_elements = total / 4;
    const int num_blocks = std::min((num_vec_elements + block_size - 1) / block_size, 4096);
    vectorized_fused_bn_relu_kernel<<<num_blocks, block_size>>>(
        (const float4*)x.data_ptr<float>(), (float4*)out.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(),
        static_cast<float>(eps), num_vec_elements, C, (H * W) / 4);
    return out;
}

torch::Tensor fused_bn_add_relu_cuda(torch::Tensor x, torch::Tensor identity, torch::Tensor w, torch::Tensor b, torch::Tensor m, torch::Tensor v, double eps) {
    const auto C = x.size(1), H = x.size(2), W = x.size(3);
    const auto total = x.numel();
    auto out = torch::empty_like(x);
    if (total == 0) return out;
    const int block_size = 256;
    const int num_vec_elements = total / 4;
    const int num_blocks = std::min((num_vec_elements + block_size - 1) / block_size, 4096);
    vectorized_fused_bn_add_relu_kernel<<<num_blocks, block_size>>>(
        (const float4*)x.data_ptr<float>(), (const float4*)identity.data_ptr<float>(), (float4*)out.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), m.data_ptr<float>(), v.data_ptr<float>(),
        static_cast<float>(eps), num_vec_elements, C, (H * W) / 4);
    return out;
}

torch::Tensor fused_two_bns_add_relu_cuda(
    torch::Tensor main_x, torch::Tensor identity_x,
    torch::Tensor w1, torch::Tensor b1, torch::Tensor m1, torch::Tensor v1, double eps1,
    torch::Tensor w2, torch::Tensor b2, torch::Tensor m2, torch::Tensor v2, double eps2) {
    const auto C = main_x.size(1), H = main_x.size(2), W = main_x.size(3);
    const auto total = main_x.numel();
    auto out = torch::empty_like(main_x);
    if (total == 0) return out;
    const int block_size = 256;
    const int num_vec_elements = total / 4;
    const int num_blocks = std::min((num_vec_elements + block_size - 1) / block_size, 4096);
    vectorized_fused_two_bns_add_relu_kernel<<<num_blocks, block_size>>>(
        (const float4*)main_x.data_ptr<float>(), (const float4*)identity_x.data_ptr<float>(),
        w1.data_ptr<float>(), b1.data_ptr<float>(), m1.data_ptr<float>(), v1.data_ptr<float>(), static_cast<float>(eps1),
        w2.data_ptr<float>(), b2.data_ptr<float>(), m2.data_ptr<float>(), v2.data_ptr<float>(), static_cast<float>(eps2),
        (float4*)out.data_ptr<float>(), num_vec_elements, C, (H * W) / 4);
    return out;
}

torch::Tensor fused_avgpool_flatten_cuda(torch::Tensor x) {
    const auto N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    const auto HW = H * W;
    auto out = torch::empty({N, C}, x.options());
    if (HW == 0) return out;
    const int block_size = 256;
    dim3 num_blocks(C, N);
    const size_t shared_mem_size = block_size * sizeof(float);
    fused_avgpool_flatten_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), N, C, HW);
    return out;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_bn_relu_maxpool_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor m, torch::Tensor v, double eps);
torch::Tensor fused_bn_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor m, torch::Tensor v, double eps);
torch::Tensor fused_bn_add_relu_cuda(torch::Tensor x, torch::Tensor identity, torch::Tensor w, torch::Tensor b, torch::Tensor m, torch::Tensor v, double eps);
torch::Tensor fused_two_bns_add_relu_cuda(
    torch::Tensor main_x, torch::Tensor identity_x,
    torch::Tensor w1, torch::Tensor b1, torch::Tensor m1, torch::Tensor v1, double eps1,
    torch::Tensor w2, torch::Tensor b2, torch::Tensor m2, torch::Tensor v2, double eps2);
torch::Tensor fused_avgpool_flatten_cuda(torch::Tensor x);
"""

# JIT compile all the CUDA kernels
fused_ops = load_inline(
    name="fused_resnet_ops_comprehensive",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=[
        "fused_bn_relu_maxpool_cuda",
        "fused_bn_relu_cuda",
        "fused_bn_add_relu_cuda",
        "fused_two_bns_add_relu_cuda",
        "fused_avgpool_flatten_cuda"
    ],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if not self.training:
            # Fused path for inference
            out = self.conv1(x)
            out = fused_ops.fused_bn_relu_cuda(
                out, self.bn1.weight, self.bn1.bias, self.bn1.running_mean, self.bn1.running_var, self.bn1.eps
            )
            out_conv2 = self.conv2(out)
            
            if self.downsample is not None:
                # Path with downsample: fuse (bn2, downsample_bn, add, relu)
                identity_conv_out = self.downsample[0](x)
                downsample_bn = self.downsample[1]
                out = fused_ops.fused_two_bns_add_relu_cuda(
                    out_conv2, identity_conv_out,
                    self.bn2.weight, self.bn2.bias, self.bn2.running_mean, self.bn2.running_var, self.bn2.eps,
                    downsample_bn.weight, downsample_bn.bias, downsample_bn.running_mean, downsample_bn.running_var, downsample_bn.eps
                )
            else:
                # Path without downsample: fuse (bn2, add, relu)
                identity = x
                out = fused_ops.fused_bn_add_relu_cuda(
                    out_conv2, identity, self.bn2.weight, self.bn2.bias, self.bn2.running_mean, self.bn2.running_var, self.bn2.eps
                )
        else:
            # Original PyTorch path for training
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)

        return out

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.training:
            x = self.conv1(x)
            x = fused_ops.fused_bn_relu_maxpool_cuda(
                x, self.bn1.weight, self.bn1.bias, self.bn1.running_mean, self.bn1.running_var, self.bn1.eps
            )
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = fused_ops.fused_avgpool_flatten_cuda(x)
            x = self.fc(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

# Test code
batch_size = 2
num_classes = 1000
input_shape = (batch_size, 3, 224, 224)

def get_inputs():
    # Ensure input tensor width is divisible by 4 for vectorized kernels
    return [torch.randn(input_shape, device='cuda')]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END