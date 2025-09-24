import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

def fuse_conv_bn(conv, bn):
    """
    Fuses a convolutional layer and a batch normalization layer for inference.
    """
    assert not conv.training and not bn.training, "Fusion only for eval mode"
    
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    ).to(conv.weight.device, conv.weight.dtype)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.running_var + bn.eps)))
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.size()))
    
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.out_channels, device=conv.weight.device)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_conv.bias.copy_(b_conv + b_bn)
    
    return fused_conv

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define TILE_H 16
#define TILE_W 16
#define KERNEL_DIM 3

__device__ inline float relu(float x) {
    return fmaxf(0.0f, x);
}

template<int stride>
__global__ void tiled_expand_depthwise_fused_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ exp_w,
    const float* __restrict__ exp_b,
    const float* __restrict__ dep_w,
    const float* __restrict__ dep_b,
    const int N, const int C_in, const int H_in, const int W_in,
    const int C_exp,
    const int H_out, const int W_out,
    const int padding)
{
    const int IN_TILE_H = (TILE_H - 1) * stride + KERNEL_DIM;
    const int IN_TILE_W = (TILE_W - 1) * stride + KERNEL_DIM;
    __shared__ float s_intermediate[IN_TILE_H][IN_TILE_W];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int block_h_start_out = blockIdx.y * TILE_H;
    const int block_w_start_out = blockIdx.x * TILE_W;
    
    const int group_idx = blockIdx.z;
    const int n = group_idx / C_exp;
    const int c_exp = group_idx % C_exp;

    const float* current_exp_w = exp_w + c_exp * C_in;
    const float exp_b_val = exp_b[c_exp];
    const float* current_dep_w = dep_w + c_exp * KERNEL_DIM * KERNEL_DIM;
    const float dep_b_val = dep_b[c_exp];
    const int input_batch_stride = C_in * H_in * W_in;
    const int input_channel_stride = H_in * W_in;

    for (int i = ty; i < IN_TILE_H; i += TILE_H) {
        for (int j = tx; j < IN_TILE_W; j += TILE_W) {
            int h_in = block_h_start_out * stride - padding + i;
            int w_in = block_w_start_out * stride - padding + j;

            float exp_acc = 0.0f;
            if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                const float* input_ptr = input + n * input_batch_stride + h_in * W_in + w_in;
                for (int c = 0; c < C_in; ++c) {
                    exp_acc += input_ptr[c * input_channel_stride] * current_exp_w[c];
                }
            }
            s_intermediate[i][j] = relu(exp_acc + exp_b_val);
        }
    }
    __syncthreads();

    const int h_out = block_h_start_out + ty;
    const int w_out = block_w_start_out + tx;

    if (h_out < H_out && w_out < W_out) {
        float dep_acc = 0.0f;
        
        for (int kh = 0; kh < KERNEL_DIM; ++kh) {
            for (int kw = 0; kw < KERNEL_DIM; ++kw) {
                dep_acc += s_intermediate[ty * stride + kh][tx * stride + kw] * current_dep_w[kh * KERNEL_DIM + kw];
            }
        }
        
        int out_idx = n * C_exp * H_out * W_out + c_exp * H_out * W_out + h_out * W_out + w_out;
        output[out_idx] = relu(dep_acc + dep_b_val);
    }
}

torch::Tensor expand_depthwise_fused(
    torch::Tensor input, torch::Tensor exp_w, torch::Tensor exp_b,
    torch::Tensor dep_w, torch::Tensor dep_b,
    int stride, int padding)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int C_exp = exp_w.size(0);
    const int H_out = (H_in + 2 * padding - KERNEL_DIM) / stride + 1;
    const int W_out = (W_in + 2 * padding - KERNEL_DIM) / stride + 1;

    auto output = torch::empty({N, C_exp, H_out, W_out}, input.options());
    
    dim3 block(TILE_W, TILE_H);
    dim3 grid( (W_out + TILE_W - 1) / TILE_W, (H_out + TILE_H - 1) / TILE_H, (N * C_exp) );

    if (stride == 1) {
        tiled_expand_depthwise_fused_kernel<1><<<grid, block>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            exp_w.data_ptr<float>(), exp_b.data_ptr<float>(),
            dep_w.data_ptr<float>(), dep_b.data_ptr<float>(),
            N, C_in, H_in, W_in, C_exp, H_out, W_out, padding);
    } else if (stride == 2) {
        tiled_expand_depthwise_fused_kernel<2><<<grid, block>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            exp_w.data_ptr<float>(), exp_b.data_ptr<float>(),
            dep_w.data_ptr<float>(), dep_b.data_ptr<float>(),
            N, C_in, H_in, W_in, C_exp, H_out, W_out, padding);
    } else {
        TORCH_CHECK(false, "Unsupported stride value");
    }
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor expand_depthwise_fused(
    torch::Tensor input, torch::Tensor exp_w, torch::Tensor exp_b,
    torch::Tensor dep_w, torch::Tensor dep_b,
    int stride, int padding);
"""

custom_ops = load_inline(
    name="custom_ops_tiled_v2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["expand_depthwise_fused"],
    verbose=True,
    extra_cflags=["-O3"],
)

class MBConvBlockNew(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(MBConvBlockNew, self).__init__()
        self.stride = stride
        expanded_channels = in_channels * expand_ratio

        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded_channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(expanded_channels, expanded_channels // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(expanded_channels // 4, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.fused_weights = {}
        self.fused_project_conv = None
        self.fused = False

    def _fuse_bn_weights(self):
        if self.fused: return
        self.eval()
        fused_expand = fuse_conv_bn(self.expand_conv, self.bn1)
        self.fused_weights['exp_w'] = fused_expand.weight.data.squeeze().contiguous()
        self.fused_weights['exp_b'] = fused_expand.bias.data.contiguous()
        fused_depthwise = fuse_conv_bn(self.depthwise_conv, self.bn2)
        self.fused_weights['dep_w'] = fused_depthwise.weight.data.view(fused_depthwise.out_channels, -1).contiguous()
        self.fused_weights['dep_b'] = fused_depthwise.bias.data.contiguous()
        self.fused_project_conv = fuse_conv_bn(self.project_conv, self.bn3)
        self.fused = True

    def forward(self, x):
        if self.training:
            y = F.relu(self.bn1(self.expand_conv(x)))
            y = F.relu(self.bn2(self.depthwise_conv(y)))
            y = self.se(y)
            y = self.bn3(self.project_conv(y))
            return y

        if not self.fused: self._fuse_bn_weights()
        x_depth = custom_ops.expand_depthwise_fused(
            x, self.fused_weights['exp_w'], self.fused_weights['exp_b'],
            self.fused_weights['dep_w'], self.fused_weights['dep_b'],
            self.stride, 1)
        se_out = self.se(x_depth)
        out = self.fused_project_conv(se_out)
        return out

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.mbconv1 = MBConvBlockNew(32, 96, 1, 3)
        self.mbconv2 = MBConvBlockNew(96, 144, 2, 6)
        self.mbconv3 = MBConvBlockNew(144, 192, 2, 6)
        self.mbconv4 = MBConvBlockNew(192, 288, 2, 6)
        self.mbconv5 = MBConvBlockNew(288, 384, 1, 6)
        self.conv_final = nn.Conv2d(384, 1408, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_final = nn.BatchNorm2d(1408)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1408, num_classes)
        self.fused = False

    def _fuse_model(self):
        if self.fused: return
        self.eval()
        self.fused_conv1 = fuse_conv_bn(self.conv1, self.bn1)
        self.fused_conv_final = fuse_conv_bn(self.conv_final, self.bn_final)
        for module in self.modules():
            if isinstance(module, MBConvBlockNew):
                module._fuse_bn_weights()
        self.fused = True
        del self.conv1, self.bn1, self.conv_final, self.bn_final

    def forward(self, x):
        if not self.training and not self.fused: self._fuse_model()

        if self.training:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.mbconv1(x)
            x = self.mbconv2(x)
            x = self.mbconv3(x)
            x = self.mbconv4(x)
            x = self.mbconv5(x)
            x = self.relu(self.bn_final(self.conv_final(x)))
        else:
            x = self.relu(self.fused_conv1(x))
            x = self.mbconv1(x)
            x = self.mbconv2(x)
            x = self.mbconv3(x)
            x = self.mbconv4(x)
            x = self.mbconv5(x)
            x = self.relu(self.fused_conv_final(x))

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