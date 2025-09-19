# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for fused operations.
# This version combines the vectorized BN+activation kernels with a new kernel
# that fuses the final stage of the network: BN + ReLU + AdaptiveAvgPool + Flatten.
fused_ops_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel for Fused BatchNorm + ReLU using pre-calculated scale/bias and float4 vectorization.
__global__ void batch_norm_relu_kernel_vec4(
    const float4* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    const int total_vec_elements, const int C, const int spatial_dim) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_vec_elements;
         i += gridDim.x * blockDim.x) {

        const int base_idx = i * 4;
        // All 4 elements in the vector belong to the same channel due to NCHW layout.
        const int c = (base_idx / spatial_dim) % C;
        const float s = scale[c];
        const float b = bias[c];

        float4 in_vec = input[i];
        float4 out_vec;

        out_vec.x = fmaxf(0.0f, in_vec.x * s + b);
        out_vec.y = fmaxf(0.0f, in_vec.y * s + b);
        out_vec.z = fmaxf(0.0f, in_vec.z * s + b);
        out_vec.w = fmaxf(0.0f, in_vec.w * s + b);

        output[i] = out_vec;
    }
}

// Kernel for Fused BatchNorm + ReLU6 using pre-calculated scale/bias and float4 vectorization.
__global__ void batch_norm_relu6_kernel_vec4(
    const float4* __restrict__ input,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    float4* __restrict__ output,
    const int total_vec_elements, const int C, const int spatial_dim) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_vec_elements;
         i += gridDim.x * blockDim.x) {

        const int base_idx = i * 4;
        const int c = (base_idx / spatial_dim) % C;
        const float s = scale[c];
        const float b = bias[c];

        float4 in_vec = input[i];
        float4 out_vec;

        out_vec.x = fminf(fmaxf(0.0f, in_vec.x * s + b), 6.0f);
        out_vec.y = fminf(fmaxf(0.0f, in_vec.y * s + b), 6.0f);
        out_vec.z = fminf(fmaxf(0.0f, in_vec.z * s + b), 6.0f);
        out_vec.w = fminf(fmaxf(0.0f, in_vec.w * s + b), 6.0f);

        output[i] = out_vec;
    }
}

// Kernel for Fused BatchNorm + ReLU + AdaptiveAvgPool2d + Flatten
__global__ void fused_bn_relu_pool_kernel(
    const float* __restrict__ input, // [N, C, H, W]
    const float* __restrict__ scale, // [C]
    const float* __restrict__ bias,  // [C]
    float* __restrict__ output,      // [N, C]
    int N, int C, int H, int W) {

    const int spatial_dim = H * W;
    const float inv_spatial_dim = 1.0f / spatial_dim;

    // Grid-stride loop over the N*C output elements. Each thread computes one output value.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N * C; i += gridDim.x * blockDim.x) {
        const int n = i / C;
        const int c = i % C;

        const float s = scale[c];
        const float b = bias[c];
        const float* input_plane = input + (n * C + c) * spatial_dim;

        float sum = 0.0f;
        for (int j = 0; j < spatial_dim; ++j) {
            float val = input_plane[j];
            val = fmaxf(0.0f, val * s + b); // Fused BN+ReLU
            sum += val;
        }
        output[i] = sum * inv_spatial_dim;
    }
}

// C++ Wrapper for BatchNorm + ReLU
torch::Tensor batch_norm_relu_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias) {
    const auto C = input.size(1);
    const int spatial_dim = input.size(2) * input.size(3);
    auto output = torch::empty_like(input);
    const int total_elements = input.numel();
    TORCH_CHECK(total_elements % 4 == 0, "Vectorized kernel requires numel to be divisible by 4.");
    const int total_vec_elements = total_elements / 4;

    const int block_size = 256;
    const int num_blocks = (total_vec_elements + block_size - 1) / block_size;

    batch_norm_relu_kernel_vec4<<<num_blocks, block_size>>>(
        (const float4*)input.data_ptr<float>(), scale.data_ptr<float>(), bias.data_ptr<float>(),
        (float4*)output.data_ptr<float>(), total_vec_elements, C, spatial_dim);
    return output;
}

// C++ Wrapper for BatchNorm + ReLU6
torch::Tensor batch_norm_relu6_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias) {
    const auto C = input.size(1);
    const int spatial_dim = input.size(2) * input.size(3);
    auto output = torch::empty_like(input);
    const int total_elements = input.numel();
    TORCH_CHECK(total_elements % 4 == 0, "Vectorized kernel requires numel to be divisible by 4.");
    const int total_vec_elements = total_elements / 4;

    const int block_size = 256;
    const int num_blocks = (total_vec_elements + block_size - 1) / block_size;

    batch_norm_relu6_kernel_vec4<<<num_blocks, block_size>>>(
        (const float4*)input.data_ptr<float>(), scale.data_ptr<float>(), bias.data_ptr<float>(),
        (float4*)output.data_ptr<float>(), total_vec_elements, C, spatial_dim);
    return output;
}

// C++ Wrapper for the new fused pooling kernel
torch::Tensor fused_bn_relu_pool_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias) {
    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    auto output = torch::empty({N, C}, input.options());
    const int total_work_items = N * C;
    if (total_work_items == 0) return output;
    
    const int block_size = 256;
    const int num_blocks = (total_work_items + block_size - 1) / block_size;

    fused_bn_relu_pool_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), scale.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), N, C, H, W);
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor batch_norm_relu_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias);
torch::Tensor batch_norm_relu6_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias);
torch::Tensor fused_bn_relu_pool_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops_full_fusion",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_cuda_source,
    functions=["batch_norm_relu_cuda", "batch_norm_relu6_cuda", "fused_bn_relu_pool_cuda"],
    verbose=False,
)

# Helper class to instantiate the original architecture for weight extraction
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
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x): pass


class FusedBatchNormReLU(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_bias = bn.bias - bn.running_mean * scale
        self.register_buffer('scale', scale.contiguous())
        self.register_buffer('fused_bias', fused_bias.contiguous())

    def forward(self, x):
        return fused_ops.batch_norm_relu_cuda(x, self.scale, self.fused_bias)


class FusedBatchNormReLU6(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_bias = bn.bias - bn.running_mean * scale
        self.register_buffer('scale', scale.contiguous())
        self.register_buffer('fused_bias', fused_bias.contiguous())

    def forward(self, x):
        return fused_ops.batch_norm_relu6_cuda(x, self.scale, self.fused_bias)


class FusedMBConv(nn.Module):
    def __init__(self, original_mbconv_seq: nn.Sequential):
        super().__init__()
        self.conv1 = original_mbconv_seq[0]
        self.fused_op1 = FusedBatchNormReLU6(original_mbconv_seq[1])
        self.conv2 = original_mbconv_seq[3]
        self.fused_op2 = FusedBatchNormReLU6(original_mbconv_seq[4])
        
        conv3 = original_mbconv_seq[6]
        bn3 = original_mbconv_seq[7]
        bn3.eval()
        
        scale = bn3.weight / torch.sqrt(bn3.running_var + bn3.eps)
        bias = bn3.bias - bn3.running_mean * scale
        
        self.fused_conv3 = nn.Conv2d(
            in_channels=conv3.in_channels, out_channels=conv3.out_channels,
            kernel_size=conv3.kernel_size, stride=conv3.stride,
            padding=conv3.padding, groups=conv3.groups, bias=True
        )
        
        fused_weights = conv3.weight * scale.view(-1, 1, 1, 1)
        self.fused_conv3.weight.data.copy_(fused_weights)
        self.fused_conv3.bias.data.copy_(bias)

    def forward(self, x):
        x = self.fused_op1(self.conv1(x))
        x = self.fused_op2(self.conv2(x))
        x = self.fused_conv3(x)
        return x


class FusedFinalStage(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_bias = bn.bias - bn.running_mean * scale
        self.register_buffer('scale', scale.contiguous())
        self.register_buffer('fused_bias', fused_bias.contiguous())

    def forward(self, x):
        return fused_ops.fused_bn_relu_pool_cuda(x, self.scale, self.fused_bias)


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        original_model = OriginalModel(num_classes).eval()

        self.conv1 = original_model.conv1
        self.fused_bn_relu1 = FusedBatchNormReLU(original_model.bn1)

        self.mbconv1 = FusedMBConv(original_model.mbconv1)
        self.mbconv2 = FusedMBConv(original_model.mbconv2)
        self.mbconv3 = FusedMBConv(original_model.mbconv3)
        self.mbconv4 = FusedMBConv(original_model.mbconv4)
        self.mbconv5 = FusedMBConv(original_model.mbconv5)
        self.mbconv6 = FusedMBConv(original_model.mbconv6)
        self.mbconv7 = FusedMBConv(original_model.mbconv7)
        
        self.conv2 = original_model.conv2
        # Replace the final BN+ReLU and Pool+Flatten with a single fused module
        self.fused_final_stage = FusedFinalStage(original_model.bn2)
        
        self.fc = original_model.fc

    def forward(self, x):
        x = self.fused_bn_relu1(self.conv1(x))
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        # Apply the new, fully fused final stage
        x = self.fused_final_stage(self.conv2(x))
        x = self.fc(x)
        
        return x

# Test code
batch_size = 10
input_shape = (3, 240, 240)
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END