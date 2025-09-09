# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA and C++ source for a comprehensive, vectorized fused kernel.
# This kernel handles the sequence: (Fused BatchNorm) + (Optional Residual Add) + (Optional ReLU).
# It uses float4 vectorization to maximize memory bandwidth, which is critical for this memory-bound operation.
# Fusing these operations reduces memory bandwidth usage and kernel launch overhead.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Vectorized fused kernel using float4 for higher memory throughput.
// This kernel handles: out = relu( (in * scale + bias) + add_tensor )
// It uses a grid-stride loop to be flexible for any input size.
template <typename T>
__global__ void fused_elementwise_kernel_vectorized(
    const T* __restrict__ in,
    const T* __restrict__ scale,
    const T* __restrict__ bias,
    const T* __restrict__ add_tensor,
    T* __restrict__ out,
    const bool apply_relu,
    const int C,
    const int spatial_size,
    const int total_elements)
{
    const int vec_size = 4;
    const int num_vec_elements = total_elements / vec_size;
    
    // Grid-stride loop for the vectorized part
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vec_elements; i += gridDim.x * blockDim.x) {
        const int idx = i * vec_size;
        // Since H*W for all layers is divisible by 4, a float4 load will not cross a channel boundary.
        // This makes the channel index calculation simple and correct for the whole vector.
        const int channel_idx = (idx / spatial_size) % C;

        float4 in_val = *reinterpret_cast<const float4*>(&in[idx]);
        
        const float s = scale[channel_idx];
        const float b = bias[channel_idx];

        float4 val;
        val.x = in_val.x * s + b;
        val.y = in_val.y * s + b;
        val.z = in_val.z * s + b;
        val.w = in_val.w * s + b;

        if (add_tensor != nullptr) {
            float4 add_val = *reinterpret_cast<const float4*>(&add_tensor[idx]);
            val.x += add_val.x;
            val.y += add_val.y;
            val.z += add_val.z;
            val.w += add_val.w;
        }

        if (apply_relu) {
            val.x = fmaxf(val.x, 0.0f);
            val.y = fmaxf(val.y, 0.0f);
            val.z = fmaxf(val.z, 0.0f);
            val.w = fmaxf(val.w, 0.0f);
        }
        
        *reinterpret_cast<float4*>(&out[idx]) = val;
    }

    // Grid-stride loop for the scalar remainder part (if total_elements is not a multiple of 4)
    int start_idx = num_vec_elements * vec_size + blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = start_idx; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const int channel_idx = (idx / spatial_size) % C;

        T val_s = in[idx] * scale[channel_idx] + bias[channel_idx];

        if (add_tensor != nullptr) {
            val_s += add_tensor[idx];
        }

        if (apply_relu) {
            val_s = fmaxf(val_s, static_cast<T>(0));
        }
        out[idx] = val_s;
    }
}

// C++ interface to launch the CUDA kernel.
torch::Tensor fused_ops_cuda(
    torch::Tensor in,
    torch::Tensor scale,
    torch::Tensor bias,
    c10::optional<torch::Tensor> add_tensor_opt,
    bool apply_relu)
{
    // Input validation
    TORCH_CHECK(in.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "Scale tensor must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor");
    TORCH_CHECK(in.is_contiguous(), "Input tensor must be contiguous");

    // Get tensor dimensions for kernel launch configuration
    const auto C = in.size(1);
    const auto H = in.size(2);
    const auto W = in.size(3);
    const int total_elements = in.numel();
    const int spatial_size = H * W;

    auto out = torch::empty_like(in);
    if (total_elements == 0) return out;

    const float* add_tensor_ptr = nullptr;
    if (add_tensor_opt.has_value()) {
        auto& add_tensor = add_tensor_opt.value();
        TORCH_CHECK(add_tensor.is_cuda(), "Add tensor must be a CUDA tensor");
        TORCH_CHECK(add_tensor.sizes() == in.sizes(), "Add tensor shape must match input tensor shape");
        TORCH_CHECK(add_tensor.is_contiguous(), "Add tensor must be contiguous");
        add_tensor_ptr = add_tensor.data_ptr<float>();
    }

    // Optimized kernel launch configuration using a grid-stride loop pattern.
    // A moderate block size and number of blocks are chosen to keep the GPU saturated.
    const int block_size = 512;
    const int num_blocks = 256;
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_elementwise_kernel_vectorized<float><<<num_blocks, block_size, 0, stream>>>(
        in.data_ptr<float>(),
        scale.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_tensor_ptr,
        out.data_ptr<float>(),
        apply_relu,
        C, spatial_size,
        total_elements
    );
    
    // Check for errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor in, torch::Tensor scale, torch::Tensor bias, c10::optional<torch::Tensor> add_tensor_opt, bool apply_relu);"

# JIT compile the CUDA kernel at module load time.
# Using a new name to avoid caching issues from previous versions.
fused_ops = load_inline(
    name="fused_ops_vectorized_v2",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=False,
)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.stride = stride
        
        # Attributes to hold fused components
        self.downsample_conv = None
        self.is_fused = False
        
    @torch.no_grad()
    def _fuse_bn_params(self, bn):
        # This function folds the BatchNorm parameters into a single scale and bias vector.
        # y = gamma * (x - mean) / sqrt(var + eps) + beta  ==> y = scale * x + bias
        gamma = bn.weight
        beta = bn.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        eps = bn.eps
        
        scale = gamma / torch.sqrt(running_var + eps)
        bias = beta - running_mean * scale
        return scale, bias

    @torch.no_grad()
    def fuse_modules(self):
        # This method performs the actual fusion, replacing BN layers with learned buffers.
        # It's called once at model initialization.
        if self.is_fused:
            return
            
        # Fuse bn1
        scale1, bias1 = self._fuse_bn_params(self.bn1)
        self.register_buffer('scale1', scale1)
        self.register_buffer('bias1', bias1)
        del self.bn1

        # Fuse bn2
        scale2, bias2 = self._fuse_bn_params(self.bn2)
        self.register_buffer('scale2', scale2)
        self.register_buffer('bias2', bias2)
        del self.bn2

        # Fuse bn3
        scale3, bias3 = self._fuse_bn_params(self.bn3)
        self.register_buffer('scale3', scale3)
        self.register_buffer('bias3', bias3)
        del self.bn3

        # Fuse downsample block if it exists
        if self.downsample is not None:
            self.downsample_conv = self.downsample[0]
            downsample_bn = self.downsample[1]
            ds_scale, ds_bias = self._fuse_bn_params(downsample_bn)
            self.register_buffer('ds_scale', ds_scale)
            self.register_buffer('ds_bias', ds_bias)
            self.downsample = None
        
        self.is_fused = True

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # Call fused kernel: BN-like op + ReLU
        out = fused_ops.fused_ops_cuda(out, self.scale1, self.bias1, None, True)

        out = self.conv2(out)
        # Call fused kernel: BN-like op + ReLU
        out = fused_ops.fused_ops_cuda(out, self.scale2, self.bias2, None, True)

        out = self.conv3(out)
        
        if self.downsample_conv is not None:
            identity = self.downsample_conv(x)
            # Call fused kernel for downsample path: BN-like op only, no ReLU
            identity = fused_ops.fused_ops_cuda(identity, self.ds_scale, self.ds_bias, None, False)

        # Final operation: BN-like op + residual add + ReLU
        out = fused_ops.fused_ops_cuda(out, self.scale3, self.bias3, identity, True)

        return out

class Model(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(Model, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        block = Bottleneck

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Fuse the model at initialization for inference speedup
        self.fuse_model()

    @torch.no_grad()
    def _fuse_bn_params(self, bn):
        # Helper function to extract scale and bias from a BN layer
        gamma = bn.weight
        beta = bn.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        eps = bn.eps
        scale = gamma / torch.sqrt(running_var + eps)
        bias = beta - running_mean * scale
        return scale, bias
        
    @torch.no_grad()
    def fuse_model(self):
        # Fuse the initial BatchNorm layer
        scale1, bias1 = self._fuse_bn_params(self.bn1)
        self.register_buffer('scale1', scale1)
        self.register_buffer('bias1', bias1)
        del self.bn1

        # Recursively fuse all Bottleneck blocks
        for module in self.modules():
            if isinstance(module, Bottleneck):
                module.fuse_modules()

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
        x = self.conv1(x)
        # Apply fused BN + ReLU after the first convolution
        x = fused_ops.fused_ops_cuda(x, self.scale1, self.bias1, None, True)
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
batch_size = 10
height = 224
width = 224
layers = [3, 4, 23, 3]
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [layers, num_classes]
# EVOLVE-BLOCK-END