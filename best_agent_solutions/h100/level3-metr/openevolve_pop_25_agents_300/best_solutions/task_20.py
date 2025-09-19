# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for fused operations with micro-optimizations:
# 1. Fused BiasAdd + ReLU6: Optimized with a grid-stride loop for better GPU utilization.
# 2. Fused AdaptiveAvgPool2d + Flatten: Optimized with warp-level primitives to reduce shared memory traffic.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// --- Optimized Kernel 1: Fused BiasAdd + ReLU6 with Grid-Stride Loop ---
__global__ void bias_relu6_kernel_optimized(
    const float* __restrict__ in,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int num_elements,
    int C,
    int spatial_dim) {

    // Use a grid-stride loop to ensure all elements are processed and
    // to improve performance by keeping all Streaming Multiprocessors (SMs) active.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_elements;
         idx += gridDim.x * blockDim.x) {
        
        int c = (idx / spatial_dim) % C;
        float val = in[idx] + bias[c];
        val = fmaxf(0.0f, val); // In-place ReLU
        out[idx] = fminf(6.0f, val); // In-place clamp at 6
    }
}

torch::Tensor bias_relu6_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor, but got ", input.dim());
    TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor, but got ", bias.dim());
    TORCH_CHECK(input.size(1) == bias.size(0), "Bias size must match input channel size");

    auto out = torch::empty_like(input);
    const int num_elements = input.numel();
    if (num_elements == 0) return out;

    const int C = input.size(1);
    const int spatial_dim = input.size(2) * input.size(3);

    const int block_size = 256;
    // Launch enough blocks to keep the GPU busy, the grid-stride loop handles correctness.
    const int num_blocks = std::min((num_elements + block_size - 1) / block_size, 4096);

    bias_relu6_kernel_optimized<<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements,
        C,
        spatial_dim
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// --- Optimized Kernel 2: Fused AdaptiveAvgPool2d + Flatten with Warp Primitives ---

// Helper device function for warp-level reduction using shuffle instructions.
// This is faster than using shared memory for intra-warp reductions.
__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__global__ void adaptive_avg_pool_flatten_kernel_optimized(const float* input, float* output, int N, int C, int H, int W) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Each block processes one N,C plane.
    int nc_idx = blockIdx.x;
    int plane_size = H * W;
    const float* plane_start = input + nc_idx * plane_size;

    // 1. Each thread computes a partial sum from global memory.
    float my_sum = 0.0f;
    for (int i = tid; i < plane_size; i += blockDim.x) {
        my_sum += plane_start[i];
    }

    // 2. Each warp reduces its own sums using fast shuffle instructions.
    my_sum = warpReduceSum(my_sum);

    // 3. The first thread of each warp writes its result to shared memory, reducing traffic.
    if (lane_id == 0) {
        sdata[warp_id] = my_sum;
    }

    __syncthreads();

    // 4. The first warp sums the results from all other warps (which are now in shared memory).
    if (warp_id == 0) {
        // Load from shared memory into the first warp's registers.
        my_sum = (tid < blockDim.x / 32) ? sdata[lane_id] : 0.0f;
        // Final reduction within the first warp.
        my_sum = warpReduceSum(my_sum);

        // 5. Thread 0 writes the final averaged result.
        if (lane_id == 0) {
            output[nc_idx] = my_sum / plane_size;
        }
    }
}


torch::Tensor adaptive_avg_pool_flatten_cuda(torch::Tensor input) {
    auto contiguous_input = input.contiguous();

    const auto N = contiguous_input.size(0);
    const auto C = contiguous_input.size(1);
    const auto H = contiguous_input.size(2);
    const auto W = contiguous_input.size(3);

    auto output = torch::empty({N, C}, contiguous_input.options());
    if (input.numel() == 0) return output;

    const int block_size = 256; // 8 warps per block
    const int num_blocks = N * C;
    // We only need enough shared memory for one float per warp.
    const int shared_mem_size = (block_size / 32) * sizeof(float);

    adaptive_avg_pool_flatten_kernel_optimized<<<num_blocks, block_size, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
        contiguous_input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

"""

fused_ops_cpp_source = """
torch::Tensor bias_relu6_cuda(torch::Tensor input, torch::Tensor bias);
torch::Tensor adaptive_avg_pool_flatten_cuda(torch::Tensor input);
"""

# Compile the inline CUDA code, using a unique name to avoid caching issues.
fused_ops = load_inline(
    name="fused_mobilenet_v2_ops_warp_optimized",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["bias_relu6_cuda", "adaptive_avg_pool_flatten_cuda"],
    verbose=False,
)


class FusedConvBN(nn.Module):
    """Mathematically fuses a Conv2d and BatchNorm2d into a single Conv2d for inference."""
    def __init__(self, conv, bn):
        super().__init__()
        self.conv = conv
        with torch.no_grad():
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            w_conv = self.conv.weight.clone()
            
            std = torch.sqrt(running_var + eps)
            scale_factor = gamma / std
            
            self.conv.weight.data = w_conv * scale_factor.reshape(-1, 1, 1, 1)
            fused_bias = beta - (running_mean * scale_factor)
            self.conv.bias = nn.Parameter(fused_bias)

    def forward(self, x):
        return self.conv(x)


class FusedConvBNReLU6(nn.Module):
    """Fuses Conv2d, BatchNorm2d, and calls a custom kernel for BiasAdd+ReLU6."""
    def __init__(self, conv, bn):
        super().__init__()
        self.conv = conv
        with torch.no_grad():
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            w_conv = self.conv.weight.clone()
            
            std = torch.sqrt(running_var + eps)
            scale_factor = gamma / std
            
            self.conv.weight.data = w_conv * scale_factor.reshape(-1, 1, 1, 1)
            fused_bias = beta - (running_mean * scale_factor)
            self.conv.bias = None
            self.register_buffer('fused_bias', fused_bias)

    def forward(self, x):
        x = self.conv(x)
        return fused_ops.bias_relu6_cuda(x, self.fused_bias)

def _initialize_weights(conv, bn):
    """Initializes weights for a Conv2D and BatchNorm2D pair before fusion."""
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    nn.init.ones_(bn.weight)
    nn.init.zeros_(bn.bias)

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        class InvertedResidual(nn.Module):
            def __init__(self, inp, oup, stride, expand_ratio):
                super(InvertedResidual, self).__init__()
                hidden_dim = int(round(inp * expand_ratio))
                layers = []

                if expand_ratio != 1:
                    conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
                    bn1 = nn.BatchNorm2d(hidden_dim)
                    _initialize_weights(conv1, bn1)
                    layers.append(FusedConvBNReLU6(conv1, bn1))
                
                conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
                bn2 = nn.BatchNorm2d(hidden_dim)
                _initialize_weights(conv2, bn2)
                layers.append(FusedConvBNReLU6(conv2, bn2))
                
                conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
                bn3 = nn.BatchNorm2d(oup)
                _initialize_weights(conv3, bn3)
                layers.append(FusedConvBN(conv3, bn3))
                
                self.conv = nn.Sequential(*layers)

            def forward(self, x):
                return self.conv(x)

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2],
            [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        first_conv = nn.Conv2d(3, input_channel, 3, 2, 1, bias=False)
        first_bn = nn.BatchNorm2d(input_channel)
        _initialize_weights(first_conv, first_bn)
        features.append(FusedConvBNReLU6(first_conv, first_bn))

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        last_conv = nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False)
        last_bn = nn.BatchNorm2d(last_channel)
        _initialize_weights(last_conv, last_bn)
        features.append(FusedConvBNReLU6(last_conv, last_bn))

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = fused_ops.adaptive_avg_pool_flatten_cuda(x)
        x = self.classifier(x)
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224).cuda()]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END