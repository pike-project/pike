# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for fused operations using "folded" BatchNorm parameters.
# Folding reduces BN to a single scale and bias, minimizing computation and memory access.
cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm> // For std::min

// --- Kernel 1: Folded BatchNorm + ReLU (Scalar) ---
// Fallback for tensors whose width is not a multiple of 4.
// Uses a grid-stride loop for robust performance across different problem sizes.
__global__ void bn_relu_folded_kernel_fp32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    int total_elements,
    int C,
    int spatial_dim) {

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements;
         index += blockDim.x * gridDim.x) {
        
        int c = (index / spatial_dim) % C;
        // Apply pre-calculated scale and bias, then ReLU. This is one FMA.
        output[index] = fmaxf(0.0f, input[index] * scale[c] + bias[c]);
    }
}

// --- Kernel 2: Folded BatchNorm + ReLU (Vectorized with float4) ---
// Processes 4 elements at a time for improved memory bandwidth. Also uses a grid-stride loop.
__global__ void bn_relu_folded_kernel_fp32_vec4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    int total_elements_div_4,
    int C,
    int spatial_dim) {

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total_elements_div_4;
         index += blockDim.x * gridDim.x) {
        
        int c = ((index * 4) / spatial_dim) % C;
        float s = scale[c];
        float b = bias[c];

        float4 in_val = input[index];
        float4 out_val;
        out_val.x = fmaxf(0.0f, in_val.x * s + b);
        out_val.y = fmaxf(0.0f, in_val.y * s + b);
        out_val.z = fmaxf(0.0f, in_val.z * s + b);
        out_val.w = fmaxf(0.0f, in_val.w * s + b);
        output[index] = out_val;
    }
}

// --- C++ Wrapper for Folded BN + ReLU ---
// Dispatches to the appropriate kernel based on tensor width.
// Uses grid-stride loop friendly launch parameters and ensures contiguity.
torch::Tensor bn_relu_folded_cuda(
    torch::Tensor input, torch::Tensor scale, torch::Tensor bias) {

    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat32, "Input must be a float32 CUDA tensor");
    TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == torch::kFloat32, "Scale must be a float32 CUDA tensor");
    TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == torch::kFloat32, "Bias must be a float32 CUDA tensor");
    
    input = input.contiguous(); // Ensure contiguous memory

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    auto output = torch::empty_like(input);
    const int total_elements = N * C * H * W;
    const int spatial_dim = H * W;

    if (W > 0 && W % 4 == 0) {
        const int total_elements_div_4 = total_elements / 4;
        const int block_size = 256;
        // Launch enough blocks to ensure good occupancy with grid-stride loop, but cap it.
        const int num_blocks = std::min((total_elements_div_4 + block_size - 1) / block_size, 4096);
        bn_relu_folded_kernel_fp32_vec4<<<num_blocks, block_size>>>(
            (const float4*)input.data_ptr<float>(), (float4*)output.data_ptr<float>(),
            scale.data_ptr<float>(), bias.data_ptr<float>(),
            total_elements_div_4, C, spatial_dim);
    } else {
        const int block_size = 256;
        const int num_blocks = std::min((total_elements + block_size - 1) / block_size, 4096);
        bn_relu_folded_kernel_fp32<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            scale.data_ptr<float>(), bias.data_ptr<float>(),
            total_elements, C, spatial_dim);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed for bn_relu_folded_cuda: ", cudaGetErrorString(err));
    }
    return output;
}


// --- Kernel 3: Folded BN + ReLU + AvgPool (Warp-Shuffle Optimized) ---
// Fuses all three operations using a highly efficient hierarchical reduction.

// Helper for warp-level reduction using __shfl_down_sync
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void bn_relu_avg_pool_folded_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ scale, const float* __restrict__ bias,
    int N, int C, int H, int W) {

    // Statically allocated shared memory for up to 32 warps (1024 threads)
    __shared__ float warp_sums[32];

    const int nc = blockIdx.x;
    const int c = nc % C;
    const int n = nc / C;

    const float s = scale[c];
    const float b = bias[c];

    const int spatial_dim = H * W;
    const float* input_plane = input + n * C * spatial_dim + c * spatial_dim;

    // Step 1: Each thread computes a partial sum from global memory
    float my_sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_dim; i += blockDim.x) {
        my_sum += fmaxf(0.0f, input_plane[i] * s + b);
    }

    // Step 2: Each warp reduces its partial sums
    float warp_sum = warpReduceSum(my_sum);

    // Step 3: Lane 0 of each warp writes its partial sum to shared memory
    if (threadIdx.x % 32 == 0) {
        warp_sums[threadIdx.x / 32] = warp_sum;
    }
    __syncthreads();

    // Step 4: The first warp reduces the sums from shared memory.
    float final_sum = (threadIdx.x < blockDim.x / 32) ? warp_sums[threadIdx.x] : 0.0f;

    // Final reduction within the first warp
    if (threadIdx.x < 32) {
        final_sum = warpReduceSum(final_sum);
    }

    // Step 5: Thread 0 writes the final result
    if (threadIdx.x == 0) {
        output[nc] = final_sum / (float)spatial_dim;
    }
}


// --- C++ Wrapper for Folded BN + ReLU + AvgPool ---
// Uses adaptive block size for efficiency on varying spatial dimensions.
torch::Tensor bn_relu_avg_pool_folded_cuda(
    torch::Tensor input, torch::Tensor scale, torch::Tensor bias) {

    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kFloat32, "Input must be a float32 CUDA tensor");
    TORCH_CHECK(scale.is_cuda() && scale.scalar_type() == torch::kFloat32, "Scale must be a float32 CUDA tensor");
    TORCH_CHECK(bias.is_cuda() && bias.scalar_type() == torch::kFloat32, "Bias must be a float32 CUDA tensor");

    input = input.contiguous();

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const int spatial_dim = H * W;

    auto output = torch::empty({N, C}, input.options());
    const int num_blocks = N * C;
    
    int block_size = 512;
    if (spatial_dim < 512) {
        block_size = 1;
        while (block_size < spatial_dim) block_size <<= 1;
        if (block_size < 32) block_size = 32;
    }
    if (block_size > 1024) block_size = 1024; // Cap at max threads per block
    
    bn_relu_avg_pool_folded_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        scale.data_ptr<float>(), bias.data_ptr<float>(),
        N, C, H, W);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed for bn_relu_avg_pool_folded_cuda: ", cudaGetErrorString(err));
    }
    return output;
}
"""

cpp_source = """
torch::Tensor bn_relu_folded_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias);
torch::Tensor bn_relu_avg_pool_folded_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor bias);
"""

# Compile the inline CUDA code
fused_op = load_inline(
    name="fused_op_densenet_hybrid",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["bn_relu_folded_cuda", "bn_relu_avg_pool_folded_cuda"],
    verbose=False,
)

class FusedBNReLUFolded(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(FusedBNReLUFolded, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps)
        self.register_buffer('folded_scale', torch.zeros(num_features))
        self.register_buffer('folded_bias', torch.zeros(num_features))
        self.is_folded = False

    def train(self, mode=True):
        super().train(mode)
        if mode: self.is_folded = False # Invalidate cache in train mode
        return self

    def forward(self, x):
        if self.training:
            self.is_folded = False
            return F.relu(self.bn(x), inplace=True)
        else:
            if not self.is_folded:
                with torch.no_grad():
                    inv_std = torch.rsqrt(self.bn.running_var + self.bn.eps)
                    scale = self.bn.weight * inv_std
                    bias = self.bn.bias - self.bn.running_mean * scale
                    self.folded_scale.copy_(scale)
                    self.folded_bias.copy_(bias)
                    self.is_folded = True
            return fused_op.bn_relu_folded_cuda(x, self.folded_scale, self.folded_bias)

class FusedBNReLUAvgPoolFolded(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(FusedBNReLUAvgPoolFolded, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps)
        self.register_buffer('folded_scale', torch.zeros(num_features))
        self.register_buffer('folded_bias', torch.zeros(num_features))
        self.is_folded = False

    def train(self, mode=True):
        super().train(mode)
        if mode: self.is_folded = False
        return self

    def forward(self, x):
        if self.training:
            self.is_folded = False
            x = self.bn(x)
            x = F.relu(x, inplace=True)
            return F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        else:
            if not self.is_folded:
                with torch.no_grad():
                    inv_std = torch.rsqrt(self.bn.running_var + self.bn.eps)
                    scale = self.bn.weight * inv_std
                    bias = self.bn.bias - self.bn.running_mean * scale
                    self.folded_scale.copy_(scale)
                    self.folded_bias.copy_(bias)
                    self.is_folded = True
            return fused_op.bn_relu_avg_pool_folded_cuda(x, self.folded_scale, self.folded_bias)

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            FusedBNReLUFolded(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        for layer in self.layers:
            new_feature = layer(x)
            x = torch.cat([x, new_feature], 1)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.fused_bn_relu = FusedBNReLUFolded(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.fused_bn_relu(x)
        out = self.conv(out)
        out = self.pool(out)
        return out

class Model(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(Model, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_relu0 = FusedBNReLUFolded(64)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_features = 64
        block_layers = [6, 12, 24, 16]

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2

        self.final_op = FusedBNReLUAvgPoolFolded(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.bn_relu0(x)
        x = self.pool0(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)
        
        x = self.final_op(x)
        x = self.classifier(x)
        return x

batch_size = 10
num_classes = 10
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]

# EVOLVE-BLOCK-END