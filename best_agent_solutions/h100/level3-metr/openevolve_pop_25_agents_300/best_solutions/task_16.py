# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels.
# This version builds upon the top-performing solution by introducing a targeted
# micro-optimization to the launch configuration of the final reduction kernel.
# 1. It retains the highly effective dual-path (vectorized/scalar) fused BatchNorm+ReLU kernel.
# 2. It retains the warp-shuffle reduction logic for the final fused operation.
# 3. It retains the critical memory pre-allocation strategy in DenseBlock.
# 4. NEW: It dynamically selects an optimal thread block size for the final
#    BN+ReLU+AvgPool kernel based on the small spatial size (7x7) of the input,
#    reducing thread overhead and improving efficiency compared to a fixed large block size.
densenet_fused_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// ------ KERNEL 1A: Fused BatchNorm + ReLU (Vectorized Path) ------
__global__ void fused_bn_relu_kernel_strided_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int H, int W,
    long stride_n, long stride_c, long stride_h) {

    const int c = blockIdx.y;
    const int n = blockIdx.z;

    const int hw_vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int hw = hw_vec_idx * 4;
    const int spatial_size = H * W;

    if (hw < spatial_size) {
        const float inv_std = rsqrtf(running_var[c] + eps);
        const float scale = weight[c] * inv_std;
        const float effective_bias = bias[c] - running_mean[c] * scale;

        const int h = hw / W;
        const long in_idx = n * stride_n + c * stride_c + h * stride_h;

        const int out_C = gridDim.y;
        const long out_idx = n * (out_C * spatial_size) + c * spatial_size + hw;

        const float4 in_val = *reinterpret_cast<const float4*>(input + in_idx);
        float4 out_val;

        out_val.x = fmaxf(0.f, in_val.x * scale + effective_bias);
        out_val.y = fmaxf(0.f, in_val.y * scale + effective_bias);
        out_val.z = fmaxf(0.f, in_val.z * scale + effective_bias);
        out_val.w = fmaxf(0.f, in_val.w * scale + effective_bias);

        *reinterpret_cast<float4*>(output + out_idx) = out_val;
    }
}

// ------ KERNEL 1B: Fused BatchNorm + ReLU (Scalar Path) ------
__global__ void fused_bn_relu_kernel_strided_scalar(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int H, int W,
    long stride_n, long stride_c, long stride_h, long stride_w) {

    const int hw = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    const int n = blockIdx.z;
    const int spatial_size = H * W;

    if (hw < spatial_size) {
        const float inv_std = rsqrtf(running_var[c] + eps);
        const float scale = weight[c] * inv_std;
        const float effective_bias = bias[c] - running_mean[c] * scale;

        const int h = hw / W;
        const int w = hw % W;

        const long in_idx = n * stride_n + c * stride_c + h * stride_h + w * stride_w;
        const int out_C = gridDim.y;
        const long out_idx = n * (out_C * spatial_size) + c * spatial_size + hw;

        output[out_idx] = fmaxf(0.0f, input[in_idx] * scale + effective_bias);
    }
}

// ------ C++ WRAPPER 1: fused_bn_relu_strided_cuda (Dispatcher) ------
torch::Tensor fused_bn_relu_strided_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double eps) {
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    auto output = torch::empty_like(input, torch::MemoryFormat::Contiguous);
    if (input.numel() == 0) return output;

    const int spatial_size = H * W;
    const int threads_per_block_hw = 256;
    auto strides = input.strides();

    if (W % 4 == 0 && strides[3] == 1) { // Dispatch to vectorized kernel
        const dim3 num_blocks( (spatial_size / 4 + threads_per_block_hw - 1) / threads_per_block_hw, C, N);
        const dim3 threads_per_block(threads_per_block_hw);
        fused_bn_relu_kernel_strided_vectorized<<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            static_cast<float>(eps), H, W,
            strides[0], strides[1], strides[2]);
    } else { // Fallback to scalar kernel
        const dim3 num_blocks( (spatial_size + threads_per_block_hw - 1) / threads_per_block_hw, C, N);
        const dim3 threads_per_block(threads_per_block_hw);
        fused_bn_relu_kernel_strided_scalar<<<num_blocks, threads_per_block>>>(
            input.data_ptr<float>(), output.data_ptr<float>(),
            weight.data_ptr<float>(), bias.data_ptr<float>(),
            running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            static_cast<float>(eps), H, W,
            strides[0], strides[1], strides[2], strides[3]);
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}


// ------ KERNEL 2: Fused BatchNorm + ReLU + Global Average Pool (Warp-Shuffle version) ------
__device__ __forceinline__ float warpReduceSum(float val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int lane = tid % 32;
  int wid = tid / 32;

  val = warpReduceSum(val);
  if (lane == 0) sdata[wid] = val;
  __syncthreads();

  val = (tid < blockDim.x / 32) ? sdata[lane] : 0.0f;
  if (wid == 0) val = warpReduceSum(val);
  return val;
}

__global__ void bn_relu_global_avg_pool_kernel_ws(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float epsilon,
    const int C,
    const int spatial_size) {

    const int nc_idx = blockIdx.x;
    const int c = nc_idx % C;
    const float* channel_input = input + nc_idx * spatial_size;

    const float mean = running_mean[c];
    const float var = running_var[c];
    const float w = weight ? weight[c] : 1.0f;
    const float b = bias ? bias[c] : 0.0f;
    const float inv_std = rsqrtf(var + epsilon);

    float my_sum = 0.0f;
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = channel_input[i];
        val = (val - mean) * inv_std * w + b;
        val = fmaxf(0.0f, val);
        my_sum += val;
    }
    
    my_sum = blockReduceSum(my_sum);

    if (threadIdx.x == 0) {
        output[nc_idx] = my_sum / spatial_size;
    }
}


// ------ C++ WRAPPER 2: bn_relu_global_avg_pool_cuda ------
torch::Tensor bn_relu_global_avg_pool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor running_mean, torch::Tensor running_var, double epsilon) {

    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor for this kernel");
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int spatial_size = H * W;

    auto output = torch::empty({N, C}, input.options());
    if (input.numel() == 0) return output;
    
    // OPTIMIZATION: Dynamically choose block size. For the final 7x7 feature map (spatial_size=49),
    // a small block size like 64 is much more efficient than a generic 256.
    int block_size;
    if (spatial_size <= 64) block_size = 64;
    else if (spatial_size <= 128) block_size = 128;
    else block_size = 256;

    dim3 num_blocks(N * C);
    dim3 block_dim(block_size);
    size_t shared_mem_size = (block_size / 32) * sizeof(float);
    
    const float* weight_ptr = weight.defined() ? weight.data_ptr<float>() : nullptr;
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    bn_relu_global_avg_pool_kernel_ws<<<num_blocks, block_dim, shared_mem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        weight_ptr, bias_ptr, running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        static_cast<float>(epsilon), C, spatial_size);
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

densenet_fused_kernels_cpp_source = """
torch::Tensor fused_bn_relu_strided_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, double epsilon);
torch::Tensor bn_relu_global_avg_pool_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, double epsilon);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="densenet_fused_kernels_v5",
    cpp_sources=densenet_fused_kernels_cpp_source,
    cuda_sources=densenet_fused_kernels_source,
    functions=["fused_bn_relu_strided_cuda", "bn_relu_global_avg_pool_cuda"],
    verbose=True,
)


class FusedBatchNormReLU(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps)
    
    def forward(self, x):
        if self.training:
            return F.relu(self.bn(x), inplace=True)
        else:
            return fused_ops.fused_bn_relu_strided_cuda(
                x, self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps
            )

class FusedFinalBNReLUAvgPool(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d):
        super().__init__()
        self.eps = bn_layer.eps
        self.register_buffer('running_mean', bn_layer.running_mean)
        self.register_buffer('running_var', bn_layer.running_var)
        if bn_layer.affine:
            self.register_parameter('weight', nn.Parameter(bn_layer.weight.clone()))
            self.register_parameter('bias', nn.Parameter(bn_layer.bias.clone()))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        if self.training:
            # Fallback for training mode
            x = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=True, eps=self.eps)
            x = F.relu(x, inplace=True)
            return F.adaptive_avg_pool2d(x, (1, 1))
        else:
            return fused_ops.bn_relu_global_avg_pool_cuda(
                x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
            )

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            FusedBatchNormReLU(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # Memory pre-allocation strategy for inference
            num_input_features = x.shape[1]
            num_final_features = num_input_features + self.num_layers * self.growth_rate
            B, _, H, W = x.shape
            # Create a single large buffer
            out_tensor = torch.empty((B, num_final_features, H, W), dtype=x.dtype, device=x.device)
            out_tensor.narrow(1, 0, num_input_features).copy_(x)
            
            offset = num_input_features
            for layer in self.layers:
                # Input is a non-contiguous view into the buffer
                input_tensor = out_tensor.narrow(1, 0, offset)
                new_feature = layer(input_tensor)
                # Copy the result into its designated slot
                out_tensor.narrow(1, offset, self.growth_rate).copy_(new_feature)
                offset += self.growth_rate
            return out_tensor
        else:
            # Original logic for training
            features = [x]
            for layer in self.layers:
                x_cat = torch.cat(features, 1)
                new_feature = layer(x_cat)
                features.append(new_feature)
            return torch.cat(features, 1)

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            FusedBatchNormReLU(num_input_features),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.transition(x)

class Model(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            FusedBatchNormReLU(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        num_features = 64
        block_layers = [6, 12, 48, 32]
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate
            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features = num_features // 2
        
        # Capture the original BatchNorm layer to create our fused module
        final_bn = nn.BatchNorm2d(num_features)
        self.fused_final_op = FusedFinalBNReLUAvgPool(final_bn)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)
        
        x = self.fused_final_op(x)
        x = x.view(x.size(0), -1) # Classifier expects a 2D tensor
        x = self.classifier(x)
        return x

batch_size = 10
num_classes = 10
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, 3, height, width).cuda()]

def get_init_inputs():
    return [32, num_classes]

# EVOLVE-BLOCK-END