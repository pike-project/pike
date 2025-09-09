import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for a fused BatchNorm2d -> ReLU operation, now supporting non-contiguous tensors.
# This allows it to work on tensor views, which is key to eliminating torch.cat.
bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// A fast parallel reduction in shared memory for summing up stats.
__device__ void warp_reduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__device__ void block_reduce(volatile float* sdata, int block_size, int tid) {
    for (unsigned int s = block_size / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        warp_reduce(sdata, tid);
    }
}

__global__ void batch_norm_relu_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* running_mean,
    float* running_var,
    const bool training,
    const float momentum,
    const float eps,
    const int N,
    const int C,
    const int H,
    const int W,
    const int64_t x_stride_N, const int64_t x_stride_C, const int64_t x_stride_H, const int64_t x_stride_W,
    const int64_t y_stride_N, const int64_t y_stride_C, const int64_t y_stride_H, const int64_t y_stride_W) {

    const int channel_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    extern __shared__ float sdata[];
    float* reduction_space = sdata;
    float* broadcast_space = &sdata[block_size];

    const int num_elements_per_channel = N * H * W;
    const int HW = H * W;

    float mean, var;

    if (training) {
        // === STAGE 1: Compute batch mean ===
        float sum = 0.0f;
        for (int i = tid; i < num_elements_per_channel; i += block_size) {
            int n = i / HW;
            int h = (i / W) % H;
            int w = i % W;
            sum += x[n * x_stride_N + channel_idx * x_stride_C + h * x_stride_H + w * x_stride_W];
        }
        reduction_space[tid] = sum;
        __syncthreads();
        block_reduce(reduction_space, block_size, tid);

        if (tid == 0) {
            mean = reduction_space[0] / num_elements_per_channel;
            running_mean[channel_idx] = (1.0f - momentum) * running_mean[channel_idx] + momentum * mean;
            broadcast_space[0] = mean;
        }
        __syncthreads();
        mean = broadcast_space[0];

        // === STAGE 2: Compute batch variance ===
        float sum_sq = 0.0f;
        for (int i = tid; i < num_elements_per_channel; i += block_size) {
            int n = i / HW;
            int h = (i / W) % H;
            int w = i % W;
            float val = x[n * x_stride_N + channel_idx * x_stride_C + h * x_stride_H + w * x_stride_W];
            sum_sq += (val - mean) * (val - mean);
        }
        reduction_space[tid] = sum_sq;
        __syncthreads();
        block_reduce(reduction_space, block_size, tid);

        if (tid == 0) {
            var = reduction_space[0] / num_elements_per_channel;
            running_var[channel_idx] = (1.0f - momentum) * running_var[channel_idx] + momentum * var;
            broadcast_space[1] = rsqrtf(var + eps);
        }
        __syncthreads();

    } else { // Inference mode
        if (tid == 0) {
            mean = running_mean[channel_idx];
            var = running_var[channel_idx];
            broadcast_space[0] = mean;
            broadcast_space[1] = rsqrtf(var + eps);
        }
        __syncthreads();
    }
    
    // === STAGE 3: Normalize and Apply ReLU (Fusion) ===
    mean = broadcast_space[0];
    float inv_std = broadcast_space[1];
    float gamma = weight[channel_idx];
    float beta = bias[channel_idx];
    
    for (int i = tid; i < num_elements_per_channel; i += block_size) {
        int n = i / HW;
        int h = (i / W) % H;
        int w = i % W;
        const int64_t x_idx = n * x_stride_N + channel_idx * x_stride_C + h * x_stride_H + w * x_stride_W;
        const int64_t y_idx = n * y_stride_N + channel_idx * y_stride_C + h * y_stride_H + w * y_stride_W;
        float normalized_val = gamma * (x[x_idx] - mean) * inv_std + beta;
        y[y_idx] = fmaxf(0.0f, normalized_val);
    }
}

torch::Tensor batch_norm_relu_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    double momentum,
    double eps) {
    
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "weight must be a contiguous CUDA tensor");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(), "bias must be a contiguous CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda() && running_mean.is_contiguous(), "running_mean must be a contiguous CUDA tensor");
    TORCH_CHECK(running_var.is_cuda() && running_var.is_contiguous(), "running_var must be a contiguous CUDA tensor");

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    auto y = torch::empty_like(x);

    auto x_strides = x.strides();
    auto y_strides = y.strides();

    const int block_size = 512;
    const dim3 blocks(C);
    const dim3 threads(block_size);
    const int shared_mem_size = (block_size + 4) * sizeof(float);

    batch_norm_relu_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        training,
        static_cast<float>(momentum),
        static_cast<float>(eps),
        N, C, H, W,
        x_strides[0], x_strides[1], x_strides[2], x_strides[3],
        y_strides[0], y_strides[1], y_strides[2], y_strides[3]);

    return y;
}
"""

bn_relu_cpp_source = """
torch::Tensor batch_norm_relu_forward_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    double momentum,
    double eps);
"""

custom_bn_relu_lib = load_inline(
    name="custom_bn_relu_lib_strided",
    cpp_sources=bn_relu_cpp_source,
    cuda_sources=bn_relu_source,
    functions=["batch_norm_relu_forward_cuda"],
    verbose=False,
)

class CustomBatchNormRelu(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(CustomBatchNormRelu, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and not self.track_running_stats:
            out = F.batch_norm(x, None, None, self.weight, self.bias, False, self.momentum, self.eps)
            return F.relu(out, inplace=True)
        use_custom_kernel = x.is_cuda and self.weight is not None and self.bias is not None
        if use_custom_kernel:
            return custom_bn_relu_lib.batch_norm_relu_forward_cuda(
                x, self.weight, self.bias, self.running_mean, self.running_var,
                self.training, self.momentum, self.eps
            )
        else:
            out = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                               self.training, self.momentum, self.eps)
            return F.relu(out, inplace=True)

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.num_input_features = num_input_features
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            CustomBatchNormRelu(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-allocation strategy to avoid torch.cat
        num_total_features = self.num_input_features + self.num_layers * self.growth_rate
        N, _, H, W = x.shape
        feature_map = x.new_empty((N, num_total_features, H, W))
        feature_map.narrow(1, 0, self.num_input_features).copy_(x)
        
        write_ptr = self.num_input_features
        for layer in self.layers:
            # Create a non-contiguous view for the input to the current layer
            input_view = feature_map.narrow(1, 0, write_ptr)
            # The layer computes the new feature. Our custom kernel handles the non-contiguous view.
            new_feature = layer(input_view)
            # Place the new feature in the correct slice of the feature_map
            feature_map.narrow(1, write_ptr, self.growth_rate).copy_(new_feature)
            write_ptr += self.growth_rate
            
        return feature_map

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            CustomBatchNormRelu(num_input_features),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            CustomBatchNormRelu(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
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
        self.final_bn_relu = CustomBatchNormRelu(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)
        x = self.final_bn_relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x

batch_size = 10
num_classes = 10
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, 3, height, width).cuda()]

def get_init_inputs():
    return [32, num_classes]