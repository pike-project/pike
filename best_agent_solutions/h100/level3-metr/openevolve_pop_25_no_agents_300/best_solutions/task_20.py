# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- Custom Kernel: Fused AdaptiveAvgPool2d + Flatten (Optimized) ---
# This kernel fuses global average pooling and the subsequent flatten operation.
# This is a key optimization as it reduces kernel launch overhead and memory traffic
# at the end of the feature extraction pipeline. A block size of 64 is chosen as it's
# optimal for the typical 7x7 feature maps at the end of MobileNetV2, minimizing
# thread wastage while still allowing for efficient parallel reduction in shared memory.
fused_pool_flatten_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_pool_flatten_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N,
    const int C,
    const int H,
    const int W) {

    extern __shared__ float sdata[];

    const int c = blockIdx.x; // Channel index
    const int n = blockIdx.y; // Batch index

    const int tid = threadIdx.x;
    const int threadsPerBlock = blockDim.x;

    const int map_size = H * W;
    const float inv_map_size = 1.0f / (float)map_size;
    
    // Pointer to the start of the current feature map (for batch n, channel c)
    const float* feature_map = input + (n * C + c) * map_size;

    // Each thread sums a portion of the feature map using a strided loop
    float my_sum = 0.0f;
    for (int i = tid; i < map_size; i += threadsPerBlock) {
        my_sum += feature_map[i];
    }
    sdata[tid] = my_sum;
    __syncthreads();

    // Perform parallel reduction within the block using shared memory
    for (unsigned int s = threadsPerBlock / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block writes the final averaged result to the output tensor
    if (tid == 0) {
        output[n * C + c] = sdata[0] * inv_map_size;
    }
}

torch::Tensor fused_pool_flatten_cuda(torch::Tensor input) {
    auto contiguous_input = input.contiguous();
    const auto N = contiguous_input.size(0);
    const auto C = contiguous_input.size(1);
    const auto H = contiguous_input.size(2);
    const auto W = contiguous_input.size(3);

    auto output = torch::empty({N, C}, contiguous_input.options());
    if (H == 0 || W == 0) return output.zero_();

    // Block size of 64 is optimal for small spatial dimensions like 7x7=49
    const int threadsPerBlock = 64; 
    const dim3 blocks(C, N);
    const dim3 threads(threadsPerBlock);
    const size_t shared_mem_size = threadsPerBlock * sizeof(float);

    fused_pool_flatten_kernel<<<blocks, threads, shared_mem_size>>>(
        contiguous_input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed in fused_pool_flatten: ", cudaGetErrorString(err));
    }
    return output;
}
"""

fused_pool_flatten_cpp_source = "torch::Tensor fused_pool_flatten_cuda(torch::Tensor input);"

fused_pool_flatten_ext = load_inline(
    name="fused_pool_flatten_ext_final", # Unique name to avoid compilation cache conflicts
    cpp_sources=fused_pool_flatten_cpp_source,
    cuda_sources=fused_pool_flatten_source,
    functions=["fused_pool_flatten_cuda"],
    verbose=False,
)


# --- Conv-BN Folding Utility ---
# This is the most impactful optimization for inference. It mathematically combines
# a Conv2d and its subsequent BatchNorm2d into a single Conv2d layer with a bias.
# This eliminates the entire BatchNorm operation, saving memory bandwidth and a kernel launch.
def fuse_conv_bn_eval(conv, bn):
    """
    Fuses a Conv2d and BatchNorm2d layer into a single Conv2d layer.
    """
    assert not (conv.training or bn.training), "Fusion is only for inference (eval) mode."
    
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=True
    ).to(conv.weight.device, conv.weight.dtype)

    # Pre-calculate BN parameters for fusion
    w_bn = bn.weight
    b_bn = bn.bias
    mean_bn = bn.running_mean
    var_bn = bn.running_var
    eps_bn = bn.eps
    
    # Calculate the scale and shift from BN parameters
    scale = w_bn / torch.sqrt(var_bn + eps_bn)
    
    # Fold the scale into the convolution weights
    w_conv = conv.weight
    fused_conv.weight.data = w_conv * scale.reshape(-1, 1, 1, 1)
    
    # Fold the shift into the convolution bias
    b_conv = conv.bias if conv.bias is not None else torch.zeros_like(mean_bn)
    fused_conv.bias.data = (b_conv - mean_bn) * scale + b_bn
    
    return fused_conv


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None: min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v: new_v += divisor
            return new_v

        def _get_inverted_residual_block_layers(inp, oup, stride, expand_ratio):
            hidden_dim = int(inp * expand_ratio)
            layers = []
            if expand_ratio != 1:
                # Pointwise
                layers.extend([
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True)
                ])
            layers.extend([
                # Depthwise
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])
            return layers

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2],
            [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1],
        ]

        # Step 1: Build a flat list of all original, unfused layers
        unfused_layers = []
        unfused_layers.extend([
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        ])

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                unfused_layers.extend(_get_inverted_residual_block_layers(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        unfused_layers.extend([
            nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ])

        # Step 2: Initialize weights of the original layers before fusion
        for m in unfused_layers:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Step 3: Fuse Conv-BN pairs and build the final feature extractor
        self.features = self._fuse_layers(unfused_layers)
        
        # Step 4: Assign the custom kernel for pooling and flattening
        self.pool_flatten = fused_pool_flatten_ext.fused_pool_flatten_cuda

        # Step 5: Build the final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _fuse_layers(self, unfused_layers):
        fused_features = []
        i = 0
        while i < len(unfused_layers):
            m1 = unfused_layers[i]
            # Check for a Conv2d followed by a BatchNorm2d
            if isinstance(m1, nn.Conv2d) and i + 1 < len(unfused_layers) and isinstance(unfused_layers[i+1], nn.BatchNorm2d):
                m2 = unfused_layers[i+1]
                # Set to eval mode is required for fusion as it uses running_mean/var
                m1.eval()
                m2.eval()
                fused_conv = fuse_conv_bn_eval(m1, m2)
                fused_features.append(fused_conv)
                i += 2 # Skip both original layers
            else:
                fused_features.append(m1)
                i += 1
        return nn.Sequential(*fused_features)

    def forward(self, x):
        x = self.features(x)
        x = self.pool_flatten(x)
        x = self.classifier(x)
        return x

batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224).cuda()]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END