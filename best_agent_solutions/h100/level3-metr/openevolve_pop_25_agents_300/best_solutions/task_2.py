# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This custom CUDA kernel integrates the best features from all top-performing prior attempts:
# 1. Fusion: Combines Linear + Bias + ReLU to reduce kernel launch overhead and memory traffic.
# 2. float4 Vectorization: Maximizes memory bandwidth, which is the key bottleneck for this
#    memory-bound matrix-vector multiplication (batch_size=1).
# 3. Hybrid Warp-Shuffle Reduction: Uses a highly efficient two-stage reduction that minimizes
#    shared memory usage and synchronization overhead by leveraging fast intra-warp shuffles.
# 4. Templated ReLU: Eliminates runtime branching for the ReLU application by generating
#    specialized kernels at compile time, one for hidden layers and one for the output layer.
fused_linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

template <bool APPLY_RELU>
__global__ void __launch_bounds__(256, 1) fused_linear_relu_kernel_templated(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int in_features,
    int out_features
) {
    // Shared memory for reduction across warps. One float per warp.
    extern __shared__ float sdata[];

    const int row = blockIdx.x; // Each block computes one output element
    if (row >= out_features) return;

    const int tid = threadIdx.x;
    const int warp_size = 32;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;

    float local_sum = 0.0f;
    
    // --- Stage 1: Vectorized Computation using float4 ---
    const int in_features_f4 = in_features / 4;
    const float4* x_f4 = reinterpret_cast<const float4*>(x);
    const float4* weight_row_f4 = reinterpret_cast<const float4*>(weight + row * in_features);
    
    for (int i = tid; i < in_features_f4; i += blockDim.x) {
        float4 x_val = x_f4[i];
        float4 w_val = weight_row_f4[i];
        local_sum += x_val.x * w_val.x;
        local_sum += x_val.y * w_val.y;
        local_sum += x_val.z * w_val.z;
        local_sum += x_val.w * w_val.w;
    }

    // --- Stage 2: Hybrid Parallel Reduction ---

    // 2a. Intra-Warp reduction using shuffle instructions.
    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    // 2b. First thread of each warp writes the warp's sum to shared memory.
    if (lane_id == 0) {
        sdata[warp_id] = local_sum;
    }

    __syncthreads();

    // 2c. First warp reduces the final results from all other warps.
    if (warp_id == 0) {
        local_sum = (lane_id < blockDim.x / warp_size) ? sdata[lane_id] : 0.0f;

        // Final reduction within the first warp.
        #pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }

        // 2d. Thread 0 writes the final fused result.
        if (lane_id == 0) {
            float final_sum = local_sum + bias[row];
            
            // APPLY_RELU is a compile-time constant, so this 'if' has zero runtime cost.
            if (APPLY_RELU) {
                final_sum = fmaxf(0.0f, final_sum);
            }
            out[row] = final_sum;
        }
    }
}

torch::Tensor fused_linear_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    bool apply_relu
) {
    auto x_cont = x.contiguous();
    auto weight_cont = weight.contiguous();
    
    TORCH_CHECK(x_cont.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight_cont.is_cuda(), "Input weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Input bias must be a CUDA tensor");
    
    TORCH_CHECK(x_cont.dim() == 2 && x_cont.size(0) == 1, "Input x must be 2D with batch_size=1");
    const int batch_size = x_cont.size(0);
    const int in_features = x_cont.size(1);
    const int out_features = weight_cont.size(0);

    // Check for float4 compatibility, which is crucial for performance.
    TORCH_CHECK(in_features % 4 == 0, "in_features must be divisible by 4 for float4 kernel");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x_cont.data_ptr()) % 16 == 0, "x tensor is not 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(weight_cont.data_ptr()) % 16 == 0, "weight tensor is not 16-byte aligned");

    TORCH_CHECK(weight_cont.size(1) == in_features, "Weight dimension mismatch");
    TORCH_CHECK(bias.size(0) == out_features, "Bias dimension mismatch");

    auto out = torch::empty({batch_size, out_features}, x_cont.options());

    // Kernel launch configuration
    const int block_size = 256;
    const int num_blocks = out_features;
    const int warp_size = 32;
    const size_t shared_mem_size = (block_size / warp_size) * sizeof(float);

    // Template dispatch to launch the correct, pre-compiled kernel version.
    if (apply_relu) {
        fused_linear_relu_kernel_templated<true><<<num_blocks, block_size, shared_mem_size>>>(
            x_cont.data_ptr<float>(), weight_cont.data_ptr<float>(), bias.data_ptr<float>(),
            out.data_ptr<float>(), in_features, out_features
        );
    } else {
        fused_linear_relu_kernel_templated<false><<<num_blocks, block_size, shared_mem_size>>>(
            x_cont.data_ptr<float>(), weight_cont.data_ptr<float>(), bias.data_ptr<float>(),
            out.data_ptr<float>(), in_features, out_features
        );
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    return out;
}
"""

fused_linear_relu_cpp_source = (
    "torch::Tensor fused_linear_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, bool apply_relu);"
)

# Compile the inline CUDA code, using a unique name to avoid caching issues
fused_op = load_inline(
    name="fused_op_mlp_ultimate",
    cpp_sources=fused_linear_relu_cpp_source,
    cuda_sources=fused_linear_relu_source,
    functions=["fused_linear_cuda"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(Model, self).__init__()
        
        self.fused_op = fused_op

        # Use nn.Linear layers to hold parameters, simplifying integration with PyTorch ecosystem.
        layers = []
        current_input_size = input_size
        
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            current_input_size = hidden_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        num_layers = len(self.layers)
        
        # Apply fused (Linear + ReLU) for all hidden layers
        for i in range(num_layers - 1):
            x = self.fused_op.fused_linear_cuda(
                x, self.layers[i].weight, self.layers[i].bias, True
            )
        
        # Apply fused (Linear only) for the final output layer
        x = self.fused_op.fused_linear_cuda(
            x, self.layers[-1].weight, self.layers[-1].bias, False
        )
        
        return x

# Test code
batch_size = 1
input_size = 1000
hidden_layer_sizes = [2000, 2000]
output_size = 10

def get_inputs():
    # Custom kernel requires input tensor on the CUDA device.
    return [torch.randn(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
# EVOLVE-BLOCK-END