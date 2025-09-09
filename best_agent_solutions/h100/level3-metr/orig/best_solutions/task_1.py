import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from torch.utils.cpp_extension import load_inline

# --- Custom Fully-Fused and Tiled MLP Kernel (CUDA C++) ---
# This solution improves upon the previous fully-fused MLP by introducing
# tiling for the first and largest matrix-vector multiplication.
#
# Performance Improvement:
# The primary bottleneck in the previous kernel was the repeated reading of the
# input vector `x` from global memory for the first layer's computation. For each
# of the `h1_size` output neurons, the entire `input_size` vector `x` was read.
#
# This new kernel implements a tiled matrix-vector multiplication for the first
# layer. The block of threads cooperatively loads a tile of the input vector `x`
# into fast shared memory. All threads in the block then reuse this tile for their
# partial dot-product calculations. This reduces global memory reads of `x` by
# a factor equal to the number of threads in the block (e.g., 512x), significantly
# improving memory bandwidth utilization and overall performance.
#
# For subsequent layers, the intermediate activations (`h1`, `h2`) are already
# in shared memory (a benefit of full fusion), so no further tiling is necessary for them.

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Helper for ReLU activation using the fast __fmaxf intrinsic
__device__ __forceinline__ float relu(float x) {
    return __fmaxf(0.0f, x);
}

// This optimized kernel computes the forward pass for a 3-layer MLP.
// It uses a single thread block and keeps intermediate activations in shared memory.
//
// Optimization: Layer 1 (the largest) uses a tiled approach. The input vector 'x'
// is loaded into shared memory in tiles, and these tiles are reused by all threads
// to compute the h1 activations. This dramatically reduces global memory bandwidth usage.
__global__ void fully_fused_mlp_kernel_tiled(
    const float* __restrict__ x,      // Input tensor (1, input_size)
    const float* __restrict__ w1,     // Weight 1 (h1_size, input_size)
    const float* __restrict__ b1,     // Bias 1 (h1_size)
    const float* __restrict__ w2,     // Weight 2 (h2_size, h1_size)
    const float* __restrict__ b2,     // Bias 2 (h2_size)
    const float* __restrict__ w3,     // Weight 3 (output_size, h2_size)
    const float* __restrict__ b3,     // Bias 3 (output_size)
    float* __restrict__ out,          // Output tensor (1, output_size)
    const int input_size,
    const int h1_size,
    const int h2_size,
    const int output_size,
    const int K_TILE_SIZE
) {
    // Dynamically allocated shared memory.
    // Layout: [tile for x (K_TILE_SIZE)] [h1 (h1_size)] [h2 (h2_size)]
    extern __shared__ float s_mem[];
    float* s_x_tile = s_mem;
    float* s_h1 = s_mem + K_TILE_SIZE;
    float* s_h2 = s_mem + K_TILE_SIZE + h1_size;

    // --- Layer 1: h1 = relu(W1 @ x + b1) ---
    // Tiled matrix-vector multiplication to maximize reuse of x from shared memory.
    // Each thread computes one or more output elements of h1 using a grid-stride loop.
    for (int i = threadIdx.x; i < h1_size; i += blockDim.x) {
        float accum = 0.0f;
        const float* w1_row = w1 + i * input_size;
        // Loop over tiles of the input vector dimension
        for (int k_base = 0; k_base < input_size; k_base += K_TILE_SIZE) {
            __syncthreads(); // Sync to ensure s_x_tile is not being written by next iteration

            // Determine the actual size of the tile (handles last partial tile)
            int k_tile_len = (k_base + K_TILE_SIZE > input_size) ? (input_size - k_base) : K_TILE_SIZE;

            // Cooperatively load a tile of the input vector 'x' into shared memory.
            // Each thread loads one element of the tile.
            if (threadIdx.x < k_tile_len) {
                s_x_tile[threadIdx.x] = x[k_base + threadIdx.x];
            }
            __syncthreads(); // Sync to ensure the tile is fully loaded before use

            // Compute partial dot product using the tile from shared memory.
            float psum = 0.0f;
            for (int k_tile = 0; k_tile < k_tile_len; ++k_tile) {
                psum += w1_row[k_base + k_tile] * s_x_tile[k_tile];
            }
            accum += psum;
        }
        s_h1[i] = relu(accum + b1[i]);
    }

    __syncthreads(); // IMPORTANT: All threads must finish computing h1 before starting h2

    // --- Layer 2: h2 = relu(W2 @ h1 + b2) ---
    // The input (h1) is already in shared memory. Tiling is not required.
    for (int i = threadIdx.x; i < h2_size; i += blockDim.x) {
        float accum = 0.0f;
        const float* w2_row = w2 + i * h1_size;
        for (int k = 0; k < h1_size; ++k) {
            accum += w2_row[k] * s_h1[k];
        }
        s_h2[i] = relu(accum + b2[i]);
    }

    __syncthreads(); // IMPORTANT: All threads must finish h2 before starting the final layer

    // --- Layer 3: out = W3 @ h2 + b3 ---
    // The input (h2) is already in shared memory. Tiling is not required.
    for (int i = threadIdx.x; i < output_size; i += blockDim.x) {
        float accum = 0.0f;
        const float* w3_row = w3 + i * h2_size;
        for (int k = 0; k < h2_size; ++k) {
            accum += w3_row[k] * s_h2[k];
        }
        out[i] = accum + b3[i];
    }
}

torch::Tensor fully_fused_mlp_forward_cuda(
    torch::Tensor x,
    torch::Tensor w1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor b2,
    torch::Tensor w3, torch::Tensor b3
) {
    // Input validation
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "Fused kernel only supports batch_size=1, got shape ", x.sizes());
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(w1.is_contiguous() && b1.is_contiguous(), "Layer 1 tensors must be contiguous");
    TORCH_CHECK(w2.is_contiguous() && b2.is_contiguous(), "Layer 2 tensors must be contiguous");
    TORCH_CHECK(w3.is_contiguous() && b3.is_contiguous(), "Layer 3 tensors must be contiguous");

    const int batch_size = x.size(0);
    const int input_size = x.size(1);
    const int h1_size = w1.size(0);
    const int h2_size = w2.size(0);
    const int output_size = w3.size(0);

    auto out = torch::empty({batch_size, output_size}, x.options());

    // Kernel launch configuration
    // The tile size for loading 'x' is tied to the block size for this implementation.
    // 512 is a good choice for modern GPUs, offering a good balance of parallelism and resource usage.
    const int K_TILE_SIZE = 512;
    const int block_size = K_TILE_SIZE;
    const int num_blocks = 1; // A single block computes the entire result

    // Calculate required dynamic shared memory
    const int shared_mem_size = (K_TILE_SIZE + h1_size + h2_size) * sizeof(float);
    
    // Check if requested shared memory exceeds device limits
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    TORCH_CHECK(shared_mem_size <= prop.sharedMemPerBlock, "Requested shared memory (", shared_mem_size, " bytes) exceeds device limit (", prop.sharedMemPerBlock, " bytes)");

    fully_fused_mlp_kernel_tiled<<<num_blocks, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        w1.data_ptr<float>(), b1.data_ptr<float>(),
        w2.data_ptr<float>(), b2.data_ptr<float>(),
        w3.data_ptr<float>(), b3.data_ptr<float>(),
        out.data_ptr<float>(),
        input_size, h1_size, h2_size, output_size, K_TILE_SIZE
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return out;
}
"""

cpp_source = """
torch::Tensor fully_fused_mlp_forward_cuda(
    torch::Tensor x,
    torch::Tensor w1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor b2,
    torch::Tensor w3, torch::Tensor b3
);
"""

try:
    if torch.cuda.is_available():
        fused_mlp_op = load_inline(
            name="fully_fused_mlp_tiled",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["fully_fused_mlp_forward_cuda"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        )
        HAS_CUSTOM_CUDA = True
    else:
        HAS_CUSTOM_CUDA = False
except Exception as e:
    print(f"Warning: Failed to compile custom CUDA kernel. Falling back to PyTorch. Error: {e}")
    HAS_CUSTOM_CUDA = False


class Model(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(Model, self).__init__()
        
        # The custom kernel is specialized for a 2-hidden-layer MLP.
        self.use_custom_kernel = (HAS_CUSTOM_CUDA and len(layer_sizes) == 2)
        
        if self.use_custom_kernel:
            h1_size, h2_size = layer_sizes[0], layer_sizes[1]
            
            # Manually define parameters for each layer to pass to the custom kernel
            self.w1 = nn.Parameter(torch.empty(h1_size, input_size))
            self.b1 = nn.Parameter(torch.empty(h1_size))
            self.w2 = nn.Parameter(torch.empty(h2_size, h1_size))
            self.b2 = nn.Parameter(torch.empty(h2_size))
            self.w3 = nn.Parameter(torch.empty(output_size, h2_size))
            self.b3 = nn.Parameter(torch.empty(output_size))
            
            self.reset_parameters()
        else:
            print("Warning: Custom CUDA kernel not used. Falling back to nn.Sequential.")
            layers = []
            current_input_size = input_size
            for layer_size in layer_sizes:
                layers.append(nn.Linear(current_input_size, layer_size))
                layers.append(nn.ReLU())
                current_input_size = layer_size
            layers.append(nn.Linear(current_input_size, output_size))
            self.network = nn.Sequential(*layers)

    def reset_parameters(self) -> None:
        if not hasattr(self, 'w1'):
            return
            
        init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.b1, -bound, bound)

        init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w2)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.b2, -bound, bound)

        init.kaiming_uniform_(self.w3, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w3)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.b3, -bound, bound)
    
    def forward(self, x):
        if self.use_custom_kernel and x.is_cuda:
            return fused_mlp_op.fully_fused_mlp_forward_cuda(
                x, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3
            )
        else:
            return self.network(x)

# Test code
batch_size = 1
input_size = 1000
layer_sizes = [400, 800]
output_size = 500

def get_inputs():
    # The custom kernel expects a CUDA tensor.
    return [torch.randn(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]