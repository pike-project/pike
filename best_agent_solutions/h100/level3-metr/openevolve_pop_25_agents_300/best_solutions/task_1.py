# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This solution implements the most effective combination of optimizations observed in
# prior high-performing attempts for this GEMV-bound workload.
# 1. `float4` Vectorization: Maximizes memory bandwidth by loading 4 floats per instruction.
# 2. Fused Linear + ReLU: A C++ template `template<bool APPLY_RELU>` generates specialized
#    kernels at compile-time, avoiding the overhead of a separate ReLU kernel launch and
#    intermediate memory writes.
# 3. Two-Stage Hybrid Reduction: An intra-warp reduction using fast `__shfl_down_sync`
#    primitives is combined with a simple, serial inter-warp reduction performed by a
#    single thread. This minimizes shared memory traffic and synchronization overhead.

fused_matvec_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// A block size of 256 is a robust choice for many architectures, balancing
// parallelism and resource usage. It must be a multiple of the warp size.
#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <bool APPLY_RELU>
__global__ void matvec_fused_kernel(
    const float* __restrict__ x,      // Input vector, size K
    const float* __restrict__ w,      // Weight matrix, size N x K
    const float* __restrict__ b,      // Bias vector, size N
    float* __restrict__ y,            // Output vector, size N
    int N, int K)
{
    // Each CUDA block is responsible for computing one element of the output vector y.
    const int j = blockIdx.x; // This is the output index, y[j]
    if (j >= N) return;

    // Shared memory to store the final reduced sum from each warp.
    // For a block size of 256, this is a small array of 8 floats.
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];

    const int tid = threadIdx.x;
    float thread_sum = 0.0f;

    // Reinterpret pointers to float4 for vectorized loads, assuming K is a multiple of 4.
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* w4_row = reinterpret_cast<const float4*>(w + j * K);
    const int K4 = K / 4;

    // Each thread computes a partial sum by striding through the K dimension.
    // This grid-stride loop ensures all data is processed regardless of block size.
    for (int k = tid; k < K4; k += blockDim.x) {
        float4 x_vec = x4[k];
        float4 w_vec = w4_row[k];
        // Use FMA (fused multiply-add) for performance and precision.
        thread_sum = fmaf(x_vec.x, w_vec.x, thread_sum);
        thread_sum = fmaf(x_vec.y, w_vec.y, thread_sum);
        thread_sum = fmaf(x_vec.z, w_vec.z, thread_sum);
        thread_sum = fmaf(x_vec.w, w_vec.w, thread_sum);
    }

    // --- Stage 1: Intra-warp reduction using shuffle instructions ---
    // This efficiently sums `thread_sum` across all 32 threads in a warp. The result
    // is stored in the register of the first thread (lane 0) of each warp.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
    }

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // The first thread of each warp writes its partial sum to shared memory.
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }

    // Synchronize to ensure all warp sums are written before the final reduction.
    __syncthreads();

    // --- Stage 2: Inter-warp reduction ---
    // The first thread of the block (tid 0) performs a simple serial reduction
    // over the small number of warp sums. This is more efficient than a second
    // parallel reduction for this small amount of data.
    if (tid == 0) {
        float block_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < (BLOCK_SIZE / WARP_SIZE); i++) {
            block_sum += warp_sums[i];
        }

        // Add bias and apply optional ReLU, then write the final result.
        float final_val = block_sum + b[j];
        if (APPLY_RELU) {
            y[j] = fmaxf(0.0f, final_val);
        } else {
            y[j] = final_val;
        }
    }
}


// C++ wrapper to interface with PyTorch
torch::Tensor fused_op_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    bool apply_relu)
{
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "Weight w must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "Bias b must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "Input x must be of shape (1, K)");
    TORCH_CHECK(w.dim() == 2, "Weight w must be of shape (N, K)");
    TORCH_CHECK(b.dim() == 1, "Bias b must be of shape (N)");

    const int K = x.size(1);
    const int N = w.size(0);

    TORCH_CHECK(K % 4 == 0, "Vectorized kernel requires K to be divisible by 4");
    TORCH_CHECK(w.size(1) == K, "Dimension mismatch: x.size(1) != w.size(1)");
    TORCH_CHECK(b.size(0) == N, "Dimension mismatch: w.size(0) != b.size(0)");

    // Ensure tensors are contiguous in memory
    x = x.contiguous();
    w = w.contiguous();
    b = b.contiguous();
    
    // The kernel operates on 1D vectors, so we squeeze the batch dim.
    const auto x_vec = x.squeeze(0);
    auto y = torch::empty({N}, x.options());

    // Launch configuration: one block per output element.
    const dim3 threadsPerBlock(BLOCK_SIZE);
    const dim3 numBlocks(N);

    // Launch the correct templated kernel based on the apply_relu flag.
    if (apply_relu) {
        matvec_fused_kernel<true><<<numBlocks, threadsPerBlock>>>(
            x_vec.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), N, K);
    } else {
        matvec_fused_kernel<false><<<numBlocks, threadsPerBlock>>>(
            x_vec.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), N, K);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Return the result with a batch dimension to match PyTorch's nn.Linear output shape.
    return y.unsqueeze(0);
}
"""

fused_matvec_cpp_source = """
torch::Tensor fused_op_forward(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    bool apply_relu);
"""

# JIT compile the CUDA kernel. Using a unique name avoids caching issues.
fused_op = load_inline(
    name="fused_matvec_op_hybrid_reduction",
    cpp_sources=fused_matvec_cpp_source,
    cuda_sources=fused_matvec_source,
    functions=["fused_op_forward"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(Model, self).__init__()
        
        # We still use nn.Linear modules to hold the parameters (weights and biases)
        # for easy management and compatibility with PyTorch's ecosystem.
        self.layers = nn.ModuleList()
        current_input_size = input_size
        
        all_layer_sizes = layer_sizes + [output_size]
        for layer_size in all_layer_sizes:
            self.layers.append(nn.Linear(current_input_size, layer_size))
            current_input_size = layer_size
            
    def forward(self, x):
        num_layers = len(self.layers)
        
        # Apply our custom fused operator for all layers in a single loop.
        for i, linear_layer in enumerate(self.layers):
            # The last layer should not have ReLU applied.
            apply_relu = (i < num_layers - 1)
            x = fused_op.fused_op_forward(
                x, linear_layer.weight, linear_layer.bias, apply_relu
            )
        
        return x

# Test code
batch_size = 1
input_size = 1000
layer_sizes = [400, 800]
output_size = 500

def get_inputs():
    # The custom CUDA kernel expects a CUDA tensor.
    return [torch.randn(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]
# EVOLVE-BLOCK-END