# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This solution builds upon the best prior attempts by combining the most effective
# techniques and introducing a key tuning parameter change.
#
# Key Improvements:
# 1.  Warp-per-Output Kernel: We use the elegant "warp-per-output" kernel design,
#     which computes one output element per warp. This avoids shared memory and
#     relies solely on fast intra-warp shuffle instructions for reduction.
# 2.  Increased Block Size: The number of threads per block is increased from 256
#     to 512. This means each thread block now has 16 warps and computes 16 output
#     elements. Larger blocks can improve GPU utilization, increase instruction-level
#     parallelism, and better hide memory latency.
# 3.  Robust CUDA Graph Capture: We incorporate the "warmup run" before capturing
#     the CUDA graph. This ensures that any one-time JIT compilation overhead is
#     not part of the graph, fixing the "empty graph" warning and unlocking the
#     full potential of CUDA graphs for minimizing launch overhead.
fused_mlp_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Templated kernel to compute Y = (ReLU)(X @ W.T + B) for batch size M=1 (GEMV).
// Each warp computes one output element. This version uses a larger block size.
template <bool apply_relu>
__global__ void fused_gemv_tuned_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ y, int N, int K) {
    // Calculate the global warp ID. Each warp is responsible for one output row.
    const int warp_idx = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane_id = threadIdx.x % 32;

    // Early exit for warps that are out of bounds for the output tensor.
    if (warp_idx >= N) {
        return;
    }

    float partial_sum = 0.0f;
    
    // Use float4 for vectorized memory access to maximize bandwidth.
    const float4* x_vec = (const float4*)x;
    const float4* w_row_vec = (const float4*)(w + warp_idx * K);
    const int K_vec = K / 4;

    // Each thread in the warp computes a partial sum of the dot product.
    // The loop strides by warpSize (32).
    for (int k = lane_id; k < K_vec; k += 32) {
        // Use __ldg to load from the read-only cache, as x is reused across all warps.
        float4 x_val = __ldg(&x_vec[k]);
        float4 w_val = w_row_vec[k];
        partial_sum += x_val.x * w_val.x + x_val.y * w_val.y + x_val.z * w_val.z + x_val.w * w_val.w;
    }
    
    // --- Intra-warp reduction using shuffle operations ---
    // This is highly efficient as it avoids shared memory and inter-thread-block synchronization.
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, offset);
    }
    
    // Lane 0 of each warp holds the final sum, adds the bias, applies ReLU, and writes the result.
    if (lane_id == 0) {
        float total_sum = partial_sum + b[warp_idx];

        // `if constexpr` ensures the conditional is resolved at compile time, avoiding runtime branching.
        if constexpr (apply_relu) {
            y[warp_idx] = fmaxf(0.0f, total_sum);
        } else {
            y[warp_idx] = total_sum;
        }
    }
}

// Common C++ launch function to validate inputs and launch the appropriate kernel.
torch::Tensor launch_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor b, bool with_relu) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && b.is_contiguous(), "All tensors must be contiguous");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "Input x must be of shape (1, K)");
    TORCH_CHECK(w.dim() == 2, "Weight w must be of shape (N, K)");
    TORCH_CHECK(x.size(1) == w.size(1), "Inner dimensions of x and w must match");
    TORCH_CHECK(w.size(0) == b.size(0), "Dimensions of w and b must match");
    TORCH_CHECK(x.size(1) % 4 == 0, "Inner dimension K must be a multiple of 4 for vectorization");

    const int N = w.size(0); // output features
    const int K = w.size(1); // input features
    
    auto y = torch::empty({1, N}, x.options());
    
    // TUNING: Increased threads per block to 512 for better latency hiding.
    const int threads_per_block = 512;
    const int warps_per_block = threads_per_block / 32; // 16 warps per block
    
    // Calculate grid size based on the warp-per-output mapping.
    const dim3 grid_dim((N + warps_per_block - 1) / warps_per_block);
    const dim3 block_dim(threads_per_block);

    if (with_relu) {
        fused_gemv_tuned_kernel<true><<<grid_dim, block_dim>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), N, K
        );
    } else {
        fused_gemv_tuned_kernel<false><<<grid_dim, block_dim>>>(
            x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), N, K
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return y;
}

torch::Tensor fused_linear_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    return launch_gemv(x, w, b, true);
}

torch::Tensor fused_linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    return launch_gemv(x, w, b, false);
}
"""

fused_mlp_cpp_source = """
torch::Tensor fused_linear_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
torch::Tensor fused_linear_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
"""

# JIT compile the custom CUDA kernels, giving it a new name to avoid caching issues.
fused_ops = load_inline(
    name="fused_mlp_tuned_graphed",
    cpp_sources=fused_mlp_cpp_source,
    cuda_sources=fused_mlp_source,
    functions=["fused_linear_relu_cuda", "fused_linear_cuda"],
    verbose=True,
)


class Model(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(Model, self).__init__()
        
        self.linear_layers = nn.ModuleList()
        current_input_size = input_size
        
        all_sizes = layer_sizes + [output_size]
        for size in all_sizes:
            self.linear_layers.append(nn.Linear(current_input_size, size))
            current_input_size = size
            
        # Attributes for CUDA graph caching
        self.graph = None
        self.static_input = None
        self.static_output = None

    def _forward_impl(self, x):
        """ The actual forward logic containing the sequence of kernel calls. """
        num_layers = len(self.linear_layers)
        
        # This intermediate tensor is reused to store the output of each layer.
        # This is important for CUDA graph capture, as it ensures the memory
        # addresses of intermediate results are static.
        intermediate = x 
        
        for i in range(num_layers - 1):
            layer = self.linear_layers[i]
            intermediate = fused_ops.fused_linear_relu_cuda(intermediate, layer.weight, layer.bias)
            
        final_layer = self.linear_layers[-1]
        output = fused_ops.fused_linear_cuda(intermediate, final_layer.weight, final_layer.bias)
        
        return output

    def forward(self, x):
        """
        This forward pass uses CUDA Graphs with a warmup run to eliminate kernel 
        launch overhead and JIT compilation costs from the timed region.
        """
        if self.graph is None:
            # 1. Warmup run: Execute the forward pass once to ensure any
            # JIT compilation or initialization of the custom CUDA kernels
            # is complete. This prevents these one-time costs from being
            # incorrectly captured in the graph.
            _ = self._forward_impl(x)
            torch.cuda.synchronize()

            # 2. Graph capture:
            # Create static tensors. A CUDA graph captures operations on specific
            # memory addresses, so we need static tensors to provide stable
            # addresses for the graph's inputs and outputs.
            self.static_input = x.clone()
            
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self._forward_impl(self.static_input)

        # 3. Graph replay: For all subsequent runs.
        # Copy the new input data into the static input tensor.
        self.static_input.copy_(x)
        # Replay the captured kernels. This is much faster than launching them
        # from the CPU and avoids all framework overhead.
        self.graph.replay()
        
        # The output of the replay is in self.static_output.
        return self.static_output

# Test code
batch_size = 1
input_size = 1000
layer_sizes = [400, 800]
output_size = 500

def get_inputs():
    # The custom CUDA kernel expects inputs on the GPU.
    return [torch.randn(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]
# EVOLVE-BLOCK-END