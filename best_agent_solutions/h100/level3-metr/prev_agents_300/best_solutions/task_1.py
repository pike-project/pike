import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused GEMV (MatVec) + Bias + ReLU for float32
# This kernel is heavily optimized for the batch_size=1 case.
# Key Optimizations:
# 1. Operator Fusion: Combines matrix-vector multiply, bias addition, and ReLU activation into a single kernel launch.
#    This drastically reduces kernel launch overhead and intermediate memory traffic.
# 2. Shared Memory Caching: The input vector 'x' is small and reused for every output element calculation.
#    Loading it into shared memory once per block makes subsequent accesses much faster than repeated global memory reads.
# 3. Coalesced Memory Access: The kernel is designed so that threads within a warp access contiguous memory locations
#    in the large weight matrix 'w'. This is the most efficient way to read from global memory.
# CORRECTNESS FIX: This version operates entirely in float32 precision. The original version used float16,
# which led to significant precision loss and a large numerical difference from the baseline. By switching
# all inputs, computations, and outputs to float32, we align perfectly with the baseline's numerics,
# ensuring correctness while still benefiting from fusion and other CUDA optimizations.
fused_gemv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h> // Header for C10_CUDA_KERNEL_LAUNCH_CHECK
#include <cmath> // For fmaxf

// --- CUDA Kernel ---
// Computes y = activation(x @ W + b) for batch size 1 (GEMV).
// x: (1, K), W: (K, N), b: (N,), y: (1, N)
// W is assumed to be contiguous and row-major with shape (K, N).
// This layout results from weight.T.contiguous() in PyTorch on a (N, K) nn.Linear weight.
template <int BLOCK_THREADS>
__global__ void fused_gemv_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int K,
    const int N,
    const bool apply_relu
) {
    // Use dynamic shared memory for caching the input vector x, as its size K is a runtime variable.
    extern __shared__ float s_x[];

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;

    // Collaboratively load x into shared memory. Each thread loads multiple elements if K > BLOCK_THREADS.
    for (int i = tid; i < K; i += BLOCK_THREADS) {
        s_x[i] = x[i];
    }
    __syncthreads(); // Synchronize to ensure x is fully loaded before proceeding.

    // Each thread in the block computes one output element.
    const int j = block_id * BLOCK_THREADS + tid;

    if (j < N) {
        float accumulator = 0.0f;
        for (int i = 0; i < K; ++i) {
            // Accessing W column-wise (w[i * N + j]) is perfectly coalesced across threads in a warp.
            accumulator += s_x[i] * w[i * N + j];
        }

        // Add bias
        accumulator += bias[j];

        // Apply ReLU activation if requested
        if (apply_relu) {
            accumulator = fmaxf(accumulator, 0.0f);
        }

        // Store the final result back to global memory.
        out[j] = accumulator;
    }
}

// --- C++ Wrapper for PyTorch ---
torch::Tensor fused_gemv_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    bool apply_relu
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "All inputs must be float32 tensors");
    TORCH_CHECK(w.dtype() == torch::kFloat32, "All inputs must be float32 tensors");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "All inputs must be float32 tensors");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "x must be of shape (1, K)");
    TORCH_CHECK(w.dim() == 2, "w must be of shape (K, N)");
    TORCH_CHECK(x.size(1) == w.size(0), "Inner dimensions of x and w must match");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == w.size(1), "bias must be of shape (N,)");

    const int K = x.size(1);
    const int N = w.size(1);

    auto out = torch::empty({1, N}, x.options());

    // Kernel launch configuration. A block size of 256 is a robust choice for many GPUs.
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    // Calculate required dynamic shared memory: K elements of float type.
    const size_t shared_mem_size = K * sizeof(float);

    // Launch the specific templated version of the kernel.
    fused_gemv_kernel<block_size><<<num_blocks, block_size, shared_mem_size>>>(
        (const float*)x.data_ptr(),
        (const float*)w.data_ptr(),
        (const float*)bias.data_ptr(),
        (float*)out.data_ptr(),
        K,
        N,
        apply_relu
    );

    // Check for any CUDA errors during kernel launch.
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
"""

fused_gemv_cpp_source = (
    "torch::Tensor fused_gemv_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor bias, bool apply_relu);"
)

# Compile the inline CUDA C++ code
fused_gemv_op = load_inline(
    name="fused_gemv_op",
    cpp_sources=fused_gemv_cpp_source,
    cuda_sources=fused_gemv_source,
    functions=["fused_gemv_cuda"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()
        # CORRECTNESS FIX: Use float32 to match the baseline model's precision.
        self.dtype = torch.float32

        all_sizes = [input_size] + layer_sizes + [output_size]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(len(all_sizes) - 1):
            in_features = all_sizes[i]
            out_features = all_sizes[i+1]

            # Initialize parameters on CPU in FP32
            temp_weight = torch.empty(out_features, in_features)
            nn.init.kaiming_uniform_(temp_weight, a=math.sqrt(5))

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(temp_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            temp_bias = torch.empty(out_features)
            nn.init.uniform_(temp_bias, -bound, bound)

            # Transpose weight for optimal memory layout (K, N) for the kernel,
            # ensure contiguity, and move to GPU with the specified dtype.
            self.weights.append(nn.Parameter(temp_weight.T.contiguous().cuda().to(self.dtype)))
            self.biases.append(nn.Parameter(temp_bias.contiguous().cuda().to(self.dtype)))

        # Placeholders for CUDA Graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def _forward_impl(self, x):
        """ The actual computation logic using the custom CUDA kernel. """
        # The input x from get_inputs is already float32. This call is a no-op but ensures dtype.
        x = x.to(dtype=self.dtype)

        # Apply fused layers with ReLU
        for i in range(len(self.weights) - 1):
            x = fused_gemv_op.fused_gemv_cuda(x, self.weights[i], self.biases[i], True)

        # The last layer has no activation.
        # FIX: The original code used self.biases[i] which was the incorrect index from the
        # previous loop. It must be self.biases[-1] to match self.weights[-1].
        x = fused_gemv_op.fused_gemv_cuda(x, self.weights[-1], self.biases[-1], False)

        # The output is already float32, so no final cast is needed.
        return x

    def forward(self, x):
        """
        Forward pass with CUDA Graph optimization to minimize CPU overhead.
        For small and fast kernels, CPU launch overhead can be a major bottleneck.
        CUDA Graphs capture the entire sequence of kernel launches and allow them to be
        replayed with a single command, almost eliminating CPU overhead.
        """
        # CUDA Graphs require a static shape. If shape changes, the graph must be recaptured.
        if self.graph is None or not hasattr(self, 'static_input') or self.static_input.shape != x.shape:
            # Create a static input tensor. The actual input content will be copied here.
            self.static_input = x.clone()

            # Warmup: Run the implementation once to ensure any one-time setup is done.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._forward_impl(self.static_input)
            torch.cuda.current_stream().wait_stream(s)

            # Capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self._forward_impl(self.static_input)

        # For subsequent runs with the same shape, copy the new input data
        # into the static tensor and replay the captured graph.
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()


# Parameters from the problem description
batch_size = 1
input_size = 1000
layer_sizes = [400, 800]
output_size = 500

def get_inputs():
    # Model is optimized for CUDA inputs.
    return [torch.randn(batch_size, input_size).cuda()]

def get_init_inputs():
    # The Model constructor uses these parameters to build the network.
    return [input_size, layer_sizes, output_size]