import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and C++ wrappers for fused linear operations
fused_linear_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath> // For fmaxf

// A generic kernel that computes Y = activation(X @ W.T + B)
// Each CUDA block computes one element of the output matrix.
// Threads within the block cooperate to compute the dot product via parallel reduction.
__global__ void fused_linear_generic_kernel(const float* __restrict__ x,
                                            const float* __restrict__ w,
                                            const float* __restrict__ b,
                                            float* __restrict__ y,
                                            const int batch_size,
                                            const int in_features,
                                            const int out_features,
                                            const bool apply_relu) {
    // Identify the output element this block is responsible for
    const int b_idx = blockIdx.x;      // Batch index
    const int out_idx = blockIdx.y;    // Output feature index

    // Grid boundary check
    if (b_idx >= batch_size || out_idx >= out_features) {
        return;
    }

    // Shared memory for the reduction
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;

    // Pointers to the start of the relevant rows in the input and weight matrices
    const float* x_row = x + b_idx * in_features;
    const float* w_row = w + out_idx * in_features;

    // Each thread computes a partial sum of the dot product using a grid-stride loop
    float partial_sum = 0.0f;
    for (int i = tid; i < in_features; i += block_dim) {
        partial_sum += x_row[i] * w_row[i];
    }
    sdata[tid] = partial_sum;
    __syncthreads(); // Ensure all partial sums are in shared memory

    // Perform the reduction in shared memory
    for (unsigned int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // Synchronize after each reduction step
    }

    // Thread 0 of the block computes the final result and writes it to global memory
    if (tid == 0) {
        float result = sdata[0] + b[out_idx];
        if (apply_relu) {
            result = fmaxf(0.0f, result);
        }
        y[b_idx * out_features + out_idx] = result;
    }
}

// C++ launcher function that handles tensor checks and kernel invocation
torch::Tensor fused_linear_launcher(torch::Tensor x, torch::Tensor w, torch::Tensor b, bool apply_relu) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor X must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "Weight tensor W must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "Bias tensor B must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be a float32 tensor");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32, "Weight must be a float32 tensor");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "Bias must be a float32 tensor");
    TORCH_CHECK(x.dim() == 2, "Input tensor X must be 2D");
    TORCH_CHECK(w.dim() == 2, "Weight tensor W must be 2D");
    TORCH_CHECK(b.dim() == 1, "Bias tensor B must be 1D");

    // Ensure tensors are contiguous
    x = x.contiguous();
    w = w.contiguous();
    b = b.contiguous();

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int out_features = w.size(0);

    // Dimension checks
    TORCH_CHECK(in_features == w.size(1), "Input and weight feature dimensions must match");
    TORCH_CHECK(out_features == b.size(0), "Weight and bias output dimensions must match");

    // Create the output tensor
    auto y = torch::empty({batch_size, out_features}, x.options());

    // CUDA launch configuration
    const int block_threads = 256;
    const dim3 threads(block_threads);
    const dim3 blocks(batch_size, out_features);
    const size_t smem_size = block_threads * sizeof(float);

    // Launch the generic kernel
    fused_linear_generic_kernel<<<blocks, threads, smem_size>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        apply_relu
    );

    // Check for any asynchronous CUDA errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}

// C++ wrapper for fused linear + relu
torch::Tensor fused_linear_relu_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    return fused_linear_launcher(x, w, b, true);
}

// C++ wrapper for fused linear (matmul + bias)
torch::Tensor fused_linear_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    return fused_linear_launcher(x, w, b, false);
}
"""

fused_linear_cpp_source = """
torch::Tensor fused_linear_relu_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b);
torch::Tensor fused_linear_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_linear_ops",
    cpp_sources=fused_linear_cpp_source,
    cuda_sources=fused_linear_source,
    functions=["fused_linear_relu_forward", "fused_linear_forward"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        Initializes the model by creating standard nn.Linear layers to hold the parameters.
        The forward pass will use custom fused kernels instead of the default nn.Linear forward method.
        """
        super(ModelNew, self).__init__()
        
        self.hidden_layers = nn.ModuleList()
        current_input_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_layer_sizes:
            self.hidden_layers.append(nn.Linear(current_input_size, hidden_size))
            current_input_size = hidden_size
        
        # Create the final output layer
        self.output_layer = nn.Linear(current_input_size, output_size)
    
    def forward(self, x):
        """
        Performs the forward pass using custom fused kernels for each layer.
        This replaces the sequence of (Linear -> ReLU) with a single kernel call.
        """
        # Process through hidden layers using the fused linear + ReLU kernel
        for layer in self.hidden_layers:
            x = fused_ops.fused_linear_relu_forward(x, layer.weight, layer.bias)
        
        # Process through the final layer using the fused linear kernel (no ReLU)
        x = fused_ops.fused_linear_forward(x, self.output_layer.weight, self.output_layer.bias)
        
        return x

# Test code configuration
batch_size = 1
input_size = 1000
hidden_layer_sizes = [2000, 2000]
output_size = 10

def get_inputs():
    return [torch.randn(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]