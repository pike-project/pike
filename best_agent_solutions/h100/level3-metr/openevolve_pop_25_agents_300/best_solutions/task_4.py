# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- Fused BiasAdd + ReLU + MaxPool2D Kernel ---
# This is the most aggressive fusion for the convolutional blocks. It combines three
# operations: bias addition, ReLU activation, and max pooling. This saves significant
# memory bandwidth by eliminating two intermediate tensors (post-bias, post-relu).
# Each thread computes one output pixel from a 2x2 input window.
fused_bias_relu_maxpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat> // For FLT_MAX

__global__ void bias_relu_maxpool2d_kernel(const float* __restrict__ input, const float* __restrict__ bias, float* __restrict__ output,
                                           int num_outputs, int C, int H_in, int W_in, int H_out, int W_out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_outputs) {
        // De-linearize index to get (n, c, h_out, w_out) coordinates
        int w_out = index % W_out;
        int h_out = (index / W_out) % H_out;
        int c = (index / (W_out * H_out)) % C;
        int n = index / (W_out * H_out * C);

        // Calculate the top-left corner of the 2x2 pooling window
        int h_start = h_out * 2;
        int w_start = w_out * 2;

        float max_val = -FLT_MAX;
        const float bias_val = bias[c];

        // Pointer to the start of the current feature map.
        const float* input_ptr = input + (n * C + c) * H_in * W_in;

        // Unrolled loop over the 2x2 pooling window for performance
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int h_in = h_start + i;
                int w_in = w_start + j;
                // Fuse bias add, relu, and max in one step
                float val = fmaxf(0.0f, input_ptr[h_in * W_in + w_in] + bias_val);
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        output[index] = max_val;
    }
}

torch::Tensor bias_relu_maxpool2d_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");

    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    // Hardcoded for kernel_size=2, stride=2 pooling, as in LeNet
    int H_out = H_in / 2;
    int W_out = W_in / 2;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    int num_outputs = N * C * H_out * W_out;
    const int threads_per_block = 256;
    const int num_blocks = (num_outputs + threads_per_block - 1) / threads_per_block;

    bias_relu_maxpool2d_kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        num_outputs, C, H_in, W_in, H_out, W_out
    );
    return output;
}
"""

# --- Fused Linear + ReLU Kernel (Matrix-Vector specialization) ---
# This kernel is highly optimized for batch_size=1, where the operation becomes
# a matrix-vector product. Each CUDA block computes one element of the output vector.
# Threads within the block cooperate on the dot product, using shared memory for
# an efficient parallel reduction.
fused_linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For fmaxf

__global__ void linear_relu_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ y, int C_in, int C_out) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int j = blockIdx.x; // Output feature index
    unsigned int block_dim = blockDim.x;

    if (j >= C_out) return;

    float partial_sum = 0.0f;
    for (int k = tid; k < C_in; k += block_dim) {
        partial_sum += w[j * C_in + k] * x[k];
    }
    sdata[tid] = partial_sum;
    __syncthreads();

    // Reduction in shared memory. Requires block_dim to be a power of 2.
    for (unsigned int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of the block adds bias, applies ReLU, and writes the final result.
    if (tid == 0) {
        y[j] = fmaxf(sdata[0] + b[j], 0.0f);
    }
}

torch::Tensor linear_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "Input x must be of shape [1, C_in]");
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda(), "All tensors must be on CUDA");

    const int C_in = x.size(1);
    const int C_out = w.size(0);
    auto y = torch::empty({1, C_out}, x.options());

    const int threads_per_block = 256; // Must be a power of 2 for reduction
    const int num_blocks = C_out;
    const int shared_mem_size = threads_per_block * sizeof(float);

    linear_relu_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(),
        y.data_ptr<float>(), C_in, C_out
    );
    return y;
}
"""

cpp_sources = """
torch::Tensor bias_relu_maxpool2d_cuda(torch::Tensor input, torch::Tensor bias);
torch::Tensor linear_relu_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
"""

cuda_sources = fused_bias_relu_maxpool_source + fused_linear_relu_source

# Use a global variable to cache the compiled module, avoiding recompilation.
_fused_ops_module = None

def get_fused_ops():
    global _fused_ops_module
    if _fused_ops_module is None:
        _fused_ops_module = load_inline(
            name="fused_lenet_ops_v4_aggressive_fixed", # New name to avoid cache conflicts
            cpp_sources=cpp_sources,
            cuda_sources=cuda_sources,
            functions=["bias_relu_maxpool2d_cuda", "linear_relu_cuda"],
            verbose=False,
        )
    return _fused_ops_module

class Model(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture with aggressive fusion of (Bias+ReLU+MaxPool) and (Linear+ReLU).
        """
        super(Model, self).__init__()
        
        self.fused_ops = get_fused_ops()
        
        # Original layers are kept for their weights and biases.
        # The convolutions themselves are still performed by cuDNN via PyTorch,
        # as it is highly optimized.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        """
        Forward pass using custom fused operators.
        """
        # --- Block 1 ---
        # Perform convolution without bias, then apply the fused (bias_add + relu + maxpool) kernel.
        # This matches the baseline's 'valid' padding behavior.
        # Input 32x32 -> conv(5x5) -> 28x28 -> fused_op -> 14x14
        conv1_out = F.conv2d(x, self.conv1.weight, bias=None, stride=1)
        x = self.fused_ops.bias_relu_maxpool2d_cuda(conv1_out, self.conv1.bias)
        
        # --- Block 2 ---
        # Repeat for the second convolutional block.
        # Input 14x14 -> conv(5x5) -> 10x10 -> fused_op -> 5x5
        conv2_out = F.conv2d(x, self.conv2.weight, bias=None, stride=1)
        x = self.fused_ops.bias_relu_maxpool2d_cuda(conv2_out, self.conv2.bias)
        
        # --- Fully Connected Layers ---
        # Flatten the output for the fully connected layers.
        x = x.view(1, -1) # Use 1 for batch size, -1 to auto-infer size
        
        # Fused linear + relu for fc1
        x = self.fused_ops.linear_relu_cuda(x, self.fc1.weight, self.fc1.bias)
        
        # Fused linear + relu for fc2
        x = self.fused_ops.linear_relu_cuda(x, self.fc2.weight, self.fc2.bias)
        
        # Final fully connected layer (no fusion as there's no subsequent ReLU)
        x = self.fc3(x)
        
        return x

# Boilerplate for evaluation
batch_size = 1
num_classes = 10

def get_inputs():
    # Input tensor must be on the CUDA device for custom kernels.
    return [torch.randn(batch_size, 1, 32, 32).cuda()]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END