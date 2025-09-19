# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused (cat + linear + tanh).
# This version evolves the top-performing kernel with two key refinements:
# 1. Swapped Grid Dimensions: The launch grid is changed from (B, H) to (H, B).
#    This makes adjacent blocks work on the same input sample, improving cache
#    locality for the x and hidden tensors.
# 2. Faster Intrinsic: Replaces `tanhf` with the `__tanhf` CUDA intrinsic,
#    which can be faster on compatible hardware.
rnn_cell_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <c10/cuda/CUDAException.h> // Header for C10_CUDA_KERNEL_LAUNCH_CHECK

// In-line device function for performing a reduction sum over a single warp
// using shuffle-down instructions, which are faster than shared memory.
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Helper device function for dot product of two float4 vectors.
__device__ __forceinline__ float dot(const float4& a, const float4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__global__ void fused_rnn_cell_kernel_optimized(
    const float* __restrict__ x, 
    const float* __restrict__ hidden, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ new_hidden,
    const int B, // Batch size
    const int I, // Input size
    const int H  // Hidden size
) {
    // Each block computes one element of the output new_hidden tensor.
    // Grid is (H, B) for better cache locality on x and hidden.
    const int row = blockIdx.y; // Batch index
    const int col = blockIdx.x; // Hidden size index

    float thread_sum = 0.0f;
    const int tid = threadIdx.x;

    // --- Vectorized dot product part 1: corresponding to x ---
    const float4* x_row_ptr4 = reinterpret_cast<const float4*>(x + row * I);
    const int combined_size = I + H;
    const float4* weight_row_ptr4 = reinterpret_cast<const float4*>(weight + col * combined_size);
    for (int k = tid; k < I / 4; k += blockDim.x) {
        thread_sum += dot(x_row_ptr4[k], weight_row_ptr4[k]);
    }

    // --- Vectorized dot product part 2: corresponding to hidden ---
    const float4* hidden_row_ptr4 = reinterpret_cast<const float4*>(hidden + row * H);
    const float4* weight_row_ptr_h4 = reinterpret_cast<const float4*>(weight + col * combined_size + I);
    for (int k = tid; k < H / 4; k += blockDim.x) {
        thread_sum += dot(hidden_row_ptr4[k], weight_row_ptr_h4[k]);
    }
    
    // --- Hybrid Warp-Shuffle + Shared Memory Reduction ---
    // 1. Each warp reduces its own partial sums.
    float warp_sum = warpReduceSum(thread_sum);

    // Shared memory to aggregate results from each warp.
    extern __shared__ float smem[];
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // 2. The first thread of each warp writes its result to shared memory.
    if (lane_id == 0) {
        smem[warp_id] = warp_sum;
    }

    __syncthreads();

    // 3. The first thread of the block performs the final, unrolled reduction.
    if (tid == 0) {
        // Unrolled sum for 4 warps (128 threads / 32 threads/warp)
        float final_sum = smem[0] + smem[1] + smem[2] + smem[3];
        final_sum += bias[col];
        // Use the faster __tanhf intrinsic
        new_hidden[row * H + col] = __tanhf(final_sum);
    }
}

torch::Tensor fused_cat_linear_tanh_optimized(
    torch::Tensor x, 
    torch::Tensor hidden, 
    torch::Tensor weight, 
    torch::Tensor bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(hidden.is_cuda(), "hidden must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    const int B = x.size(0);
    const int I = x.size(1);
    const int H = hidden.size(1);

    TORCH_CHECK(I % 4 == 0, "Input size (I) must be divisible by 4 for vectorization.");
    TORCH_CHECK(H % 4 == 0, "Hidden size (H) must be divisible by 4 for vectorization.");

    auto new_hidden = torch::empty_like(hidden);

    const int threads_per_block = 128;
    // Swapped grid dimensions for better cache locality
    const dim3 blocks(H, B);
    const dim3 threads(threads_per_block);
    // Shared memory size is one float per warp in the block.
    const size_t shared_mem_size = (threads_per_block / 32) * sizeof(float);

    fused_rnn_cell_kernel_optimized<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        hidden.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        new_hidden.data_ptr<float>(),
        B, I, H
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return new_hidden;
}
"""

rnn_cell_fused_cpp_source = (
    "torch::Tensor fused_cat_linear_tanh_optimized(torch::Tensor x, torch::Tensor hidden, torch::Tensor weight, torch::Tensor bias);"
)

# Compile the inline CUDA code
fused_rnn_op = load_inline(
    name="fused_rnn_op_optimized", # Unique name to avoid cache conflicts
    cpp_sources=rnn_cell_fused_cpp_source,
    cuda_sources=rnn_cell_fused_source,
    functions=["fused_cat_linear_tanh_optimized"],
    verbose=True,
)


class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model with a custom CUDA kernel.
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Using register_buffer is good practice for state that should be part of the model
        # but is not a trainable parameter. It ensures correct device placement.
        self.register_buffer('hidden', torch.randn((batch_size, hidden_size)))
        
        # Create a standard layer to get correctly initialized weights, then discard the layer itself.
        _i2h = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Register weights and biases as parameters for our custom operation.
        self.i2h_weight = nn.Parameter(_i2h.weight.clone())
        self.i2h_bias = nn.Parameter(_i2h.bias.clone())
        
        # The second linear layer remains a standard PyTorch module, as it's a GEMM
        # operation for which cuBLAS is highly optimized.
        self.h2o = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        """
        Forward pass of the Vanilla RNN using the custom fused kernel.
        """
        current_hidden = self.hidden
        if initial_hidden is not None:
             # The provided initial_hidden from get_inputs is used for the forward pass.
             current_hidden = initial_hidden
        current_hidden = current_hidden.to(x.device)
        
        # The fused kernel replaces torch.cat, the first nn.Linear, and nn.Tanh.
        new_hidden_state = fused_rnn_op.fused_cat_linear_tanh_optimized(
            x, current_hidden, self.i2h_weight, self.i2h_bias
        )
        
        # Update the model's persistent hidden state.
        self.hidden = new_hidden_state
        
        # The second linear layer is applied as usual.
        output = self.h2o(self.hidden)
        return output

batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256

def get_inputs():
    # Ensure inputs are on CUDA and contiguous for the custom kernel.
    # The second tensor is the initial_hidden state passed to the forward method.
    return [
        torch.randn(batch_size, input_size).cuda().contiguous(),
        torch.randn(batch_size, hidden_size).cuda().contiguous()
    ]

def get_init_inputs():
    return [input_size, hidden_size, output_size]

# EVOLVE-BLOCK-END