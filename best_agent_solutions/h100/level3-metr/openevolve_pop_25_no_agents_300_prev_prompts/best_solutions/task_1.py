# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# This solution combines the best features from all top-performing programs to
# push the performance further.
# 1. CUDA Graphs: Preserved from the original program, this is the most critical
#    optimization for eliminating CPU overhead from kernel launches.
# 2. float4 Vectorization: Preserved from the original program to maximize memory
#    bandwidth, which is a key bottleneck in this GEMV operation.
# 3. Warp-Shuffle Reduction: Adopted from inspiration programs, this replaces the
#    slower shared memory reduction. It uses a two-stage process: an ultra-fast
#    intra-warp reduction using shuffle instructions (no shared memory), followed
#    by a parallel reduction of warp results by the first warp. This minimizes
#    shared memory usage and synchronization overhead.
# 4. Increased Block Size: The block size is increased from 256 to 512, which can
#    improve GPU occupancy and help hide memory latency, a common practice for
#    reduction-heavy kernels.
# 5. Compiler Optimizations: Added `-O3` and `--use_fast_math` flags to instruct
#    the compiler to perform aggressive optimizations.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For fmaxf

// Templated Fused GEMV (for batch_size=1) + Bias Add + optional ReLU Kernel
// This version combines float4 vectorization with a two-stage warp-shuffle reduction.
template<bool apply_relu>
__global__ void fused_gemv_kernel_float4_ws(const float* __restrict__ x, 
                                            const float* __restrict__ W, 
                                            const float* __restrict__ b, 
                                            float* __restrict__ y, 
                                            const int in_features, 
                                            const int out_features) {
    // Each block calculates one output element y[i]
    const int i = blockIdx.x; // Corresponds to the output feature index
    if (i >= out_features) return;

    const int tid = threadIdx.x;
    
    float partial_sum = 0.0f;

    // Cast raw float pointers to float4 pointers for vectorized loads.
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* W_row4 = reinterpret_cast<const float4*>(W + i * in_features);
    const int in_features4 = in_features / 4;

    // Each thread computes a part of the dot product using float4.
    for (int k = tid; k < in_features4; k += blockDim.x) {
        const float4 x_vec = x4[k];
        const float4 w_vec = W_row4[k];
        partial_sum += x_vec.x * w_vec.x + x_vec.y * w_vec.y + x_vec.z * w_vec.z + x_vec.w * w_vec.w;
    }
    
    // --- Stage 1: Intra-Warp Reduction ---
    // Reduce partial sums within each warp using fast shuffle instructions.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    // --- Stage 2: Inter-Warp Reduction ---
    extern __shared__ float sdata[];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Lane 0 of each warp writes its reduced sum to shared memory.
    if (lane_id == 0) {
        sdata[warp_id] = partial_sum;
    }
    __syncthreads();

    // The first warp (warp_id == 0) reduces the results from all other warps.
    if (warp_id == 0) {
        const int num_warps = blockDim.x / 32;
        float final_sum = (lane_id < num_warps) ? sdata[lane_id] : 0.0f;
        
        // Final reduction within the first warp using shuffles.
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_sum += __shfl_down_sync(0xffffffff, final_sum, offset);
        }
        
        // Thread 0 writes the final result after adding bias and applying ReLU.
        if (lane_id == 0) {
            final_sum += b[i];
            if (apply_relu) {
                y[i] = fmaxf(0.0f, final_sum);
            } else {
                y[i] = final_sum;
            }
        }
    }
}

// Common C++ launcher logic, templated to reduce code duplication
template<bool apply_relu>
inline torch::Tensor fused_gemv_launcher(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias)
{
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "Input must be a [1, C] tensor for this optimized kernel");
    
    const int in_features = x.size(1);
    const int out_features = weight.size(0);
    TORCH_CHECK(in_features % 4 == 0, "in_features must be divisible by 4 for float4 kernel");

    auto output = torch::empty({1, out_features}, x.options());

    // A larger block size can improve occupancy and hide memory latency.
    const int block_size = 512;
    const int num_blocks = out_features;
    const int num_warps = block_size / 32;
    // Shared memory is only needed to store one float per warp.
    const size_t shmem_size = num_warps * sizeof(float);

    fused_gemv_kernel_float4_ws<apply_relu><<<num_blocks, block_size, shmem_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        in_features,
        out_features
    );
    
    return output;
}

// C++ wrappers for Python, explicitly instantiating the templates
torch::Tensor fused_linear_relu_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    return fused_gemv_launcher<true>(x, weight, bias);
}

torch::Tensor fused_linear_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    return fused_gemv_launcher<false>(x, weight, bias);
}
"""

cpp_source = """
torch::Tensor fused_linear_relu_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
torch::Tensor fused_linear_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias);
"""

# JIT compile the inline CUDA code. A unique name prevents caching issues.
cuda_module = load_inline(
    name="fused_gemv_module_graph_float4_ws",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_linear_relu_forward", "fused_linear_forward"],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

# A base class for our fused layers to avoid code duplication
class FusedLinearBase(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Mimic nn.Linear's default initialization for numerical consistency.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

class FusedLinearReLU(FusedLinearBase):
    def forward(self, x):
        return cuda_module.fused_linear_relu_forward(x, self.weight, self.bias)

class FusedLinear(FusedLinearBase):
    def forward(self, x):
        return cuda_module.fused_linear_forward(x, self.weight, self.bias)

class Model(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(Model, self).__init__()
        
        self.network = nn.Sequential(
            FusedLinearReLU(input_size, layer_sizes[0]),
            FusedLinearReLU(layer_sizes[0], layer_sizes[1]),
            FusedLinear(layer_sizes[1], output_size)
        )
        
        # Attributes for CUDA Graph optimization
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, x):
        # The first time forward is called, we capture the model execution in a CUDA graph.
        # This reduces kernel launch overhead on subsequent calls.
        if self.graph is None:
            # Create a static input tensor with the same properties as the real input.
            self.static_input = x.clone()
            
            # Warmup: Run the model a few times to ensure all CUDA kernels are initialized
            # before we start capturing the graph.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.static_output = self.network(self.static_input)
            torch.cuda.current_stream().wait_stream(s)
            
            # Capture the graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self.network(self.static_input)

        # For all calls, copy the input data to our static tensor
        self.static_input.copy_(x)
        
        # Replay the captured graph. This is much faster than launching the kernels individually.
        self.graph.replay()
        
        # Return the static output tensor. Its data has been updated by the graph replay.
        return self.static_output

# Test code
batch_size = 1
input_size = 1000
layer_sizes = [400, 800]
output_size = 500

def get_inputs():
    # The custom kernel requires input tensors to be on the GPU
    return [torch.randn(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]
# EVOLVE-BLOCK-END