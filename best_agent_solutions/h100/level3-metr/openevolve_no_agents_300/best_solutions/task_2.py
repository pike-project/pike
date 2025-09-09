# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# The CUDA kernel remains the same as it is already highly optimized with vectorized
# memory access and a hybrid shuffle-based reduction. The key change is in the C++
# wrapper to support CUDA graphs by accepting a pre-allocated output tensor.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__global__ void fused_linear_kernel_hybrid_reduction(const float* __restrict__ input,
                                                     const float* __restrict__ weight,
                                                     const float* __restrict__ bias,
                                                     float* __restrict__ output,
                                                     const int in_features,
                                                     const int out_features,
                                                     const bool apply_relu) {
    // Each CUDA block computes one element of the output vector.
    const int out_idx = blockIdx.x;
    if (out_idx >= out_features) {
        return;
    }

    // Vectorized memory access using float4.
    const float4* input_vec = reinterpret_cast<const float4*>(input);
    const float4* weight_row_vec = reinterpret_cast<const float4*>(weight + out_idx * in_features);

    float partial_sum = 0.0f;
    const int vec_in_features = in_features / 4;

    for (int i = threadIdx.x; i < vec_in_features; i += blockDim.x) {
        float4 in_val = input_vec[i];
        float4 wt_val = weight_row_vec[i];
        partial_sum += in_val.x * wt_val.x + in_val.y * wt_val.y + in_val.z * wt_val.z + in_val.w * wt_val.w;
    }

    // --- Hybrid Reduction Strategy ---
    // Step 1: Intra-warp reduction using shuffle instructions.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, offset);
    }

    // Step 2: Write warp-level sums to shared memory.
    extern __shared__ float sdata[];
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
        sdata[warp_id] = partial_sum;
    }
    __syncthreads();

    // Step 3: Final reduction in shared memory by the first thread.
    if (threadIdx.x == 0) {
        float total_sum = 0.0f;
        int num_warps = blockDim.x / WARP_SIZE;
        for (int i = 0; i < num_warps; i++) {
            total_sum += sdata[i];
        }
        
        float final_sum = total_sum + bias[out_idx];
        if (apply_relu) {
            output[out_idx] = fmaxf(0.0f, final_sum);
        } else {
            output[out_idx] = final_sum;
        }
    }
}

// MODIFIED C++ wrapper: Takes a pre-allocated output tensor. This is essential for CUDA Graphs.
void fused_linear_cuda_out(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, bool apply_relu) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "Output must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2 && input.size(0) == 1, "Input must be of shape [1, in_features]");
    TORCH_CHECK(input.size(1) % 4 == 0, "Input features must be a multiple of 4");
    
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    TORCH_CHECK(output.dim() == 2 && output.size(0) == 1 && output.size(1) == out_features, "Output tensor has incorrect shape");

    const dim3 blocks(out_features);
    const dim3 threads(BLOCK_SIZE);
    const int shared_mem_size = (BLOCK_SIZE / WARP_SIZE) * sizeof(float);

    fused_linear_kernel_hybrid_reduction<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(), // Writes to the provided output tensor
        in_features,
        out_features,
        apply_relu
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
"""

# Update C++ source to reflect the new wrapper signature
cpp_source = "void fused_linear_cuda_out(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor output, bool apply_relu);"

# JIT compile, using a new name to ensure recompilation
fused_op = load_inline(
    name="fused_linear_graph_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_linear_cuda_out"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(Model, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_layer_sizes[0])
        self.fc2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.fc3 = nn.Linear(hidden_layer_sizes[1], output_size)

        # Buffers for CUDA Graph static memory. register_buffer ensures they move with the model (.cuda(), etc.)
        self.register_buffer('static_input', torch.randn(1, input_size, device='cuda'))
        self.register_buffer('static_hidden1', torch.empty(1, hidden_layer_sizes[0], device='cuda'))
        self.register_buffer('static_hidden2', torch.empty(1, hidden_layer_sizes[1], device='cuda'))
        self.register_buffer('static_output', torch.empty(1, output_size, device='cuda'))
        
        self.graph = None

    def forward(self, x):
        # On the first run, capture the graph.
        if self.graph is None:
            # Warm up with a real input to initialize memory pools and capture valid operations.
            self.static_input.copy_(x)
            
            # Create a graph object
            self.graph = torch.cuda.CUDAGraph()
            
            # Begin capture. All operations within this context are recorded.
            with torch.cuda.graph(self.graph):
                # Layer 1: input -> hidden1
                fused_op.fused_linear_cuda_out(self.static_input, self.fc1.weight, self.fc1.bias, self.static_hidden1, True)
                
                # Layer 2: hidden1 -> hidden2
                fused_op.fused_linear_cuda_out(self.static_hidden1, self.fc2.weight, self.fc2.bias, self.static_hidden2, True)
                
                # Layer 3: hidden2 -> output
                fused_op.fused_linear_cuda_out(self.static_hidden2, self.fc3.weight, self.fc3.bias, self.static_output, False)
        
        # For all subsequent runs, copy the new input and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()
        
        # Return a clone of the output. The buffer itself is static and will be overwritten on the next call.
        return self.static_output.clone()

# Test code configuration
batch_size = 1
input_size = 1000
hidden_layer_sizes = [2000, 2000]
output_size = 10

def get_inputs():
    return [torch.randn(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
# EVOLVE-BLOCK-END