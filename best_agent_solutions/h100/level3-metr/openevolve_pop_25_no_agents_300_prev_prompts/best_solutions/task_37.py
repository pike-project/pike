# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for a fused slice and linear layer operation.
# This kernel uses a parallel reduction strategy with several optimizations:
# 1. float4 vectorization for high memory bandwidth.
# 2. A grid-stride loop to allow flexible thread block sizes and improve occupancy.
# 3. A hybrid reduction strategy: fast shared memory reduction followed by an
#    even faster warp-level reduction using shuffle instructions to minimize synchronization.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// This kernel computes the fused slice and fully-connected layer using a parallel reduction.
// Each thread block computes a single element in the output tensor.
__global__ void reduction_slice_fc_kernel_vec4_shuffle(
    const float* __restrict__ input,      // Input tensor from LSTM: shape (B, S, H)
    const float* __restrict__ weight,     // Weight matrix of the linear layer: shape (O, H)
    const float* __restrict__ bias,       // Bias vector of the linear layer: shape (O)
    float* __restrict__ output,           // Output tensor: shape (B, O)
    int B, int S, int H, int O) {

    // Each block computes one element of the output matrix (b, o).
    const int o = blockIdx.x;
    const int b = blockIdx.y;

    extern __shared__ float s_partials[];

    const int thread_id = threadIdx.x;
    const int block_size = blockDim.x; // Block size is a launch parameter (e.g., 128)

    // Pointers to the start of the relevant data rows, cast to float4 for vectorized loads.
    const float4* input_slice_ptr = reinterpret_cast<const float4*>(input + b * S * H + (S - 1) * H);
    const float4* weight_row_ptr = reinterpret_cast<const float4*>(weight + o * H);

    // Step 1: Each thread computes a partial sum using a grid-stride loop.
    // This makes the kernel flexible to block size and improves instruction-level parallelism.
    float partial_sum = 0.0f;
    for (int k_vec = thread_id; k_vec < (H / 4); k_vec += block_size) {
        float4 in_vec = input_slice_ptr[k_vec];
        float4 wt_vec = weight_row_ptr[k_vec];
        partial_sum += in_vec.x * wt_vec.x + in_vec.y * wt_vec.y + in_vec.z * wt_vec.z + in_vec.w * wt_vec.w;
    }
    s_partials[thread_id] = partial_sum;
    __syncthreads();

    // Step 2: Reduce in shared memory down to a single warp's worth of data (32 values).
    // This reduces the number of expensive __syncthreads() calls.
    for (int stride = block_size / 2; stride >= 32; stride >>= 1) {
        if (thread_id < stride) {
            s_partials[thread_id] += s_partials[thread_id + stride];
        }
        __syncthreads();
    }

    // Step 3: The first warp (threads 0-31) performs the final, fast reduction using warp shuffle.
    if (thread_id < 32) {
        float final_sum = s_partials[thread_id];

        // Warp-level reduction using __shfl_down_sync. This is faster than using shared memory.
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            final_sum += __shfl_down_sync(0xffffffff, final_sum, offset);
        }
        
        // Lane 0 of the warp holds the final sum and writes it to global memory.
        if (thread_id == 0) {
            output[b * O + o] = final_sum + bias[o];
        }
    }
}

// C++ wrapper function to launch the CUDA kernel
torch::Tensor slice_and_fc_cuda(
    torch::Tensor lstm_out,
    torch::Tensor weight,
    torch::Tensor bias) {

    TORCH_CHECK(lstm_out.is_cuda(), "lstm_out must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    
    const auto B = lstm_out.size(0);
    const auto S = lstm_out.size(1);
    const auto H = lstm_out.size(2);
    const auto O = weight.size(0);

    TORCH_CHECK(H % 4 == 0, "H must be divisible by 4 for the vec4 kernel");
    
    auto output = torch::empty({B, O}, lstm_out.options());

    // Configure kernel launch parameters.
    // A block size of 128 is a good default to improve latency hiding on the GPU.
    const int block_size = 128;
    const dim3 threads(block_size);
    // Grid of O*B blocks provides high parallelism for this problem's dimensions.
    const dim3 blocks(O, B);

    // Shared memory size must match the block size.
    size_t shared_mem_size = block_size * sizeof(float);

    reduction_slice_fc_kernel_vec4_shuffle<<<blocks, threads, shared_mem_size>>>(
        lstm_out.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, S, H, O
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel failed with error: ", cudaGetErrorString(err));
    }

    return output;
}
"""

cpp_source = "torch::Tensor slice_and_fc_cuda(torch::Tensor lstm_out, torch::Tensor weight, torch::Tensor bias);"

# JIT compile the CUDA kernel
# Using a unique name to avoid caching issues between different solutions.
custom_ops = load_inline(
    name="fused_slice_graph_op_v2_shuffle",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["slice_and_fc_cuda"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # CUDA Graph related attributes
        self.graph = None
        self.static_inputs = None
        self.static_output_c = None

    def forward(self, x, h0, c0):
        # On the first run, capture the model execution in a CUDA graph.
        if self.graph is None:
            # Warmup runs are recommended before capturing.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    out_warmup, state_warmup = self.lstm(x, (h0, c0))
                    # The custom op performs the slice internally
                    _ = custom_ops.slice_and_fc_cuda(out_warmup, self.fc.weight, self.fc.bias)
            torch.cuda.current_stream().wait_stream(s)

            # Create static tensors for inputs. The graph will be bound to these tensors.
            self.static_inputs = [x.clone(), h0.clone(), c0.clone()]
            
            # Capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                out_g, (h_g, c_g) = self.lstm(self.static_inputs[0], (self.static_inputs[1], self.static_inputs[2]))
                # The fused kernel is captured in the graph. It takes the full LSTM output.
                _ = custom_ops.slice_and_fc_cuda(out_g, self.fc.weight, self.fc.bias)
                # Save the output tensor from within the graph to retrieve results after replay.
                self.static_output_c = c_g

        # Copy the current input data to the static tensors.
        self.static_inputs[0].copy_(x)
        self.static_inputs[1].copy_(h0)
        self.static_inputs[2].copy_(c0)
        
        # Replay the graph. This is much faster than executing the operations individually.
        self.graph.replay()
        
        # Return a clone of the output tensor to prevent user modification of graph memory.
        return self.static_output_c.clone()

# Test code
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    # Ensure inputs are on the correct CUDA device
    return [
        torch.randn(batch_size, sequence_length, input_size).cuda(),
        torch.randn((num_layers, batch_size, hidden_size)).cuda(),
        torch.randn((num_layers, batch_size, hidden_size)).cuda()
    ]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
# EVOLVE-BLOCK-END