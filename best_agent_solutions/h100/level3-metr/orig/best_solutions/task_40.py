import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------- CUDA / C++ Code -----------------

cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>
#include <string>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

// Device functions for activations on half2 vectors
__device__ __forceinline__ half2 sigmoid_h2(half2 x) {
    float f_low = __half2float(__low2half(x));
    float f_high = __half2float(__high2half(x));
    f_low = 1.0f / (1.0f + expf(-f_low));
    f_high = 1.0f / (1.0f + expf(-f_high));
    return __floats2half2_rn(f_low, f_high);
}

__device__ __forceinline__ half2 tanh_h2(half2 x) {
    float f_low = __half2float(__low2half(x));
    float f_high = __half2float(__high2half(x));
    return __floats2half2_rn(tanhf(f_low), tanhf(f_high));
}

// "Mega-Kernel" that processes an entire layer's sequence in one launch.
// This kernel eliminates the C++ loop over sequence length, avoiding
// significant launch overhead and intermediate data copies. It performs
// the h-h matrix multiplication inside the kernel with coalesced memory access.
// Grid: (batch_size), Block: (hidden_size / 2)
__global__ void gru_layer_forward_kernel(
    const half2* __restrict__ x_gates_all_timesteps, // Precomputed (X*W_ih + b_ih). Shape: (seq, batch, 3*h/2)
    const half2* __restrict__ h_init,                // Initial hidden state for this layer. Shape: (batch, h/2)
    const half2* __restrict__ W_hh_T,                // TRANSPOSED hidden-hidden weight. Shape: (h, 3*h/2)
    const half2* __restrict__ b_hh,                  // Hidden-hidden bias. Shape: (3*h/2)
    half2* __restrict__ h_n_out,                     // Final hidden state output. Shape: (batch, h/2)
    half2* __restrict__ layer_output,                // Full layer output (for stacking). Shape: (seq, batch, h/2)
    const int seq_len,
    const int batch_size,
    const int hidden_size,
    const bool store_full_output
) {
    const int hidden_size_div_2 = hidden_size / 2;

    // One block per batch item
    const int batch_idx = blockIdx.x;
    
    // One thread per hidden_size/2 element
    const int hid_idx = threadIdx.x;

    // Shared memory to hold the current hidden state h_t for this batch item.
    extern __shared__ half2 h_s[];

    // --- Initialization ---
    // Load initial hidden state from global to shared memory.
    h_s[hid_idx] = h_init[batch_idx * hidden_size_div_2 + hid_idx];
    __syncthreads();

    const int W_hh_T_row_stride = 3 * hidden_size_div_2;

    // --- Main Loop over Time Steps ---
    for (int t = 0; t < seq_len; ++t) {
        // --- In-kernel GEMV: h_gates = h_{t-1} @ W_hh.T + b_hh ---
        // h_{t-1} is in h_s.
        // My thread 'hid_idx' computes the 'hid_idx'-th half2 element of the r, z, and n gates.
        // This is done by iterating over the input dimension (h) and accumulating the results.
        
        half2 r_h_gate_acc = __floats2half2_rn(0.0f, 0.0f);
        half2 z_h_gate_acc = __floats2half2_rn(0.0f, 0.0f);
        half2 n_h_gate_acc = __floats2half2_rn(0.0f, 0.0f);

        for (int k_h2 = 0; k_h2 < hidden_size_div_2; ++k_h2) {
            half2 h_prev_val = h_s[k_h2]; // Read from shared mem

            half h_prev_low  = __low2half(h_prev_val);
            half h_prev_high = __high2half(h_prev_val);

            // Row in W_hh_T corresponding to input h_prev_low
            const int row_idx_low = 2 * k_h2;
            const half2* W_row_low_ptr = W_hh_T + row_idx_low * W_hh_T_row_stride;
            
            // Row in W_hh_T corresponding to input h_prev_high
            const int row_idx_high = 2 * k_h2 + 1;
            const half2* W_row_high_ptr = W_hh_T + row_idx_high * W_hh_T_row_stride;

            // FMA for the low part of h_prev. Broadcast low part to both halves of a half2.
            half2 h_prev_low_bcast = __halves2half2(h_prev_low, h_prev_low);
            r_h_gate_acc = __hfma2(h_prev_low_bcast, W_row_low_ptr[hid_idx], r_h_gate_acc);
            z_h_gate_acc = __hfma2(h_prev_low_bcast, W_row_low_ptr[hidden_size_div_2 + hid_idx], z_h_gate_acc);
            n_h_gate_acc = __hfma2(h_prev_low_bcast, W_row_low_ptr[2 * hidden_size_div_2 + hid_idx], n_h_gate_acc);
            
            // FMA for the high part of h_prev. Broadcast high part to both halves of a half2.
            half2 h_prev_high_bcast = __halves2half2(h_prev_high, h_prev_high);
            r_h_gate_acc = __hfma2(h_prev_high_bcast, W_row_high_ptr[hid_idx], r_h_gate_acc);
            z_h_gate_acc = __hfma2(h_prev_high_bcast, W_row_high_ptr[hidden_size_div_2 + hid_idx], z_h_gate_acc);
            n_h_gate_acc = __hfma2(h_prev_high_bcast, W_row_high_ptr[2 * hidden_size_div_2 + hid_idx], n_h_gate_acc);
        }

        // Add bias
        half2 h_gates_r = __hadd2(r_h_gate_acc, b_hh[hid_idx]);
        half2 h_gates_z = __hadd2(z_h_gate_acc, b_hh[hidden_size_div_2 + hid_idx]);
        half2 h_gates_n = __hadd2(n_h_gate_acc, b_hh[2 * hidden_size_div_2 + hid_idx]);

        // ---- Fetch x_gates ----
        const int x_gates_offset = (t * batch_size + batch_idx) * 3 * hidden_size_div_2;
        const half2 x_gates_r = x_gates_all_timesteps[x_gates_offset + hid_idx];
        const half2 x_gates_z = x_gates_all_timesteps[x_gates_offset + hidden_size_div_2 + hid_idx];
        const half2 x_gates_n = x_gates_all_timesteps[x_gates_offset + 2 * hidden_size_div_2 + hid_idx];

        // ---- Fused Gate Logic ----
        half2 r_t = sigmoid_h2(__hadd2(x_gates_r, h_gates_r));
        half2 z_t = sigmoid_h2(__hadd2(x_gates_z, h_gates_z));
        half2 n_t = tanh_h2(__hadd2(x_gates_n, __hmul2(r_t, h_gates_n)));

        // ---- Compute new h_t and store to shared memory ----
        half2 h_t_minus_1 = h_s[hid_idx];
        const half2 one = __float2half2_rn(1.0f);
        half2 h_t = __hadd2(__hmul2(__hsub2(one, z_t), n_t), __hmul2(z_t, h_t_minus_1));
        h_s[hid_idx] = h_t;

        // ---- Store full output if needed for stacked layers ----
        if (store_full_output) {
            const int layer_output_offset = (t * batch_size + batch_idx) * hidden_size_div_2;
            layer_output[layer_output_offset + hid_idx] = h_t;
        }
        
        __syncthreads(); // Wait for all threads to update h_s before next t
    }

    // --- Finalization ---
    // After loop, h_s contains the final hidden state h_n. Write it to global memory.
    h_n_out[batch_idx * hidden_size_div_2 + hid_idx] = h_s[hid_idx];
}


// C++ function that orchestrates the forward pass for all layers.
torch::Tensor gru_forward_cuda(
    torch::Tensor input,
    torch::Tensor h_0,
    const std::vector<torch::Tensor>& weights_ih,
    const std::vector<torch::Tensor>& weights_hh_t, // Expects pre-transposed weights
    const std::vector<torch::Tensor>& biases_ih,
    const std::vector<torch::Tensor>& biases_hh
) {
    const auto seq_len = input.size(0);
    const auto batch_size = input.size(1);
    const auto num_layers = weights_ih.size();
    const auto hidden_size = h_0.size(2);
    const int hidden_size_div_2 = hidden_size / 2;

    TORCH_CHECK(hidden_size % 2 == 0, "hidden_size must be a multiple of 2 for half2 vectorization.");
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on a CUDA device.");

    const auto stream = at::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);

    auto options_fp16_cuda = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);

    auto current_x_fp16 = input.to(torch::kFloat16);
    auto h_n_all_layers = h_0.to(torch::kFloat16);

    // Buffer for intermediate layer outputs (full sequence)
    auto layer_output_fp16 = torch::empty({seq_len, batch_size, hidden_size}, options_fp16_cuda);
    
    // --- Kernel Launch Config ---
    dim3 grid(batch_size);
    dim3 block(hidden_size_div_2);
    size_t shared_mem_size = hidden_size_div_2 * sizeof(half2);
    TORCH_CHECK(block.x <= 1024, "Hidden size too large for a single block");

    for (int layer = 0; layer < num_layers; ++layer) {
        // Pre-compute input-gate matrix product for all timesteps
        auto x_gates_all_timesteps = torch::addmm(
            biases_ih[layer],
            current_x_fp16.view({-1, current_x_fp16.size(2)}),
            weights_ih[layer].t()
        ).view({seq_len, batch_size, -1});
        
        // This slice is contiguous, safe to use its data_ptr
        auto h_n_layer = h_n_all_layers.select(0, layer);
        
        bool store_full_output = (layer < num_layers - 1);
        
        gru_layer_forward_kernel<<<grid, block, shared_mem_size, stream.stream()>>>(
            (const half2*)x_gates_all_timesteps.data_ptr(),
            (const half2*)h_n_layer.data_ptr(), // h_init for the layer
            (const half2*)weights_hh_t[layer].data_ptr(),
            (const half2*)biases_hh[layer].data_ptr(),
            (half2*)h_n_layer.data_ptr(), // Output for final h_n of the layer
            (half2*)layer_output_fp16.data_ptr(),
            seq_len,
            batch_size,
            hidden_size,
            store_full_output
        );
        
        // The output of the current layer becomes the input for the next
        if (store_full_output) {
            current_x_fp16 = layer_output_fp16;
        }
    }

    return h_n_all_layers.to(torch::kFloat32);
}
"""

cpp_source = """
#include <torch/extension.h>
#include <vector>

torch::Tensor gru_forward_cuda(
    torch::Tensor input, torch::Tensor h_0,
    const std::vector<torch::Tensor>& weights_ih,
    const std::vector<torch::Tensor>& weights_hh_t,
    const std::vector<torch::Tensor>& biases_ih,
    const std::vector<torch::Tensor>& biases_hh);
"""

# Jit-compile the C++/CUDA code
custom_gru_module = load_inline(
    name="mega_kernel_gru_v4", # Changed name to avoid cache issues
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["gru_forward_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_70"], # sm_70 for half precision ops
)


# ----------------- New Model Definition -----------------

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()

        if batch_first:
            raise NotImplementedError("batch_first=True is not supported by this custom GRU kernel.")
        if hidden_size % 128 != 0:
            # Block size is hidden_size/2, must be a multiple of warp size (32)
            # A multiple of 128 for hidden_size (64 for hidden_size/2) is safer and common
            raise ValueError("hidden_size must be a multiple of 128 for this kernel.")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        # Create a temporary standard GRU to easily initialize weights
        _gru = nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=False)

        self.weights_ih = nn.ParameterList()
        self.weights_hh_t = nn.ParameterList() # Store transposed weights
        if bias:
            self.biases_ih = nn.ParameterList()
            self.biases_hh = nn.ParameterList()

        for layer in range(num_layers):
            # Extract, convert to FP16, and store weights
            weight_ih = getattr(_gru, f'weight_ih_l{layer}')
            self.weights_ih.append(nn.Parameter(weight_ih.contiguous().half()))
            
            weight_hh = getattr(_gru, f'weight_hh_l{layer}')
            # Pre-transpose W_hh for coalesced memory access in the kernel
            self.weights_hh_t.append(nn.Parameter(weight_hh.t().contiguous().half()))
            
            if bias:
                self.biases_ih.append(nn.Parameter(getattr(_gru, f'bias_ih_l{layer}').half()))
                self.biases_hh.append(nn.Parameter(getattr(_gru, f'bias_hh_l{layer}').half()))
        
        self.custom_gru_module = custom_gru_module

    def forward(self, x, h0):
        # The C++ function expects lists of tensors
        w_ih_list = [p for p in self.weights_ih]
        w_hh_t_list = [p for p in self.weights_hh_t]

        if self.bias:
            b_ih_list = [p for p in self.biases_ih]
            b_hh_list = [p for p in self.biases_hh]
        else:
            # Create zero biases if the layer is configured without them
            zero_bias_ih = torch.zeros(3 * self.hidden_size, dtype=torch.float16, device=x.device)
            zero_bias_hh = torch.zeros(3 * self.hidden_size, dtype=torch.float16, device=x.device)
            b_ih_list = [zero_bias_ih] * self.num_layers
            b_hh_list = [zero_bias_hh] * self.num_layers

        h_n = self.custom_gru_module.gru_forward_cuda(
            x.contiguous(),
            h0.contiguous(),
            w_ih_list,
            w_hh_t_list,
            b_ih_list,
            b_hh_list
        )
        
        return h_n

# ----------------- I/O Definition -----------------
input_size = 128
hidden_size = 256
num_layers = 6
batch_size = 10
seq_len = 512

def get_inputs():
    x = torch.randn(seq_len, batch_size, input_size).cuda()
    h0 = torch.randn(num_layers, batch_size, hidden_size).cuda()
    return [x, h0]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]