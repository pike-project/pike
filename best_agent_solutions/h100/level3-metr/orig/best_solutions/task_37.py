import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# --- Combined CUDA & C++ Source ---
# By combining all C++ and CUDA code into a single source string compiled by NVCC,
# we prevent redefinition errors that can occur when torch's JIT compiler
# handles separate .cpp and .cu files.

# Optimization v3:
# 1. Fused Dropout: The main LSTM cell kernel now optionally performs dropout.
#    When dropout is enabled for a layer, a dropout mask for the entire sequence
#    is pre-generated on the host. A view of this mask for the current time step
#    is passed to the kernel. The kernel then fuses the dropout application
#    (element-wise multiplication by the scaled mask) with the calculation of h_next.
# 2. Reduced Kernel Launches: This fusion eliminates a separate, large kernel launch
#    for `torch::dropout` for each layer (except the last), significantly reducing
#    kernel launch overhead during training.
# 3. Unified Kernel: A single kernel handles both the dropout and no-dropout cases
#    by checking if the passed mask pointer is null, avoiding code duplication.
# 4. Retains v2 Optimizations: This builds upon the direct output write, strided
#    I/O, and vectorized FP16 math from the previous version.


combined_source = """
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

// --- FP16 Math Helpers ---
__device__ __forceinline__ __half hsigmoid(__half x) {
    const __half one = __float2half(1.0f);
    return __hdiv(one, __hadd(one, hexp(__hneg(x))));
}

__device__ __forceinline__ __half2 hsigmoid2(__half2 x) {
    __half lo = hsigmoid(__low2half(x));
    __half hi = hsigmoid(__high2half(x));
    return __halves2half2(lo, hi);
}

__device__ __forceinline__ __half2 htanh2(__half2 x) {
    __half lo = htanh(__low2half(x));
    __half hi = htanh(__high2half(x));
    return __halves2half2(lo, hi);
}

// --- Fused LSTM Cell Kernel (FP16, Vectorized, Strided I/O, Fused Dropout) ---
__global__ void lstm_cell_fused_fp16_kernel(
    const __half* __restrict__ h_gates,
    const __half* __restrict__ i_gates_ptr,
    const int64_t i_gates_stride_b,
    const __half* __restrict__ c_prev,
    __half* __restrict__ h_next_ptr,
    const int64_t h_next_stride_b,
    __half* __restrict__ c_next,
    const __half* __restrict__ dropout_mask, // Contiguous (B,H) mask for current time step. Can be nullptr.
    const int hidden_size) {

    const int H = hidden_size;
    const int H_div_2 = H / 2;
    const int batch_idx = blockIdx.x;

    const __half2* h_gates_h2 = reinterpret_cast<const __half2*>(h_gates);
    const __half2* c_prev_h2 = reinterpret_cast<const __half2*>(c_prev);
    __half2* c_next_h2 = reinterpret_cast<__half2*>(c_next);

    const __half* i_gates_for_batch = i_gates_ptr + batch_idx * i_gates_stride_b;
    const __half2* i_gates_for_batch_h2 = reinterpret_cast<const __half2*>(i_gates_for_batch);

    __half* h_next_for_batch = h_next_ptr + batch_idx * h_next_stride_b;
    __half2* h_next_for_batch_h2 = reinterpret_cast<__half2*>(h_next_for_batch);

    const __half2* dropout_mask_h2 = reinterpret_cast<const __half2*>(dropout_mask);

    for (int i = threadIdx.x; i < H_div_2; i += blockDim.x) {
        const int h_gates_base_idx = batch_idx * 4 * H_div_2;
        const __half2 i_h = h_gates_h2[h_gates_base_idx + i];
        const __half2 f_h = h_gates_h2[h_gates_base_idx + i + H_div_2];
        const __half2 g_h = h_gates_h2[h_gates_base_idx + i + 2*H_div_2];
        const __half2 o_h = h_gates_h2[h_gates_base_idx + i + 3*H_div_2];

        const __half2 i_i = i_gates_for_batch_h2[i];
        const __half2 f_i = i_gates_for_batch_h2[i + H_div_2];
        const __half2 g_i = i_gates_for_batch_h2[i + 2*H_div_2];
        const __half2 o_i = i_gates_for_batch_h2[i + 3*H_div_2];

        const __half2 i_act = hsigmoid2(__hadd2(i_h, i_i));
        const __half2 f_act = hsigmoid2(__hadd2(f_h, f_i));
        const __half2 g_act = htanh2(__hadd2(g_h, g_i));
        const __half2 o_act = hsigmoid2(__hadd2(o_h, o_i));

        const int state_base_idx = batch_idx * H_div_2;
        const __half2 c_prev_val = c_prev_h2[state_base_idx + i];

        const __half2 c_next_val = __hadd2(__hmul2(f_act, c_prev_val), __hmul2(i_act, g_act));
        c_next_h2[state_base_idx + i] = c_next_val;

        __half2 h_next_val = __hmul2(o_act, htanh2(c_next_val));

        // *** OPTIMIZATION: Fused Dropout Application ***
        if (dropout_mask != nullptr) {
            // Mask is pre-scaled by 1/(1-p) on the host.
            // It's a contiguous (B, H) tensor for the current time step.
            const __half2 mask_val = dropout_mask_h2[batch_idx * H_div_2 + i];
            h_next_val = __hmul2(h_next_val, mask_val);
        }

        h_next_for_batch_h2[i] = h_next_val;
    }
}

void launch_lstm_cell_kernel_fp16(
    const torch::Tensor& h_gates,
    const torch::Tensor& i_gates,
    const torch::Tensor& c_prev,
    torch::Tensor& h_next,
    torch::Tensor& c_next,
    const torch::Tensor& dropout_mask // Pass a defined tensor for dropout, or undefined for none.
) {
    const int batch_size = h_gates.size(0);
    const int hidden_size = c_prev.size(1);
    TORCH_CHECK(hidden_size % 2 == 0, "Hidden size must be divisible by 2 for __half2 vectorization.");

    const int threads_per_block = 256;
    const dim3 threads(threads_per_block);
    const dim3 blocks(batch_size);

    const int64_t i_gates_stride_b = i_gates.stride(0);
    const int64_t h_next_stride_b = h_next.stride(0);

    // If dropout_mask is not a defined tensor, its data_ptr() is nullptr.
    const __half* mask_ptr = dropout_mask.defined() ?
        reinterpret_cast<const __half*>(dropout_mask.data_ptr<at::Half>()) :
        nullptr;

    lstm_cell_fused_fp16_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(h_gates.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(i_gates.data_ptr<at::Half>()),
        i_gates_stride_b,
        reinterpret_cast<const __half*>(c_prev.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(h_next.data_ptr<at::Half>()),
        h_next_stride_b,
        reinterpret_cast<__half*>(c_next.data_ptr<at::Half>()),
        mask_ptr,
        hidden_size
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// --- C++ Main Logic ---

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_HALF(x) TORCH_CHECK(x.scalar_type() == torch::kHalf, #x " must be a Half tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_HALF(x);

std::vector<torch::Tensor> lstm_forward_fp16(
    torch::Tensor input,
    const torch::Tensor& h_init,
    const torch::Tensor& c_init,
    const std::vector<torch::Tensor>& w_ih,
    const std::vector<torch::Tensor>& w_hh,
    const std::vector<torch::Tensor>& b_ih,
    const std::vector<torch::Tensor>& b_hh,
    double dropout_p,
    bool is_training
) {
    CHECK_INPUT(input);
    CHECK_INPUT(h_init);
    CHECK_INPUT(c_init);

    const auto num_layers = w_ih.size();
    auto h_states_init = h_init.unbind(0);
    auto c_states_init = c_init.unbind(0);

    auto current_x = input.contiguous();

    std::vector<torch::Tensor> final_h_states;
    final_h_states.reserve(num_layers);
    std::vector<torch::Tensor> final_c_states;
    final_c_states.reserve(num_layers);

    for (size_t i = 0; i < num_layers; ++i) {
        CHECK_CUDA(w_ih[i]); CHECK_HALF(w_ih[i]);
        const auto batch_size = current_x.size(0);
        const auto seq_len = current_x.size(1);
        const auto hidden_size = h_states_init[i].size(1);
        const auto options = current_x.options();

        auto precomputed_igates = torch::addmm(b_ih[i], current_x.flatten(0, 1), w_ih[i].t()).view({batch_size, seq_len, -1});

        auto output_sequence = torch::empty({batch_size, seq_len, hidden_size}, options);
        auto h_gates = torch::empty({batch_size, 4 * hidden_size}, options);

        auto h_curr = h_states_init[i];
        auto c_curr = c_states_init[i].clone();
        auto c_next = torch::empty_like(c_curr);

        // *** OPTIMIZATION: Pre-generate dropout mask for the whole sequence ***
        torch::Tensor dropout_mask; // An undefined tensor
        bool const apply_dropout = (dropout_p > 0.0 && is_training && i < num_layers - 1);
        if (apply_dropout) {
            const float scale = 1.0f / (1.0f - dropout_p);
            // Create mask for the output sequence (B, S, H) and apply scaling factor
            dropout_mask = (at::bernoulli(torch::full_like(output_sequence, 1.0 - dropout_p)) * scale);
        }

        for (int64_t t = 0; t < seq_len; ++t) {
            auto i_gates_t = precomputed_igates.select(1, t);
            auto h_next_t = output_sequence.select(1, t);
            
            torch::Tensor mask_t; // Undefined tensor
            if (apply_dropout) {
                mask_t = dropout_mask.select(1, t);
            }

            torch::addmm_out(h_gates, b_hh[i], h_curr, w_hh[i].t());
            
            // Launch kernel, which now handles dropout internally if mask_t is defined.
            launch_lstm_cell_kernel_fp16(h_gates, i_gates_t, c_curr, h_next_t, c_next, mask_t);

            h_curr = h_next_t;
            std::swap(c_curr, c_next);
        }

        final_h_states.push_back(h_curr.contiguous());
        final_c_states.push_back(c_curr);
        
        // output_sequence now contains the (potentially dropped-out) outputs
        current_x = output_sequence;
    }

    auto final_h = torch::stack(final_h_states, 0);
    auto final_c = torch::stack(final_c_states, 0);

    return {current_x, final_h, final_c};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_fp16", &lstm_forward_fp16, "Hyper-Optimized FP16 Multi-Layer LSTM Forward v3 (CUDA, Fused Dropout)");
}
"""

# JIT compilation
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
    lstm_cuda_module = load_inline(
        name='lstm_cuda_fp16_fused_dropout_v4', # Changed name to avoid cache issues
        cpp_sources='',
        cuda_sources=combined_source,
        verbose=False,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++17']
    )
else:
    lstm_cuda_module = None
    if not torch.cuda.is_available():
        print("Warning: CUDA not available.")
    else:
        print(f"Warning: Current GPU (compute capability {torch.cuda.get_device_capability()}) does not support FP16 operations efficiently. Custom LSTM disabled.")


class CustomLSTMImpl(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2 for __half2 vectorization."
        
        self.w_ih = nn.ParameterList()
        self.w_hh = nn.ParameterList()
        self.b_ih = nn.ParameterList()
        self.b_hh = nn.ParameterList()

        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.w_ih.append(nn.Parameter(torch.empty(4 * hidden_size, layer_input_size, dtype=torch.half)))
            self.w_hh.append(nn.Parameter(torch.empty(4 * hidden_size, hidden_size, dtype=torch.half)))
            self.b_ih.append(nn.Parameter(torch.empty(4 * hidden_size, dtype=torch.half)))
            self.b_hh.append(nn.Parameter(torch.empty(4 * hidden_size, dtype=torch.half)))

    def forward(self, x, states):
        h0, c0 = states
        outputs = lstm_cuda_module.forward_fp16(
            x, h0, c0,
            list(self.w_ih), list(self.w_hh),
            list(self.b_ih), list(self.b_hh),
            self.dropout, self.training
        )
        return outputs[0], (outputs[1], outputs[2])

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        
        use_custom_lstm = lstm_cuda_module is not None
        if use_custom_lstm:
            print("Using custom high-performance FP16 CUDA LSTM v3 (Fused Dropout).")
            self.lstm = CustomLSTMImpl(input_size, hidden_size, num_layers, dropout)
            
            # Initialize custom weights from a standard PyTorch LSTM to ensure correctness
            pytorch_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
            with torch.no_grad():
                for i in range(num_layers):
                    self.lstm.w_ih[i].data.copy_(getattr(pytorch_lstm, f'weight_ih_l{i}').half())
                    self.lstm.w_hh[i].data.copy_(getattr(pytorch_lstm, f'weight_hh_l{i}').half())
                    self.lstm.b_ih[i].data.copy_(getattr(pytorch_lstm, f'bias_ih_l{i}').half())
                    self.lstm.b_hh[i].data.copy_(getattr(pytorch_lstm, f'bias_hh_l{i}').half())
        else:
             print("Warning: Custom CUDA LSTM not available or supported. Falling back to nn.LSTM.")
             self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)

        self.fc = nn.Linear(hidden_size, output_size)
        self.use_custom = use_custom_lstm
    
    def forward(self, x, h0, c0):
        if self.use_custom:
            x_in = x.half()
            h0_in = h0.half()
            c0_in = c0.half()
        else:
            x_in, h0_in, c0_in = x, h0, c0

        out, state = self.lstm(x_in, (h0_in, c0_in))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :].float())
        
        return state[1].float()

# --- Model Configuration ---
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FP32 inputs are cast to FP16 inside the custom module's forward pass
    return [
        torch.randn(batch_size, sequence_length, input_size, device=device),
        torch.randn(num_layers, batch_size, hidden_size, device=device),
        torch.randn(num_layers, batch_size, hidden_size, device=device)
    ]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]