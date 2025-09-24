import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os
import math

# Define the custom CUDA kernel source for the recurrent part of a single LSTM layer.
# Version 6: Hybrid Approach
# - This version separates the non-recurrent (input-to-hidden) and recurrent (hidden-to-hidden) computations.
# - The large (input * weight_ih) GEMM is now performed outside the kernel using torch.matmul, leveraging cuBLAS for maximum efficiency.
# - This kernel is responsible only for the recurrent loop over the time sequence for a single layer.
# - It takes the pre-computed input gates and iteratively computes the hidden and cell states.
# - Core computation within the loop is a tiled GEMV for (h_t-1 * weight_hh) fused with the addition of pre-computed gates, biases, and element-wise activations.
# - Thread count is reduced to 512 per block to potentially improve SM occupancy and overall GPU utilization.

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// --- Tiling configuration for GEMV ---
constexpr int TILE_N_THREADS = 16;
constexpr int TILE_K_THREADS = 32;
constexpr int BLOCK_THREADS = TILE_N_THREADS * TILE_K_THREADS; // 512 threads

// --- Vectorized activation functions for half2 (unchanged from v5) ---
__device__ inline half2 sigmoidf_gpu_h2(half2 x) {
    float2 val_f2 = __half22float2(x);
    val_f2.x = 1.f / (1.f + expf(-val_f2.x));
    val_f2.y = 1.f / (1.f + expf(-val_f2.y));
    return __float22half2_rn(val_f2);
}

__device__ inline half2 tanhf_gpu_h2(half2 x) {
    float2 val_f2 = __half22float2(x);
    // Clamping to avoid precision issues with large inputs in tanhf
    val_f2.x = fmaxf(-20.0f, fminf(20.0f, val_f2.x));
    val_f2.y = fmaxf(-20.0f, fminf(20.0f, val_f2.y));
    val_f2.x = tanhf(val_f2.x);
    val_f2.y = tanhf(val_f2.y);
    return __float22half2_rn(val_f2);
}

// Optimized Tiled GEMV for 512 threads
__device__ void tiled_gemv_512_fp16(
    const half* __restrict__ vec, const half* __restrict__ mat, half* __restrict__ out, const int N, const int K
) {
    const int thread_id = threadIdx.x;
    const int tile_n_id = thread_id / TILE_K_THREADS; // 0..15
    const int tile_k_id = thread_id % TILE_K_THREADS; // 0..31

    __shared__ half s_vec_tile[TILE_K_THREADS];
    __shared__ half s_mat_tile[TILE_N_THREADS][TILE_K_THREADS];

    for (int j = tile_n_id; j < N; j += TILE_N_THREADS) {
        float accumulator = 0.0f;
        for (int k_base = 0; k_base < K; k_base += TILE_K_THREADS) {
            const int k = k_base + tile_k_id;
            if (k < K) {
                if (tile_n_id == 0) s_vec_tile[tile_k_id] = vec[k];
                s_mat_tile[tile_n_id][tile_k_id] = mat[j * K + k];
            } else {
                if (tile_n_id == 0) s_vec_tile[tile_k_id] = __float2half(0.0f);
                s_mat_tile[tile_n_id][tile_k_id] = __float2half(0.0f);
            }
            __syncthreads();

            #pragma unroll
            for (int k_tile = 0; k_tile < TILE_K_THREADS; ++k_tile) {
                accumulator += __half2float(s_mat_tile[tile_n_id][k_tile]) * __half2float(s_vec_tile[k_tile]);
            }
            __syncthreads();
        }
        out[j] = __float2half(accumulator);
    }
}

__global__ void lstm_recurrent_loop_kernel_v6(
    const half* __restrict__ gates_ih,
    const half* __restrict__ h_init,
    const half* __restrict__ c_init,
    const half* __restrict__ w_hh,
    const half* __restrict__ b_hh,
    half* __restrict__ h_out,
    half* __restrict__ h_n,
    half* __restrict__ c_n,
    const int seq_len, const int hidden_size
) {
    const int batch_idx = blockIdx.x;
    const int gate_size = 4 * hidden_size;
    const int hidden_size_div_2 = hidden_size / 2;

    // Shared memory for hidden state (h), cell state (c), and hh gate results
    extern __shared__ char s_mem_raw[];
    float* s_c = (float*)s_mem_raw;
    half* s_h = (half*)(s_mem_raw + hidden_size * sizeof(float));
    half* s_gates_hh = s_h + hidden_size;

    // Pointers for the current batch item
    const size_t batch_offset_gates = (size_t)batch_idx * seq_len * gate_size;
    const size_t batch_offset_hidden = (size_t)batch_idx * hidden_size;
    const size_t batch_offset_out = (size_t)batch_idx * seq_len * hidden_size;

    // Initialize h and c from global memory
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        s_h[i] = h_init[batch_offset_hidden + i];
        s_c[i] = __half2float(c_init[batch_offset_hidden + i]);
    }
    __syncthreads();

    // Loop over sequence length
    for (int t = 0; t < seq_len; ++t) {
        // Compute hidden-to-hidden gates: s_h @ w_hh^T
        tiled_gemv_512_fp16(s_h, w_hh, s_gates_hh, gate_size, hidden_size);
        __syncthreads();

        // --- Vectorized cell update ---
        half2* s_gates_hh_h2 = reinterpret_cast<half2*>(s_gates_hh);
        float2* s_c_f2 = reinterpret_cast<float2*>(s_c);
        half2* s_h_h2 = reinterpret_cast<half2*>(s_h);
        const half2* gates_ih_t_h2 = reinterpret_cast<const half2*>(gates_ih + batch_offset_gates + (size_t)t * gate_size);
        const half2* b_hh_h2 = reinterpret_cast<const half2*>(b_hh);
        half2* h_out_t_h2 = reinterpret_cast<half2*>(h_out + batch_offset_out + (size_t)t * hidden_size);

        for (int i = threadIdx.x; i < hidden_size_div_2; i += blockDim.x) {
            // Fuse the addition of (precomputed input gates) + (hh gates) + (hh bias)
            float2 gates_ih_f2 = __half22float2(gates_ih_t_h2[i]);
            float2 gates_hh_f2 = __half22float2(s_gates_hh_h2[i]);
            float2 bias_hh_f2  = __half22float2(b_hh_h2[i]);
            half2 i_gate_h2 = __float22half2_rn({gates_ih_f2.x + gates_hh_f2.x + bias_hh_f2.x, gates_ih_f2.y + gates_hh_f2.y + bias_hh_f2.y});

            gates_ih_f2 = __half22float2(gates_ih_t_h2[i + hidden_size_div_2]);
            gates_hh_f2 = __half22float2(s_gates_hh_h2[i + hidden_size_div_2]);
            bias_hh_f2  = __half22float2(b_hh_h2[i + hidden_size_div_2]);
            half2 f_gate_h2 = __float22half2_rn({gates_ih_f2.x + gates_hh_f2.x + bias_hh_f2.x, gates_ih_f2.y + gates_hh_f2.y + bias_hh_f2.y});

            gates_ih_f2 = __half22float2(gates_ih_t_h2[i + 2 * hidden_size_div_2]);
            gates_hh_f2 = __half22float2(s_gates_hh_h2[i + 2 * hidden_size_div_2]);
            bias_hh_f2  = __half22float2(b_hh_h2[i + 2 * hidden_size_div_2]);
            half2 g_gate_h2 = __float22half2_rn({gates_ih_f2.x + gates_hh_f2.x + bias_hh_f2.x, gates_ih_f2.y + gates_hh_f2.y + bias_hh_f2.y});
            
            gates_ih_f2 = __half22float2(gates_ih_t_h2[i + 3 * hidden_size_div_2]);
            gates_hh_f2 = __half22float2(s_gates_hh_h2[i + 3 * hidden_size_div_2]);
            bias_hh_f2  = __half22float2(b_hh_h2[i + 3 * hidden_size_div_2]);
            half2 o_gate_h2 = __float22half2_rn({gates_ih_f2.x + gates_hh_f2.x + bias_hh_f2.x, gates_ih_f2.y + gates_hh_f2.y + bias_hh_f2.y});

            // Apply activations and update states
            float2 i_gate_f2 = __half22float2(sigmoidf_gpu_h2(i_gate_h2));
            float2 f_gate_f2 = __half22float2(sigmoidf_gpu_h2(f_gate_h2));
            float2 g_gate_f2 = __half22float2(tanhf_gpu_h2(g_gate_h2));
            float2 o_gate_f2 = __half22float2(sigmoidf_gpu_h2(o_gate_h2));

            float2 s_c_val_f2 = s_c_f2[i];
            s_c_val_f2.x = f_gate_f2.x * s_c_val_f2.x + i_gate_f2.x * g_gate_f2.x;
            s_c_val_f2.y = f_gate_f2.y * s_c_val_f2.y + i_gate_f2.y * g_gate_f2.y;
            s_c_f2[i] = s_c_val_f2;

            float2 s_h_val_f2 = {o_gate_f2.x * tanhf(s_c_val_f2.x), o_gate_f2.y * tanhf(s_c_val_f2.y)};
            half2 s_h_val_h2 = __float22half2_rn(s_h_val_f2);
            s_h_h2[i] = s_h_val_h2;

            // Write output hidden state for this time step
            h_out_t_h2[i] = s_h_val_h2;
        }
        __syncthreads();
    }

    // Write final h_n and c_n to global memory
    half2* h_n_h2 = reinterpret_cast<half2*>(h_n);
    half2* c_n_h2 = reinterpret_cast<half2*>(c_n);
    float2* s_c_f2 = reinterpret_cast<float2*>(s_c);
    half2* s_h_h2 = reinterpret_cast<half2*>(s_h);
    for (int i = threadIdx.x; i < hidden_size_div_2; i += blockDim.x) {
        size_t offset = batch_idx * hidden_size_div_2 + i;
        h_n_h2[offset] = s_h_h2[i];
        c_n_h2[offset] = __float22half2_rn(s_c_f2[i]);
    }
}
"""

cpp_source = """
#include <vector>
#include <ATen/ATen.h>

void lstm_recurrent_loop_v6(
    torch::Tensor gates_ih, torch::Tensor h_init, torch::Tensor c_init,
    torch::Tensor w_hh, torch::Tensor b_hh,
    torch::Tensor h_out, torch::Tensor h_n, torch::Tensor c_n);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("recurrent_loop", &lstm_recurrent_loop_v6, "LSTM Recurrent Loop v6 (FP16 Hybrid CUDA)");
}

void lstm_recurrent_loop_v6(
    torch::Tensor gates_ih, torch::Tensor h_init, torch::Tensor c_init,
    torch::Tensor w_hh, torch::Tensor b_hh,
    torch::Tensor h_out, torch::Tensor h_n, torch::Tensor c_n
) {
    CHECK_INPUT(gates_ih); CHECK_INPUT(h_init); CHECK_INPUT(c_init);
    CHECK_INPUT(w_hh); CHECK_INPUT(b_hh);
    CHECK_INPUT(h_out); CHECK_INPUT(h_n); CHECK_INPUT(c_n);
    TORCH_CHECK(gates_ih.scalar_type() == at::kHalf, "Inputs must be half tensors");

    const int batch_size = gates_ih.size(0);
    const int seq_len = gates_ih.size(1);
    const int hidden_size = h_init.size(1);
    
    TORCH_CHECK(hidden_size % 2 == 0, "hidden_size must be even for vectorized kernel");

    const dim3 blocks(batch_size);
    const dim3 threads(BLOCK_THREADS);

    const size_t shared_mem_size = (hidden_size * sizeof(float)) + // s_c
                                   (hidden_size * sizeof(at::Half)) + // s_h
                                   (4 * hidden_size * sizeof(at::Half)); // s_gates_hh

    lstm_recurrent_loop_kernel_v6<<<blocks, threads, shared_mem_size>>>(
        gates_ih.data_ptr<at::Half>(), h_init.data_ptr<at::Half>(), c_init.data_ptr<at::Half>(),
        w_hh.data_ptr<at::Half>(), b_hh.data_ptr<at::Half>(),
        h_out.data_ptr<at::Half>(), h_n.data_ptr<at::Half>(), c_n.data_ptr<at::Half>(),
        seq_len, hidden_size
    );
    
    if (cudaGetLastError() != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed in lstm_recurrent_loop_v6");
    }
}
"""

try:
    build_dir = "/tmp/cuda_ext_build_lstm_fused_v6"
    os.makedirs(build_dir, exist_ok=True)
    fused_lstm_cuda_lib_v6 = load_inline(
        name="fused_lstm_cuda_lib_v6",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['recurrent_loop'],
        verbose=False,
        build_directory=build_dir,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_75", "-std=c++17"]
    )
    print("Successfully compiled custom FP16 Hybrid LSTM CUDA kernel.")
except Exception as e:
    print(f"Failed to compile custom CUDA kernel: {e}")
    fused_lstm_cuda_lib_v6 = None


class FusedLSTMv6(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.weight_ih = nn.ParameterList()
        self.weight_hh = nn.ParameterList()
        self.bias_ih = nn.ParameterList()
        self.bias_hh = nn.ParameterList()

        gate_size = 4 * hidden_size
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.weight_ih.append(nn.Parameter(torch.empty(gate_size, layer_input_size)))
            self.weight_hh.append(nn.Parameter(torch.empty(gate_size, hidden_size)))
            self.bias_ih.append(nn.Parameter(torch.empty(gate_size)))
            self.bias_hh.append(nn.Parameter(torch.empty(gate_size)))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def load_weights_from_original(self, lstm_module: nn.LSTM):
        with torch.no_grad():
            params = [p.data for p in lstm_module.parameters()]
            for l in range(self.num_layers):
                w_ih, w_hh, b_ih, b_hh = params[l*4 : l*4 + 4]
                self.weight_ih[l].copy_(w_ih)
                self.weight_hh[l].copy_(w_hh)
                self.bias_ih[l].copy_(b_ih)
                self.bias_hh[l].copy_(b_hh)
        print("Loaded weights from nn.LSTM into custom FusedLSTMv6.")

    def forward(self, x, states):
        h0, c0 = states
        # Unbind states for per-layer processing
        h_inits = h0.unbind(0)
        c_inits = c0.unbind(0)
        
        bs, sl = x.shape[0], x.shape[1]
        dtype = x.dtype
        device = x.device

        final_h_n_list, final_c_n_list = [], []
        layer_input = x
        
        for l in range(self.num_layers):
            layer_input_size = self.input_size if l == 0 else self.hidden_size
            
            # 1. Non-recurrent GEMM: Handled by torch.matmul (cuBLAS)
            gates_ih = torch.addmm(self.bias_ih[l], layer_input.reshape(-1, layer_input_size), self.weight_ih[l].t())
            gates_ih = gates_ih.view(bs, sl, -1)
            
            # Prepare tensors for the recurrent kernel
            h_init_l = h_inits[l].contiguous()
            c_init_l = c_inits[l].contiguous()
            h_out_l = torch.empty(bs, sl, self.hidden_size, device=device, dtype=dtype)
            h_n_l = torch.empty(bs, self.hidden_size, device=device, dtype=dtype)
            c_n_l = torch.empty(bs, self.hidden_size, device=device, dtype=dtype)

            # 2. Recurrent part: Handled by custom CUDA kernel
            fused_lstm_cuda_lib_v6.recurrent_loop(
                gates_ih.contiguous(), h_init_l, c_init_l,
                self.weight_hh[l].contiguous(), self.bias_hh[l].contiguous(),
                h_out_l, h_n_l, c_n_l
            )
            
            layer_input = h_out_l
            final_h_n_list.append(h_n_l)
            final_c_n_list.append(c_n_l)

        # Stack final states
        h_n_final = torch.stack(final_h_n_list, dim=0)
        c_n_final = torch.stack(final_c_n_list, dim=0)

        return layer_input, (h_n_final, c_n_final)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        use_custom_kernel = (fused_lstm_cuda_lib_v6 is not None and 
                             dropout == 0.0 and 
                             hidden_size % 2 == 0)
        
        if use_custom_kernel:
            # Use the new hybrid custom LSTM implementation
            self.lstm = FusedLSTMv6(input_size, hidden_size, num_layers)
        else:
            # Fallback conditions
            if fused_lstm_cuda_lib_v6 is None:
              print("Warning: Custom CUDA LSTM kernel failed to compile. Falling back to torch.nn.LSTM.")
            if dropout > 0.0:
              print("Warning: Dropout > 0 is not supported by custom kernel. Falling back to torch.nn.LSTM.")
            if hidden_size % 2 != 0:
              print("Warning: hidden_size must be even for vectorized kernel. Falling back to torch.nn.LSTM.")
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        
        self.use_custom_kernel = use_custom_kernel
        # The fc layer from the original model was dead code as its output was not returned.
        # It is removed for efficiency, matching the behavior of the original return statement.

    def forward(self, x, h0, c0):
        # The model's actual return is the final cell state `c_n`
        _, state = self.lstm(x, (h0, c0))
        return state[1]