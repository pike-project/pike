import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# CUDA and C++ source code for the optimized custom GRU forward pass
# This version uses a hybrid approach: cuBLAS for all GEMMs and a custom
# fused kernel for the element-wise operations.
gru_source = """
#include <torch/extension.h>
#include <vector>
#include <stdexcept>

// CUDA specific includes
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// Error checking macros
#define CUDA_CHECK(call)                                                         \\
  do {                                                                           \\
    cudaError_t err = call;                                                      \\
    if (err != cudaSuccess) {                                                    \\
      fprintf(stderr, "CUDA error at %s %d: %s\\n", __FILE__, __LINE__,          \\
              cudaGetErrorString(err));                                          \\
      throw std::runtime_error("CUDA error");                                    \\
    }                                                                            \\
  } while (0)

#define CUBLAS_CHECK(call)                                                       \\
  do {                                                                           \\
    cublasStatus_t status = call;                                                \\
    if (status != CUBLAS_STATUS_SUCCESS) {                                       \\
      fprintf(stderr, "cuBLAS error at %s %d\\n", __FILE__, __LINE__);           \\
      throw std::runtime_error("cuBLAS error");                                  \\
    }                                                                            \\
  } while (0)

// Device-side activation functions using fast math intrinsics
__device__ inline float sigmoidf(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// tanhf is usually a hardware instruction, but __expf can be faster with --use_fast_math
__device__ inline float tanhf_custom(float x) {
    // This is equivalent to tanhf, but can be faster with fast-math.
    // computes (e^2x - 1) / (e^2x + 1)
    float exp2x = __expf(2.0f * x);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}


/**
 * @brief A fused, vectorized kernel for the element-wise portion of a GRU time step.
 *
 * This kernel is launched for each time step after the GEMMs for `x @ W_ih` and `h @ W_hh`
 * have been computed by cuBLAS. It performs the remaining operations:
 * - Adds biases to the gate values.
 * - Applies sigmoid and tanh activations.
 * - Computes the new hidden state `h_new`.
 *
 * Key Optimizations:
 * 1. Fused Operations: All element-wise ops for a time step are in one kernel.
 * 2. Vectorization: All memory I/O and computations use `__half2` and `float2`,
 *    doubling memory bandwidth and arithmetic throughput.
 * 3. Simplified Logic: No complex loops or shared memory needed, as all inputs are
 *    read once from global memory.
 *
 * Threading Model:
 * - Grid: (batch_size, 1, 1). Each block processes one sequence in the batch.
 * - Block: (threads_per_block, 1, 1). Threads cooperate on the `hidden_size` dimension for a sequence.
 */
__global__ void gru_elementwise_fused_vectorized_kernel(
    const __half2* gates_x_t, // Pre-calculated input-gate GEMM for this time step. Shape: (batch_size, 3 * hidden_size / 2).
    const __half2* gates_h,   // Pre-calculated hidden-gate GEMM. Shape: (batch_size, 3 * hidden_size / 2).
    const __half2* bias_ih,   // Input-hidden bias. Shape: (3 * hidden_size / 2).
    const __half2* bias_hh,   // Hidden-hidden bias. Shape: (3 * hidden_size / 2).
    __half2* h_prev,          // Previous hidden state (IN/OUT parameter). Shape: (batch_size, hidden_size / 2).
    __half2* output_t,        // Output for the current time step. Shape: (batch_size, hidden_size / 2).
    const int hidden_size_half) {

    const int batch_idx = blockIdx.x;
    const int h_idx_pair_start = threadIdx.x;
    const int stride = blockDim.x;

    // Calculate base pointers for the current batch item to simplify indexing
    const __half2* gates_x_t_base = gates_x_t + batch_idx * 3 * hidden_size_half;
    const __half2* gates_h_base   = gates_h + batch_idx * 3 * hidden_size_half;
    __half2* h_prev_base          = h_prev + batch_idx * hidden_size_half;
    __half2* output_t_base        = output_t + batch_idx * hidden_size_half;

    // Loop over the hidden dimension, where each thread handles one `__half2` element at a time
    for (int i = h_idx_pair_start; i < hidden_size_half; i += stride) {
        // Find corresponding indices for reset (r), update (z), and new (n) gates
        const int r_idx = i;
        const int z_idx = i + hidden_size_half;
        const int n_idx = i + 2 * hidden_size_half;

        // --- Load all necessary data in vectorized form ---
        float2 gates_x_r = __half22float2(gates_x_t_base[r_idx]);
        float2 gates_x_z = __half22float2(gates_x_t_base[z_idx]);
        float2 gates_x_n = __half22float2(gates_x_t_base[n_idx]);

        float2 gates_h_r = __half22float2(gates_h_base[r_idx]);
        float2 gates_h_z = __half22float2(gates_h_base[z_idx]);
        float2 gates_h_n = __half22float2(gates_h_base[n_idx]);

        // FIX: 'float2 bias_ih_r f' was a typo. Corrected to 'float2 bias_ih_r_f'.
        float2 bias_ih_r_f = __half22float2(bias_ih[r_idx]);
        float2 bias_hh_r_f = __half22float2(bias_hh[r_idx]);
        float2 bias_ih_z_f = __half22float2(bias_ih[z_idx]);
        float2 bias_hh_z_f = __half22float2(bias_hh[z_idx]);
        float2 bias_ih_n_f = __half22float2(bias_ih[n_idx]);
        float2 bias_hh_n_f = __half22float2(bias_hh[n_idx]);

        // --- Perform computations in FP32 for precision ---
        float r1 = sigmoidf(gates_x_r.x + gates_h_r.x + bias_ih_r_f.x + bias_hh_r_f.x);
        float r2 = sigmoidf(gates_x_r.y + gates_h_r.y + bias_ih_r_f.y + bias_hh_r_f.y);

        float z1 = sigmoidf(gates_x_z.x + gates_h_z.x + bias_ih_z_f.x + bias_hh_z_f.x);
        float z2 = sigmoidf(gates_x_z.y + gates_h_z.y + bias_ih_z_f.y + bias_hh_z_f.y);

        float n1 = tanhf_custom(gates_x_n.x + bias_ih_n_f.x + r1 * (gates_h_n.x + bias_hh_n_f.x));
        float n2 = tanhf_custom(gates_x_n.y + bias_ih_n_f.y + r2 * (gates_h_n.y + bias_hh_n_f.y));

        float2 h_prev_f = __half22float2(h_prev_base[i]);

        float h_new1 = (1.0f - z1) * n1 + z1 * h_prev_f.x;
        float h_new2 = (1.0f - z2) * n2 + z2 * h_prev_f.y;

        __half2 h_new_h2 = __float22half2_rn(make_float2(h_new1, h_new2));

        // --- Write results back to global memory ---
        output_t_base[i] = h_new_h2;
        h_prev_base[i] = h_new_h2; // Update h_prev in-place for the next time step
    }
}


// C++ host function to orchestrate the GRU forward pass
torch::Tensor gru_forward_cuda(
    torch::Tensor x,                     // (seq_len, batch_size, input_size), FP32
    torch::Tensor h0,                    // (num_layers, batch_size, hidden_size), FP32
    std::vector<torch::Tensor> w_ih,     // List of (3*h, in_i), FP16
    std::vector<torch::Tensor> w_hh,     // List of (3*h, h), FP16
    std::vector<torch::Tensor> b_ih,     // List of (3*h), FP16
    std::vector<torch::Tensor> b_hh) {   // List of (3*h), FP16

    const int64_t seq_len = x.size(0);
    const int64_t batch_size = x.size(1);
    const int64_t num_layers = h0.size(0);
    const int64_t hidden_size = h0.size(2);
    const int64_t hidden_size_half = hidden_size / 2;

    TORCH_CHECK(hidden_size % 2 == 0, "Custom GRU kernel requires hidden_size to be a multiple of 2 for __half2 vectorization.");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    auto x_fp16 = x.to(torch::kHalf);
    auto h_states_fp16 = h0.clone().to(torch::kHalf);
    auto layer_input_fp16 = x_fp16;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int layer = 0; layer < num_layers; ++layer) {
        auto w_ih_l = w_ih[layer];
        auto w_hh_l = w_hh[layer];
        auto b_ih_l = b_ih[layer];
        auto b_hh_l = b_hh[layer];

        const int64_t current_input_size = layer_input_fp16.size(2);
        auto layer_input_reshaped = layer_input_fp16.view({-1, current_input_size});

        // OPTIMIZATION 1: Pre-compute the entire `X @ W_ih^T` GEMM.
        auto gates_x = torch::empty({seq_len * batch_size, 3 * hidden_size}, x_fp16.options());
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  /*n*/ 3 * hidden_size, /*m*/ seq_len * batch_size, /*k*/ current_input_size,
                                  &alpha,
                                  w_ih_l.data_ptr<at::Half>(), CUDA_R_16F, /*lda*/ current_input_size,
                                  layer_input_reshaped.data_ptr<at::Half>(), CUDA_R_16F, /*ldb*/ current_input_size,
                                  &beta,
                                  gates_x.data_ptr<at::Half>(), CUDA_R_16F, /*ldc*/ 3 * hidden_size,
                                  CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        
        auto h_prev = h_states_fp16.select(0, layer);
        auto layer_output_seq = torch::empty({seq_len, batch_size, hidden_size}, x_fp16.options());

        // Host-side loop over the time sequence
        for (int t = 0; t < seq_len; ++t) {
            // OPTIMIZATION 2: Compute `H_prev @ W_hh^T` with cuBLAS.
            auto gates_h = torch::empty({batch_size, 3 * hidden_size}, x_fp16.options());
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                      /*n*/ 3 * hidden_size, /*m*/ batch_size, /*k*/ hidden_size,
                                      &alpha,
                                      w_hh_l.data_ptr<at::Half>(), CUDA_R_16F, /*lda*/ hidden_size,
                                      h_prev.data_ptr<at::Half>(), CUDA_R_16F, /*ldb*/ hidden_size,
                                      &beta,
                                      gates_h.data_ptr<at::Half>(), CUDA_R_16F, /*ldc*/ 3 * hidden_size,
                                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            auto gates_x_t = gates_x.slice(0, t * batch_size, (t + 1) * batch_size);
            auto output_t = layer_output_seq.select(0, t);

            // OPTIMIZATION 3: Launch the fast, fused, vectorized element-wise kernel.
            dim3 blocks(batch_size);
            dim3 threads(256);
            gru_elementwise_fused_vectorized_kernel<<<blocks, threads>>>(
                reinterpret_cast<const __half2*>(gates_x_t.data_ptr<at::Half>()),
                reinterpret_cast<const __half2*>(gates_h.data_ptr<at::Half>()),
                reinterpret_cast<const __half2*>(b_ih_l.data_ptr<at::Half>()),
                reinterpret_cast<const __half2*>(b_hh_l.data_ptr<at::Half>()),
                reinterpret_cast<__half2*>(h_prev.data_ptr<at::Half>()),
                reinterpret_cast<__half2*>(output_t.data_ptr<at::Half>()),
                hidden_size_half);
        }
        CUDA_CHECK(cudaGetLastError());
        
        layer_input_fp16 = layer_output_seq;
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    return layer_input_fp16.to(torch::kFloat32);
}
"""

# C++ source for function binding
gru_cpp_source = """
#include <vector>
torch::Tensor gru_forward_cuda(
    torch::Tensor x,
    torch::Tensor h0,
    std::vector<torch::Tensor> w_ih,
    std::vector<torch::Tensor> w_hh,
    std::vector<torch::Tensor> b_ih,
    std::vector<torch::Tensor> b_hh);
"""

# JIT compile the custom CUDA kernel
custom_gru_module = load_inline(
    name="custom_gru_module_hybrid",
    cpp_sources=gru_cpp_source,
    cuda_sources=gru_source,
    functions=["gru_forward_cuda"],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75']
)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(ModelNew, self).__init__()
        if batch_first:
            raise NotImplementedError("batch_first=True is not supported by this custom kernel.")
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be a multiple of 2 for the vectorized CUDA kernel.")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        # Create parameters for each layer
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            # Weights are (out_features, in_features) which is (3 * hidden_size, layer_input_size)
            setattr(self, f'weight_ih_l{i}', nn.Parameter(torch.empty(3 * hidden_size, layer_input_size)))
            setattr(self, f'weight_hh_l{i}', nn.Parameter(torch.empty(3 * hidden_size, hidden_size)))
            if bias:
                setattr(self, f'bias_ih_l{i}', nn.Parameter(torch.empty(3 * hidden_size)))
                setattr(self, f'bias_hh_l{i}', nn.Parameter(torch.empty(3 * hidden_size)))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1:
                nn.init.uniform_(weight, -stdv, stdv)
            else: # biases
                nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, h0):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size)
        :param h0: The initial hidden state, shape (num_layers, batch_size, hidden_size)
        :return: output: The output features, shape (seq_len, batch_size, hidden_size)
        """
        # Collect parameters and convert to FP16 for the custom kernel.
        # This conversion is fast and done on the GPU.
        w_ih = [getattr(self, f'weight_ih_l{i}').half() for i in range(self.num_layers)]
        w_hh = [getattr(self, f'weight_hh_l{i}').half() for i in range(self.num_layers)]

        if self.bias:
            b_ih = [getattr(self, f'bias_ih_l{i}').half() for i in range(self.num_layers)]
            b_hh = [getattr(self, f'bias_hh_l{i}').half() for i in range(self.num_layers)]
        else:
            # Create zero biases if they are disabled
            b_ih = [torch.zeros(3 * self.hidden_size, device=x.device, dtype=torch.half) for _ in range(self.num_layers)]
            b_hh = [torch.zeros(3 * self.hidden_size, device=x.device, dtype=torch.half) for _ in range(self.num_layers)]
        
        output = custom_gru_module.gru_forward_cuda(x, h0, w_ih, w_hh, b_ih, b_hh)
        # The final hidden state is implicitly handled and updated inside the CUDA code
        # but only the output sequence is returned, matching the original model's forward signature.
        return output