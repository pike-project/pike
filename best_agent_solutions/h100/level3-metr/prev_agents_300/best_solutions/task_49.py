import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# This version builds upon v6 with two primary optimizations:
# 1. Pre-computation of Inner Decays: The original kernel computed expf(-cumsum(A))
#    for every element of B, resulting in L*N redundant exponential calculations
#    per chunk. This version pre-computes the L unique values of expf(-cumsum(A))
#    into a dedicated shared memory array (s_decay_inner). This reduces the number
#    of expensive expf calls from L*N (e.g., 1024) to just L (e.g., 64).
# 2. Manual Loop Unrolling: The main state reduction loop over the sequence length `L`
#    is manually unrolled by a factor of 2. This increases instruction-level
#    parallelism (ILP), allowing the GPU scheduler to better hide the latency of
#    memory access and arithmetic operations, leading to higher throughput.
mamba_fused_scan_cuda_source_v7 = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// __device__ function for warp-level inclusive scan (prefix sum).
// Uses __shfl_up_sync for efficient data exchange within a warp.
// Note: This implementation is correct for up to 32 elements.
__device__ __forceinline__ void warp_inclusive_scan(int lane_id, float& val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float other = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane_id >= offset) {
            val += other;
        }
    }
}


__global__ void mamba_fused_scan_kernel_v7(
    // Inputs
    const float* __restrict__ a_in,         // (B, H, C, L)
    const float* __restrict__ b_in,         // (B, H, C, L, N)
    const float* __restrict__ x_in,         // (B, H, C, L, P)
    const float* __restrict__ initial_states_in, // (B, H, P, N)
    // Output
    float* __restrict__ final_states_out, // (B, H, P, N)
    // Dimensions
    int B, int H, int C, int L, int N, int P)
{
    const int VEC_SIZE = 4;
    // Grid: (H, B), Block: (P, N/VEC_SIZE)
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;

    const int p = threadIdx.x;        // Corresponds to d_head dimension
    const int n_base = threadIdx.y;   // Corresponds to d_state dimension (vectorized)

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int TPB = blockDim.x * blockDim.y; // P * (N/VEC_SIZE)

    extern __shared__ float s_mem[];
    float* s_A = s_mem;                                 // Size: L
    float* s_B = s_A + L;                               // Size: L * N
    float* s_X = s_B + L * N;                           // Size: L * P
    float* s_decay_inner = s_X + L * P;                 // Size: L (New)

    const float* a = a_in + (batch_idx * H + head_idx) * C * L;
    const float* b = b_in + (batch_idx * H + head_idx) * C * L * N;
    const float* x = x_in + (batch_idx * H + head_idx) * C * L * P;

    // Load initial state vector into registers using float4
    const int state_offset = (batch_idx * H + head_idx) * P * N + p * N + n_base * VEC_SIZE;
    float4 h_vec = *((const float4*)&initial_states_in[state_offset]);

    for (int c = 0; c < C; ++c) {
        const int chunk_offset_L = c * L;
        const int chunk_offset_LN = c * L * N;
        const int chunk_offset_LP = c * L * P;

        // Step 1: Cooperatively load chunk data into shared memory.
        __syncthreads();
        for (int i = tid; i < L / 4; i += TPB) {
            ((float4*)s_A)[i] = ((const float4*)(a + chunk_offset_L))[i];
        }
        for (int i = tid; i < (L * N) / 4; i += TPB) {
            ((float4*)s_B)[i] = ((const float4*)(b + chunk_offset_LN))[i];
        }
        for (int i = tid; i < (L * P) / 4; i += TPB) {
            ((float4*)s_X)[i] = ((const float4*)(x + chunk_offset_LP))[i];
        }
        __syncthreads();

        // Step 2: Perform fast parallel inclusive scan for cumsum(A).
        // This scan is specialized for L=64 (2 warps).
        if (tid < L) {
            float val = s_A[tid];
            int lane_id = tid % 32;
            warp_inclusive_scan(lane_id, val);
            s_A[tid] = val;
        }
        __syncthreads();

        if (tid >= 32 && tid < L) {
            s_A[tid] += s_A[31]; // Block-level correction for the 2nd warp
        }
        __syncthreads();
        float* s_A_cumsum = s_A;

        // Step 2b (New): Pre-compute inner decay factors exp(-cumsum(A)) to avoid redundant expf calls.
        if (tid < L) {
            s_decay_inner[tid] = expf(-s_A_cumsum[tid]);
        }
        __syncthreads();

        // Step 2c (Optimized): Apply pre-computed decays to B.
        for (int i = tid; i < L * N; i += TPB) {
            int s = i / N;
            s_B[i] *= s_decay_inner[s];
        }
        __syncthreads();

        // Step 3 (Optimized): Vectorized GEMM for state reduction (X^T @ B) with 2x manual unrolling.
        float4 state_reduction = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int s = 0; s < L; s += 2) {
            // Iteration s
            const float x_val0 = s_X[s * P + p];
            const float4 b_vec0 = ((const float4*)s_B)[s * (N / VEC_SIZE) + n_base];
            state_reduction.x = fmaf(b_vec0.x, x_val0, state_reduction.x);
            state_reduction.y = fmaf(b_vec0.y, x_val0, state_reduction.y);
            state_reduction.z = fmaf(b_vec0.z, x_val0, state_reduction.z);
            state_reduction.w = fmaf(b_vec0.w, x_val0, state_reduction.w);

            // Iteration s+1
            const float x_val1 = s_X[(s + 1) * P + p];
            const float4 b_vec1 = ((const float4*)s_B)[(s + 1) * (N / VEC_SIZE) + n_base];
            state_reduction.x = fmaf(b_vec1.x, x_val1, state_reduction.x);
            state_reduction.y = fmaf(b_vec1.y, x_val1, state_reduction.y);
            state_reduction.z = fmaf(b_vec1.z, x_val1, state_reduction.z);
            state_reduction.w = fmaf(b_vec1.w, x_val1, state_reduction.w);
        }

        // Step 4: Apply the recurrence relation to update the hidden state vector.
        const float a_sum = s_A_cumsum[L - 1];
        const float decay_c = expf(a_sum);
        h_vec.x = decay_c * (h_vec.x + state_reduction.x);
        h_vec.y = decay_c * (h_vec.y + state_reduction.y);
        h_vec.z = decay_c * (h_vec.z + state_reduction.z);
        h_vec.w = decay_c * (h_vec.w + state_reduction.w);
    }

    // Write final state vector back to global memory
    *((float4*)&final_states_out[state_offset]) = h_vec;
}

void mamba_fused_scan_cuda_v7(
    torch::Tensor a, torch::Tensor b, torch::Tensor x, torch::Tensor initial_states,
    torch::Tensor final_states)
{
    const int B = a.size(0);
    const int H = a.size(1);
    const int C = a.size(2);
    const int L = a.size(3);
    const int N = b.size(4);
    const int P = x.size(4);
    const int VEC_SIZE = 4;

    dim3 grid_dim(H, B);
    dim3 block_dim(P, N / VEC_SIZE);

    const size_t shared_mem_size = (L + L*N + L*P + L) * sizeof(float);
    if (shared_mem_size > 49152) {
      throw std::runtime_error("Required shared memory exceeds the 48KB limit.");
    }

    mamba_fused_scan_kernel_v7<<<grid_dim, block_dim, shared_mem_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        x.data_ptr<float>(),
        initial_states.data_ptr<float>(),
        final_states.data_ptr<float>(),
        B, H, C, L, N, P
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
"""

mamba_fused_scan_cpp_source_v7 = "void mamba_fused_scan_cuda_v7(torch::Tensor a, torch::Tensor b, torch::Tensor x, torch::Tensor initial_states, torch::Tensor final_states);";


mamba_fused_op_v7 = load_inline(
    name="mamba_fused_op_v7",
    cpp_sources=mamba_fused_scan_cpp_source_v7,
    cuda_sources=mamba_fused_scan_cuda_source_v7,
    functions=["mamba_fused_scan_cuda_v7"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(ModelNew, self).__init__()
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        assert d_head * (d_state // 4) <= 1024, "d_head * (d_state/4) must be <= 1024 for this kernel"
        assert block_len == 64, "This optimized kernel is hardcoded for block_len=64"
        assert d_head % 4 == 0 and d_state % 4 == 0, "d_head and d_state must be divisible by 4 for vectorized loads"
        
        shared_mem_needed = (block_len + block_len * d_state + block_len * d_head + block_len) * 4
        if shared_mem_needed > 49152:
            raise ValueError(f"Required shared memory ({shared_mem_needed} bytes) exceeds the typical 48KB limit.")

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
        self.mamba_fused_op = mamba_fused_op_v7

    def forward(self, X, initial_states=None):
        X_blocks, A_blocks, B_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B)
        ]
        
        # Rearrange and make contiguous for safe casting to float4 and optimal memory access in CUDA
        A_fused = rearrange(A_blocks, "b c l h -> b h c l").contiguous()
        B_fused = rearrange(B_blocks, "b c l h n -> b h c l n").contiguous()
        X_fused = rearrange(X_blocks, "b c l h p -> b h c l p").contiguous()

        if initial_states is None:
            initial_states = torch.zeros(self.batch_size, self.n_heads, self.d_head, self.d_state, device=X.device, dtype=X.dtype)
        else:
            if initial_states.dim() == 5:
                initial_states = initial_states.squeeze(1)
            initial_states = initial_states.contiguous()

        final_states_fused = torch.empty_like(initial_states)
        
        self.mamba_fused_op.mamba_fused_scan_cuda_v7(
            A_fused, B_fused, X_fused, initial_states, final_states_fused
        )
        
        return final_states_fused

# Test parameters
batch_size = 16
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    return [torch.randn(batch_size, seq_length, n_heads, d_head).cuda()]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]