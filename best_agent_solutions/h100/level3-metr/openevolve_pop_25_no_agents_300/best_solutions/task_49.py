# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# Combined CUDA source for fused Mamba operations.
# This version includes two highly optimized kernels:
# 1. A kernel that fuses the calculation of decay_states and its subsequent einsum.
# 2. An exp(segsum(x)) kernel optimized with coalesced memory writes.
mamba_combined_fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

// Kernel 1: Fuses the calculation of decay_states and the subsequent einsum.
// This is a major optimization that avoids materializing the large intermediate decay_states tensor.
// It computes: states[b,c,h,p,n] = sum_l( B[b,c,l,h,n] * exp(A_cumsum[b,h,c,L-1] - A_cumsum[b,h,c,l]) * X[b,c,l,h,p] )
__global__ void fused_decay_einsum_kernel(
    const float* __restrict__ B,      // Shape: (B, C, L, H, N)
    const float* __restrict__ A_cumsum, // Shape: (B, H, C, L)
    const float* __restrict__ X,      // Shape: (B, C, L, H, P)
    float* __restrict__ states,       // Shape: (B, C, H, P, N)
    int B_dim, int C_dim, int L_dim, int H_dim, int P_dim, int N_dim
) {
    // Each block computes one (P, N) output matrix for a given (b, c, h) grid coordinate.
    const int b = blockIdx.z / (C_dim * H_dim);
    const int h = blockIdx.z % H_dim;
    const int c = (blockIdx.z / H_dim) % C_dim;

    // Thread indices map to the output matrix dimensions (p, n).
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (p >= P_dim || n >= N_dim) return;

    // Pre-fetch the last value of A_cumsum for this chunk, used for all 'l' iterations.
    const float a_cumsum_last = A_cumsum[(long long)b * H_dim * C_dim * L_dim +
                                         (long long)h * C_dim * L_dim +
                                         (long long)c * L_dim +
                                         (L_dim - 1)];

    float sum = 0.0f;
    // Loop over the reduction dimension 'l'.
    for (int l = 0; l < L_dim; ++l) {
        // Calculate flat indices for the contiguous input tensors.
        long long b_idx = (long long)b * C_dim * L_dim * H_dim * N_dim + (long long)c * L_dim * H_dim * N_dim + (long long)l * H_dim * N_dim + (long long)h * N_dim + n;
        long long a_idx = (long long)b * H_dim * C_dim * L_dim + (long long)h * C_dim * L_dim + (long long)c * L_dim + l;
        long long x_idx = (long long)b * C_dim * L_dim * H_dim * P_dim + (long long)c * L_dim * H_dim * P_dim + (long long)l * H_dim * P_dim + (long long)h * P_dim + p;

        // Compute the decay factor on-the-fly.
        float decay_factor = expf(a_cumsum_last - A_cumsum[a_idx]);

        // Accumulate the product.
        sum += B[b_idx] * decay_factor * X[x_idx];
    }

    // Write the final result to the output tensor.
    long long states_idx = (long long)b * C_dim * H_dim * P_dim * N_dim + (long long)c * H_dim * P_dim * N_dim + (long long)h * P_dim * N_dim + (long long)p * N_dim + n;
    states[states_idx] = sum;
}


// Kernel 2: Fused exponential segment sum (exp(segsum(x))) with coalesced writes.
// This version uses a sequential scan (efficient for small L) and has each thread
// compute one column of the output matrix to ensure coalesced global memory writes.
__global__ void fused_exp_segsum_columnwise_kernel(const float* __restrict__ x, float* __restrict__ out,
                                                   const int num_sequences, const int L) {
    const int seq_idx = blockIdx.x;
    if (seq_idx >= num_sequences) return;

    const int tid = threadIdx.x;
    const float* x_ptr = x + seq_idx * L;
    float* out_ptr = out + seq_idx * L * L;

    extern __shared__ float s_cumsum[];

    if (tid == 0) {
        float current_sum = 0.0f;
        for (int i = 0; i < L; ++i) {
            current_sum += x_ptr[i];
            s_cumsum[i] = current_sum;
        }
    }
    __syncthreads();

    const int col = tid;
    if (col < L) {
        const float cumsum_col = s_cumsum[col];
        for (int row = 0; row < L; ++row) {
            if (row >= col) {
                const float cumsum_row = s_cumsum[row];
                out_ptr[row * L + col] = expf(cumsum_row - cumsum_col);
            } else {
                out_ptr[row * L + col] = 0.0f;
            }
        }
    }
}


// C++ Wrapper for Kernel 1
torch::Tensor fused_decay_einsum_cuda(torch::Tensor B, torch::Tensor A_cumsum, torch::Tensor X) {
    TORCH_CHECK(B.is_cuda() && A_cumsum.is_cuda() && X.is_cuda(), "Inputs must be CUDA tensors");
    
    const int B_dim = B.size(0);
    const int C_dim = B.size(1);
    const int L_dim = B.size(2);
    const int H_dim = B.size(3);
    const int N_dim = B.size(4);
    const int P_dim = X.size(4);

    auto states = torch::empty({B_dim, C_dim, H_dim, P_dim, N_dim}, B.options());
    
    dim3 threads(16, 16); // 256 threads per block in a 2D grid.
    dim3 blocks(
        (P_dim + threads.x - 1) / threads.x,
        (N_dim + threads.y - 1) / threads.y,
        B_dim * C_dim * H_dim // Launch one block for each (b, c, h) item.
    );
    
    fused_decay_einsum_kernel<<<blocks, threads>>>(
        B.data_ptr<float>(), A_cumsum.data_ptr<float>(), X.data_ptr<float>(), states.data_ptr<float>(),
        B_dim, C_dim, L_dim, H_dim, P_dim, N_dim
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return states;
}


// C++ Wrapper for Kernel 2
torch::Tensor exp_segsum_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    const int L = x.size(-1);
    const int64_t num_sequences = (x.numel() == 0) ? 0 : x.numel() / L;

    auto out_shape = x.sizes().vec();
    out_shape.push_back(L);
    auto out = torch::empty(out_shape, x.options());

    if (num_sequences == 0) return out;

    auto x_flat = x.view({num_sequences, L});
    auto out_flat = out.view({num_sequences, L, L});
    
    TORCH_CHECK(L <= 1024, "L must be <= 1024 for this kernel implementation");
    const dim3 block_size(L > 0 ? L : 1);
    const dim3 grid_size(num_sequences);
    const size_t shared_mem_size = L * sizeof(float);

    if (L > 0) {
        fused_exp_segsum_columnwise_kernel<<<grid_size, block_size, shared_mem_size>>>(
            x_flat.data_ptr<float>(), out_flat.data_ptr<float>(), num_sequences, L);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

mamba_combined_fused_cpp_source = """
torch::Tensor fused_decay_einsum_cuda(torch::Tensor B, torch::Tensor A_cumsum, torch::Tensor X);
torch::Tensor exp_segsum_cuda(torch::Tensor x);
"""

# Compile the inline CUDA code
mamba_kernels = load_inline(
    name="mamba_fused_kernels_final",
    cpp_sources=mamba_combined_fused_cpp_source,
    cuda_sources=mamba_combined_fused_source,
    functions=["fused_decay_einsum_cuda", "exp_segsum_cuda"],
    verbose=True,
)


class Model(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(Model, self).__init__()
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Initialize parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
    def forward(self, X, initial_states=None):
        # Rearrange inputs into blocks/chunks. Ensure contiguity for CUDA kernels.
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l").contiguous()
        A_cumsum = torch.cumsum(A_blocks, dim=-1).contiguous()
        
        # OPTIMIZATION 1: Dead code (`Y_diag`) is eliminated.
        
        # OPTIMIZATION 2: The intra-chunk state computation is replaced with a single fused CUDA kernel.
        # This kernel fuses the `exp`, broadcasted subtraction, and `einsum` operations,
        # avoiding large intermediate tensors.
        states = mamba_kernels.fused_decay_einsum_cuda(
            B_blocks.contiguous(), A_cumsum, X_blocks.contiguous()
        )
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        # OPTIMIZATION 3: Use the improved fused kernel for the `exp(segsum)` operation.
        # This kernel uses coalesced writes for maximum memory bandwidth.
        padded_A = F.pad(A_cumsum[:, :, :, -1], (1, 0)).contiguous()
        decay_chunk = mamba_kernels.exp_segsum_cuda(padded_A)
        
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        return new_states[:, -1]

# Test parameters
batch_size = 16
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    # Input tensors must be on the GPU for CUDA kernels.
    return [torch.randn(batch_size, seq_length, n_heads, d_head).cuda()]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]

# EVOLVE-BLOCK-END