# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# Inline CUDA source code
cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Kernel 1: Compute exp(segsum) from a cumsum'd input
// This replaces the creation of a large intermediate tensor and fuses operations
__global__ void exp_segsum_from_cumsum_kernel(float* out, const float* cumsum, int N, int T) {
    int row_idx_in_matrix = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx_in_matrix = blockIdx.x * blockDim.x + threadIdx.x;
    
    int batch_idx = blockIdx.z;

    if (batch_idx < N && row_idx_in_matrix < T && col_idx_in_matrix < T) {
        const float* cumsum_ptr = cumsum + batch_idx * T;
        float* out_ptr = out + batch_idx * T * T;

        if (row_idx_in_matrix >= col_idx_in_matrix) {
            float val = expf(cumsum_ptr[row_idx_in_matrix] - cumsum_ptr[col_idx_in_matrix]);
            out_ptr[row_idx_in_matrix * T + col_idx_in_matrix] = val;
        } else {
            out_ptr[row_idx_in_matrix * T + col_idx_in_matrix] = 0.0f;
        }
    }
}

// Kernel 2: Fused computation of intra-chunk states
// Fuses the calculation of decay_states and the subsequent einsum.
// states[b,c,h,p,n] = sum_l( B[b,c,l,h,n] * exp(A_cumsum_last - A_cumsum_l) * X[b,c,l,h,p] )
__global__ void fused_states_computation_kernel(
    const float* B_blocks,      // Shape: (B, C, L, H, N)
    const float* A_cumsum,      // Shape: (B, H, C, L)
    const float* X_blocks,      // Shape: (B, C, L, H, P)
    float* states,              // Shape: (B, C, H, P, N)
    int B, int C, int L, int H, int P, int N
) {
    // Grid maps to (P, N, B*C*H)
    // Block maps to a tile of size (TILE_P, TILE_N)

    // Map blockIdx.z to batch dimensions (b, c, h)
    int bch_idx = blockIdx.z;
    int b = bch_idx / (C * H);
    int c = (bch_idx / H) % C;
    int h = bch_idx % H;

    // Map threadIdx to output matrix tile dimensions (p, n)
    int p = blockIdx.x * blockDim.x + threadIdx.x; // d_head dimension
    int n = blockIdx.y * blockDim.y + threadIdx.y; // d_state dimension

    if (p >= P || n >= N) {
        return;
    }

    // Pointer to the start of the relevant A_cumsum chunk for this (b, h, c)
    const float* a_cumsum_ptr = A_cumsum + b * (H * C * L) + h * (C * L) + c * L;
    
    // Read the last value of the cumsum for this chunk once per thread. Stored in register.
    const float a_cumsum_last = a_cumsum_ptr[L - 1];

    float accumulator = 0.0f;

    // Loop over the reduction dimension L (block_len)
    for (int l = 0; l < L; ++l) {
        // On-the-fly decay calculation
        float decay = expf(a_cumsum_last - a_cumsum_ptr[l]);

        // Get B and X values from global memory
        const int b_idx = b * (C * L * H * N) + c * (L * H * N) + l * (H * N) + h * N + n;
        const int x_idx = b * (C * L * H * P) + c * (L * H * P) + l * (H * P) + h * P + p;

        accumulator += B_blocks[b_idx] * decay * X_blocks[x_idx];
    }

    // Write final result to the output states tensor
    const int out_idx = b * (C * H * P * N) + c * (H * P * N) + h * (P * N) + p * N + n;
    states[out_idx] = accumulator;
}


// C++ wrapper functions
torch::Tensor exp_segsum_from_cumsum_cuda(torch::Tensor cumsum) {
    TORCH_CHECK(cumsum.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(cumsum.is_contiguous(), "Input must be contiguous");

    const auto T = cumsum.size(-1);
    const int64_t N = cumsum.numel() / T;

    auto out_shape = cumsum.sizes().vec();
    out_shape.push_back(T);
    auto out = torch::empty(out_shape, cumsum.options());

    auto cumsum_flat = cumsum.reshape({N, T});
    auto out_flat = out.reshape({N, T, T});

    dim3 block_size(16, 16, 1);
    dim3 grid_size(
        (T + block_size.x - 1) / block_size.x,
        (T + block_size.y - 1) / block_size.y,
        N
    );

    exp_segsum_from_cumsum_kernel<<<grid_size, block_size>>>(
        out_flat.data_ptr<float>(),
        cumsum_flat.data_ptr<float>(),
        N,
        T
    );
    return out;
}

torch::Tensor fused_states_computation_cuda(
    torch::Tensor B_blocks, // (B, C, L, H, N)
    torch::Tensor A_cumsum, // (B, H, C, L)
    torch::Tensor X_blocks  // (B, C, L, H, P)
) {
    const auto B = B_blocks.size(0);
    const auto C = B_blocks.size(1);
    const auto L = B_blocks.size(2);
    const auto H = B_blocks.size(3);
    const auto N = B_blocks.size(4); // d_state
    const auto P = X_blocks.size(4); // d_head

    auto states = torch::empty({B, C, H, P, N}, B_blocks.options());

    auto B_cont = B_blocks.contiguous();
    auto A_cont = A_cumsum.contiguous();
    auto X_cont = X_blocks.contiguous();

    const int TILE_DIM_P = 16;
    const int TILE_DIM_N = 16;
    dim3 block_size(TILE_DIM_P, TILE_DIM_N, 1);
    dim3 grid_size(
        (P + TILE_DIM_P - 1) / TILE_DIM_P,
        (N + TILE_DIM_N - 1) / TILE_DIM_N,
        B * C * H
    );

    fused_states_computation_kernel<<<grid_size, block_size>>>(
        B_cont.data_ptr<float>(),
        A_cont.data_ptr<float>(),
        X_cont.data_ptr<float>(),
        states.data_ptr<float>(),
        B, C, L, H, P, N
    );

    return states;
}
"""

cpp_source = """
torch::Tensor exp_segsum_from_cumsum_cuda(torch::Tensor cumsum);
torch::Tensor fused_states_computation_cuda(
    torch::Tensor B_blocks,
    torch::Tensor A_cumsum,
    torch::Tensor X_blocks
);
"""

# JIT compile the CUDA kernels
cuda_kernels = load_inline(
    name="mamba_kernels_v2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["exp_segsum_from_cumsum_cuda", "fused_states_computation_cuda"],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model implementation for benchmarking.
        
        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
        """
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
        """
        Forward pass implementing the SSD operation.
        
        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :param initial_states: Optional initial states
        :return: Output tensor Y and final state
        """
        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks_permuted = rearrange(A_blocks, "b c l h -> b h c l").contiguous()
        A_cumsum = torch.cumsum(A_blocks_permuted, dim=-1)
        
        # 1. Compute diagonal block outputs
        # Replace torch.exp(segsum(A_blocks)) with a fused CUDA kernel
        L = cuda_kernels.exp_segsum_from_cumsum_cuda(A_cumsum)
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                             C_blocks, B_blocks, L, X_blocks)
        
        # 2. Compute intra-chunk states
        # Replace decay calculation and einsum with a single fused kernel.
        # This avoids materializing the intermediate `decay_states` tensor.
        states = cuda_kernels.fused_states_computation_cuda(B_blocks, A_cumsum, X_blocks)
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        # Replace torch.exp(segsum(...)) for decay_chunk calculation
        padded_chunk_sum = F.pad(A_cumsum[:, :, :, -1], (1, 0))
        padded_chunk_sum_cumsum = torch.cumsum(padded_chunk_sum, dim=-1)
        decay_chunk = cuda_kernels.exp_segsum_from_cumsum_cuda(padded_chunk_sum_cumsum)
        
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
    return [torch.randn(batch_size, seq_length, n_heads, d_head).cuda()]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]

# EVOLVE-BLOCK-END