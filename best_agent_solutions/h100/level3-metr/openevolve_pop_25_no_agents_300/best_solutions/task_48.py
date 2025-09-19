# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels, combining the best element-wise kernels
# with a new, highly-fused kernel for state calculation.
mamba_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

// Kernel 1: L = exp(segsum(A)) from a pre-computed cumsum tensor.
// From the top-performing solution.
__global__ void compute_L_kernel(const float* __restrict__ a_cumsum, float* __restrict__ l_out, int L_dim, int num_matrices) {
    int mat_idx = blockIdx.x;
    if (mat_idx >= num_matrices) return;
    
    const float* cumsum_slice = a_cumsum + mat_idx * L_dim;
    float* l_slice = l_out + mat_idx * L_dim * L_dim;

    extern __shared__ float s_cumsum[];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;

    for (int i = tid; i < L_dim; i += block_threads) {
        s_cumsum[i] = cumsum_slice[i];
    }
    __syncthreads();

    for (int l = threadIdx.y; l < L_dim; l += blockDim.y) {
        for (int s = threadIdx.x; s < L_dim; s += blockDim.x) {
            if (l >= s) {
                l_slice[l * L_dim + s] = expf(s_cumsum[l] - s_cumsum[s]);
            } else {
                l_slice[l * L_dim + s] = 0.0f; // exp(-inf) from masking
            }
        }
    }
}

torch::Tensor compute_L_cuda(torch::Tensor a_cumsum) {
    auto a_cumsum_contig = a_cumsum.contiguous();

    auto original_shape_vec = a_cumsum_contig.sizes().vec();
    const int L_dim = original_shape_vec.back();
    original_shape_vec.pop_back();
    long num_matrices = 1;
    for (auto s : original_shape_vec) {
        num_matrices *= s;
    }

    original_shape_vec.push_back(L_dim);
    original_shape_vec.push_back(L_dim);
    auto l_out = torch::empty(original_shape_vec, a_cumsum_contig.options());
    if (num_matrices == 0) return l_out;

    auto a_cumsum_flat = a_cumsum_contig.view({num_matrices, L_dim});
    auto l_out_flat = l_out.view({num_matrices, L_dim, L_dim});

    dim3 grid_dim(num_matrices);
    dim3 block_dim(16, 16);
    size_t shared_mem_size = L_dim * sizeof(float);

    compute_L_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        a_cumsum_flat.data_ptr<float>(),
        l_out_flat.data_ptr<float>(),
        L_dim,
        num_matrices
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return l_out;
}


// Kernel 2: Fused kernel for intra-chunk state computation
// Computes: states = einsum("bclhn,bhcl,bclhp->bchpn", B, exp(A_last - A), X)
// This fuses the decay calculation with the einsum, avoiding an intermediate tensor.
__global__ void fused_intra_chunk_states_kernel(
    const float* __restrict__ a_cumsum, // B, H, C, L
    const float* __restrict__ b_blocks, // B, C, L, H, N
    const float* __restrict__ x_blocks, // B, C, L, H, P
    float* __restrict__ states_out,     // B, C, H, P, N
    int B, int C, int L, int H, int N, int P
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = blockIdx.z;

    int p = threadIdx.x;
    int n = threadIdx.y;

    if (b >= B || c >= C || h >= H || p >= P || n >= N) return;

    const float* a_slice = a_cumsum + b * (H * C * L) + h * (C * L) + c * L;
    float a_last = a_slice[L - 1];

    float sum = 0.0f;
    for (int l = 0; l < L; ++l) {
        float decay = expf(a_last - a_slice[l]);
        
        long b_idx = b * (C*L*H*N) + c * (L*H*N) + l * (H*N) + h * N + n;
        long x_idx = b * (C*L*H*P) + c * (L*H*P) + l * (H*P) + h * P + p;

        sum += b_blocks[b_idx] * decay * x_blocks[x_idx];
    }

    long out_idx = b * (C*H*P*N) + c * (H*P*N) + h * (P*N) + p * N + n;
    states_out[out_idx] = sum;
}

torch::Tensor fused_intra_chunk_states_cuda(
    torch::Tensor a_cumsum, 
    torch::Tensor b_blocks, 
    torch::Tensor x_blocks
) {
    auto a_cumsum_c = a_cumsum.contiguous();
    auto b_blocks_c = b_blocks.contiguous();
    auto x_blocks_c = x_blocks.contiguous();

    const auto B = a_cumsum_c.size(0);
    const auto H = a_cumsum_c.size(1);
    const auto C = a_cumsum_c.size(2);
    const auto L = a_cumsum_c.size(3);
    const auto N = b_blocks_c.size(4);
    const auto P = x_blocks_c.size(4);

    auto opts = a_cumsum_c.options();
    auto states_out = torch::empty({B, C, H, P, N}, opts);
    
    dim3 grid_dim(B, C, H);
    dim3 block_dim(P, N); // P=d_head, N=d_state

    fused_intra_chunk_states_kernel<<<grid_dim, block_dim>>>(
        a_cumsum_c.data_ptr<float>(),
        b_blocks_c.data_ptr<float>(),
        x_blocks_c.data_ptr<float>(),
        states_out.data_ptr<float>(),
        B, C, L, H, N, P
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return states_out;
}


// Kernel 3: Simple element-wise exp
// From the top-performing solution.
__global__ void exp_kernel(const float* __restrict__ in, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = expf(in[idx]);
    }
}

torch::Tensor exp_cuda(torch::Tensor in) {
    auto in_contiguous = in.contiguous();
    auto out = torch::empty_like(in_contiguous);
    const int size = in.numel();
    if (size == 0) return out;
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    exp_kernel<<<num_blocks, block_size>>>(
        in_contiguous.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return out;
}
"""

mamba_kernels_cpp_source = """
torch::Tensor compute_L_cuda(torch::Tensor a_cumsum);
torch::Tensor fused_intra_chunk_states_cuda(torch::Tensor a_cumsum, torch::Tensor b_blocks, torch::Tensor x_blocks);
torch::Tensor exp_cuda(torch::Tensor in);
"""

# Compile the inline CUDA code
mamba_kernels = load_inline(
    name="mamba_kernels_v4",
    cpp_sources=mamba_kernels_cpp_source,
    cuda_sources=mamba_kernels_source,
    functions=["compute_L_cuda", "fused_intra_chunk_states_cuda", "exp_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
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
        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # 1. Compute diagonal block outputs
        L = mamba_kernels.compute_L_cuda(A_cumsum)
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                             C_blocks, B_blocks, L, X_blocks)
        
        # 2. Compute intra-chunk states with a single fused kernel
        # This replaces the creation of `decay_states` and a subsequent `einsum`.
        states = mamba_kernels.fused_intra_chunk_states_cuda(A_cumsum, B_blocks, X_blocks)
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        padded_A_last_cumsum = torch.cumsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), dim=-1)
        decay_chunk = mamba_kernels.compute_L_cuda(padded_A_last_cumsum)
        
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]
        
        # 4. Compute state-to-output conversion
        state_decay_out = mamba_kernels.exp_cuda(A_cumsum)
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', 
                           C_blocks, states, state_decay_out)
        
        # Combine diagonal and off-diagonal terms
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        
        return Y

# Test parameters
batch_size = 16
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    return [torch.randn(batch_size, seq_length, n_heads, d_head)]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]

# EVOLVE-BLOCK-END