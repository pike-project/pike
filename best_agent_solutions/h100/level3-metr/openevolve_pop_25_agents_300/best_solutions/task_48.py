# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernels for Mamba operations
# This combines the most effective kernels from prior top-performing solutions.
mamba_fused_kernels_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <c10/cuda/CUDAException.h>

// ==========================================================================================
// KERNEL 1: Fused Cumsum + Segsum + Exp for L matrix
// This kernel uses a parallel scan in shared memory for maximum efficiency.
// ==========================================================================================
__global__ void fused_segsum_exp_kernel(const float* a_in, float* l_out, int block_len) {
    int seq_idx = blockIdx.x;
    extern __shared__ float s_data[];

    const float* a_seq = a_in + seq_idx * block_len;
    float* l_matrix = l_out + seq_idx * block_len * block_len;
    int tid = threadIdx.x;

    // Load sequence into shared memory
    if (tid < block_len) {
        s_data[tid] = a_seq[tid];
    }
    __syncthreads();

    // In-place inclusive parallel scan (Kogge-Stone) to compute cumsum
    for (int offset = 1; offset < block_len; offset *= 2) {
        __syncthreads();
        float temp = 0;
        if (tid >= offset) {
            temp = s_data[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            s_data[tid] += temp;
        }
    }
    __syncthreads(); // Ensure all threads have finished the scan

    // Compute the output L matrix using the cumsum results in shared memory
    if (tid < block_len) { // Each thread computes one row
        float cumsum_l = s_data[tid];
        for (int s = 0; s < block_len; ++s) {
            if (tid >= s) {
                l_matrix[tid * block_len + s] = expf(cumsum_l - s_data[s]);
            } else {
                l_matrix[tid * block_len + s] = 0.0f; // exp(-inf) = 0
            }
        }
    }
}

// ==========================================================================================
// KERNEL 2: Fused State Update (Intra-chunk)
// Fuses: decay = exp(A_cs_last - A_cs); states = einsum(B, decay, X)
// ==========================================================================================
__global__ void fused_state_update_kernel(
    const float* A_cumsum, const float* B, const float* X, float* states,
    int B_dim, int C_dim, int L_dim, int H_dim, int N_dim, int P_dim) {

    // Map grid/block dimensions to tensor dimensions for clarity and efficiency
    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int h = blockIdx.z;
    const int p = threadIdx.x;
    const int n = threadIdx.y;

    extern __shared__ float s_A_cs[]; // Shared memory for one slice of A_cumsum

    // Cooperatively load the relevant A_cumsum slice into shared memory
    const float* A_cs_in_ptr = A_cumsum + b*H_dim*C_dim*L_dim + h*C_dim*L_dim + c*L_dim;
    int tid_flat = threadIdx.y * blockDim.x + threadIdx.x;
    for(int i = tid_flat; i < L_dim; i += blockDim.x * blockDim.y) {
        s_A_cs[i] = A_cs_in_ptr[i];
    }
    __syncthreads();

    const float A_cs_last = s_A_cs[L_dim - 1];
    float state_pn = 0.0f;

    // Main loop to compute the reduction
    for (int l = 0; l < L_dim; ++l) {
        const float decay = expf(A_cs_last - s_A_cs[l]);
        // B layout: b,c,l,h,n. X layout: b,c,l,h,p
        long b_idx = (long)b*C_dim*L_dim*H_dim*N_dim + c*L_dim*H_dim*N_dim + l*H_dim*N_dim + h*N_dim + n;
        long x_idx = (long)b*C_dim*L_dim*H_dim*P_dim + c*L_dim*H_dim*P_dim + l*H_dim*P_dim + h*P_dim + p;
        state_pn += B[b_idx] * decay * X[x_idx];
    }

    // Write the final result to global memory
    long state_out_idx = (long)b*C_dim*H_dim*P_dim*N_dim + c*H_dim*P_dim*N_dim + h*P_dim*N_dim + p*N_dim + n;
    states[state_out_idx] = state_pn;
}

// ==========================================================================================
// KERNEL 3: Fused State-to-Output Conversion (Off-diagonal)
// Fuses: decay = exp(A_cs); Y_off = einsum(C, states, decay)
// ==========================================================================================
__global__ void fused_y_off_kernel(
    const float* A_cumsum, const float* C, const float* states, float* Y_off,
    int B_dim, int C_dim, int L_dim, int H_dim, int N_dim, int P_dim) {

    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int l = blockIdx.z;
    const int p = threadIdx.x;
    const int h = threadIdx.y;

    if (b >= B_dim || c >= C_dim || l >= L_dim || h >= H_dim || p >= P_dim) return;

    extern __shared__ float s_C_block[]; // Shared memory for a block of C vectors

    // Cooperatively load C vectors for all heads (h) in this block.
    // Threads with p < N_dim participate in loading for their respective h.
    if (threadIdx.x < N_dim) {
        const long C_base_idx = (long)b*C_dim*L_dim*H_dim*N_dim + c*L_dim*H_dim*N_dim + l*H_dim*N_dim + h*N_dim;
        s_C_block[h * N_dim + threadIdx.x] = C[C_base_idx + threadIdx.x];
    }
    __syncthreads();

    const long states_base_idx = (long)b*C_dim*H_dim*P_dim*N_dim + c*H_dim*P_dim*N_dim + h*P_dim*N_dim + p*N_dim;

    // Compute dot product between C and states
    float dot_product = 0.0f;
    for (int n = 0; n < N_dim; ++n) {
        dot_product += s_C_block[h * N_dim + n] * states[states_base_idx + n];
    }

    // Apply final decay and write to output
    const float decay = expf(A_cumsum[(long)b*H_dim*C_dim*L_dim + h*C_dim*L_dim + c*L_dim + l]);
    long y_off_idx = (long)b*C_dim*L_dim*H_dim*P_dim + c*L_dim*H_dim*P_dim + l*H_dim*P_dim + h*P_dim + p;
    Y_off[y_off_idx] = dot_product * decay;
}


// --- C++ Wrappers ---

torch::Tensor fused_segsum_exp_wrapper(torch::Tensor a_in) {
    auto original_shape = a_in.sizes();
    const int block_len = original_shape.back();
    auto a_flat = a_in.reshape({-1, block_len});
    const int num_sequences = a_flat.size(0);
    auto opts = a_in.options();
    auto l_flat = torch::empty({num_sequences, block_len, block_len}, opts);
    if (block_len == 0 || num_sequences == 0) return l_flat.reshape({original_shape[0], original_shape[1], original_shape[2], block_len, block_len});

    fused_segsum_exp_kernel<<<num_sequences, block_len, block_len * sizeof(float)>>>(
        a_flat.data_ptr<float>(), l_flat.data_ptr<float>(), block_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<int64_t> out_shape;
    for(size_t i = 0; i < original_shape.size() - 1; ++i) out_shape.push_back(original_shape[i]);
    out_shape.push_back(block_len);
    out_shape.push_back(block_len);
    return l_flat.reshape(out_shape);
}

torch::Tensor fused_state_update_wrapper(torch::Tensor A_cumsum, torch::Tensor B, torch::Tensor X) {
    const int B_dim = B.size(0), C_dim = B.size(1), L_dim = B.size(2), H_dim = B.size(3), N_dim = B.size(4);
    const int P_dim = X.size(4);
    TORCH_CHECK(P_dim * N_dim <= 1024, "d_head * d_state exceeds max block size");

    auto states = torch::empty({B_dim, C_dim, H_dim, P_dim, N_dim}, X.options());
    const dim3 grid_dim(B_dim, C_dim, H_dim);
    const dim3 block_dim(P_dim, N_dim);
    size_t shmem_size = L_dim * sizeof(float);

    fused_state_update_kernel<<<grid_dim, block_dim, shmem_size>>>(
        A_cumsum.data_ptr<float>(), B.data_ptr<float>(), X.data_ptr<float>(), states.data_ptr<float>(),
        B_dim, C_dim, L_dim, H_dim, N_dim, P_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return states;
}

torch::Tensor fused_y_off_wrapper(torch::Tensor A_cumsum, torch::Tensor C, torch::Tensor states) {
    const int B_dim = C.size(0), C_dim = C.size(1), L_dim = C.size(2), H_dim = C.size(3), N_dim = C.size(4);
    const int P_dim = states.size(3);
    auto Y_off = torch::empty({B_dim, C_dim, L_dim, H_dim, P_dim}, C.options());
    
    const dim3 grid_dim(B_dim, C_dim, L_dim);
    const dim3 block_dim(P_dim, H_dim);
    TORCH_CHECK(P_dim * H_dim <= 1024, "d_head * n_heads exceeds max block size");

    size_t shmem_size = H_dim * N_dim * sizeof(float);

    fused_y_off_kernel<<<grid_dim, block_dim, shmem_size>>>(
        A_cumsum.data_ptr<float>(), C.data_ptr<float>(), states.data_ptr<float>(), Y_off.data_ptr<float>(),
        B_dim, C_dim, L_dim, H_dim, N_dim, P_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return Y_off;
}

"""

mamba_fused_kernels_cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_segsum_exp_wrapper(torch::Tensor a_in);
torch::Tensor fused_state_update_wrapper(torch::Tensor A_cumsum, torch::Tensor B, torch::Tensor X);
torch::Tensor fused_y_off_wrapper(torch::Tensor A_cumsum, torch::Tensor C, torch::Tensor states);
"""

# JIT compile the CUDA kernels
mamba_fused_ops = load_inline(
    name="mamba_fused_ops",
    cpp_sources=mamba_fused_kernels_cpp_source,
    cuda_sources=mamba_fused_kernels_source,
    functions=["fused_segsum_exp_wrapper", "fused_state_update_wrapper", "fused_y_off_wrapper"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)


class Model(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super(Model, self).__init__()
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        self.batch_size, self.seq_length, self.n_heads, self.d_head, self.d_state, self.block_len = \
            batch_size, seq_length, n_heads, d_head, d_state, block_len
        
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

    def forward(self, X, initial_states=None):
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len).contiguous()
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks_permuted = rearrange(A_blocks, "b c l h -> b h c l").contiguous()
        A_cumsum = torch.cumsum(A_blocks_permuted, dim=-1)
        
        # 1. Compute diagonal block outputs (L matrix via CUDA, einsum in PyTorch)
        L = mamba_fused_ops.fused_segsum_exp_wrapper(A_blocks_permuted)
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", 
                             C_blocks, B_blocks, L, X_blocks)
        
        # 2. Compute intra-chunk states with fused CUDA kernel
        states = mamba_fused_ops.fused_state_update_wrapper(A_cumsum, B_blocks, X_blocks)
        
        # 3. Compute inter-chunk recurrence (PyTorch is fine for this smaller part)
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states_padded = torch.cat([initial_states, states], dim=1)
        
        padded_A_cumsum_last = F.pad(A_cumsum[:, :, :, -1], (1, 0)).contiguous()
        decay_chunk = mamba_fused_ops.fused_segsum_exp_wrapper(padded_A_cumsum_last)
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states_padded)
        states = new_states[:, :-1].contiguous()
        
        # 4. Compute state-to-output conversion with fused CUDA kernel
        Y_off = mamba_fused_ops.fused_y_off_wrapper(A_cumsum, C_blocks, states)
        
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
    return [torch.randn(batch_size, seq_length, n_heads, d_head).cuda()]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]

# EVOLVE-BLOCK-END