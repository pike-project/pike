import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from einops import rearrange

# CUDA source for the new, fully-fused selective scan kernel.
# This version integrates all pre-processing (chunking, cumsum) into a single kernel launch.
mamba_kernels_source_v3 = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath> // For expf

// TILE dimensions for shared memory sizing and thread block configuration.
constexpr int TILE_L = 64; // Max block_len
constexpr int TILE_P = 64; // Max d_head
constexpr int TILE_N = 16; // Max d_state

// =================================================================================================
// Fully Fused Mamba Selective Scan Kernel
//
// Optimization vs. previous version:
// This kernel eliminates all pre-processing steps in PyTorch by fusing them directly into the
// CUDA implementation. Previously, permute, rearrange, and cumsum operations were performed
// before the kernel call, creating significant memory and kernel launch overhead.
//
// 1. **Fused Data Loading**: The kernel now reads directly from the original, non-permuted,
//    non-chunked input tensors (X, A, B). It calculates the correct offsets on-the-fly to
//    load the data for each chunk into shared memory. This avoids writing multiple
//    intermediate tensors to global memory.
//
// 2. **In-Kernel Cumsum**: The cumulative sum of the 'A' parameter, required for the decay
//    term, is now computed inside the kernel using shared memory after 'A' is loaded. For
//    the small 'block_len' of 64, a simple sequential computation by a single thread is
//    efficient and avoids a complex parallel scan implementation.
//
// This "end-to-end fusion" approach drastically reduces memory bandwidth usage and kernel
// launch latency, leading to a substantial performance gain.
// =================================================================================================
__global__ void mamba_selective_scan_fused_kernel(
    float* __restrict__ final_state,
    const float* __restrict__ X_in,      // (B, S, H, P)
    const float* __restrict__ A_in,      // (B, S, H)
    const float* __restrict__ B_in,      // (B, S, H, N)
    const float* __restrict__ initial_states, // (B, H, N, P)
    const int B_dim, const int S_dim, const int H_dim,
    const int N_dim, const int P_dim, const int block_len) {

    // Grid: (B_dim * H_dim) -> Each block processes one (b, h) recurrence chain.
    const int bh_idx = blockIdx.x;
    const int b = bh_idx / H_dim;
    const int h = bh_idx % H_dim;

    // Block: (TILE_P, TILE_N) -> Each thread handles one element (n,p) of the state matrix.
    const int p = threadIdx.x;
    const int n = threadIdx.y;

    const int thread_id = n * blockDim.x + p;
    const int block_size = blockDim.x * blockDim.y;

    // Shared memory for one chunk.
    __shared__ float sA[TILE_L];
    __shared__ float sA_cs[TILE_L];
    __shared__ float sB[TILE_L][TILE_N];
    __shared__ float sX[TILE_L][TILE_P];
    __shared__ float sA_exp_neg[TILE_L];

    // Pointers to the start of the data for the current (b, h) pair in the original tensors.
    const long B_S_stride = (long)S_dim * H_dim * N_dim;
    const long X_S_stride = (long)S_dim * H_dim * P_dim;
    const long A_S_stride = (long)S_dim * H_dim;

    const float* A_src = A_in + b * A_S_stride + h;
    const float* B_src = B_in + b * B_S_stride + h * N_dim;
    const float* X_src = X_in + b * X_S_stride + h * P_dim;

    // State pointers.
    const int bh_stride_H = N_dim * P_dim;
    const float* init_h_ptr = initial_states + bh_idx * bh_stride_H;
    float* final_h_ptr = final_state + bh_idx * bh_stride_H;

    // Each thread loads its element of the initial state into a private register.
    float running_h_np = 0.0f;
    if (n < N_dim && p < P_dim) {
        running_h_np = init_h_ptr[n * P_dim + p];
    }

    const int C_dim = S_dim / block_len;

    // Main Recurrence Loop over Chunks
    for (int c = 0; c < C_dim; ++c) {
        const int s_offset = c * block_len;

        // 1. FUSED PRE-PROCESSING: Load data and compute cumsum on-the-fly.
        // Load sA (strided access, but small size: block_len)
        for (int i = thread_id; i < block_len; i += block_size) {
            sA[i] = A_src[(s_offset + i) * H_dim];
        }
        // Load sB (coalesced access)
        for (int i = thread_id; i < block_len * N_dim; i += block_size) {
            sB[i/N_dim][i%N_dim] = B_src[(s_offset + i/N_dim) * H_dim * N_dim + (i%N_dim)];
        }
        // Load sX (coalesced access)
        for (int i = thread_id; i < block_len * P_dim; i += block_size) {
            sX[i/P_dim][i%P_dim] = X_src[(s_offset + i/P_dim) * H_dim * P_dim + (i%P_dim)];
        }
        __syncthreads();

        // In-kernel cumsum on sA, performed by a single thread.
        if (thread_id == 0) {
            float current_sum = 0.0f;
            for (int l = 0; l < block_len; ++l) {
                current_sum += sA[l];
                sA_cs[l] = current_sum;
            }
        }
        __syncthreads();

        // 2. MAIN SCAN: Same optimized computation as before.
        // Cooperatively pre-compute exponentials.
        for (int l = thread_id; l < block_len; l += block_size) {
            sA_exp_neg[l] = expf(-sA_cs[l]);
        }
        __syncthreads();

        // Compute intra-chunk state with manual unrolling for ILP.
        if (n < N_dim && p < P_dim) {
            float inner_sum_np_0 = 0.0f, inner_sum_np_1 = 0.0f, inner_sum_np_2 = 0.0f, inner_sum_np_3 = 0.0f;
            for (int l = 0; l < block_len; l += 4) {
                inner_sum_np_0 += sA_exp_neg[l + 0] * sB[l + 0][n] * sX[l + 0][p];
                inner_sum_np_1 += sA_exp_neg[l + 1] * sB[l + 1][n] * sX[l + 1][p];
                inner_sum_np_2 += sA_exp_neg[l + 2] * sB[l + 2][n] * sX[l + 2][p];
                inner_sum_np_3 += sA_exp_neg[l + 3] * sB[l + 3][n] * sX[l + 3][p];
            }
            const float inner_sum_np = inner_sum_np_0 + inner_sum_np_1 + inner_sum_np_2 + inner_sum_np_3;
            const float a_cs_last = sA_cs[block_len - 1];
            const float inter_chunk_decay = expf(a_cs_last);
            running_h_np = (running_h_np + inner_sum_np) * inter_chunk_decay;
        }
        __syncthreads();
    }

    // 3. Write final state from registers to global memory.
    if (n < N_dim && p < P_dim) {
        final_h_ptr[n * P_dim + p] = running_h_np;
    }
}

// C++ host function to launch the kernel.
torch::Tensor mamba_selective_scan_fused(
    torch::Tensor X,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor initial_states,
    const int block_len
) {
    const int B_dim = X.size(0);
    const int S_dim = X.size(1);
    const int H_dim = X.size(2);
    const int P_dim = X.size(3); // d_head
    const int N_dim = B.size(3); // d_state

    TORCH_CHECK(block_len <= TILE_L, "block_len must be <= TILE_L (", TILE_L, ")");
    TORCH_CHECK(P_dim <= TILE_P, "d_head must be <= TILE_P (", TILE_P, ")");
    TORCH_CHECK(N_dim <= TILE_N, "d_state must be <= TILE_N (", TILE_N, ")");
    TORCH_CHECK(block_len > 0 && block_len % 4 == 0, "block_len must be a multiple of 4 for unrolling.");
    TORCH_CHECK(S_dim % block_len == 0, "seq_length must be divisible by block_len.");

    auto final_state = torch::empty({B_dim, H_dim, N_dim, P_dim}, X.options());

    dim3 grid(B_dim * H_dim);
    dim3 block(TILE_P, TILE_N);

    mamba_selective_scan_fused_kernel<<<grid, block>>>(
        final_state.data_ptr<float>(),
        X.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        initial_states.data_ptr<float>(),
        B_dim, S_dim, H_dim, N_dim, P_dim, block_len
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed in mamba_selective_scan_fused");

    return final_state;
}
"""

# C++ source for the Pybind11 module definition.
mamba_kernels_cpp_source_v3 = """
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Forward declaration of the C++ host function defined in the CUDA source.
torch::Tensor mamba_selective_scan_fused(
    torch::Tensor X, torch::Tensor A, torch::Tensor B,
    torch::Tensor initial_states, const int block_len
);

// Wrapper function with input checks.
torch::Tensor mamba_selective_scan_fused_wrapper(
    torch::Tensor X, torch::Tensor A, torch::Tensor B,
    torch::Tensor initial_states, const int block_len
) {
    CHECK_INPUT(X);
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(initial_states);
    return mamba_selective_scan_fused(X, A, B, initial_states, block_len);
}

// Pybind11 module definition.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mamba_selective_scan_fused", &mamba_selective_scan_fused_wrapper, "Fully Fused Mamba Selective Scan Kernel");
}
"""

# JIT compile the new fully-fused kernel.
mamba_fused_kernel = load_inline(
    name="mamba_kernels_v7_fully_fused",
    cpp_sources=mamba_kernels_cpp_source_v3,
    cuda_sources=mamba_kernels_source_v3,
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Hyper-optimized Mamba implementation using a single, fully-fused CUDA kernel
        that handles all pre-processing (chunking, cumsum) on-the-fly.
        """
        super(ModelNew, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        assert block_len % 4 == 0, "block_len must be a multiple of 4 for this kernel version"
        assert d_head <= 64, "d_head must be <= 64 for this kernel version"
        assert d_state <= 16, "d_state must be <= 16 for this kernel version"
        assert block_len <= 64, "block_len must be <= 64 for this kernel version"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len
        
        # Parameters with original (B, S, H, ...) layout
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        # C is not used in the state recurrence, so it's kept for API consistency.
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
    
    def forward(self, X, initial_states=None):
        """
        Forward pass implemented with a single, end-to-end fused CUDA kernel call.
        No more permute, rearrange, or cumsum in PyTorch.
        """
        if initial_states is None:
            # The kernel expects initial_states in (B, H, N, P) layout
            initial_states = torch.zeros(
                self.batch_size, self.n_heads, self.d_state, self.d_head,
                device=X.device, dtype=X.dtype
            )
        
        # Call the fully fused kernel. It handles all data layout transformations and
        # pre-computations internally.
        # Input Tensors:
        # X: (B, S, H, P)
        # A: (B, S, H)
        # B: (B, S, H, N)
        # initial_states: (B, H, N, P)
        final_state_internal = mamba_fused_kernel.mamba_selective_scan_fused(
            X.contiguous(),
            self.A.contiguous(),
            self.B.contiguous(),
            initial_states.contiguous(),
            self.block_len
        )
        # Kernel returns state in (B, H, N, P) layout.
        
        # Permute to match original model's state layout (B, H, P, N)
        final_state = final_state_internal.permute(0, 1, 3, 2).contiguous()
        return final_state

# Test parameters
batch_size = 16
seq_length = 128
n_heads = 8
d_head = 64
d_state = 16
block_len = 64

def get_inputs():
    # Input shape is (batch, seq_len, n_heads, d_head)
    X = torch.randn(batch_size, seq_length, n_heads, d_head, device='cuda', dtype=torch.float32)
    return [X]

def get_init_inputs():
    return [batch_size, seq_length, n_heads, d_head, d_state, block_len]