# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# This solution enhances the top-performing program by combining its vectorization techniques
# with layout optimization and extending vectorization to all major kernels.
#
# Key Improvements:
# 1.  Layout Optimization: The state computation kernel now outputs the `states` tensor in the
#     (b,c,h,n,p) layout, which is optimal for the Y_off kernel. This completely eliminates
#     a costly `permute` operation from the forward pass.
# 2.  Vectorized State Computation: The `fused_state_computation_kernel` is now vectorized
#     using `float4` over the P_size (d_head) dimension, significantly increasing its throughput.
# 3.  Vectorized Inter-Chunk Recurrence: The `fused_inter_chunk_kernel` is also adapted for
#     the new layout and vectorized with `float4`, improving its memory and compute efficiency.
#
# This holistic approach minimizes data movement and maximizes kernel performance.

mamba_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// --- Kernel 1: Fused Diagonal Block Calculation (Vectorized Dot Product) --- (Unchanged)
// Fuses: L = torch.exp(self.segsum(A_blocks))
// and:   Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_blocks, B_blocks, L, X_blocks)
// Improvement: The dot product over N_SIZE is vectorized using float4.
__global__ void fused_mamba_diag_kernel(
    const float* __restrict__ A_cumsum, // Shape: (b, h, c, l)
    const float* __restrict__ B,        // Shape: (b, c, l, h, n)
    const float* __restrict__ C,        // Shape: (b, c, l, h, n)
    const float* __restrict__ X,        // Shape: (b, c, l, h, p)
    float* __restrict__ Y,              // Shape: (b, c, l, h, p)
    const int B_SIZE, const int H_SIZE, const int C_SIZE, const int L_SIZE, const int P_SIZE, const int N_SIZE
) {
    const int block_idx = blockIdx.x;
    const int p = threadIdx.x;

    if (p >= P_SIZE) return;

    const int h = block_idx % H_SIZE;
    const int l = (block_idx / H_SIZE) % L_SIZE;
    const int c = (block_idx / (H_SIZE * L_SIZE)) % C_SIZE;
    const int b = block_idx / (H_SIZE * L_SIZE * C_SIZE);

    const long A_stride_b = (long)H_SIZE * C_SIZE * L_SIZE;
    const long A_stride_h = (long)C_SIZE * L_SIZE;
    const long A_stride_c = L_SIZE;

    const long BC_stride_b = (long)C_SIZE * L_SIZE * H_SIZE * N_SIZE;
    const long BC_stride_c = (long)L_SIZE * H_SIZE * N_SIZE;
    const long BC_stride_l = (long)H_SIZE * N_SIZE;
    const long BC_stride_h = N_SIZE;

    const long XY_stride_b = (long)C_SIZE * L_SIZE * H_SIZE * P_SIZE;
    const long XY_stride_c = (long)L_SIZE * H_SIZE * P_SIZE;
    const long XY_stride_l = (long)H_SIZE * P_SIZE;
    const long XY_stride_h = P_SIZE;

    const float* A_ptr = A_cumsum + b * A_stride_b + h * A_stride_h + c * A_stride_c;
    const float A_cumsum_l = A_ptr[l];

    extern __shared__ float cb_dot_products[];

    if (threadIdx.x < L_SIZE) {
        int s = threadIdx.x;
        if (s <= l) {
            float temp_sum = 0.0f;
            const long c_base_idx = b * BC_stride_b + c * BC_stride_c + l * BC_stride_l + h * BC_stride_h;
            const long b_base_idx = b * BC_stride_b + c * BC_stride_c + s * BC_stride_l + h * BC_stride_h;
            
            for (int n_vec = 0; n_vec < N_SIZE / 4; ++n_vec) {
                const float4 c_vec = ((const float4*)(C + c_base_idx))[n_vec];
                const float4 b_vec = ((const float4*)(B + b_base_idx))[n_vec];
                temp_sum += c_vec.x * b_vec.x + c_vec.y * b_vec.y + c_vec.z * b_vec.z + c_vec.w * b_vec.w;
            }
            cb_dot_products[s] = temp_sum;
        }
    }
    __syncthreads();

    float p_val = 0.0f;
    for (int s = 0; s <= l; ++s) {
        const float cb_dot_s = cb_dot_products[s];
        const float A_cumsum_s = A_ptr[s];
        const float l_val = expf(A_cumsum_l - A_cumsum_s);
        const long x_idx = b * XY_stride_b + c * XY_stride_c + s * XY_stride_l + h * XY_stride_h + p;
        p_val += l_val * cb_dot_s * X[x_idx];
    }
    
    const long y_idx = b * XY_stride_b + c * XY_stride_c + l * XY_stride_l + h * XY_stride_h + p;
    Y[y_idx] = p_val;
}


// --- Kernel 2: Fused State Computation (Vectorized & Layout-Optimized) ---
// Improvement: Vectorized over P_SIZE using float4. Outputs in (b,c,h,n,p) layout to avoid permute.
__global__ void fused_state_computation_kernel(
    const float* __restrict__ B,           // b,c,l,h,n
    const float* __restrict__ A_cumsum,    // b,h,c,l
    const float* __restrict__ X,           // b,c,l,h,p
    float* __restrict__ states,            // b,c,h,n,p (output)
    const int B_size, const int C_size, const int L_size, 
    const int H_size, const int N_size, const int P_size
) {
    const int b = blockIdx.x, c = blockIdx.y, h = blockIdx.z;
    const int p_vec = threadIdx.x, n = threadIdx.y;

    if (p_vec >= P_size / 4 || n >= N_size) return;

    const long a_base_idx = (long)b * H_size * C_size * L_size + (long)h * C_size * L_size + (long)c * L_size;
    const float a_last = A_cumsum[a_base_idx + L_size - 1];

    float4 sum_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int l = 0; l < L_size; ++l) {
        const float decay = expf(a_last - A_cumsum[a_base_idx + l]);
        const long b_idx = (long)b * C_size * L_size * H_size * N_size + (long)c * L_size * H_size * N_size + (long)l * H_size * N_size + (long)h * N_size + n;
        const float b_val = B[b_idx];
        
        const long x_base_idx = (long)b * C_size * L_size * H_size * P_size + (long)c * L_size * H_size * P_size + (long)l * H_size * P_size + (long)h * P_size;
        const float4 x_vec = ((const float4*)(X + x_base_idx))[p_vec];

        float intermediate = b_val * decay;
        sum_vec.x += intermediate * x_vec.x;
        sum_vec.y += intermediate * x_vec.y;
        sum_vec.z += intermediate * x_vec.z;
        sum_vec.w += intermediate * x_vec.w;
    }
    
    const long states_base_idx = (long)b * C_size * H_size * N_size * P_size + (long)c * H_size * N_size * P_size + (long)h * N_size * P_size + (long)n * P_size;
    ((float4*)(states + states_base_idx))[p_vec] = sum_vec;
}


// --- Kernel 3: Fused inter-chunk recurrence (Vectorized & Layout-Optimized) ---
// Improvement: Vectorized over P_SIZE. Operates on (b,c,h,n,p) layout.
__global__ void fused_inter_chunk_kernel(
    const float* __restrict__ A_last_padded, // b,h,c+1
    const float* __restrict__ states,        // b,c+1,h,n,p
    float* __restrict__ new_states,          // b,c+1,h,n,p
    const int B_size, const int H_size, const int C_padded, 
    const int P_size, const int N_size
) {
    const int b = blockIdx.x, h = blockIdx.y, z = blockIdx.z;
    const int p_vec = threadIdx.x, n = threadIdx.y;

    if (p_vec >= P_size / 4 || n >= N_size) return;

    extern __shared__ float s_A_cumsum[];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        const float* a_ptr = A_last_padded + b * H_size * C_padded + h * C_padded;
        float current_sum = 0.0f;
        for (int i = 0; i < C_padded; ++i) {
            current_sum += a_ptr[i];
            s_A_cumsum[i] = current_sum;
        }
    }
    __syncthreads();

    float4 sum_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int c = 0; c <= z; ++c) {
        const float decay = expf(s_A_cumsum[z] - s_A_cumsum[c]);
        const long states_base_idx = (long)b * C_padded * H_size * N_size * P_size + (long)c * H_size * N_size * P_size + (long)h * N_size * P_size + (long)n * P_size;
        const float4 s_vec = ((const float4*)(states + states_base_idx))[p_vec];

        sum_vec.x += decay * s_vec.x;
        sum_vec.y += decay * s_vec.y;
        sum_vec.z += decay * s_vec.z;
        sum_vec.w += decay * s_vec.w;
    }
    
    const long out_base_idx = (long)b * C_padded * H_size * N_size * P_size + (long)z * H_size * N_size * P_size + (long)h * N_size * P_size + (long)n * P_size;
    ((float4*)(new_states + out_base_idx))[p_vec] = sum_vec;
}


// --- Kernel 4: Fused Y_off computation (Vectorized & Restructured) --- (Unchanged)
// Fuses: state_decay_out = torch.exp(A_cumsum)
// and:   Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C_blocks, states, state_decay_out)
// Improvement: Vectorized over P_SIZE using float4. Grid/block structure changed to increase occupancy.
__global__ void fused_Y_off_coalesced_kernel(
    const float* __restrict__ C,                 // b,c,l,h,n
    const float* __restrict__ states,            // b,c,h,n,p
    const float* __restrict__ A_cumsum,          // b,h,c,l
    float* __restrict__ Y_off,                   // b,c,l,h,p
    const int B_size, const int C_size, const int L_size, const int H_size, 
    const int P_size, const int N_size
) {
    const int p_vec = threadIdx.x;
    const int h = threadIdx.y;

    if (p_vec >= P_size / 4 || h >= H_size) return;

    const int bcl_idx = blockIdx.x;
    const int l = bcl_idx % L_size;
    const int c = (bcl_idx / L_size) % C_size;
    const int b = bcl_idx / (L_size * C_size);

    const long c_idx_base = (long)b * C_size * L_size * H_size * N_size + (long)c * L_size * H_size * N_size + (long)l * H_size * N_size + (long)h * N_size;
    const long states_idx_base = (long)b * C_size * H_size * N_size * P_size + (long)c * H_size * N_size * P_size + (long)h * N_size * P_size;

    float4 dot_prod_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int n = 0; n < N_size; ++n) {
        const float c_val = C[c_idx_base + n];
        const float4 s_vec = ((const float4*)(states + states_idx_base + n * P_size))[p_vec];
        dot_prod_vec.x += c_val * s_vec.x;
        dot_prod_vec.y += c_val * s_vec.y;
        dot_prod_vec.z += c_val * s_vec.z;
        dot_prod_vec.w += c_val * s_vec.w;
    }

    const long a_idx = (long)b * H_size * C_size * L_size + (long)h * C_size * L_size + (long)c * L_size + l;
    const float decay = expf(A_cumsum[a_idx]);

    const long y_off_idx_base = (long)b * C_size * L_size * H_size * P_size + (long)c * L_size * H_size * P_size + (long)l * H_size * P_size + (long)h * P_size;

    float4 y_vec;
    y_vec.x = decay * dot_prod_vec.x;
    y_vec.y = decay * dot_prod_vec.y;
    y_vec.z = decay * dot_prod_vec.z;
    y_vec.w = decay * dot_prod_vec.w;
    ((float4*)(Y_off + y_off_idx_base))[p_vec] = y_vec;
}


// --- C++ Wrappers ---

torch::Tensor fused_mamba_diag(torch::Tensor A_cumsum, torch::Tensor B, torch::Tensor C, torch::Tensor X) {
    auto A_c = A_cumsum.contiguous(); auto B_c = B.contiguous(); auto C_c = C.contiguous(); auto X_c = X.contiguous();
    const int B_SIZE = X_c.size(0), C_SIZE = X_c.size(1), L_SIZE = X_c.size(2);
    const int H_SIZE = X_c.size(3), P_SIZE = X_c.size(4), N_SIZE = B_c.size(4);
    TORCH_CHECK(L_SIZE <= 64, "This kernel assumes block_len (L_SIZE) <= 64 for shared memory.");
    TORCH_CHECK(N_SIZE % 4 == 0, "N_SIZE must be divisible by 4 for vectorization.");
    auto Y = torch::empty_like(X_c);
    const dim3 block_dim(std::max(P_SIZE, L_SIZE));
    const dim3 grid_dim(B_SIZE * C_SIZE * L_SIZE * H_SIZE);
    const size_t shared_mem_size = L_SIZE * sizeof(float);
    fused_mamba_diag_kernel<<<grid_dim, block_dim, shared_mem_size>>>(A_c.data_ptr<float>(), B_c.data_ptr<float>(), C_c.data_ptr<float>(), X_c.data_ptr<float>(), Y.data_ptr<float>(), B_SIZE, H_SIZE, C_SIZE, L_SIZE, P_SIZE, N_SIZE);
    return Y;
}

torch::Tensor fused_state_computation(torch::Tensor B, torch::Tensor A_cumsum, torch::Tensor X) {
    auto B_cont = B.contiguous(); auto A_cumsum_cont = A_cumsum.contiguous(); auto X_cont = X.contiguous();
    const auto B_size = B_cont.size(0), C_size = B_cont.size(1), L_size = B_cont.size(2);
    const auto H_size = B_cont.size(3), N_size = B_cont.size(4), P_size = X_cont.size(4);
    auto states = torch::empty({B_size, C_size, H_size, N_size, P_size}, B_cont.options());
    TORCH_CHECK(P_size % 4 == 0, "P_size must be divisible by 4 for vectorization.");
    const dim3 grid_dim(B_size, C_size, H_size);
    const dim3 block_dim(P_size / 4, N_size);
    TORCH_CHECK((P_size / 4 * N_size) <= 1024, "Block size (P*N/4) exceeds 1024");
    fused_state_computation_kernel<<<grid_dim, block_dim>>>(B_cont.data_ptr<float>(), A_cumsum_cont.data_ptr<float>(), X_cont.data_ptr<float>(), states.data_ptr<float>(), B_size, C_size, L_size, H_size, N_size, P_size);
    return states;
}

torch::Tensor fused_inter_chunk(torch::Tensor A_last_padded, torch::Tensor states) {
    auto A_cont = A_last_padded.contiguous(); auto states_cont = states.contiguous();
    const auto B_size = states_cont.size(0), C_padded = states_cont.size(1), H_size = states_cont.size(2);
    const auto N_size = states_cont.size(3), P_size = states_cont.size(4);
    auto new_states = torch::empty_like(states_cont);
    const dim3 grid_dim(B_size, H_size, C_padded);
    TORCH_CHECK(P_size % 4 == 0, "P_size must be divisible by 4 for vectorization.");
    const dim3 block_dim(P_size / 4, N_size);
    TORCH_CHECK((P_size / 4 * N_size) <= 1024, "Block size (P*N/4) exceeds 1024");
    const size_t shared_mem_size = C_padded * sizeof(float);
    fused_inter_chunk_kernel<<<grid_dim, block_dim, shared_mem_size>>>(A_cont.data_ptr<float>(), states_cont.data_ptr<float>(), new_states.data_ptr<float>(), B_size, H_size, C_padded, P_size, N_size);
    return new_states;
}

torch::Tensor fused_Y_off_coalesced(torch::Tensor C, torch::Tensor states, torch::Tensor A_cumsum) {
    auto C_cont = C.contiguous(); auto A_cumsum_cont = A_cumsum.contiguous();
    auto states_cont = states.contiguous(); // No permute needed, states is already (b,c,h,n,p)
    const auto B_size = C_cont.size(0), C_size = C_cont.size(1), L_size = C_cont.size(2);
    const auto H_size = C_cont.size(3), N_size = C_cont.size(4);
    const auto P_size = states_cont.size(4);
    auto Y_off = torch::empty({B_size, C_size, L_size, H_size, P_size}, C_cont.options());
    TORCH_CHECK(P_size % 4 == 0, "P_size must be divisible by 4 for vectorization.");
    const dim3 grid_dim(B_size * C_size * L_size);
    const dim3 block_dim(P_size / 4, H_size);
    TORCH_CHECK((P_size / 4 * H_size) <= 1024, "Block size exceeds 1024");
    fused_Y_off_coalesced_kernel<<<grid_dim, block_dim>>>(C_cont.data_ptr<float>(), states_cont.data_ptr<float>(), A_cumsum_cont.data_ptr<float>(), Y_off.data_ptr<float>(), B_size, C_size, L_size, H_size, P_size, N_size);
    return Y_off;
}
"""

mamba_ops_cpp_source = """
torch::Tensor fused_mamba_diag(torch::Tensor A_cumsum, torch::Tensor B, torch::Tensor C, torch::Tensor X);
torch::Tensor fused_state_computation(torch::Tensor B, torch::Tensor A_cumsum, torch::Tensor X);
torch::Tensor fused_inter_chunk(torch::Tensor A_last_padded, torch::Tensor states);
torch::Tensor fused_Y_off_coalesced(torch::Tensor C, torch::Tensor states, torch::Tensor A_cumsum);
"""

mamba_ops = load_inline(
    name="mamba_ops_v3",
    cpp_sources=mamba_ops_cpp_source,
    cuda_sources=mamba_ops_source,
    functions=["fused_mamba_diag", "fused_state_computation", "fused_inter_chunk", "fused_Y_off_coalesced"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
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
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        
    def forward(self, X, initial_states=None):
        # Rearrange into blocks/chunks and ensure contiguous memory layout for kernel calls
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        A_blocks_permuted = rearrange(A_blocks, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks_permuted, dim=-1)
        
        # 1. Compute diagonal block outputs with a single, highly-fused kernel
        Y_diag = mamba_ops.fused_mamba_diag(A_cumsum, B_blocks, C_blocks, X_blocks)
        
        # 2. Compute intra-chunk states with a fused, vectorized kernel producing the optimal layout
        states = mamba_ops.fused_state_computation(B_blocks, A_cumsum, X_blocks)
        
        # 3. Compute inter-chunk recurrence with a fully-fused, vectorized kernel
        if initial_states is None:
            # Zeros must be created with the new layout (b,c,h,n,p)
            initial_states = torch.zeros_like(states[:, :1])
        states_cat = torch.cat([initial_states, states], dim=1)
        
        A_last_padded = F.pad(A_cumsum[:, :, :, -1], (1, 0))
        
        new_states = mamba_ops.fused_inter_chunk(A_last_padded, states_cat)
        states = new_states[:, :-1]
        
        # 4. Compute state-to-output conversion. No permute is needed due to layout optimization.
        Y_off = mamba_ops.fused_Y_off_coalesced(C_blocks, states, A_cumsum)
        
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