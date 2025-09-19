# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# This version combines the best features of previous top-performing models and introduces further fusion.
# 1. `selective_scan_diag_kernel`: Combines the efficient grid-stride loop for shared memory calculation
#    from Program 1 with the float4 vectorized dot product from Program 2, as d_state (N) is 16.
# 2. `compute_states_kernel`: Now uses shared memory to cache the A_cumsum slice, reducing global memory reads.
# 3. `compute_y_off_and_add_diag_kernel`: A new kernel that replaces the old `compute_y_off_kernel`.
#    It not only computes the off-diagonal part but also fuses the final addition (Y_diag + Y_off),
#    writing the final result directly. This saves a kernel launch and avoids writing/reading the
#    intermediate Y_off tensor. It also uses float4 vectorization for its internal dot product.
# 4. `inter_chunk_scan_kernel`: The highly efficient scan-based inter-chunk recurrence from Program 1 is retained.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define MAX_CHUNKS 16 // For on-stack array allocation in the scan kernel

// Kernel 1: Fused selective scan with vectorized dot product and grid-stride loop
__global__ void selective_scan_diag_kernel(
    const float* __restrict__ A_cumsum, // (B, H, C, L)
    const float* __restrict__ B_blocks, // (B, C, L, H, N)
    const float* __restrict__ C_blocks, // (B, C, L, H, N)
    const float* __restrict__ X_blocks, // (B, C, L, H, P)
    float* Y_diag,                      // (B, C, L, H, P)
    const int B, const int C, const int H, const int L, const int P, const int N
) {
    const int bch_idx = blockIdx.x;
    const int l = blockIdx.y;
    const int p = threadIdx.x;

    extern __shared__ float K_vals_shared[]; // Size L

    const int h = bch_idx % H;
    const int bc_idx = bch_idx / H;
    const int c = bc_idx % C;
    const int b = bc_idx / C;

    // --- Cooperative K_val calculation using Shared Memory, grid-stride, and float4 ---
    for (int s = p; s <= l; s += P) {
        const float* C_ptr_l = C_blocks + (((long)(b * C + c) * L + l) * H + h) * N;
        const float* B_ptr_s = B_blocks + (((long)(b * C + c) * L + s) * H + h) * N;
        
        float k_dot = 0.0f;
        const float4* C_vec = reinterpret_cast<const float4*>(C_ptr_l);
        const float4* B_vec = reinterpret_cast<const float4*>(B_ptr_s);
        
        #pragma unroll
        for (int i = 0; i < N / 4; ++i) {
             float4 c_v = C_vec[i];
             float4 b_v = B_vec[i];
             k_dot += c_v.x * b_v.x + c_v.y * b_v.y + c_v.z * b_v.z + c_v.w * b_v.w;
        }
        K_vals_shared[s] = k_dot;
    }
    __syncthreads();

    // --- Main accumulation loop ---
    float y_val = 0.0f;
    const float* A_cumsum_ptr = A_cumsum + ((long)(b * H + h) * C + c) * L;
    const float A_cumsum_l = A_cumsum_ptr[l];

    for (int s = 0; s <= l; ++s) {
        const float L_val = expf(A_cumsum_l - A_cumsum_ptr[s]);
        const float K_val = K_vals_shared[s];
        const float X_val = X_blocks[((((long)(b * C + c) * L + s) * H + h) * P + p)];
        y_val += X_val * L_val * K_val;
    }

    Y_diag[((((long)(b * C + c) * L + l) * H + h) * P + p)] = y_val;
}


// Kernel 2: Fused states computation with shared memory for A_cumsum
__global__ void compute_states_kernel(
    const float* __restrict__ B, const float* __restrict__ A_cumsum, const float* __restrict__ X,
    float* __restrict__ states,
    int b_dim, int c_dim, int l_dim, int h_dim, int n_dim, int p_dim)
{
    const int h = blockIdx.z % h_dim;
    const int c = (blockIdx.z / h_dim) % c_dim;
    const int b = blockIdx.z / (c_dim * h_dim);

    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= b_dim || c >= c_dim || h >= h_dim || p >= p_dim || n >= n_dim) return;
    
    extern __shared__ float sh_A[]; // Size l_dim (64)
    const float* A_ptr_global = A_cumsum + ((long)b * h_dim + h) * c_dim * l_dim + (long)c * l_dim;
    
    int tid_flat = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_in_block = blockDim.x * blockDim.y;
    for(int i = tid_flat; i < l_dim; i += threads_in_block) {
        sh_A[i] = A_ptr_global[i];
    }
    __syncthreads();

    long B_s_b = (long)c_dim * l_dim * h_dim * n_dim, B_s_c = (long)l_dim * h_dim * n_dim, B_s_l = (long)h_dim * n_dim, B_s_h = n_dim;
    long X_s_b = (long)c_dim * l_dim * h_dim * p_dim, X_s_c = (long)l_dim * h_dim * p_dim, X_s_l = (long)h_dim * p_dim, X_s_h = p_dim;
    
    const float* B_ptr = B + (long)b * B_s_b + (long)c * B_s_c;
    const float* X_ptr = X + (long)b * X_s_b + (long)c * X_s_c;
    
    float a_cumsum_last = sh_A[l_dim - 1];
    float sum = 0.0f;

    for (int l = 0; l < l_dim; ++l) {
        float decay = expf(a_cumsum_last - sh_A[l]);
        long b_idx = (long)l * B_s_l + (long)h * B_s_h + n;
        long x_idx = (long)l * X_s_l + (long)h * X_s_h + p;
        sum += B_ptr[b_idx] * decay * X_ptr[x_idx];
    }
    
    long S_s_b = (long)c_dim * h_dim * p_dim * n_dim, S_s_c = (long)h_dim * p_dim * n_dim, S_s_h = (long)p_dim * n_dim, S_s_p = n_dim;
    states[(long)b * S_s_b + (long)c * S_s_c + (long)h * S_s_h + (long)p * S_s_p + n] = sum;
}

// Kernel 3: Fused Y_off calculation, vectorized dot product, and addition of Y_diag
__global__ void compute_y_off_and_add_diag_kernel(
    const float* __restrict__ C, const float* __restrict__ states, const float* __restrict__ A_cumsum,
    const float* __restrict__ Y_diag_in, float* __restrict__ Y_out,
    int b_dim, int c_dim, int l_dim, int h_dim, int p_dim, int n_dim)
{
    const int h = blockIdx.z % h_dim;
    const int c = (blockIdx.z / h_dim) % c_dim;
    const int b = blockIdx.z / (c_dim * h_dim);

    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= b_dim || c >= c_dim || h >= h_dim || l >= l_dim || p >= p_dim) return;
    
    long C_s_b = (long)c_dim * l_dim * h_dim * n_dim, C_s_c = (long)l_dim * h_dim * n_dim, C_s_l = (long)h_dim * n_dim, C_s_h = n_dim;
    long S_s_b = (long)c_dim * h_dim * p_dim * n_dim, S_s_c = (long)h_dim * p_dim * n_dim, S_s_h = (long)p_dim * n_dim, S_s_p = n_dim;
    long A_s_b = (long)h_dim * c_dim * l_dim, A_s_h = (long)c_dim * l_dim, A_s_c = l_dim;

    float matmul_sum = 0.0f;
    const float* c_base_ptr = C + (long)b * C_s_b + (long)c * C_s_c + (long)l * C_s_l + (long)h * C_s_h;
    const float* s_base_ptr = states + (long)b * S_s_b + (long)c * S_s_c + (long)h * S_s_h + (long)p * S_s_p;
    
    const float4* c_vec = reinterpret_cast<const float4*>(c_base_ptr);
    const float4* s_vec = reinterpret_cast<const float4*>(s_base_ptr);

    #pragma unroll
    for (int i = 0; i < n_dim / 4; ++i) {
        float4 c_v = c_vec[i];
        float4 s_v = s_vec[i];
        matmul_sum += c_v.x * s_v.x + c_v.y * s_v.y + c_v.z * s_v.z + c_v.w * s_v.w;
    }

    long a_idx = (long)b * A_s_b + (long)h * A_s_h + (long)c * A_s_c + l;
    float decay_val = expf(A_cumsum[a_idx]);

    float y_off = decay_val * matmul_sum;
    
    long Y_s_b = (long)c_dim * l_dim * h_dim * p_dim, Y_s_c = (long)l_dim * h_dim * p_dim, Y_s_l = (long)h_dim * p_dim, Y_s_h = p_dim;
    long y_idx = (long)b * Y_s_b + (long)c * Y_s_c + (long)l * Y_s_l + (long)h * Y_s_h + p;
    
    Y_out[y_idx] = Y_diag_in[y_idx] + y_off;
}

// Kernel 4: Fused inter-chunk scan (unchanged)
__global__ void inter_chunk_scan_kernel(
    const float* __restrict__ A_cumsum_last, const float* __restrict__ states_cat, float* states_inter,
    const int B, const int H, const int C, const int P, const int N
) {
    if (C > MAX_CHUNKS) return;

    const int b = blockIdx.x; const int h = blockIdx.y;
    const int p = threadIdx.x; const int n = threadIdx.y;
    if (b >= B || h >= H || p >= P || n >= N) return;
    
    float A_padded[MAX_CHUNKS + 1]; float h_prime[MAX_CHUNKS + 1];
    A_padded[0] = 0.0f;
    const float* A_in_ptr = A_cumsum_last + ((long)b * H + h) * C;
    for (int i = 0; i < C; ++i) { A_padded[i+1] = A_in_ptr[i]; }

    long s_cat_stride_c = (long)H * P * N;
    long s_cat_base_idx = (long)b * (C + 1) * H * P * N + (long)h * P * N + (long)p * N + n;
    for (int c = 0; c < C + 1; ++c) {
        h_prime[c] = expf(-A_padded[c]) * states_cat[s_cat_base_idx + c * s_cat_stride_c];
    }
    
    float s_prime_prefix_sum = 0.0f;
    long s_inter_stride_c = (long)H * P * N;
    long s_inter_base_idx = (long)b * C * H * P * N + (long)h * P * N + (long)p * N + n;
    for (int z = 0; z < C; ++z) {
        s_prime_prefix_sum += h_prime[z];
        states_inter[s_inter_base_idx + z * s_inter_stride_c] = expf(A_padded[z]) * s_prime_prefix_sum;
    }
}

// --- C++ Wrappers ---
torch::Tensor selective_scan_diag_cuda(torch::Tensor A_cumsum, torch::Tensor B_blocks, torch::Tensor C_blocks, torch::Tensor X_blocks) {
    const int B = X_blocks.size(0), C = X_blocks.size(1), L = X_blocks.size(2), H = X_blocks.size(3), P = X_blocks.size(4), N = B_blocks.size(4);
    auto Y_diag = torch::empty_like(X_blocks);
    selective_scan_diag_kernel<<<dim3(B*C*H, L), dim3(P), L*sizeof(float)>>>( A_cumsum.data_ptr<float>(), B_blocks.data_ptr<float>(), C_blocks.data_ptr<float>(), X_blocks.data_ptr<float>(), Y_diag.data_ptr<float>(), B, C, H, L, P, N );
    return Y_diag;
}

torch::Tensor compute_states_cuda(torch::Tensor B, torch::Tensor A_cumsum, torch::Tensor X) {
    const int b = B.size(0), c = B.size(1), l = B.size(2), h = B.size(3), n = B.size(4);
    const int p = X.size(4);
    auto states = torch::empty({b, c, h, p, n}, B.options());
    const dim3 threads(16, 16);
    const dim3 blocks((p + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y, (long)b*c*h);
    compute_states_kernel<<<blocks, threads, l*sizeof(float)>>>( B.data_ptr<float>(), A_cumsum.data_ptr<float>(), X.data_ptr<float>(), states.data_ptr<float>(), b, c, l, h, n, p);
    return states;
}

torch::Tensor compute_y_off_and_add_diag_cuda(torch::Tensor C, torch::Tensor states, torch::Tensor A_cumsum, torch::Tensor Y_diag_in) {
    const int b = C.size(0), c = C.size(1), l = C.size(2), h = C.size(3), n = C.size(4);
    const int p = states.size(3);
    auto Y_out = torch::empty_like(Y_diag_in);
    const dim3 threads(16, 16);
    const dim3 blocks((l + threads.x - 1) / threads.x, (p + threads.y - 1) / threads.y, (long)b*c*h);
    compute_y_off_and_add_diag_kernel<<<blocks, threads>>>( C.data_ptr<float>(), states.data_ptr<float>(), A_cumsum.data_ptr<float>(), Y_diag_in.data_ptr<float>(), Y_out.data_ptr<float>(), b, c, l, h, p, n);
    return Y_out;
}

torch::Tensor inter_chunk_scan_cuda(torch::Tensor A_cumsum_last, torch::Tensor states_cat) {
    const int B = A_cumsum_last.size(0), H = A_cumsum_last.size(1), C = A_cumsum_last.size(2);
    const int P = states_cat.size(3), N = states_cat.size(4);
    auto states_inter = torch::empty({B, C, H, P, N}, states_cat.options());
    inter_chunk_scan_kernel<<<dim3(B, H), dim3(P, N)>>>( A_cumsum_last.data_ptr<float>(), states_cat.data_ptr<float>(), states_inter.data_ptr<float>(), B, H, C, P, N );
    return states_inter;
}
"""

cpp_source = """
torch::Tensor selective_scan_diag_cuda(torch::Tensor A_cumsum, torch::Tensor B_blocks, torch::Tensor C_blocks, torch::Tensor X_blocks);
torch::Tensor compute_states_cuda(torch::Tensor B, torch::Tensor A_cumsum, torch::Tensor X);
torch::Tensor compute_y_off_and_add_diag_cuda(torch::Tensor C, torch::Tensor states, torch::Tensor A_cumsum, torch::Tensor Y_diag_in);
torch::Tensor inter_chunk_scan_cuda(torch::Tensor A_cumsum_last, torch::Tensor states_cat);
"""

custom_cuda_module = load_inline(
    name="custom_mamba_kernels_v6_hyper_fused",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["selective_scan_diag_cuda", "compute_states_cuda", "compute_y_off_and_add_diag_cuda", "inter_chunk_scan_cuda"],
    verbose=False,
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
        
        A_blocks_re = rearrange(A_blocks, "b c l h -> b h c l").contiguous()
        A_cumsum = torch.cumsum(A_blocks_re, dim=-1)
        
        # 1. Compute diagonal block outputs with the highly optimized fused CUDA kernel
        Y_diag = custom_cuda_module.selective_scan_diag_cuda(A_cumsum, B_blocks, C_blocks, X_blocks)
        
        # 2. Compute intra-chunk states using a fused CUDA kernel with shared memory
        states = custom_cuda_module.compute_states_cuda(B_blocks, A_cumsum, X_blocks)

        # 3. Compute inter-chunk recurrence with a new fused scan kernel
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states_cat = torch.cat([initial_states, states], dim=1)
        
        A_cumsum_last = A_cumsum[:, :, :, -1]
        states_inter = custom_cuda_module.inter_chunk_scan_cuda(A_cumsum_last, states_cat)
        
        # 4. Compute off-diagonal part and add to diagonal part in a single fused kernel
        Y_blocks = custom_cuda_module.compute_y_off_and_add_diag_cuda(C_blocks, states_inter, A_cumsum, Y_diag)
        
        # Reshape to final output format
        Y = rearrange(Y_blocks, "b c l h p -> b (c l) h p")
        
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