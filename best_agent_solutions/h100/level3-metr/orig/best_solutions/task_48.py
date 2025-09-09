import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for the selective scan operation
# Corrected kernel logic for state update
selective_scan_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <ATen/cuda/CUDAContext.h> // For AT_CUDA_CHECK
#include <c10/cuda/CUDAStream.h>   // For getCurrentCUDAStream

#define D_STATE 16 // d_state is fixed at 16 for this model, optimal for unrolling

__global__ void selective_scan_fwd_kernel(
    const float* A, const float* B, const float* C, const float* X, float* Y,
    int B_DIM, int C_CHUNKS, int L, int H, int P
) {
    // Each block handles a single scan sequence (b, c, h)
    // Each thread in the block handles a single feature in d_head (p)
    const int p = threadIdx.x;
    const int block_idx = blockIdx.x;
    
    // De-flatten block_idx to get (b, c, h)
    const int h = block_idx % H;
    const int c = (block_idx / H) % C_CHUNKS;
    const int b = block_idx / (H * C_CHUNKS);

    // Per-thread state array of size d_state
    float state[D_STATE];
    #pragma unroll
    for (int i = 0; i < D_STATE; ++i) {
        state[i] = 0.0f;
    }

    // Loop over the sequence length L within the block
    for (int s = 0; s < L; ++s) {
        // Pre-calculate base indices for the current step 's'
        // A is (b, h, c, l)
        long a_idx = (long)b * H * C_CHUNKS * L + (long)h * C_CHUNKS * L + (long)c * L + s;
        // B, C are (b, c, l, h, n)
        long bc_base_idx = (long)b * C_CHUNKS * L * H * D_STATE + (long)c * L * H * D_STATE + (long)s * H * D_STATE + (long)h * D_STATE;
        // X, Y are (b, c, l, h, p)
        long xp_base_idx = (long)b * C_CHUNKS * L * H * P + (long)c * L * H * P + (long)s * H * P + (long)h * P;

        // Load inputs for the current step
        const float a_val = A[a_idx];
        const float x_val = X[xp_base_idx + p];

        // Use CUDA's fast math intrinsic for exp
        const float exp_a = __expf(a_val);

        float y_val = 0.0f;
        
        // Recurrent state update and output calculation
        #pragma unroll
        for (int n = 0; n < D_STATE; ++n) {
            // Corrected update rule to match parallel scan logic: h_t = exp(A_t) * h_{t-1} + B_t * x_t
            // The previous version incorrectly multiplied exp(A_t) with the B_t * x_t term.
            state[n] = exp_a * state[n] + B[bc_base_idx + n] * x_val;
            // Compute output: y_t = C_t * h_t
            y_val += C[bc_base_idx + n] * state[n];
        }
        
        // Write output for the current step
        Y[xp_base_idx + p] = y_val;
    }
}

torch::Tensor selective_scan_fwd(
    torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor X
) {
    // Ensure inputs are contiguous. This is crucial for kernel performance and correctness.
    A = A.contiguous();
    B = B.contiguous();
    C = C.contiguous();
    X = X.contiguous();
    
    // Input shapes
    // A: (b, h, c, l)
    // B: (b, c, l, h, n)
    // C: (b, c, l, h, n)
    // X: (b, c, l, h, p)
    
    const auto b_dim = X.size(0);
    const auto c_chunks = X.size(1);
    const auto l = X.size(2);
    const auto h = X.size(3);
    const auto p = X.size(4); // d_head
    const auto n = B.size(4); // d_state

    // Dimension checks
    TORCH_CHECK(n == D_STATE, "d_state must be equal to the compiled D_STATE");
    // The following check used blockDim.x, which is a CUDA device variable and cannot be accessed from host code.
    // It has been removed as it was causing the "undefined symbol" error.
    // The logic is implicitly enforced by setting block.x = p.
    TORCH_CHECK(A.dim() == 4 && A.size(0) == b_dim && A.size(1) == h && A.size(2) == c_chunks && A.size(3) == l, "A tensor dimension mismatch");
    TORCH_CHECK(B.dim() == 5 && B.size(0) == b_dim && B.size(1) == c_chunks && B.size(2) == l && B.size(3) == h && B.size(4) == n, "B tensor dimension mismatch");
    TORCH_CHECK(C.dim() == 5 && C.size(0) == b_dim && C.size(1) == c_chunks && C.size(2) == l && C.size(3) == h && C.size(4) == n, "C tensor dimension mismatch");

    // Output tensor
    auto Y = torch::empty_like(X);

    // CUDA launch configuration
    // Grid dimension: one block per (b, c, h) tuple
    dim3 grid(b_dim * c_chunks * h);
    // Block dimension: one thread per element in d_head (p)
    dim3 block(p);

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    selective_scan_fwd_kernel<<<grid, block, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), X.data_ptr<float>(), Y.data_ptr<float>(),
        b_dim, c_chunks, l, h, p
    );
    
    // Check for any CUDA errors
    AT_CUDA_CHECK(cudaGetLastError());

    return Y;
}
"""

selective_scan_cpp_source = (
    "torch::Tensor selective_scan_fwd(torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor X);"
)

# JIT compile the inline CUDA code.
# This will be done once when the Python module is first imported.
selective_scan_cuda = load_inline(
    name="selective_scan_cuda",
    cpp_sources=selective_scan_cpp_source,
    cuda_sources=selective_scan_source,
    functions=["selective_scan_fwd"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        """
        Mamba Structured State Space model with custom CUDA kernel for selective scan.
        
        :param batch_size: Size of the batch
        :param seq_length: Length of the input sequence
        :param n_heads: Number of attention heads
        :param d_head: Dimension of each head
        :param d_state: Dimension of the state space
        :param block_len: Length of each block for chunked computation
        """
        super(Model, self).__init__()
        
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"
        assert d_state == 16, "This custom kernel is hardcoded for d_state=16"
        assert d_head <= 1024, "d_head must be <= 1024 for CUDA block size"


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

        # Store the JIT-compiled CUDA function
        self.selective_scan_fwd = selective_scan_cuda.selective_scan_fwd

    def segsum(self, x):
        """Naive segment sum calculation (kept for other parts of the model)."""
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum
    
    def forward(self, X, initial_states=None):
        """
        Forward pass implementing the SSD operation with the custom CUDA kernel.
        
        :param X: Input tensor of shape (batch, length, n_heads, d_head)
        :param initial_states: Optional initial states
        :return: Output tensor Y
        """
        # Rearrange into blocks/chunks
        X_blocks, A_blocks_untransposed, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]
        
        # Correctly transpose A_blocks to (b, h, c, l) for all subsequent calculations,
        # matching the original model's logic and fixing the einsum shape mismatch.
        A_blocks = rearrange(A_blocks_untransposed, "b c l h -> b h c l")
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        
        # 1. Compute diagonal block outputs using the fused selective scan kernel
        # Kernel expects A: (b, h, c, l), B: (b, c, l, h, n), C: (b, c, l, h, n), X: (b, c, l, h, p)
        Y_diag = self.selective_scan_fwd(A_blocks, B_blocks, C_blocks, X_blocks)
        
        # 2. Compute intra-chunk states
        # A_cumsum now has shape (b, h, c, l), which matches the 'bhcl' in the einsum.
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", 
                            B_blocks, decay_states, X_blocks)
        
        # 3. Compute inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        
        # A_cumsum[:, :, :, -1] has shape (b, h, c)
        decay_chunk = torch.exp(self.segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]
        
        # 4. Compute state-to-output conversion
        # A_cumsum now has shape (b, h, c, l), which matches the 'bhcl' in the einsum.
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', 
                           C_blocks, states, state_decay_out)
        
        # Combine diagonal and off-diagonal terms
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        
        return Y