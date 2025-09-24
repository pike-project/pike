import torch
import torch.nn as nn
import math
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # Specialized configs for small M, N and medium K (K=256 for each GEMM)
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 1}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K_TOTAL'],
)
@triton.jit
def fused_cat_gemm_bias_kernel(
    a_ptr, b_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K_TOTAL,
    stride_a_m, stride_a_k,
    stride_b_m, stride_b_k,
    stride_w_k, stride_w_n,
    stride_out_m, stride_out_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    """
    Triton kernel for a fused operation equivalent to:
    `output = torch.cat((A, B), dim=1) @ W + bias`
    This is implemented as `A @ W_A + B @ W_B + bias` to avoid materializing the concatenation.
    A: Final forward hidden state, shape (M, K_TOTAL/2)
    B: Final backward hidden state, shape (M, K_TOTAL/2)
    W: Weight matrix, shape (K_TOTAL, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    K_HALF = K_TOTAL // 2

    # --- First GEMM: A @ W[:K_HALF, :] ---
    for k_base in range(0, K_HALF, BLOCK_SIZE_K):
        offs_k = k_base + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k)
        wA_ptrs = weight_ptr + (offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
        
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K_HALF), other=0.0)
        wA = tl.load(wA_ptrs, mask=(offs_k[:, None] < K_HALF) & (offs_n[None, :] < N), other=0.0)
        
        accumulator += tl.dot(a, wA)

    # --- Second GEMM: B @ W[K_HALF:, :] ---
    for k_base in range(0, K_HALF, BLOCK_SIZE_K):
        offs_k = k_base + tl.arange(0, BLOCK_SIZE_K)
        b_ptrs = b_ptr + (offs_m[:, None] * stride_b_m + offs_k[None, :] * stride_b_k)
        offs_wk_b = K_HALF + offs_k
        wB_ptrs = weight_ptr + (offs_wk_b[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
        
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K_HALF), other=0.0)
        wB = tl.load(wB_ptrs, mask=(offs_wk_b[:, None] < K_TOTAL) & (offs_n[None, :] < N), other=0.0)
        
        accumulator += tl.dot(b, wB)

    # Add bias and store
    bias_ptrs = bias_ptr + offs_n
    bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
    accumulator = accumulator + bias[None, :].to(tl.float32)

    output_ptrs = output_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, accumulator.to(output_ptr.dtype.element_ty), mask=output_mask)


def fused_cat_gemm_bias(h_forward: torch.Tensor, h_backward: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor, output_dtype: torch.dtype):
    assert h_forward.is_cuda and h_backward.is_cuda and weight_t.is_cuda and bias.is_cuda
    # Sliced tensors might not be contiguous, but our kernel can handle non-contiguous inputs via strides.
    # However, for simplicity and potential performance benefits, we'll enforce contiguous inputs.
    h_forward = h_forward.contiguous()
    h_backward = h_backward.contiguous()
    
    M, K_HALF = h_forward.shape
    K_TOTAL, N = weight_t.shape
    assert K_TOTAL == K_HALF * 2, f"K_TOTAL {K_TOTAL} must be twice K_HALF {K_HALF}"
    
    output = torch.empty((M, N), device=h_forward.device, dtype=output_dtype)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    fused_cat_gemm_bias_kernel[grid](
        h_forward, h_backward, weight_t, bias, output,
        M, N, K_TOTAL,
        h_forward.stride(0), h_forward.stride(1),
        h_backward.stride(0), h_backward.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        output.stride(0), output.stride(1),
    )
    return output


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.compute_dtype = torch.float16
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.lstm.to(self.compute_dtype)
        # It's good practice to flatten parameters after moving the model to a new device or dtype.
        self.lstm.flatten_parameters()

        # Initialize linear layer parameters manually for custom kernel
        in_features = hidden_size * 2
        tmp_weight = torch.empty(output_size, in_features, dtype=torch.float32)
        tmp_bias = torch.empty(output_size, dtype=torch.float32)
        nn.init.kaiming_uniform_(tmp_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tmp_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(tmp_bias, -bound, bound)

        # Store weight transposed for coalesced memory access in the kernel
        self.fc_weight_t = nn.Parameter(tmp_weight.T.contiguous().to(self.compute_dtype))
        self.fc_bias = nn.Parameter(tmp_bias.to(self.compute_dtype))

        # CUDA Graph parameters
        self.graph = None
        self.static_x = None
        self.static_h0 = None
        self.static_c0 = None
        self.static_out = None

    def _graph_capture(self, x, h0, c0, output_dtype):
        self.static_x = torch.empty_like(x)
        self.static_h0 = torch.empty_like(h0)
        self.static_c0 = torch.empty_like(c0)

        # Warm-up run
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                # Correctness fix: Use the LSTM's `out` tensor to get the hidden states
                # from the last time step, which matches the original model's behavior.
                out_warmup, _ = self.lstm(self.static_x, (self.static_h0, self.static_c0))
                last_timestep_out = out_warmup[:, -1, :]
                h_fwd = last_timestep_out[:, :self.hidden_size]
                h_bwd = last_timestep_out[:, self.hidden_size:]
                _ = fused_cat_gemm_bias(h_fwd, h_bwd, self.fc_weight_t, self.fc_bias, output_dtype)
        torch.cuda.current_stream().wait_stream(s)

        # Graph capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            # Correctness fix: The captured graph must also replicate the correct logic.
            out_captured, _ = self.lstm(self.static_x, (self.static_h0, self.static_c0))
            last_timestep_out = out_captured[:, -1, :]
            
            # The input to the custom kernel is a concatenation of the forward and backward
            # hidden states from the last layer at the final time step.
            h_fwd_captured = last_timestep_out[:, :self.hidden_size]
            h_bwd_captured = last_timestep_out[:, self.hidden_size:]
            
            # Call the fused kernel on the correctly sliced hidden states.
            self.static_out = fused_cat_gemm_bias(h_fwd_captured, h_bwd_captured, self.fc_weight_t, self.fc_bias, output_dtype)

    def forward(self, x, h0, c0):
        original_dtype = x.dtype
        x_casted = x.to(self.compute_dtype)
        h0_casted = h0.to(self.compute_dtype)
        c0_casted = c0.to(self.compute_dtype)

        # Re-capture graph if input shape changes
        if self.graph is None or self.static_x.shape != x.shape:
            self._graph_capture(x_casted, h0_casted, c0_casted, original_dtype)

        # Copy current inputs to static tensors for the graph
        self.static_x.copy_(x_casted)
        self.static_h0.copy_(h0_casted)
        self.static_c0.copy_(c0_casted)

        # Replay the captured graph
        self.graph.replay()
        
        return self.static_out