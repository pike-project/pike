import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# --- Triton Kernel for Fused GEMM and Bias Addition (FP16 optimized) ---
# This kernel is highly efficient for the fused 1x1 convolutions. It takes a (C, NHW) formatted
# input and weight matrix to produce a (C_out, NHW) output.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    A, B, C, bias, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, ADD_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m, num_pid_n = tl.cdiv(M, BLOCK_SIZE_M), tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    if ADD_BIAS:
        bias_ptrs = bias + offs_am
        bias_vals = tl.load(bias_ptrs, mask=offs_am < M, other=0.0)
        accumulator = accumulator + bias_vals[:, None]
        
    c_ptrs = C + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(C.dtype.element_ty), mask=c_mask)


# --- Triton Kernel for Fused MaxPool 3x3 (s1, p1) + 1x1 Convolution ---
# This kernel reads NCHW input, performs the maxpool operation on-the-fly, and multiplies
# by the 1x1 conv weights, writing the output to a (C_out, NHW) buffer. This avoids
# materializing the intermediate pooled tensor.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N_gemm', 'K'],
)
@triton.jit
def maxpool_conv1x1_bias_kernel(
    X, W_ptr, B_ptr, Y,
    N_dim, C_in, H_dim, W_dim, C_out,
    stride_xn, stride_xc, stride_xh, stride_xw, stride_ym, stride_yn, stride_wc, stride_wk,
    M: tl.constexpr, N_gemm: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, ADD_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m, num_pid_n = tl.cdiv(M, BLOCK_SIZE_M), tl.cdiv(N_gemm, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = W_ptr + (offs_am[:, None] * stride_wc + offs_k[None, :] * stride_wk)

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_n_dim = offs_bn // (H_dim * W_dim)
    offs_h_dim = (offs_bn % (H_dim * W_dim)) // W_dim
    offs_w_dim = offs_bn % W_dim

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_channels = (k_start * BLOCK_SIZE_K) + offs_k
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (k_channels[None, :] < K), other=0.0)
        max_val = tl.full((BLOCK_SIZE_K, BLOCK_SIZE_N), value=-float('inf'), dtype=X.dtype.element_ty)
        for dh in range(0, 3):
            for dw in range(0, 3):
                h_, w_ = offs_h_dim[None, :] + dh - 1, offs_w_dim[None, :] + dw - 1
                mask_hw = (h_ >= 0) & (h_ < H_dim) & (w_ >= 0) & (w_ < W_dim)
                load_mask = (k_channels[:, None] < K) & (offs_bn[None, :] < N_gemm) & mask_hw
                x_ptrs = X + (offs_n_dim[None, :] * stride_xn + k_channels[:, None] * stride_xc + h_ * stride_xh + w_ * stride_xw)
                current_vals = tl.load(x_ptrs, mask=load_mask, other=-float('inf'))
                max_val = tl.maximum(max_val, current_vals)
        accumulator += tl.dot(a, max_val)
        a_ptrs += BLOCK_SIZE_K * stride_wk

    if ADD_BIAS:
        bias_ptrs = B_ptr + offs_am
        bias_vals = tl.load(bias_ptrs, mask=offs_am < M, other=0.0)
        accumulator += bias_vals[:, None]

    c_ptrs = Y + offs_am[:, None] * stride_ym + offs_bn[None, :] * stride_yn
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N_gemm)
    tl.store(c_ptrs, accumulator.to(Y.dtype.element_ty), mask=c_mask)

# --- Python Wrappers for Triton Kernels ---

def conv1x1_bias_triton_gemm_inplace(x_gemm, weight, bias, out_tensor):
    M, K = weight.shape
    _, N_gemm = x_gemm.shape
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N_gemm, META['BLOCK_SIZE_N']),)
    gemm_bias_kernel[grid](
        weight, x_gemm, out_tensor, bias,
        M, N_gemm, K,
        weight.stride(0), weight.stride(1),
        x_gemm.stride(0), x_gemm.stride(1),
        out_tensor.stride(0), out_tensor.stride(1),
        ADD_BIAS=(bias is not None)
    )

def maxpool_conv1x1_triton_forward_inplace(x, weight, bias, out_tensor):
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    weight_reshaped = weight.view(C_out, C_in)
    M, K, N_gemm = C_out, C_in, N * H * W
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N_gemm, META['BLOCK_SIZE_N']),)
    out_stride_c, out_stride_nhw = out_tensor.stride()
    maxpool_conv1x1_bias_kernel[grid](
        x, weight_reshaped, bias, out_tensor,
        N, C_in, H, W, C_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out_stride_c, out_stride_nhw,
        weight_reshaped.stride(0), weight_reshaped.stride(1), M, N_gemm, K,
        ADD_BIAS=(bias is not None)
    )

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(ModelNew, self).__init__()
        
        self.out_1x1, self.out_3x3, self.out_5x5, self.pool_proj = out_1x1, out_3x3, out_5x5, pool_proj
        
        # To ensure correctness, we must replicate the exact weight initialization sequence
        # of the original model. We create temporary layers in the same order as the original
        # nn.Module to capture their initialized weights, assuming a fixed random seed.
        
        # Original order: branch1x1, branch3x3 (reduce, conv), branch5x5 (reduce, conv), branch_pool (conv)
        _branch1x1_conv = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        _branch3x3_reduce = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        _branch3x3_conv = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        _branch5x5_reduce = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        _branch5x5_conv = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        _pool_conv = nn.Conv2d(in_channels, pool_proj, kernel_size=1)

        # Now, create the optimized layers and parameters, populating them with the captured weights.
        
        # Fuse the first 1x1 convolutions from three branches for a single large GEMM
        self.fused_1x1_weight = nn.Parameter(torch.cat([
            _branch1x1_conv.weight, _branch3x3_reduce.weight, _branch5x5_reduce.weight
        ], dim=0).squeeze().half())
        self.fused_1x1_bias = nn.Parameter(torch.cat([
            _branch1x1_conv.bias, _branch3x3_reduce.bias, _branch5x5_reduce.bias
        ], dim=0).half())
        self.split_sizes = [out_1x1, reduce_3x3, reduce_5x5]
        
        # Keep 3x3 and 5x5 convs as standard PyTorch modules to leverage cuDNN
        self.conv3x3 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        
        # Initialize them with the weights from the corresponding temporary layers
        self.conv3x3.weight.data = _branch3x3_conv.weight.half()
        self.conv3x3.bias.data = _branch3x3_conv.bias.half()
        self.conv5x5.weight.data = _branch5x5_conv.weight.half()
        self.conv5x5.bias.data = _branch5x5_conv.bias.half()

        # Store weights for the pooling branch's custom kernel
        self.pool_weight = nn.Parameter(_pool_conv.weight.half())
        self.pool_bias = nn.Parameter(_pool_conv.bias.half())
    
    def forward(self, x):
        N, C_in, H, W = x.shape
        x_fp16 = x.half()
        
        # --- Stage 1: Pre-allocation and Output Views ---
        total_out_channels = self.out_1x1 + self.out_3x3 + self.out_5x5 + self.pool_proj
        # Allocate the final output buffer in the efficient (C, NHW) format
        output_gemm = torch.empty((total_out_channels, N * H * W), device=x.device, dtype=torch.float16)

        # Create views into the output buffer for each branch to write results in-place
        c_offset = 0
        branch1_out_slice = output_gemm[c_offset:c_offset + self.out_1x1, :]
        c_offset += self.out_1x1
        branch3_out_slice = output_gemm[c_offset:c_offset + self.out_3x3, :]
        c_offset += self.out_3x3
        branch5_out_slice = output_gemm[c_offset:c_offset + self.out_5x5, :]
        c_offset += self.out_5x5
        pool_out_slice = output_gemm[c_offset:, :]

        # --- Stage 2: Execute All Branches ---

        # Branch Pool: Use the fused maxpool+conv1x1 kernel. It efficiently reads NCHW input
        # and writes directly to its slice in the (C, NHW) output buffer.
        maxpool_conv1x1_triton_forward_inplace(x_fp16, self.pool_weight, self.pool_bias, pool_out_slice)
        
        # Reshape input from NCHW to (C, NHW) for the main GEMM operation.
        x_gemm = x_fp16.view(N, C_in, H * W).permute(1, 0, 2).reshape(C_in, N * H * W)
        
        # Fused 1x1 convolutions for branches 1, 3, and 5
        fused_1x1_out_channels = self.fused_1x1_weight.shape[0]
        intermediate_gemm = torch.empty((fused_1x1_out_channels, N * H * W), device=x.device, dtype=torch.float16)
        conv1x1_bias_triton_gemm_inplace(x_gemm, self.fused_1x1_weight, self.fused_1x1_bias, intermediate_gemm)
        
        # Split the intermediate tensor into views for each branch (no data is copied)
        branch1_out_gemm, branch3_in_gemm, branch5_in_gemm = torch.split(intermediate_gemm, self.split_sizes, dim=0)
        
        # Branch 1: The result is ready, just copy it to the final output buffer slice.
        branch1_out_slice.copy_(branch1_out_gemm)
        
        # Branch 3: Reshape intermediate from (C,NHW) to NCHW, apply cuDNN-backed Conv, reshape back, and copy.
        reduce_3x3 = self.split_sizes[1]
        branch3_in_nchw = branch3_in_gemm.view(reduce_3x3, N, H, W).permute(1, 0, 2, 3)
        branch3_out_nchw = self.conv3x3(branch3_in_nchw)
        branch3_out_slice.copy_(branch3_out_nchw.permute(1, 0, 2, 3).reshape(self.out_3x3, N * H * W))

        # Branch 5: Same logic as Branch 3.
        reduce_5x5 = self.split_sizes[2]
        branch5_in_nchw = branch5_in_gemm.view(reduce_5x5, N, H, W).permute(1, 0, 2, 3)
        branch5_out_nchw = self.conv5x5(branch5_in_nchw)
        branch5_out_slice.copy_(branch5_out_nchw.permute(1, 0, 2, 3).reshape(self.out_5x5, N * H * W))

        # --- Stage 3: Final Reshape ---
        # Reshape the completed (C, NHW) buffer back to NCHW and cast to float32
        return output_gemm.view(total_out_channels, N, H, W).permute(1, 0, 2, 3).float()

# --- Test code (provided in the problem description) ---
in_channels = 480
out_1x1 = 192
reduce_3x3 = 96
out_3x3 = 208
reduce_5x5 = 16
out_5x5 = 48
pool_proj = 64
batch_size = 10
height = 224
width = 224

def get_inputs():
    # Model expects float32 input and handles half-precision internally
    return [torch.randn(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]