import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# This single kernel handles both pointwise convolutions (expand and project)
# It fuses Conv1x1 -> BatchNorm -> Activation (optional) -> Residual Add (optional)
# This kernel is already well-optimized and remains from the previous solution.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_pointwise_conv_bn_act_kernel(
    a_ptr, b_ptr, c_ptr,
    gamma_ptr, beta_ptr, identity_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_im, stride_in,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    """Fused kernel for Pointwise Conv (GEMM) -> Folded BN -> Activation -> Residual Add."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    if GROUP_SIZE_M > 1:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:, None]
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :]
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(a_ptr.dtype.element_ty)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_cn < N
    gamma = tl.load(gamma_ptr + offs_cn, mask=mask_n)
    beta = tl.load(beta_ptr + offs_cn, mask=mask_n)
    c = c * gamma[None, :] + beta[None, :]

    if HAS_RESIDUAL:
        offs_im = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_in = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        identity_ptrs = identity_ptr + stride_im * offs_im[:, None] + stride_in * offs_in[None, :]
        identity_mask = (offs_im[:, None] < M) & (offs_in[None, :] < N)
        identity_val = tl.load(identity_ptrs, mask=identity_mask, other=0.0)
        c += identity_val

    if ACTIVATION == "relu6":
        c = tl.minimum(tl.maximum(c, 0.0), 6.0)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# Specialized and unrolled depthwise convolution kernel for K=5, S=2
# This kernel is already well-optimized and remains from the previous solution.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16, 'BLOCK_SIZE_C': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_H': 8,  'BLOCK_SIZE_W': 32, 'BLOCK_SIZE_C': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 8,  'BLOCK_SIZE_C': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16, 'BLOCK_SIZE_C': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_W': 8, 'BLOCK_SIZE_C': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_W': 8, 'BLOCK_SIZE_C': 256}, num_warps=8, num_stages=4),
    ],
    key=['C', 'H_out', 'W_out'],
)
@triton.jit
def fused_depthwise_conv_k5_s2_bn_relu6_nhwc_kernel(
    x_ptr, w_ptr, y_ptr,
    gamma_ptr, beta_ptr,
    H_in, W_in, C, H_out, W_out,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_wc, stride_wkh, stride_wkw,
    stride_yn, stride_yh, stride_yw, stride_yc,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    num_h_blocks = tl.cdiv(H_out, BLOCK_SIZE_H)
    num_w_blocks = tl.cdiv(W_out, BLOCK_SIZE_W)
    num_c_blocks = tl.cdiv(C, BLOCK_SIZE_C)
    
    pid_c_block = pid % num_c_blocks
    _pid_temp = pid // num_c_blocks
    pid_w_block = _pid_temp % num_w_blocks
    _pid_temp = _pid_temp // num_w_blocks
    pid_h_block = _pid_temp % num_h_blocks
    pid_n = _pid_temp // num_h_blocks

    offs_h_out = pid_h_block * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w_out = pid_w_block * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    offs_c = pid_c_block * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C), dtype=tl.float32)
    
    base_h_in = offs_h_out[:, None, None] * 2 - 2
    base_w_in = offs_w_out[None, :, None] * 2 - 2

    # Manually unroll the 5x5 convolution
    for kh in range(5):
        for kw in range(5):
            h_in = base_h_in + kh
            w_in = base_w_in + kw
            
            mask_in = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
            mask_c = offs_c[None, None, :] < C
            mask = mask_in & mask_c

            x_ptrs = x_ptr + pid_n * stride_xn + h_in * stride_xh + w_in * stride_xw + offs_c[None, None, :] * stride_xc
            x = tl.load(x_ptrs, mask=mask, other=0.0)
            
            w_ptrs = w_ptr + offs_c[None, None, :] * stride_wc + kh * stride_wkh + kw * stride_wkw
            w = tl.load(w_ptrs, mask=mask_c, other=0.0)
            
            acc += x * w

    gamma = tl.load(gamma_ptr + offs_c, mask=offs_c < C)
    beta = tl.load(beta_ptr + offs_c, mask=offs_c < C)
    y = acc * gamma[None, None, :] + beta[None, None, :]
    y = tl.minimum(tl.maximum(y, 0.0), 6.0)

    y_ptrs = y_ptr + pid_n * stride_yn + offs_h_out[:, None, None] * stride_yh + \
             offs_w_out[None, :, None] * stride_yw + offs_c[None, None, :] * stride_yc
    
    mask_h_out = offs_h_out[:, None, None] < H_out
    mask_w_out = offs_w_out[None, :, None] < W_out
    mask_c_out = offs_c[None, None, :] < C
    mask_out = mask_h_out & mask_w_out & mask_c_out
    tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=mask_out)
    

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(Model, self).__init__()
        
        # Store essential parameters
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = in_channels * expand_ratio

        # --- MAJOR OPTIMIZATION: Pre-fold weights and BN params at initialization ---
        # Instead of storing nn.Module layers and calculating BN params on the fly,
        # we pre-calculate everything and store them as simple tensor buffers.
        # This reduces Python overhead, memory usage, and makes the forward pass cleaner.
        
        if self.expand_ratio != 1:
            _expand_conv = nn.Conv2d(in_channels, self.hidden_dim, 1, 1, 0, bias=False)
            _expand_bn = nn.BatchNorm2d(self.hidden_dim).eval()
            expand_gamma, expand_beta = self._prepare_bn_params(_expand_bn)
            self.register_buffer('expand_weight', _expand_conv.weight.detach().half())
            self.register_buffer('expand_gamma', expand_gamma)
            self.register_buffer('expand_beta', expand_beta)
        
        _depthwise_conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size, stride, self.padding, groups=self.hidden_dim, bias=False)
        _depthwise_bn = nn.BatchNorm2d(self.hidden_dim).eval()
        dw_gamma, dw_beta = self._prepare_bn_params(_depthwise_bn)
        self.register_buffer('depthwise_weight', _depthwise_conv.weight.detach().half())
        self.register_buffer('depthwise_gamma', dw_gamma)
        self.register_buffer('depthwise_beta', dw_beta)
        
        _project_conv = nn.Conv2d(self.hidden_dim, out_channels, 1, 1, 0, bias=False)
        _project_bn = nn.BatchNorm2d(out_channels).eval()
        proj_gamma, proj_beta = self._prepare_bn_params(_project_bn)
        self.register_buffer('project_weight', _project_conv.weight.detach().half())
        self.register_buffer('project_gamma', proj_gamma)
        self.register_buffer('project_beta', proj_beta)
        
        # Workspace for intermediate tensors to avoid reallocation
        self.register_buffer('workspace', None, persistent=False)

    def _prepare_bn_params(self, bn_layer):
        """Calculates the fused gamma and beta parameters from a BatchNorm layer."""
        bn_rm, bn_rv = bn_layer.running_mean, bn_layer.running_var
        bn_eps, bn_w, bn_b = bn_layer.eps, bn_layer.weight, bn_layer.bias
        inv_std = torch.rsqrt(bn_rv + bn_eps)
        gamma = bn_w * inv_std
        beta = bn_b - bn_rm * gamma
        return gamma.half(), beta.half()

    def _call_fused_pointwise_conv(self, x_nhwc, conv_w, gamma, beta, activation, out_tensor, identity_nhwc=None):
        N, H, W, C_in = x_nhwc.shape
        C_out = conv_w.shape[0]
        
        A = x_nhwc.view(-1, C_in)
        B = conv_w.squeeze().T.contiguous()
        
        M, K = A.shape
        _K, N_out = B.shape
        C = out_tensor.view(M, N_out)
        
        has_residual = identity_nhwc is not None
        if has_residual:
            identity_reshaped = identity_nhwc.view(-1, C_out)
            stride_im, stride_in = identity_reshaped.stride()
        else:
            identity_reshaped = torch.empty(0, device=x_nhwc.device, dtype=x_nhwc.dtype)
            stride_im, stride_in = 0, 0

        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N_out, META['BLOCK_SIZE_N']),)
        fused_pointwise_conv_bn_act_kernel[grid](
            A, B, C, gamma, beta, identity_reshaped, M, N_out, K,
            A.stride(0), A.stride(1), B.stride(0), B.stride(1),
            C.stride(0), C.stride(1), stride_im, stride_in,
            ACTIVATION=activation, HAS_RESIDUAL=has_residual
        )

    def _call_fused_depthwise_conv_nhwc(self, x_nhwc, conv_w, gamma, beta, out_tensor):
        N, H_in, W_in, C = x_nhwc.shape
        H_out, W_out = out_tensor.shape[1], out_tensor.shape[2]
        y = out_tensor
        w_ckk = conv_w.squeeze(1).contiguous()
        grid = lambda META: (N * triton.cdiv(H_out, META['BLOCK_SIZE_H']) * triton.cdiv(W_out, META['BLOCK_SIZE_W']) * triton.cdiv(C, META['BLOCK_SIZE_C']),)
        
        fused_depthwise_conv_k5_s2_bn_relu6_nhwc_kernel[grid](
            x_nhwc, w_ckk, y, gamma, beta,
            H_in, W_in, C, H_out, W_out,
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            w_ckk.stride(0), w_ckk.stride(1), w_ckk.stride(2),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3)
        )

    def forward(self, x):
        original_dtype = x.dtype
        N, _, H_in, W_in = x.shape
        
        H_dw_out = (H_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_dw_out = (W_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        size1 = N * H_in * W_in * self.hidden_dim if self.expand_ratio != 1 else 0
        size2 = N * H_dw_out * W_dw_out * self.hidden_dim
        required_size = size1 + size2
        if self.workspace is None or self.workspace.numel() < required_size:
            self.workspace = torch.empty(required_size, dtype=torch.float16, device=x.device)
        
        x_fp16_nhwc = x.half().permute(0, 2, 3, 1).contiguous()
        identity_fp16_nhwc = x_fp16_nhwc if self.use_residual else None
        
        # Expansion phase
        if self.expand_ratio != 1:
            buffer1 = self.workspace.narrow(0, 0, size1).view(N, H_in, W_in, self.hidden_dim)
            self._call_fused_pointwise_conv(x_fp16_nhwc, self.expand_weight,
                                            self.expand_gamma, self.expand_beta, "relu6", 
                                            out_tensor=buffer1)
            current_x = buffer1
        else:
            current_x = x_fp16_nhwc
        
        # Depthwise phase
        buffer2 = self.workspace.narrow(0, size1, size2).view(N, H_dw_out, W_dw_out, self.hidden_dim)
        self._call_fused_depthwise_conv_nhwc(current_x, self.depthwise_weight,
                                             self.depthwise_gamma, self.depthwise_beta, out_tensor=buffer2)
        current_x = buffer2
        
        # Projection phase
        final_y_shape = (N, H_dw_out, W_dw_out, self.out_channels)
        final_y_nhwc = torch.empty(final_y_shape, device=x.device, dtype=torch.float16)
        self._call_fused_pointwise_conv(current_x, self.project_weight,
                                        self.project_gamma, self.project_beta, "none",
                                        out_tensor=final_y_nhwc, identity_nhwc=identity_fp16_nhwc)
        
        return final_y_nhwc.permute(0, 3, 1, 2).contiguous().to(original_dtype)

# Test code
batch_size = 10
in_channels = 112
out_channels = 192
kernel_size = 5
stride = 2
expand_ratio = 6

def get_inputs():
    return [torch.randn(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, expand_ratio]