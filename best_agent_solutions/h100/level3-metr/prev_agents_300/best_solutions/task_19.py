import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# This kernel fuses a standard 2D convolution, batch normalization, and ReLU activation.
# It is optimized for channels-last (NHWC) memory format and fp16/bf16 data types,
# leveraging Tensor Cores for the main GEMM operation.
@triton.autotune(
    configs=[
        # Expanded and tuned configs
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=16),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=16),
    ],
    key=['C_in', 'C_out', 'stride_h', 'H_in'],
)
@triton.jit
def fused_conv_bn_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N_in, H_in, W_in, C_in, C_out, H_out, W_out,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_yn, stride_yh, stride_yw, stride_yc,
    stride_wc_out, stride_wc_in, stride_wkh, stride_wkw,
    stride_h, stride_w, padding_h, padding_w,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N_in * H_out * W_out, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(C_out, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    offs_hw = offs_m % (H_out * W_out)
    offs_n_img = offs_m // (H_out * W_out)
    offs_h_out = offs_hw // W_out
    offs_w_out = offs_hw % W_out
    
    offs_h_in_start = offs_h_out * stride_h - padding_h
    offs_w_in_start = offs_w_out * stride_w - padding_w
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k_c_base in range(0, C_in * KERNEL_H * KERNEL_W, BLOCK_SIZE_K):
        k_c_offsets = k_c_base + tl.arange(0, BLOCK_SIZE_K)
        
        offs_k_cin = k_c_offsets % C_in
        offs_k_khkw = k_c_offsets // C_in
        kh = offs_k_khkw // KERNEL_W
        kw = offs_k_khkw % KERNEL_W
        
        x_h = offs_h_in_start[:, None] + kh[None, :]
        x_w = offs_w_in_start[:, None] + kw[None, :]

        x_ptr = X_ptr + offs_n_img[:, None] * stride_xn + x_h * stride_xh + x_w * stride_xw + offs_k_cin[None, :] * stride_xc
        x_mask = (offs_n_img[:, None] < N_in) & (x_h >= 0) & (x_h < H_in) & (x_w >= 0) & (x_w < W_in) & (k_c_offsets[None, :] < C_in * KERNEL_H * KERNEL_W)
        a = tl.load(x_ptr, mask=x_mask, other=0.0)

        w_ptr = W_ptr + offs_n[:, None] * stride_wc_out + kh[None, :] * stride_wkh + kw[None, :] * stride_wkw + offs_k_cin[None, :] * stride_wc_in
        w_mask = (offs_n[:, None] < C_out) & (k_c_offsets[None, :] < C_in * KERNEL_H * KERNEL_W)
        b = tl.load(w_ptr, mask=w_mask, other=0.0)
        
        acc += tl.dot(a, tl.trans(b))
    
    b_ptr = B_ptr + offs_n
    bias = tl.load(b_ptr, mask=offs_n < C_out, other=0.0)
    result = acc + bias[None, :].to(tl.float32)
    result = tl.maximum(result, 0.0)

    y_ptr = Y_ptr + offs_n_img[:, None] * stride_yn + offs_h_out[:, None] * stride_yh + offs_w_out[:, None] * stride_yw + offs_n[None, :] * stride_yc
    y_mask = (offs_m[:, None] < N_in * H_out * W_out) & (offs_n[None, :] < C_out)
    tl.store(y_ptr, result.to(Y_ptr.dtype.element_ty), mask=y_mask)


# This kernel fuses the entire depthwise separable block:
# DepthwiseConv -> BatchNorm -> ReLU -> PointwiseConv -> BatchNorm -> ReLU
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16, 'BLOCK_SIZE_C_OUT': 64, 'BLOCK_SIZE_C_IN': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16, 'BLOCK_SIZE_C_OUT': 128,'BLOCK_SIZE_C_IN': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 8,  'BLOCK_SIZE_W': 8,  'BLOCK_SIZE_C_OUT': 256,'BLOCK_SIZE_C_IN': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 32, 'BLOCK_SIZE_C_OUT': 32, 'BLOCK_SIZE_C_IN': 32}, num_stages=5, num_warps=16),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16, 'BLOCK_SIZE_C_OUT': 256,'BLOCK_SIZE_C_IN': 128}, num_stages=3, num_warps=16),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 16, 'BLOCK_SIZE_C_OUT': 128,'BLOCK_SIZE_C_IN': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 32, 'BLOCK_SIZE_C_OUT': 128,'BLOCK_SIZE_C_IN': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16, 'BLOCK_SIZE_C_OUT': 256,'BLOCK_SIZE_C_IN': 256}, num_stages=3, num_warps=16),
    ],
    key=['H_out', 'W_out', 'C_in', 'C_out', 'dw_stride_h'],
)
@triton.jit
def fused_dw_pw_conv_bn_relu_kernel(
    X_ptr, DW_W_ptr, DW_B_ptr, PW_W_ptr, PW_B_ptr, Y_ptr,
    N_in, H_in, W_in, C_in, N_out, H_out, W_out, C_out,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_yn, stride_yh, stride_yw, stride_yc,
    dw_stride_h, dw_stride_w, dw_padding_h, dw_padding_w,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C_OUT: tl.constexpr, BLOCK_SIZE_C_IN: tl.constexpr,
):
    pid_h, pid_w, pid_nc_out = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    num_c_out_blocks = tl.cdiv(C_out, BLOCK_SIZE_C_OUT)
    n, c_out_block_idx = pid_nc_out // num_c_out_blocks, pid_nc_out % num_c_out_blocks
    h_start, w_start = pid_h * BLOCK_SIZE_H, pid_w * BLOCK_SIZE_W
    h_offsets, w_offsets = h_start + tl.arange(0, BLOCK_SIZE_H), w_start + tl.arange(0, BLOCK_SIZE_W)
    c_out_offsets = c_out_block_idx * BLOCK_SIZE_C_OUT + tl.arange(0, BLOCK_SIZE_C_OUT)
    
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C_OUT), dtype=tl.float32)
    
    for k_start in range(0, C_in, BLOCK_SIZE_C_IN):
        c_in_offsets = k_start + tl.arange(0, BLOCK_SIZE_C_IN)
        intermediate = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C_IN), dtype=X_ptr.dtype.element_ty)
        
        for kh in range(KERNEL_H):
            for kw in range(KERNEL_W):
                x_h = h_offsets[:, None] * dw_stride_h - dw_padding_h + kh
                x_w = w_offsets[None, :] * dw_stride_w - dw_padding_w + kw
                x_mask = (x_h >= 0) & (x_h < H_in) & (x_w >= 0) & (x_w < W_in)
                cin_mask = c_in_offsets[None, None, :] < C_in
                
                x_ptr = X_ptr + n*stride_xn + x_h[:,:,None]*stride_xh + x_w[:,:,None]*stride_xw + c_in_offsets[None,None,:]*stride_xc
                dw_w_ptr = DW_W_ptr + kh*KERNEL_W*C_in + kw*C_in + c_in_offsets
                
                x_patch = tl.load(x_ptr, mask=x_mask[:, :, None] & cin_mask, other=0.0)
                dw_w = tl.load(dw_w_ptr, mask=c_in_offsets < C_in, other=0.0)
                intermediate += x_patch * dw_w[None, None, :]
        
        dw_b = tl.load(DW_B_ptr + c_in_offsets, mask=c_in_offsets < C_in, other=0.0)
        intermediate = tl.maximum(intermediate + dw_b[None, None, :], 0)

        pw_w = tl.load(PW_W_ptr + c_in_offsets[:, None]*C_out + c_out_offsets[None, :], mask=(c_in_offsets[:,None] < C_in) & (c_out_offsets[None,:] < C_out), other=0.0)
        
        intermediate_reshaped = tl.reshape(intermediate, (BLOCK_SIZE_H * BLOCK_SIZE_W, BLOCK_SIZE_C_IN))
        pointwise_result = tl.dot(intermediate_reshaped, pw_w)
        acc += tl.reshape(pointwise_result, (BLOCK_SIZE_H, BLOCK_SIZE_W, BLOCK_SIZE_C_OUT))

    pw_b = tl.load(PW_B_ptr + c_out_offsets, mask=c_out_offsets < C_out, other=0.0)
    acc = tl.maximum(acc + pw_b[None, None, :].to(tl.float32), 0.0)
    
    y_ptr = Y_ptr + n*stride_yn + h_offsets[:,None,None]*stride_yh + w_offsets[None,:,None]*stride_yw + c_out_offsets[None,None,:]*stride_yc
    y_mask = (h_offsets[:,None,None] < H_out) & (w_offsets[None,:,None] < W_out) & (c_out_offsets[None,None,:] < C_out)
    tl.store(y_ptr, acc.to(Y_ptr.dtype.element_ty), mask=y_mask)

# **IMPROVED KERNEL**
# This kernel fuses the final global average pooling and the fully connected layer.
# It uses tl.dot for the matrix multiplication by processing a block of the batch dimension
# at once, which is significantly more efficient and allows leveraging Tensor Cores.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_C': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_C': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_C': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_C': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_C': 128}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 512, 'BLOCK_SIZE_C': 128}, num_warps=16, num_stages=3),
    ],
    key=['C', 'K', 'N'],
)
@triton.jit
def fused_pool_fc_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, C, K,
    stride_xn, stride_xh, stride_xw, stride_xc,
    stride_wc, stride_wk, stride_yn, stride_yk,
    H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_C: tl.constexpr,
):
    INV_HW: tl.constexpr = 1.0 / (H * W)
    pid_n_block, pid_k_block = tl.program_id(0), tl.program_id(1)

    n_offsets = pid_n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = pid_k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    n_mask = n_offsets < N
    k_mask = k_offsets < K

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

    for c_start in range(0, C, BLOCK_SIZE_C):
        c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
        c_mask = c_offsets < C

        # Perform pooling for a block of (N, C)
        pool_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_C), dtype=tl.float32)
        for h_idx in range(H):
            for w_idx in range(W):
                x_ptr = (X_ptr + 
                         n_offsets[:, None] * stride_xn + 
                         h_idx * stride_xh + 
                         w_idx * stride_xw + 
                         c_offsets[None, :] * stride_xc)
                mask = n_mask[:, None] & c_mask[None, :]
                patch_vec = tl.load(x_ptr, mask=mask, other=0.0)
                pool_acc += patch_vec.to(tl.float32)

        pooled_vals = (pool_acc * INV_HW).to(W_ptr.dtype.element_ty)

        # Load weights and compute dot product
        w_ptr = W_ptr + c_offsets[:, None] * stride_wc + k_offsets[None, :] * stride_wk
        w_mask = c_mask[:, None] & k_mask[None, :]
        w_block = tl.load(w_ptr, mask=w_mask, other=0.0)
        
        acc += tl.dot(pooled_vals, w_block)
    
    if B_ptr is not None:
        bias = tl.load(B_ptr + k_offsets, mask=k_mask, other=0.0)
        acc += bias[None, :].to(tl.float32)
    
    y_ptr = Y_ptr + n_offsets[:, None] * stride_yn + k_offsets[None, :] * stride_yk
    y_mask = n_mask[:, None] & k_mask[None, :]
    tl.store(y_ptr, acc.to(Y_ptr.dtype.element_ty), mask=y_mask)


class FusedConvBNReLU(nn.Module):
    def __init__(self, conv, bn):
        super().__init__()
        conv.eval()
        bn.eval()
        with torch.no_grad():
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            fused_weight = conv.weight * scale.view(-1, 1, 1, 1)
            fused_bias = bn.bias - bn.running_mean * scale
        self.register_buffer('fused_weight', fused_weight.permute(0, 2, 3, 1).contiguous().half())
        self.register_buffer('fused_bias', fused_bias.contiguous().half())
        self.stride, self.padding = conv.stride, conv.padding
        self.in_channels, self.out_channels = conv.in_channels, conv.out_channels
        self.kernel_size = conv.kernel_size

    def forward(self, x):
        N, H_in, W_in, C_in = x.shape
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        kernel_h, kernel_w = self.kernel_size
        H_out = (H_in + 2 * padding_h - kernel_h) // stride_h + 1
        W_out = (W_in + 2 * padding_w - kernel_w) // stride_w + 1
        y = torch.empty((N, H_out, W_out, self.out_channels), device=x.device, dtype=x.dtype)
        
        grid = lambda meta: (triton.cdiv(N * H_out * W_out, meta['BLOCK_SIZE_M']) * triton.cdiv(self.out_channels, meta['BLOCK_SIZE_N']),)
        w_stride = self.fused_weight.stride()
        fused_conv_bn_relu_kernel[grid](
            x, self.fused_weight, self.fused_bias, y,
            N, H_in, W_in, C_in, self.out_channels, H_out, W_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            w_stride[0], w_stride[3], w_stride[1], w_stride[2],
            stride_h, stride_w, padding_h, padding_w,
            KERNEL_H=kernel_h, KERNEL_W=kernel_w
        )
        return y


class FusedDepthwiseSeparable(nn.Module):
    def __init__(self, dw_conv, dw_bn, pw_conv, pw_bn):
        super().__init__()
        self.in_channels, self.out_channels = dw_conv.in_channels, pw_conv.out_channels
        self.stride, self.padding = dw_conv.stride, dw_conv.padding
        dw_conv.eval(), dw_bn.eval(), pw_conv.eval(), pw_bn.eval()
        with torch.no_grad():
            scale_dw = dw_bn.weight / torch.sqrt(dw_bn.running_var + dw_bn.eps)
            fused_dw_weight = dw_conv.weight * scale_dw.view(-1, 1, 1, 1)
            fused_dw_bias = dw_bn.bias - dw_bn.running_mean * scale_dw
            
            scale_pw = pw_bn.weight / torch.sqrt(pw_bn.running_var + pw_bn.eps)
            fused_pw_weight = pw_conv.weight * scale_pw.view(-1, 1, 1, 1)
            fused_pw_bias = pw_bn.bias - pw_bn.running_mean * scale_pw
        
        self.register_buffer('fused_dw_weight', fused_dw_weight.permute(2, 3, 0, 1).squeeze(3).contiguous().half())
        self.register_buffer('fused_dw_bias', fused_dw_bias.contiguous().half())
        self.register_buffer('fused_pw_weight', fused_pw_weight.squeeze().t().contiguous().half())
        self.register_buffer('fused_pw_bias', fused_pw_bias.contiguous().half())

    def forward(self, x):
        N, H_in, W_in, C_in = x.shape
        stride_h, stride_w = self.stride
        padding_h, padding_w = self.padding
        H_out = (H_in + 2*padding_h - 3)//stride_h + 1
        W_out = (W_in + 2*padding_w - 3)//stride_w + 1
        y = torch.empty((N, H_out, W_out, self.out_channels), device=x.device, dtype=x.dtype)
        
        grid = lambda meta: (triton.cdiv(H_out, meta['BLOCK_SIZE_H']), triton.cdiv(W_out, meta['BLOCK_SIZE_W']), N * triton.cdiv(self.out_channels, meta['BLOCK_SIZE_C_OUT']))
        fused_dw_pw_conv_bn_relu_kernel[grid](
            x, self.fused_dw_weight, self.fused_dw_bias, self.fused_pw_weight, self.fused_pw_bias, y,
            N, H_in, W_in, C_in, N, H_out, W_out, self.out_channels,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            stride_h, stride_w, padding_h, padding_w, KERNEL_H=3, KERNEL_W=3
        )
        return y


class FusedPoolFC(nn.Module):
    def __init__(self, fc_layer):
        super().__init__()
        self.in_features, self.out_features = fc_layer.in_features, fc_layer.out_features
        self.register_buffer('weight', fc_layer.weight.t().contiguous().half())
        self.register_buffer('bias', fc_layer.bias.contiguous().half() if fc_layer.bias is not None else None)

    def forward(self, x):
        N, H, W, C = x.shape
        y = torch.empty((N, self.out_features), device=x.device, dtype=x.dtype)
        
        def grid(meta):
            return (triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(self.out_features, meta['BLOCK_SIZE_K']))

        fused_pool_fc_kernel[grid](
            x, self.weight, self.bias, y,
            N, C, self.out_features,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.weight.stride(0), self.weight.stride(1),
            y.stride(0), y.stride(1),
            H=H, W=W
        )
        return y


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        super(Model, self).__init__()
        
        def conv_dw(inp, oup, stride):
            dw_conv = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
            dw_bn = nn.BatchNorm2d(inp)
            pw_conv = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
            pw_bn = nn.BatchNorm2d(oup)
            return FusedDepthwiseSeparable(dw_conv, dw_bn, pw_conv, pw_bn)
        
        first_conv = nn.Conv2d(input_channels, int(32 * alpha), 3, 2, 1, bias=False)
        first_bn = nn.BatchNorm2d(int(32 * alpha))
        fused_first_layer = FusedConvBNReLU(first_conv, first_bn)

        self.features = nn.Sequential(
            fused_first_layer,
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
        )
        
        fc = nn.Linear(int(1024 * alpha), num_classes)
        self.pool_fc = FusedPoolFC(fc)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.graphed_output = None
    
    def _forward_impl(self, x):
        x = self.features(x)
        x = self.pool_fc(x)
        return x

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last).half()

        if self.graph is None or x.shape != self.static_input.shape:
            # First run or shape change: capture the graph
            self.static_input = x.clone()
            
            # Warmup runs
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._forward_impl(self.static_input)
            torch.cuda.current_stream().wait_stream(s)
            
            # Graph capture
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.graphed_output = self._forward_impl(self.static_input)

        # Replay the graph
        self.static_input.copy_(x)
        self.graph.replay()
        return self.graphed_output.clone().float()

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000
alpha = 1.0

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width, device='cuda', dtype=torch.float16)]

def get_init_inputs():
    return [num_classes, input_channels, alpha]