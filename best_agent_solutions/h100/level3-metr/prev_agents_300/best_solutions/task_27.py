import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# --- Autotuner Configurations ---
# Expanded configurations for the convolution kernels to provide a wider search space.
# Added more options for BLOCK_K_IN and increased num_stages for larger tiles to better hide memory latency.
CONV_AUTOTUNER_CONFIGS = [
    triton.Config({'BLOCK_M_PX': 16, 'BLOCK_N_C': 32, 'BLOCK_K_IN': 32}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_M_PX': 32, 'BLOCK_N_C': 64, 'BLOCK_K_IN': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M_PX': 64, 'BLOCK_N_C': 32, 'BLOCK_K_IN': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M_PX': 32, 'BLOCK_N_C': 128, 'BLOCK_K_IN': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M_PX': 128, 'BLOCK_N_C': 32, 'BLOCK_K_IN': 32}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M_PX': 64, 'BLOCK_N_C': 64, 'BLOCK_K_IN': 64}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M_PX': 64, 'BLOCK_N_C': 128, 'BLOCK_K_IN': 32}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M_PX': 128, 'BLOCK_N_C': 64, 'BLOCK_K_IN': 32}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M_PX': 128, 'BLOCK_N_C': 128, 'BLOCK_K_IN': 32}, num_warps=8, num_stages=4),
    # Added configs for more thorough tuning
    triton.Config({'BLOCK_M_PX': 64, 'BLOCK_N_C': 64, 'BLOCK_K_IN': 32}, num_warps=4, num_stages=5),
    triton.Config({'BLOCK_M_PX': 128, 'BLOCK_N_C': 64, 'BLOCK_K_IN': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M_PX': 64, 'BLOCK_N_C': 128, 'BLOCK_K_IN': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M_PX': 128, 'BLOCK_N_C': 128, 'BLOCK_K_IN': 64}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_M_PX': 256, 'BLOCK_N_C': 64, 'BLOCK_K_IN': 32}, num_warps=8, num_stages=4),
]

# --- Triton Kernel for Fused Conv-ReLU ---
@triton.autotune(configs=CONV_AUTOTUNER_CONFIGS, key=['C_out', 'K_in', 'H_out', 'W_out'])
@triton.jit
def _fused_conv_relu_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N, H_in, W_in, C_in, H_out, W_out, C_out,
    KH, KW, SH, SW, PH, PW, K_in,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wk, stride_wn,
    stride_outn, stride_outc, stride_outh, stride_outw,
    BLOCK_M_PX: tl.constexpr, BLOCK_N_C: tl.constexpr, BLOCK_K_IN: tl.constexpr,
):
    pid_m_block = tl.program_id(0)
    pid_c_block = tl.program_id(1)

    c_out_offsets = pid_c_block * BLOCK_N_C + tl.arange(0, BLOCK_N_C)
    c_out_mask = c_out_offsets < C_out
    m_offsets = pid_m_block * BLOCK_M_PX + tl.arange(0, BLOCK_M_PX)
    m_mask = m_offsets < (N * H_out * W_out)
    n_offsets = m_offsets // (H_out * W_out)
    rem_m = m_offsets % (H_out * W_out)
    h_out_offsets = rem_m // W_out
    w_out_offsets = rem_m % W_out

    acc = tl.zeros((BLOCK_M_PX, BLOCK_N_C), dtype=tl.float32)

    for k_start in range(0, K_in, BLOCK_K_IN):
        k_offsets = k_start + tl.arange(0, BLOCK_K_IN)
        k_mask = k_offsets < K_in
        k_cin_val = k_offsets % C_in
        rem_k = k_offsets // C_in
        k_kh_val = rem_k % KH
        k_kw_val = rem_k // KH
        h_in_idx = h_out_offsets[:, None] * SH + k_kh_val[None, :] - PH
        w_in_idx = w_out_offsets[:, None] * SW + k_kw_val[None, :] - PW

        x_offsets = (n_offsets[:, None] * stride_xn + h_in_idx * stride_xh +
                     w_in_idx * stride_xw + k_cin_val[None, :] * stride_xc)
        x_padding_mask = (h_in_idx >= 0) & (h_in_idx < H_in) & (w_in_idx >= 0) & (w_in_idx < W_in)
        load_mask = m_mask[:, None] & k_mask[None, :] & x_padding_mask
        input_vals = tl.load(x_ptr + x_offsets, mask=load_mask, other=0.0)

        w_ptr_offsets = k_offsets[:, None] * stride_wk + c_out_offsets[None, :] * stride_wn
        weight_vals = tl.load(w_ptr + w_ptr_offsets, mask=k_mask[:, None] & c_out_mask[None, :])
        acc += tl.dot(input_vals, weight_vals, out_dtype=tl.float32)

    bias = tl.load(bias_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
    acc += bias[None, :]
    activated = tl.maximum(acc, 0.0)

    out_ptr_offsets = (n_offsets[:, None] * stride_outn + h_out_offsets[:, None] * stride_outh +
                       w_out_offsets[:, None] * stride_outw + c_out_offsets[None, :] * stride_outc)
    store_mask = m_mask[:, None] & c_out_mask[None, :]
    tl.store(out_ptr + out_ptr_offsets, activated.to(out_ptr.dtype.element_ty), mask=store_mask)

# --- NEW: Refined Triton Kernel for Fused Conv-ReLU-MaxPool ---
@triton.autotune(configs=CONV_AUTOTUNER_CONFIGS, key=['C_out', 'K_in', 'H_out', 'W_out'])
@triton.jit
def _fused_conv_relu_pool_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    N, H_in, W_in, C_in, H_out, W_out, C_out,
    KH, KW, SH, SW, PH, PW, K_in,
    POOL_KH, POOL_KW, POOL_SH, POOL_SW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wk, stride_wn,
    stride_outn, stride_outc, stride_outh, stride_outw,
    BLOCK_M_PX: tl.constexpr, BLOCK_N_C: tl.constexpr, BLOCK_K_IN: tl.constexpr,
):
    pid_m_block = tl.program_id(0)
    pid_c_block = tl.program_id(1)

    c_out_offsets = pid_c_block * BLOCK_N_C + tl.arange(0, BLOCK_N_C)
    c_out_mask = c_out_offsets < C_out
    m_offsets = pid_m_block * BLOCK_M_PX + tl.arange(0, BLOCK_M_PX)
    m_mask = m_offsets < (N * H_out * W_out)
    n_offsets = m_offsets // (H_out * W_out)
    rem_m = m_offsets % (H_out * W_out)
    h_out_offsets = rem_m // W_out
    w_out_offsets = rem_m % W_out

    acc00 = tl.zeros((BLOCK_M_PX, BLOCK_N_C), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_M_PX, BLOCK_N_C), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_M_PX, BLOCK_N_C), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_M_PX, BLOCK_N_C), dtype=tl.float32)
    
    # Calculate pointer offsets for shifting receptive field by 1 pixel in H and W
    # This assumes the convolution stride (SH, SW) is 1, which is true for this model.
    delta_h_ptr = 1 * SH * stride_xh
    delta_w_ptr = 1 * SW * stride_xw

    for k_start in range(0, K_in, BLOCK_K_IN):
        k_offsets = k_start + tl.arange(0, BLOCK_K_IN)
        k_mask = k_offsets < K_in
        
        w_ptr_offsets = k_offsets[:, None] * stride_wk + c_out_offsets[None, :] * stride_wn
        weight_vals = tl.load(w_ptr + w_ptr_offsets, mask=k_mask[:, None] & c_out_mask[None, :])
        
        k_cin_val = k_offsets % C_in
        rem_k = k_offsets // C_in
        k_kh_val = rem_k % KH
        k_kw_val = rem_k // KH

        # --- Base offsets for top-left of 2x2 pooling window ---
        h_intermediate_base = h_out_offsets * POOL_SH
        w_intermediate_base = w_out_offsets * POOL_SW
        h_in_base = h_intermediate_base[:, None] * SH + k_kh_val[None, :] - PH
        w_in_base = w_intermediate_base[:, None] * SW + k_kw_val[None, :] - PW

        base_x_offsets = (n_offsets[:, None] * stride_xn + h_in_base * stride_xh + 
                          w_in_base * stride_xw + k_cin_val[None, :] * stride_xc)
        base_padding_mask = (h_in_base >= 0) & (h_in_base < H_in) & (w_in_base >= 0) & (w_in_base < W_in)

        # --- Load and compute for all 4 pooling locations ---
        # Location (0,0)
        mask00 = m_mask[:, None] & k_mask[None, :] & base_padding_mask
        input_vals00 = tl.load(x_ptr + base_x_offsets, mask=mask00, other=0.0)
        acc00 += tl.dot(input_vals00, weight_vals, out_dtype=tl.float32)

        # Location (0,1) - shift W
        x_offsets01 = base_x_offsets + delta_w_ptr
        padding_mask01 = (h_in_base >= 0) & (h_in_base < H_in) & ((w_in_base + 1) >= 0) & ((w_in_base + 1) < W_in)
        mask01 = m_mask[:, None] & k_mask[None, :] & padding_mask01
        input_vals01 = tl.load(x_ptr + x_offsets01, mask=mask01, other=0.0)
        acc01 += tl.dot(input_vals01, weight_vals, out_dtype=tl.float32)
        
        # Location (1,0) - shift H
        x_offsets10 = base_x_offsets + delta_h_ptr
        padding_mask10 = ((h_in_base + 1) >= 0) & ((h_in_base + 1) < H_in) & (w_in_base >= 0) & (w_in_base < W_in)
        mask10 = m_mask[:, None] & k_mask[None, :] & padding_mask10
        input_vals10 = tl.load(x_ptr + x_offsets10, mask=mask10, other=0.0)
        acc10 += tl.dot(input_vals10, weight_vals, out_dtype=tl.float32)
        
        # Location (1,1) - shift H and W
        x_offsets11 = base_x_offsets + delta_h_ptr + delta_w_ptr
        padding_mask11 = ((h_in_base + 1) >= 0) & ((h_in_base + 1) < H_in) & ((w_in_base + 1) >= 0) & ((w_in_base + 1) < W_in)
        mask11 = m_mask[:, None] & k_mask[None, :] & padding_mask11
        input_vals11 = tl.load(x_ptr + x_offsets11, mask=mask11, other=0.0)
        acc11 += tl.dot(input_vals11, weight_vals, out_dtype=tl.float32)

    bias = tl.load(bias_ptr + c_out_offsets, mask=c_out_mask, other=0.0)
    res00, res01 = tl.maximum(acc00 + bias[None, :], 0.0), tl.maximum(acc01 + bias[None, :], 0.0)
    res10, res11 = tl.maximum(acc10 + bias[None, :], 0.0), tl.maximum(acc11 + bias[None, :], 0.0)
    max_vals = tl.maximum(tl.maximum(res00, res01), tl.maximum(res10, res11))

    out_ptr_offsets = (n_offsets[:, None] * stride_outn + h_out_offsets[:, None] * stride_outh +
                       w_out_offsets[:, None] * stride_outw + c_out_offsets[None, :] * stride_outc)
    store_mask = m_mask[:, None] & c_out_mask[None, :]
    tl.store(out_ptr + out_ptr_offsets, max_vals.to(out_ptr.dtype.element_ty), mask=store_mask)

# --- Fused Global Average Pool + Linear Kernel ---
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_SPATIAL': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_SPATIAL': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_SPATIAL': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_SPATIAL': 64}, num_warps=8),
        triton.Config({'BLOCK_C': 128, 'BLOCK_SPATIAL': 32}, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'BLOCK_SPATIAL': 16}, num_warps=8),
    ],
    key=['C_in', 'SPATIAL_DIM', 'C_out'],
)
@triton.jit
def _fused_gap_fc_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W, C_out, SPATIAL_DIM,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc,
    BLOCK_C: tl.constexpr, BLOCK_SPATIAL: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_cout = tl.program_id(1)
    final_acc = tl.zeros((), tl.float32)

    for c_base in range(0, C_in, BLOCK_C):
        c_offsets = c_base + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C_in
        spatial_sum_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for s_base in range(0, SPATIAL_DIM, BLOCK_SPATIAL):
            s_offsets = s_base + tl.arange(0, BLOCK_SPATIAL)
            s_mask = s_offsets < SPATIAL_DIM
            h = s_offsets // W
            w = s_offsets % W
            x_addrs = (x_ptr + pid_n * stride_xn +
                       h[:, None] * stride_xh +
                       w[:, None] * stride_xw +
                       c_offsets[None, :] * stride_xc)
            x_tile = tl.load(x_addrs, mask=s_mask[:, None] & c_mask[None, :], other=0.0)
            spatial_sum_acc += tl.sum(x_tile, axis=0)
        
        w_addrs = w_ptr + pid_cout * stride_wn + c_offsets * stride_wc
        weights = tl.load(w_addrs, mask=c_mask, other=0.0)
        final_acc += tl.sum(spatial_sum_acc * weights, axis=0)

    final_avg = final_acc / SPATIAL_DIM
    bias = tl.load(b_ptr + pid_cout)
    final_avg += bias
    out_addr = out_ptr + pid_n * C_out + pid_cout
    tl.store(out_addr, final_avg.to(out_ptr.dtype.element_ty))

# --- Base class for Fused Modules with BN Folding ---
class FusedOpBase(nn.Module):
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size if isinstance(conv.kernel_size, tuple) else (conv.kernel_size, conv.kernel_size)
        self.stride = conv.stride if isinstance(conv.stride, tuple) else (conv.stride, conv.stride)
        self.padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)

        scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
        fused_bias = (conv.bias.detach() if conv.bias is not None else 0) - bn.running_mean.detach()
        fused_bias = fused_bias * scale + bn.bias.detach()
        fused_weight = conv.weight.detach() * scale.view(-1, 1, 1, 1)

        weight_data = fused_weight.permute(3, 2, 1, 0).contiguous().view(-1, self.out_channels)
        self.weight = nn.Parameter(weight_data, requires_grad=False)
        self.bias = nn.Parameter(fused_bias, requires_grad=False)

class FusedConvReLU(FusedOpBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H_in, W_in = x.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding
        H_out, W_out = (H_in + 2 * PH - KH) // SH + 1, (W_in + 2 * PW - KW) // SW + 1
        
        out = torch.empty((N, self.out_channels, H_out, W_out), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
        grid = lambda meta: (triton.cdiv(N * H_out * W_out, meta['BLOCK_M_PX']), triton.cdiv(self.out_channels, meta['BLOCK_N_C']))
        
        _fused_conv_relu_kernel[grid](
            x, self.weight, self.bias, out,
            N, H_in, W_in, self.in_channels, H_out, W_out, self.out_channels,
            KH, KW, SH, SW, PH, PW, self.in_channels * KH * KW,
            *x.stride(), *self.weight.stride(), *out.stride(),
        )
        return out

class FusedConvReLUPool(FusedOpBase):
    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d, pool: nn.MaxPool2d):
        super().__init__(conv, bn)
        self.pool_kh, self.pool_kw = (pool.kernel_size, pool.kernel_size) if isinstance(pool.kernel_size, int) else pool.kernel_size
        self.pool_sh, self.pool_sw = (pool.stride, pool.stride) if isinstance(pool.stride, int) else pool.stride
        assert self.pool_kh == 2 and self.pool_kw == 2, "Optimized kernel only supports 2x2 pooling"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H_in, W_in = x.shape
        KH, KW = self.kernel_size
        SH, SW = self.stride
        PH, PW = self.padding
        H_intermediate = (H_in + 2 * PH - KH) // SH + 1
        W_intermediate = (W_in + 2 * PW - KW) // SW + 1
        H_out, W_out = (H_intermediate - self.pool_kh) // self.pool_sh + 1, (W_intermediate - self.pool_kw) // self.pool_sw + 1

        out = torch.empty((N, self.out_channels, H_out, W_out), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
        grid = lambda meta: (triton.cdiv(N * H_out * W_out, meta['BLOCK_M_PX']), triton.cdiv(self.out_channels, meta['BLOCK_N_C']))

        _fused_conv_relu_pool_kernel[grid](
            x, self.weight, self.bias, out,
            N, H_in, W_in, self.in_channels, H_out, W_out, self.out_channels,
            KH, KW, SH, SW, PH, PW, self.in_channels * KH * KW,
            self.pool_kh, self.pool_kw, self.pool_sh, self.pool_sw,
            *x.stride(), *self.weight.stride(), *out.stride(),
        )
        return out

class FusedStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        bn1 = nn.BatchNorm2d(out_channels)
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        bn2 = nn.BatchNorm2d(out_channels)
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.block1 = FusedConvReLU(conv1, bn1)
        self.block2 = FusedConvReLUPool(conv2, bn2, pool)

    def forward(self, x):
        return self.block2(self.block1(x))

class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        super(ModelNew, self).__init__()
        layers = []
        current_channels = input_channels
        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        
        self.feature_extractor = nn.Sequential(*layers)
        self.fc = nn.Linear(block_widths[-1], output_classes)
        
        self.half()
        self.to(memory_format=torch.channels_last)
        self.eval()

    def _make_stage(self, in_channels, out_channels):
        return FusedStage(in_channels, out_channels)

    def forward(self, x):
        x = x.to(device=self.fc.weight.device, dtype=self.fc.weight.dtype, memory_format=torch.channels_last)
        x = self.feature_extractor(x)
        
        N, C, H, W = x.shape
        C_out = self.fc.out_features
        out_final = torch.empty((N, C_out), dtype=x.dtype, device=x.device)
        
        grid = (N, C_out)
        _fused_gap_fc_kernel[grid](
            x, self.fc.weight, self.fc.bias, out_final,
            N, C, H, W, C_out, H * W,
            *x.stride(), *self.fc.weight.stride(),
        )
        
        return out_final.float()

# --- Boilerplate for Testing ---
batch_size = 8
input_channels = 3
image_height, image_width = 224, 224
stages = 3
block_widths = [64, 128, 256]
output_classes = 10

def get_inputs():
    """ Generates random input tensor of shape (batch_size, input_channels, height, width) """
    return [torch.randn(batch_size, input_channels, image_height, image_width)]

def get_init_inputs():
    """ Initializes model parameters """
    return [input_channels, stages, block_widths, output_classes]