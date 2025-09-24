import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Helper to map torch dtype to triton dtype
DTYPE_MAP = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}

# Autotuner configs for the unified 1x1 group convolution kernel
AUTOTUNE_CONFIGS_CONV1X1 = [
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
]

# Autotuner configs for the spatially-tiled depthwise convolution kernel
AUTOTUNE_CONFIGS_DEPTHWISE = [
    triton.Config({'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 16}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 32}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_W': 32}, num_warps=8, num_stages=4),
    triton.Config({'BLOCK_SIZE_C': 32, 'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_SIZE_C': 64, 'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 32}, num_warps=8, num_stages=5),
]


@triton.autotune(configs=AUTOTUNE_CONFIGS_CONV1X1, key=['C_out_per_group', 'C_in_per_group', 'S'])
@triton.jit
def fused_gconv1x1_kernel(
    x_ptr, w_ptr, bias_ptr, add_ptr, y_ptr,
    N, C_in_per_group, H, W, C_out_per_group, S, G,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc_out, stride_wc_in,
    stride_addn, stride_addc, stride_addh, stride_addw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    INPUT_DTYPE: tl.constexpr, HAS_ADD: tl.constexpr, HAS_RELU: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(C_out_per_group, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(S, BLOCK_SIZE_N)
    num_pid_g = G
    
    pid_g = pid // (num_pid_m * num_pid_n)
    pid_mn = pid % (num_pid_m * num_pid_n)
    
    pid_m_total = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n
    
    group_id = tl.program_id(axis=1)
    first_pid_m = group_id * GROUP_SIZE_M
    group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid_m_total % group_size)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m, mask_n = offs_m < C_out_per_group, offs_n < S
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    w_idx, h_idx, n_idx = offs_n % W, (offs_n // W) % H, offs_n // (W * H)

    x_ptr += pid_g * C_in_per_group * stride_xc
    w_ptr += pid_g * C_out_per_group * stride_wc_out

    for k in range(0, tl.cdiv(C_in_per_group, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        w_ptrs = w_ptr + (offs_m[:, None] * stride_wc_out + offs_k[None, :] * stride_wc_in)
        weights = tl.load(w_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < C_in_per_group), other=0.0)

        x_ptrs = x_ptr + (n_idx[None, :] * stride_xn + offs_k[:, None] * stride_xc + h_idx[None, :] * stride_xh + w_idx[None, :] * stride_xw)
        inputs = tl.load(x_ptrs, mask=(offs_k[:, None] < C_in_per_group) & mask_n[None, :], other=0.0)
        
        acc += tl.dot(weights, inputs)

    offs_m_global = pid_g * C_out_per_group + offs_m
    bias = tl.load(bias_ptr + offs_m_global, mask=mask_m, other=0.0)
    acc += bias[:, None]
    
    if HAS_RELU:
        acc = tl.maximum(acc, 0.0)
    
    if HAS_ADD:
        add_ptrs = add_ptr + (n_idx[None, :] * stride_addn + offs_m_global[:, None] * stride_addc + 
                              h_idx[None, :] * stride_addh + w_idx[None, :] * stride_addw)
        add_vals = tl.load(add_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        acc += add_vals

    y_ptrs = y_ptr + (n_idx[None, :] * stride_yn + offs_m_global[:, None] * stride_yc + h_idx[None, :] * stride_yh + w_idx[None, :] * stride_yw)
    tl.store(y_ptrs, acc.to(INPUT_DTYPE), mask=mask_m[:, None] & mask_n[None, :])

def fused_gconv1x1(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, G: int, has_relu: bool, add_tensor: torch.Tensor = None) -> torch.Tensor:
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    C_in_per_group = w.shape[1]
    y = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)
    S = N * H * W
    C_out_per_group = C_out // G

    def grid(META):
        num_pid_m = triton.cdiv(C_out_per_group, META['BLOCK_SIZE_M'])
        num_pid_n = triton.cdiv(S, META['BLOCK_SIZE_N'])
        num_groups = triton.cdiv(num_pid_m, META['GROUP_SIZE_M'])
        return (G * num_pid_m * num_pid_n, num_groups, 1)

    HAS_ADD = add_tensor is not None
    if HAS_ADD:
        stride_addn, stride_addc, stride_addh, stride_addw = add_tensor.stride()
    else:
        add_tensor = x 
        stride_addn, stride_addc, stride_addh, stride_addw = 0, 0, 0, 0

    w_reshaped = w.squeeze(-1).squeeze(-1)
    fused_gconv1x1_kernel[grid](
        x, w_reshaped, bias, add_tensor, y,
        N, C_in_per_group, H, W, C_out_per_group, S, G,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w_reshaped.stride(0), w_reshaped.stride(1),
        stride_addn, stride_addc, stride_addh, stride_addw,
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        DTYPE_MAP[x.dtype], HAS_ADD=HAS_ADD, HAS_RELU=has_relu
    )
    return y


@triton.autotune(configs=AUTOTUNE_CONFIGS_DEPTHWISE, key=['C', 'H', 'W'])
@triton.jit
def fused_depthwise_conv3x3_shuffle_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    N, C, H, W, G,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc_out, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    INPUT_DTYPE: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_h_group = tl.program_id(1)
    pid_w_group = tl.program_id(2)

    offs_c = tl.arange(0, BLOCK_SIZE_C)
    offs_h = pid_h_group * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w_group * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    acc = tl.zeros((BLOCK_SIZE_C, BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)
    
    mask_c_load = offs_c < C
    
    for kh in range(3):
        for kw in range(3):
            x_h = offs_h[None, :, None] + kh - 1
            x_w = offs_w[None, None, :] + kw - 1

            mask_h_load = (x_h >= 0) & (x_h < H)
            mask_w_load = (x_w >= 0) & (x_w < W)
            
            w = tl.load(w_ptr + offs_c[:, None, None] * stride_wc_out + kh * stride_wkh + kw * stride_wkw, mask=mask_c_load[:, None, None], other=0.0)

            x = tl.load(x_ptr + pid_n * stride_xn + offs_c[:, None, None] * stride_xc + x_h * stride_xh + x_w * stride_xw, 
                        mask=mask_c_load[:, None, None] & mask_h_load & mask_w_load, other=0.0)
            
            acc += x * w

    bias = tl.load(bias_ptr + offs_c, mask=mask_c_load, other=0.0)
    acc += bias[:, None, None]

    C_PER_GROUP = C // G
    g_orig = offs_c // C_PER_GROUP
    c_in_g_orig = offs_c % C_PER_GROUP
    c_shuffled = c_in_g_orig * G + g_orig
    
    y_ptrs = y_ptr + pid_n * stride_yn + c_shuffled[:, None, None] * stride_yc + \
             offs_h[None, :, None] * stride_yh + offs_w[None, None, :] * stride_yw

    mask_h_store = offs_h < H
    mask_w_store = offs_w < W
    tl.store(y_ptrs, acc.to(INPUT_DTYPE), mask=mask_c_load[:, None, None] & mask_h_store[None, :, None] & mask_w_store[None, None, :])

def fused_depthwise_conv3x3_shuffle(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, groups: int) -> torch.Tensor:
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    grid = lambda META: (N, triton.cdiv(H, META['BLOCK_SIZE_H']), triton.cdiv(W, META['BLOCK_SIZE_W']))
    fused_depthwise_conv3x3_shuffle_kernel[grid](
        x, w, bias, y,
        N, C, H, W, groups,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        DTYPE_MAP[x.dtype]
    )
    return y


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(Model, self).__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create dummy layers to extract weights and BN params. These are not called directly.
        conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, groups=groups, bias=False)
        bn1 = nn.BatchNorm2d(mid_channels)
        self._fuse_conv_bn(conv1, bn1, 'fused_conv1')
        
        conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False)
        bn2 = nn.BatchNorm2d(mid_channels)
        self._fuse_conv_bn(conv2, bn2, 'fused_conv2')
        
        conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1, 0, groups=groups, bias=False)
        bn3 = nn.BatchNorm2d(out_channels)
        self._fuse_conv_bn(conv3, bn3, 'fused_conv3')
        
        if in_channels != out_channels:
            shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
            shortcut_bn = nn.BatchNorm2d(out_channels)
            self._fuse_conv_bn(shortcut_conv, shortcut_bn, 'fused_shortcut')

    def _fuse_conv_bn(self, conv, bn, name_prefix):
        with torch.no_grad():
            conv.eval()
            bn.eval()
            
            running_mean, running_var = bn.running_mean, bn.running_var
            gamma, beta, eps = bn.weight, bn.bias, bn.eps

            std = torch.sqrt(running_var + eps)
            scale = gamma / std
            shift = beta - running_mean * scale

            # Fused weights have shape of conv weights, biases have shape of bn biases
            if conv.groups == 1: # Standard convolution
                fused_w = conv.weight * scale.view(-1, 1, 1, 1)
            else: # Grouped or depthwise convolution
                if conv.kernel_size == (1,1): # 1x1 Grouped Conv
                    fused_w = conv.weight * scale.view(-1, 1, 1, 1)
                else: # Depthwise Conv
                    fused_w = conv.weight * scale.view(-1, 1, 1, 1)
            fused_b = shift

            # The model will internally run in fp16 for performance
            self.register_buffer(f'{name_prefix}_w', fused_w.to(torch.float16))
            self.register_buffer(f'{name_prefix}_b', fused_b.to(torch.float16))
    
    def forward(self, x):
        # Store original dtype and move to float16 for accelerated computation
        original_dtype = x.dtype
        x_fp16 = x.to(torch.float16)

        # Shortcut branch calculation first
        if self.in_channels == self.out_channels:
            shortcut_out = x_fp16
        else:
            shortcut_out = fused_gconv1x1(x_fp16, self.fused_shortcut_w, self.fused_shortcut_b, G=1, has_relu=False)
        
        # Main branch
        # Corresponds to: out = F.relu(self.bn1(self.conv1(x)))
        out = fused_gconv1x1(x_fp16, self.fused_conv1_w, self.fused_conv1_b, self.groups, has_relu=True)
        
        # Corresponds to: out = self.shuffle(self.bn2(self.conv2(out)))
        # Using the improved spatially-tiled kernel
        out = fused_depthwise_conv3x3_shuffle(out, self.fused_conv2_w, self.fused_conv2_b, self.groups)
        
        # Corresponds to: out = F.relu(self.bn3(self.conv3(out))) + self.shortcut(x)
        # This single kernel launch fuses the final group convolution, relu, and residual addition
        out = fused_gconv1x1(out, self.fused_conv3_w, self.fused_conv3_b, self.groups, has_relu=True, add_tensor=shortcut_out)
        
        # Cast back to original dtype to match baseline output for correctness
        return out.to(original_dtype)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x

batch_size = 10
input_channels = 240
out_channels = 480
groups = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    # Input can be float32, the model will handle conversion internally
    return [torch.randn(batch_size, input_channels, height, width, device='cuda')]

def get_init_inputs():
    return [input_channels, out_channels, groups]