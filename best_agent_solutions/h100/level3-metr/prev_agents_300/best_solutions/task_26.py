import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# --- Kernel 1: Fused 3x3 Standard Convolution + BatchNorm + ReLU (for the entry layer) ---
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 32, 'BLOCK_CIN': 8}, num_warps=4),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16, 'BLOCK_CIN': 8}, num_warps=4),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 64, 'BLOCK_CIN': 8}, num_warps=4),
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 64, 'BLOCK_CIN': 4}, num_warps=8),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 32, 'BLOCK_CIN': 4}, num_warps=8),
    ],
    key=['H_out', 'W_out', 'C_out', 'C_in'],
)
@triton.jit
def fused_conv3x3_bn_relu_kernel(
    X_ptr, Y_ptr, W_ptr, BN_SCALE_ptr, BN_BIAS_ptr,
    N, C_in, H_in, W_in, C_out, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    stride_wc_out, stride_wc_in, stride_wkh, stride_wkw,
    stride_h, stride_w, padding_h, padding_w,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_CIN: tl.constexpr
):
    pid_nc = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    
    n = pid_nc // C_out
    c_out = pid_nc % C_out

    num_blocks_w = tl.cdiv(W_out, BLOCK_W)
    block_h_idx = pid_spatial // num_blocks_w
    block_w_idx = pid_spatial % num_blocks_w
    
    offs_h_out = block_h_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_w_out = block_w_idx * BLOCK_W + tl.arange(0, BLOCK_W)

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    w_ptr_base = W_ptr + c_out * stride_wc_out
    
    for c_in_start in range(0, C_in, BLOCK_CIN):
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_CIN)
        c_in_mask = c_in_offsets < C_in
        
        for kh in range(3):
            for kw in range(3):
                # Load weights for this filter patch
                w_offs = c_in_offsets[None, :] * stride_wc_in + kh * stride_wkh + kw * stride_wkw
                w_vals = tl.load(w_ptr_base + w_offs, mask=c_in_mask[None, :], other=0.0)

                # Calculate input coordinates, broadcasting 1D ranges to 3D tensors
                # Shape: [BLOCK_H, 1, 1]
                h_in = (offs_h_out[:, None, None] * stride_h) + kh - padding_h
                # Shape: [1, BLOCK_W, 1]
                w_in = (offs_w_out[None, :, None] * stride_w) + kw - padding_w
                
                # Calculate input offsets for loading a 3D block of input data
                x_offs = (n * stride_xn +
                          c_in_offsets[None, None, :] * stride_xc +
                          h_in * stride_xh +
                          w_in * stride_xw)

                # Calculate the load mask for the 3D block
                mask_h = (h_in >= 0) & (h_in < H_in)
                mask_w = (w_in >= 0) & (w_in < W_in)
                x_mask = mask_h & mask_w & c_in_mask[None, None, :]
                
                # Load 3D block of input data
                x_vals = tl.load(X_ptr + x_offs, mask=x_mask, other=0.0)
                
                # Accumulate: multiply input block with weights and sum over the channel dimension
                acc += tl.sum(x_vals.to(tl.float32) * w_vals.to(tl.float32), axis=2)

    bn_scale = tl.load(BN_SCALE_ptr + c_out).to(tl.float32)
    bn_bias = tl.load(BN_BIAS_ptr + c_out).to(tl.float32)
    result = acc * bn_scale + bn_bias
    result = tl.maximum(result, 0.0)

    y_offs_h = offs_h_out[:, None]
    y_offs_w = offs_w_out[None, :]
    store_mask = (y_offs_h < H_out) & (y_offs_w < W_out)

    y_offs = n * stride_yn + c_out * stride_yc + y_offs_h * stride_yh + y_offs_w * stride_yw
    tl.store(Y_ptr + y_offs, result.to(Y_ptr.dtype.element_ty), mask=store_mask)


def fused_conv3x3_bn_relu_wrapper(x, conv, bn):
    N, C_in, H_in, W_in = x.shape
    conv_w = conv.weight
    C_out = conv_w.shape[0]

    H_out = (H_in + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
    W_out = (W_in + 2 * conv.padding[1] - conv.kernel_size[1]) // conv.stride[1] + 1
    
    bn_w, bn_b, bn_rm, bn_rv, bn_eps = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
    bn_scale = bn_w / torch.sqrt(bn_rv + bn_eps)
    bn_bias_fused = bn_b - bn_rm * bn_scale
    
    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (
        N * C_out,
        triton.cdiv(H_out, META['BLOCK_H']) * triton.cdiv(W_out, META['BLOCK_W'])
    )
    
    fused_conv3x3_bn_relu_kernel[grid](
        x, y, conv_w, bn_scale, bn_bias_fused,
        N, C_in, H_in, W_in, C_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        conv_w.stride(0), conv_w.stride(1), conv_w.stride(2), conv_w.stride(3),
        conv.stride[0], conv.stride[1], conv.padding[0], conv.padding[1]
    )
    return y


# --- Kernel 2: General-purpose Fused 1x1 Grouped Conv + BN + optional Residual/ReLU ---
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['N', 'H', 'W', 'C_in', 'C_out', 'groups'],
)
@triton.jit
def fused_gconv_bn_kernel(
    X_ptr, Y_ptr, W_ptr,
    BN_SCALE_ptr, BN_BIAS_ptr,
    RESIDUAL_ptr,
    N, H, W, C_in, C_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    stride_wn, stride_wk,
    stride_resn, stride_resc, stride_resh, stride_resw,
    groups,
    APPLY_RELU: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    RELU_ORDER_IS_BEFORE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    pid_group = tl.program_id(axis=1)

    C_in_per_group = C_in // groups
    C_out_per_group = C_out // groups
    
    num_pid_m = tl.cdiv(N * H * W, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(C_out_per_group, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    n_idx = offs_m // (H * W)
    rem_m = offs_m % (H * W)
    h_idx = rem_m // W
    w_idx = rem_m % W
    
    m_mask = offs_m < (N * H * W)

    for k in range(0, tl.cdiv(C_in_per_group, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = offs_k < C_in_per_group
        
        x_offs_c = pid_group * C_in_per_group + offs_k[None, :]
        x_offs = (n_idx[:, None] * stride_xn + x_offs_c * stride_xc + 
                  h_idx[:, None] * stride_xh + w_idx[:, None] * stride_xw)
        x_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(X_ptr + x_offs, mask=x_mask, other=0.0)

        w_offs_c_out = pid_group * C_out_per_group + offs_n[None, :]
        w_offs = (w_offs_c_out * stride_wn + offs_k[:, None] * stride_wk)
        w_mask = (w_offs_c_out < C_out) & k_mask[:, None]
        b = tl.load(W_ptr + w_offs, mask=w_mask, other=0.0)
        
        accumulator += tl.dot(a, b, allow_tf32=True)

    c_out_idx = pid_group * C_out_per_group + offs_n
    n_mask = c_out_idx < C_out
    
    scale = tl.load(BN_SCALE_ptr + c_out_idx, mask=n_mask, other=0.0).to(tl.float32)
    bias = tl.load(BN_BIAS_ptr + c_out_idx, mask=n_mask, other=0.0).to(tl.float32)
    result = accumulator * scale[None, :] + bias[None, :]
    
    if APPLY_RELU and RELU_ORDER_IS_BEFORE:
        result = tl.maximum(result, 0.0)

    if ADD_RESIDUAL:
        res_offs = (n_idx[:, None] * stride_resn + c_out_idx[None, :] * stride_resc + 
                    h_idx[:, None] * stride_resh + w_idx[:, None] * stride_resw)
        res_mask = m_mask[:, None] & n_mask[None, :]
        residual = tl.load(RESIDUAL_ptr + res_offs, mask=res_mask, other=0.0)
        result += residual.to(tl.float32)

    if APPLY_RELU and not RELU_ORDER_IS_BEFORE:
        result = tl.maximum(result, 0.0)

    y_offs = (n_idx[:, None] * stride_yn + c_out_idx[None, :] * stride_yc + 
              h_idx[:, None] * stride_yh + w_idx[:, None] * stride_yw)
    y_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(Y_ptr + y_offs, result.to(Y_ptr.dtype.element_ty), mask=y_mask)

def fused_gconv_bn_wrapper(x, conv, bn, groups, residual=None, apply_relu=False, relu_order='after'):
    N, C_in, H, W = x.shape
    conv_w = conv.weight
    C_out = conv_w.shape[0]
    
    bn_w, bn_b, bn_rm, bn_rv, bn_eps = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
    bn_scale = bn_w / torch.sqrt(bn_rv + bn_eps)
    bn_bias_fused = bn_b - bn_rm * bn_scale
    
    y = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)
    
    add_residual = residual is not None
    if not add_residual:
        residual = torch.empty(0, device=x.device, dtype=x.dtype)
        stride_resn, stride_resc, stride_resh, stride_resw = 0, 0, 0, 0
    else:
        stride_resn, stride_resc, stride_resh, stride_resw = residual.stride()
        
    conv_w_squeezed = conv_w.squeeze().contiguous()
    
    grid = lambda META: (
        triton.cdiv(N * H * W, META['BLOCK_SIZE_M']) * triton.cdiv(C_out // groups, META['BLOCK_SIZE_N']),
        groups
    )
    
    fused_gconv_bn_kernel[grid](
        x, y, conv_w_squeezed, bn_scale, bn_bias_fused, residual,
        N, H, W, C_in, C_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        conv_w_squeezed.stride(0), conv_w_squeezed.stride(1),
        stride_resn, stride_resc, stride_resh, stride_resw,
        groups=groups, APPLY_RELU=apply_relu, ADD_RESIDUAL=add_residual,
        RELU_ORDER_IS_BEFORE=(relu_order == 'before')
    )
    return y

# --- Kernel 3: Fused 3x3 Depthwise Convolution + BN + Channel Shuffle ---
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 8, 'BLOCK_W': 32}, num_warps=4),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 64}, num_warps=4),
        triton.Config({'BLOCK_H': 1, 'BLOCK_W': 128}, num_warps=4),
        triton.Config({'BLOCK_H': 1, 'BLOCK_W': 256}, num_warps=8),
        triton.Config({'BLOCK_H': 16, 'BLOCK_W': 16}, num_warps=4),
    ],
    key=['H', 'W', 'C'],
)
@triton.jit
def fused_dwconv_bn_shuffle_kernel(
    X_ptr, Y_ptr, W_ptr, BN_SCALE_ptr, BN_BIAS_ptr,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    stride_wc,
    groups,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr
):
    pid_nc = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    
    n = pid_nc // C
    c_out = pid_nc % C
    
    channels_per_group = C // groups
    group_idx = c_out % groups
    channel_in_group_idx = c_out // groups
    c_in = group_idx * channels_per_group + channel_in_group_idx
    
    num_blocks_w = tl.cdiv(W, BLOCK_W)
    block_h_idx = pid_spatial // num_blocks_w
    block_w_idx = pid_spatial % num_blocks_w
    
    h_start = block_h_idx * BLOCK_H
    w_start = block_w_idx * BLOCK_W
    
    offs_h = h_start + tl.arange(0, BLOCK_H)
    offs_w = w_start + tl.arange(0, BLOCK_W)
    
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    
    w_ptr_base = W_ptr + c_in * stride_wc

    for kh in range(3):
        for kw in range(3):
            w_val = tl.load(w_ptr_base + kh * 3 + kw).to(tl.float32)
            
            h_in = offs_h[:, None] + kh - 1
            w_in = offs_w[None, :] + kw - 1
            
            mask_h = (h_in >= 0) & (h_in < H)
            mask_w = (w_in >= 0) & (w_in < W)
            load_mask = mask_h & mask_w
            
            x_offs = (n * stride_xn + c_in * stride_xc + h_in * stride_xh + w_in * stride_xw)
            x_val = tl.load(X_ptr + x_offs, mask=load_mask, other=0.0)
            acc += x_val.to(tl.float32) * w_val

    bn_scale = tl.load(BN_SCALE_ptr + c_in).to(tl.float32)
    bn_bias = tl.load(BN_BIAS_ptr + c_in).to(tl.float32)
    result = acc * bn_scale + bn_bias
    
    y_offs_h = offs_h[:, None]
    y_offs_w = offs_w[None, :]
    store_mask = (y_offs_h < H) & (y_offs_w < W)
    y_offs = (n * stride_yn + c_out * stride_yc + y_offs_h * stride_yh + y_offs_w * stride_yw)
    tl.store(Y_ptr + y_offs, result.to(Y_ptr.dtype.element_ty), mask=store_mask)

def fused_dwconv_bn_shuffle_wrapper(x, conv, bn, groups):
    N, C, H, W = x.shape
    conv_w = conv.weight
    
    bn_w, bn_b, bn_rm, bn_rv, bn_eps = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
    bn_scale = bn_w / torch.sqrt(bn_rv + bn_eps)
    bn_bias_fused = bn_b - bn_rm * bn_scale
    
    y = torch.empty_like(x)
    
    grid = lambda META: (
        N * C,
        triton.cdiv(H, META['BLOCK_H']) * triton.cdiv(W, META['BLOCK_W'])
    )
    
    fused_dwconv_bn_shuffle_kernel[grid](
        x, y, conv_w, bn_scale, bn_bias_fused,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        conv_w.stride(0),
        groups=groups,
    )
    return y

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

# --- New ShuffleNet Architecture with Fused Kernels and CUDA Graphs ---
class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnit, self).__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1, 0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.groups = groups
        self.shuffle = ChannelShuffle(groups)
        self.has_shortcut_conv = in_channels != out_channels
        
        if self.has_shortcut_conv:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        if self.training:
            if self.has_shortcut_conv:
                shortcut_val = self.shortcut_bn(self.shortcut_conv(x))
            else:
                shortcut_val = x
            
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.shuffle(out) # Note: Original ShuffleNetV1 shuffles after first GConv
            out = self.bn2(self.conv2(out))
            out = self.bn3(self.conv3(out)) # Note: Original has GConv -> BN
            out = F.relu(out + shortcut_val) # Note: ReLU after add
            return out
        else:
            # Fused inference path
            if self.has_shortcut_conv:
                shortcut_val = fused_gconv_bn_wrapper(x, self.shortcut_conv, self.shortcut_bn, groups=1, apply_relu=False)
            else:
                shortcut_val = x

            out = fused_gconv_bn_wrapper(x, self.conv1, self.bn1, groups=self.groups, apply_relu=True)
            out = fused_dwconv_bn_shuffle_wrapper(out, self.conv2, self.bn2, groups=self.groups)
            out = fused_gconv_bn_wrapper(out, self.conv3, self.bn3, groups=self.groups, 
                                         residual=shortcut_val, apply_relu=True, relu_order='before')
            return out

class Model(nn.Module):
    def __init__(self, num_classes=1000, groups=3, stages_repeats=[3, 7, 3], stages_out_channels=[24, 240, 480, 960]):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, stages_out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stages_out_channels[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage2 = self._make_stage(stages_out_channels[0], stages_out_channels[1], stages_repeats[0], groups)
        self.stage3 = self._make_stage(stages_out_channels[1], stages_out_channels[2], stages_repeats[1], groups)
        self.stage4 = self._make_stage(stages_out_channels[2], stages_out_channels[3], stages_repeats[2], groups)
        
        self.conv5 = nn.Conv2d(stages_out_channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.fc = nn.Linear(1024, num_classes)
        
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = [ShuffleNetUnit(in_channels, out_channels, groups)]
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def _forward_eager(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _forward_impl(self, x):
        x = fused_conv3x3_bn_relu_wrapper(x, self.conv1, self.bn1)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = fused_gconv_bn_wrapper(x, self.conv5, self.bn5, groups=1, apply_relu=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x):
        if self.training:
            return self._forward_eager(x)
        
        if self.graph is None or x.shape != self.static_input.shape:
            self.static_input = x
            
            # Warmup
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.static_output = self._forward_impl(self.static_input)
            torch.cuda.current_stream().wait_stream(s)
            
            # Trace
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self._forward_impl(self.static_input)

        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

# Test code
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width, dtype=torch.float16).cuda()]

def get_init_inputs():
    return [num_classes]