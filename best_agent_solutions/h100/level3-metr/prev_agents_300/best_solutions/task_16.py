import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from typing import List
import triton
import triton.language as tl

# --- Triton Kernel for Fused Final Block (BN + ReLU + AdaptiveAvgPool) ---
# This kernel from the previous solution is well-optimized and remains unchanged.
@triton.jit
def _fused_final_block_kernel_nhwc(
    X_ptr, Y_ptr,
    Weight_ptr, Bias_ptr, Mean_ptr, Var_ptr,
    N, C, H, W,
    X_S_N, X_S_H, X_S_W, X_S_C,
    Y_S_N, Y_S_C,
    EPS: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    n_idx = tl.program_id(0)
    c_group_idx = tl.program_id(1)

    c_offsets = c_group_idx * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C
    
    mean = tl.load(Mean_ptr + c_offsets, mask=c_mask, other=0.0)
    var = tl.load(Var_ptr + c_offsets, mask=c_mask, other=0.0)
    weight = tl.load(Weight_ptr + c_offsets, mask=c_mask, other=0.0)
    bias = tl.load(Bias_ptr + c_offsets, mask=c_mask, other=0.0)
    inv_std = tl.math.rsqrt(var + EPS)

    acc = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    
    x_batch_ptr = X_ptr + n_idx * X_S_N
    for h in range(H):
        for w_start in range(0, W, BLOCK_SIZE_W):
            w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
            w_mask = w_offsets < W
            x_ptr = x_batch_ptr + h * X_S_H + w_offsets[:, None] * X_S_W + c_offsets[None, :] * X_S_C
            x = tl.load(x_ptr, mask=w_mask[:, None] & c_mask[None, :], other=0.0)
            
            normalized = (x - mean[None, :]) * inv_std[None, :]
            scaled = normalized * weight[None, :] + bias[None, :]
            activated = tl.maximum(scaled, 0.0)
            
            acc += tl.sum(activated, axis=0)
            
    avg = acc / (H * W)
    y_ptr = Y_ptr + n_idx * Y_S_N + c_offsets * Y_S_C
    tl.store(y_ptr, avg, mask=c_mask)

def fused_final_block(x: torch.Tensor, bn: nn.BatchNorm2d) -> torch.Tensor:
    if not x.is_contiguous(memory_format=torch.channels_last) or x.dtype != torch.float32:
        x = bn(x)
        x = F.relu(x, inplace=True)
        return F.adaptive_avg_pool2d(x, (1, 1))

    N, C, H, W = x.shape
    y = torch.empty((N, C), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_C = 64
    BLOCK_SIZE_W = 64
    grid = (N, triton.cdiv(C, BLOCK_SIZE_C))
    
    strides = x.stride()
    
    _fused_final_block_kernel_nhwc[grid](
        x, y,
        bn.weight, bn.bias, bn.running_mean, bn.running_var,
        N, C, H, W,
        strides[0], strides[2], strides[3], strides[1],
        y.stride(0), y.stride(1),
        EPS=bn.eps,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        num_warps=4
    )
    return y.view(N, C, 1, 1)

# --- Triton Kernel for Fused Transition Layer (BN -> ReLU -> Conv1x1 -> AvgPool) ---
# This kernel from the previous solution is also well-optimized and remains unchanged.
@triton.jit
def _fused_bn_relu_conv1x1_avgpool_kernel_nhwc(
    X_ptr, Y_ptr, W_ptr,
    BN_W_ptr, BN_B_ptr, BN_M_ptr, BN_V_ptr,
    N, C_in, C_out, H, W,
    X_S_N, X_S_H, X_S_W, X_S_C,
    Y_S_N, Y_S_H, Y_S_W, Y_S_C,
    W_S_CO, W_S_CI,
    EPS: tl.constexpr,
    BLOCK_SIZE_CO: tl.constexpr,
    BLOCK_SIZE_CI: tl.constexpr,
    BLOCK_SIZE_W_OUT: tl.constexpr
):
    pid_nhw_group = tl.program_id(0)
    pid_co_group = tl.program_id(1)

    H_out, W_out = H // 2, W // 2
    
    W_OUT_GROUPS = tl.cdiv(W_out, BLOCK_SIZE_W_OUT)
    w_out_group_idx = pid_nhw_group % W_OUT_GROUPS
    h_out = (pid_nhw_group // W_OUT_GROUPS) % H_out
    n = pid_nhw_group // (W_OUT_GROUPS * H_out)

    co_offsets = pid_co_group * BLOCK_SIZE_CO + tl.arange(0, BLOCK_SIZE_CO)
    co_mask = co_offsets < C_out
    
    w_out_start = w_out_group_idx * BLOCK_SIZE_W_OUT
    w_out_offsets = w_out_start + tl.arange(0, BLOCK_SIZE_W_OUT)
    w_out_mask = w_out_offsets < W_out

    acc_conv_pooled = tl.zeros((BLOCK_SIZE_CO, BLOCK_SIZE_W_OUT), dtype=tl.float32)

    h_in_start = h_out * 2
    w_in_start = w_out_offsets * 2

    for ci_start in range(0, tl.cdiv(C_in, BLOCK_SIZE_CI)):
        ci_offsets = ci_start * BLOCK_SIZE_CI + tl.arange(0, BLOCK_SIZE_CI)
        ci_mask = ci_offsets < C_in

        w_ptr_block = W_ptr + co_offsets[:, None] * W_S_CO + ci_offsets[None, :] * W_S_CI
        w_vals = tl.load(w_ptr_block, mask=co_mask[:, None] & ci_mask[None, :], other=0.0)
        
        bn_m = tl.load(BN_M_ptr + ci_offsets, mask=ci_mask, other=0.0)
        bn_v = tl.load(BN_V_ptr + ci_offsets, mask=ci_mask, other=0.0)
        bn_w = tl.load(BN_W_ptr + ci_offsets, mask=ci_mask, other=0.0)
        bn_b = tl.load(BN_B_ptr + ci_offsets, mask=ci_mask, other=0.0)
        inv_std = tl.math.rsqrt(bn_v + EPS)

        for h_offset in range(2):
            for w_offset in range(2):
                h_in = h_in_start + h_offset
                w_in_bcast = w_in_start + w_offset
                
                x_ptr = X_ptr + n * X_S_N + h_in * X_S_H + w_in_bcast[None, :] * X_S_W + ci_offsets[:, None] * X_S_C
                x_vals = tl.load(x_ptr, mask=ci_mask[:, None] & w_out_mask[None, :], other=0.0)
                
                normalized = (x_vals - bn_m[:, None]) * inv_std[:, None]
                scaled = normalized * bn_w[:, None] + bn_b[:, None]
                activated = tl.maximum(scaled, 0.0)
                
                acc_conv_pooled += tl.dot(w_vals, activated)

    final_val = acc_conv_pooled * 0.25
    
    y_ptr = Y_ptr + n * Y_S_N + h_out * Y_S_H + w_out_offsets[None, :] * Y_S_W + co_offsets[:, None] * Y_S_C
    tl.store(y_ptr, final_val, mask=co_mask[:, None] & w_out_mask[None, :])

def fused_transition_op_correct(x: torch.Tensor, bn: nn.BatchNorm2d, conv: nn.Conv2d) -> torch.Tensor:
    N, C_in, H, W = x.shape
    C_out = conv.out_channels
    if H % 2 != 0 or W % 2 != 0 or not x.is_contiguous(memory_format=torch.channels_last) or x.dtype != torch.float32:
        x = bn(x)
        x = F.relu(x, inplace=True)
        x = conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)

    H_out, W_out = H // 2, W // 2
    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
    
    W_conv = conv.weight.squeeze()
    x_strides, y_strides = x.stride(), y.stride()

    BLOCK_SIZE_CO, BLOCK_SIZE_CI, BLOCK_SIZE_W_OUT = 32, 64, 32
    grid = (N * H_out * triton.cdiv(W_out, BLOCK_SIZE_W_OUT), triton.cdiv(C_out, BLOCK_SIZE_CO))
    
    _fused_bn_relu_conv1x1_avgpool_kernel_nhwc[grid](
        x, y, W_conv,
        bn.weight, bn.bias, bn.running_mean, bn.running_var,
        N, C_in, C_out, H, W,
        x_strides[0], x_strides[2], x_strides[3], x_strides[1],
        y_strides[0], y_strides[2], y_strides[3], y_strides[1],
        W_conv.stride(0), W_conv.stride(1),
        EPS=bn.eps,
        BLOCK_SIZE_CO=BLOCK_SIZE_CO, BLOCK_SIZE_CI=BLOCK_SIZE_CI, BLOCK_SIZE_W_OUT=BLOCK_SIZE_W_OUT,
        num_warps=4,
    )
    return y

# --- NEW Triton Kernel for Fused BN -> ReLU for DenseBlock ---
# This kernel replaces the slow `.contiguous()` call in the previous C++ implementation
# by handling strided inputs directly and fusing the BN and ReLU operations.
@triton.jit
def _fused_bn_relu_kernel_nhwc(
    X_ptr, Y_ptr,
    Weight_ptr, Bias_ptr, Mean_ptr, Var_ptr,
    N, C, H, W,
    X_S_N, X_S_H, X_S_W, X_S_C,
    Y_S_N, Y_S_H, Y_S_W, Y_S_C,
    EPS: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    n_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    cw_group_idx = tl.program_id(2)

    C_GROUPS = tl.cdiv(C, BLOCK_SIZE_C)
    c_group_idx = cw_group_idx % C_GROUPS
    w_group_idx = cw_group_idx // C_GROUPS

    c_offsets = c_group_idx * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < C

    mean = tl.load(Mean_ptr + c_offsets, mask=c_mask, other=0.0)
    var = tl.load(Var_ptr + c_offsets, mask=c_mask, other=0.0)
    weight = tl.load(Weight_ptr + c_offsets, mask=c_mask, other=0.0)
    bias = tl.load(Bias_ptr + c_offsets, mask=c_mask, other=0.0)
    inv_std = tl.math.rsqrt(var + EPS)
    
    w_offsets = w_group_idx * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    w_mask = w_offsets < W

    # Pointer calculations use strides to handle potentially non-contiguous input `X`
    x_ptr = X_ptr + n_idx * X_S_N + h_idx * X_S_H + w_offsets[None, :] * X_S_W + c_offsets[:, None] * X_S_C
    y_ptr = Y_ptr + n_idx * Y_S_N + h_idx * Y_S_H + w_offsets[None, :] * Y_S_W + c_offsets[:, None] * Y_S_C
    
    mask = c_mask[:, None] & w_mask[None, :]
    x = tl.load(x_ptr, mask=mask, other=0.0)

    # Fused BN -> ReLU
    normalized = (x - mean[:, None]) * inv_std[:, None]
    scaled = normalized * weight[:, None] + bias[:, None]
    activated = tl.maximum(scaled, 0.0)
    
    tl.store(y_ptr, activated, mask=mask)

def fused_bn_relu(x: torch.Tensor, bn: nn.BatchNorm2d) -> torch.Tensor:
    N, C, H, W = x.shape
    # Output must be contiguous for the subsequent conv layer
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype, memory_format=torch.channels_last)

    BLOCK_SIZE_C = 64
    BLOCK_SIZE_W = 64
    grid = (N, H, triton.cdiv(C, BLOCK_SIZE_C) * triton.cdiv(W, BLOCK_SIZE_W))
    
    x_strides = x.stride()
    y_strides = y.stride()

    _fused_bn_relu_kernel_nhwc[grid](
        x, y,
        bn.weight, bn.bias, bn.running_mean, bn.running_var,
        N, C, H, W,
        x_strides[0], x_strides[2], x_strides[3], x_strides[1], # input NHWC strides
        y_strides[0], y_strides[2], y_strides[3], y_strides[1], # output NHWC strides
        EPS=bn.eps,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
        num_warps=4
    )
    return y

# --- New Optimized Architecture ---

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        self.growth_rate = growth_rate
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # We omit Dropout(0.0) as it's a no-op
            layer = nn.Sequential(
                nn.BatchNorm2d(num_input_features + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_input_features + i * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
            )
            self.layers.append(layer)

    def forward(self, x):
        if self.training or not x.is_cuda or x.dtype != torch.float32 or not x.is_contiguous(memory_format=torch.channels_last):
            features = [x]
            concatenated_input = x
            for layer in self.layers:
                new_feature = layer(concatenated_input)
                features.append(new_feature)
                concatenated_input = torch.cat(features, 1)
            return concatenated_input

        # Optimized Inference Path with pre-allocation and fused BN-ReLU
        n, c_in, h, w = x.shape
        num_output_features = c_in + self.num_layers * self.growth_rate
        output = torch.empty((n, num_output_features, h, w), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
        output.narrow(1, 0, c_in).copy_(x)

        for i, layer in enumerate(self.layers):
            bn, _, conv = layer
            current_c_in = c_in + i * self.growth_rate
            input_view = output.narrow(1, 0, current_c_in)
            
            # Use fused kernel to handle non-contiguous view and create dense output for conv
            bn_relu_out = fused_bn_relu(input_view, bn)
            
            new_feature = conv(bn_relu_out)
            
            # Copy the result into the pre-allocated output tensor
            write_offset = c_in + i * self.growth_rate
            output.narrow(1, write_offset, self.growth_rate).copy_(new_feature)
            
        return output

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        if not self.training and x.is_cuda and x.dtype == torch.float32 and x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0:
             return fused_transition_op_correct(x, self.bn, self.conv)
        
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class Model(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(Model, self).__init__()
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial_bn = nn.BatchNorm2d(64)
        self.initial_relu = nn.ReLU(inplace=True)
        self.initial_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.register_buffer('initial_folded_w', None, persistent=False)
        self.register_buffer('initial_folded_b', None, persistent=False)

        num_features = 64
        block_layers = [6, 12, 48, 32]
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features += num_layers * growth_rate

            if i != len(block_layers) - 1:
                transition = TransitionLayer(num_input_features=num_features, num_output_features=num_features // 2)
                self.transition_layers.append(transition)
                num_features //= 2

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        self.to(memory_format=torch.channels_last)

        self.graph = None
        self.static_input = None
        self.static_output = None

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            if self.initial_folded_w is None:
                conv_w, bn = self.initial_conv.weight, self.initial_bn
                inv_std = torch.rsqrt(bn.running_var + bn.eps)
                scale = bn.weight * inv_std
                self.initial_folded_w = conv_w * scale.view(-1, 1, 1, 1)
                self.initial_folded_b = bn.bias - bn.running_mean * scale
            x = F.conv2d(x, self.initial_folded_w, self.initial_folded_b, stride=2, padding=3)
            x = self.initial_relu(x)
        else:
            x = self.initial_conv(x)
            x = self.initial_bn(x)
            x = self.initial_relu(x)
        x = self.initial_pool(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transition_layers):
                x = self.transition_layers[i](x)

        if not self.training and x.dim() == 4 and x.is_cuda:
            x = fused_final_block(x, self.final_bn)
        else:
            x = self.final_bn(x)
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.initial_folded_w, self.initial_folded_b = None, None
            self.graph, self.static_input, self.static_output = None, None, None
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or not x.is_cuda:
            return self._forward_impl(x)

        if self.graph is None or self.static_input.shape != x.shape:
            self.static_input = torch.empty_like(x)
            self.static_input.copy_(x)
            
            torch.cuda.synchronize()
            for _ in range(3):
                _ = self._forward_impl(self.static_input)
            torch.cuda.synchronize()
            
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self._forward_impl(self.static_input)

        self.static_input.copy_(x, non_blocking=True)
        self.graph.replay()
        return self.static_output

batch_size = 10
num_classes = 10
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, 3, height, width).to(memory_format=torch.channels_last)]

def get_init_inputs():
    return [32, num_classes]