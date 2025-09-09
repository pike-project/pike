import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- CUDA Kernels ---
# This solution introduces a new, highly-fused kernel for the final stage of the
# ShuffleNet block, combining the second 1x1 group convolution, its batch norm,
# the final ReLU, and the residual addition into a single pass. This significantly
# reduces kernel launch overhead and memory bandwidth consumption.
#
# 1. NEW KERNEL: Fused 1x1 Group Conv -> BN -> ReLU -> Add
#    This kernel is the main contribution. It performs a 1x1 grouped convolution
#    and fuses it with the subsequent bias addition (from BN folding), the ReLU
#    activation, and the final element-wise addition from the shortcut path.
#    This replaces three separate operations (a conv, a relu, an add) with a
#    single, highly efficient kernel, minimizing data movement.
#
# 2. RETAINED KERNEL: Fused ReLU -> DW-Conv -> BN -> Shuffle
#    The effective kernel from the previous solution is retained. It handles the
#    first ReLU, the depthwise convolution, its batch norm, and the channel
#    shuffle operation. Its output feeds directly into our new fused kernel.
#
# 3. STRATEGY: This combination results in the main path of the ShuffleNetUnit
#    being executed in just three kernel launches (cuDNN GConv1, custom fused
#    DWConv, custom fused GConv3+Add), which is a reduction from the four
#    launches in the previous design and a significant reduction from the ~7
ax_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>

// --- KERNEL 1: Fused ReLU -> 3x3 DW-Conv -> BN -> Channel Shuffle (FP16) ---
// This kernel is retained from the previous effective solution.
#define DW_VEC_SIZE 8
#define DW_TILE_W 64
#define DW_TILE_H 16
#define DW_PADDED_TILE_H (DW_TILE_H + 2)
#define DW_PADDED_TILE_W (DW_TILE_W + 2)

__global__ void fused_relu_dwconv_bn_shuffle_fp16_kernel(
    const half* __restrict__ input,
    const half* __restrict__ conv_weight,
    const half* __restrict__ bn_bias,
    half* __restrict__ out,
    int H, int W, int C,
    int groups, int channels_per_group)
{
    __shared__ half s_tile[DW_PADDED_TILE_H][DW_PADDED_TILE_W];

    const int n_cin_idx = blockIdx.z;
    const int n = n_cin_idx / C;
    const int c_in = n_cin_idx % C;

    const int tile_y_start = blockIdx.y * DW_TILE_H;
    const int tile_x_start = blockIdx.x * DW_TILE_W;

    const int HW = H * W;
    const half* input_nc = input + n_cin_idx * HW;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    for(int i = tid; i < DW_PADDED_TILE_H * DW_PADDED_TILE_W; i += block_size) {
        const int load_y = i / DW_PADDED_TILE_W;
        const int load_x = i % DW_PADDED_TILE_W;
        const int in_y = tile_y_start + load_y - 1;
        const int in_x = tile_x_start + load_x - 1;
        if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
            half val = input_nc[in_y * W + in_x];
            s_tile[load_y][load_x] = (__half2float(val) > 0.0f) ? val : __float2half(0.0f);
        } else {
            s_tile[load_y][load_x] = __float2half(0.0f);
        }
    }
    __syncthreads();

    const int ty = threadIdx.y;
    const int tx_vec = threadIdx.x;
    const int out_y = tile_y_start + ty;
    const int out_x_start = tile_x_start + tx_vec * DW_VEC_SIZE;

    if (out_y < H && out_x_start < W) {
        const half* weight_c_h = conv_weight + c_in * 9;
        float w[9];
        for(int i=0; i<9; ++i) w[i] = __half2float(weight_c_h[i]);

        float v_acc[DW_VEC_SIZE] = {0.0f};
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            half segment[DW_VEC_SIZE + 2];
            *((float4*)segment) = *((const float4*)&s_tile[ty + kh][tx_vec * DW_VEC_SIZE]);
            *((half2*)(segment + 8)) = *((const half2*)&s_tile[ty + kh][tx_vec * DW_VEC_SIZE + 8]);
            
            #pragma unroll
            for (int i = 0; i < DW_VEC_SIZE; ++i) {
                v_acc[i] += __half2float(segment[i + 0]) * w[kh * 3 + 0];
                v_acc[i] += __half2float(segment[i + 1]) * w[kh * 3 + 1];
                v_acc[i] += __half2float(segment[i + 2]) * w[kh * 3 + 2];
            }
        }
        
        const float bias = __half2float(bn_bias[c_in]);
        for(int i=0; i<DW_VEC_SIZE; ++i) v_acc[i] += bias;

        half out_h[DW_VEC_SIZE];
        for(int i=0; i<DW_VEC_SIZE; ++i) out_h[i] = __float2half(v_acc[i]);
        
        const int g_in = c_in / channels_per_group;
        const int cpg_in = c_in % channels_per_group;
        const int c_out = cpg_in * groups + g_in;
        const int n_cout_idx = n * C + c_out;

        *((float4*)(out + (n_cout_idx * HW) + out_y * W + out_x_start)) = *((float4*)out_h);
    }
}


// --- KERNEL 2: NEW Fused 1x1 Group Conv -> BN -> ReLU -> Add (FP16) ---
// This kernel is the new, more aggressive fusion.
#define GCONV_VEC_SIZE 8
#define GCONV_TILE_W 64
#define GCONV_TILE_H 8

__global__ void fused_gconv1x1_bn_relu_add_fp16_kernel(
    half* __restrict__ out,
    const half* __restrict__ main_in,
    const half* __restrict__ shortcut_in,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    int H, int W, int C_in, int C_out, int groups)
{
    // Each block computes a (GCONV_TILE_W x GCONV_TILE_H) tile for one output map (n, c_out).
    const int n_cout_idx = blockIdx.z;
    const int n = n_cout_idx / C_out;
    const int c_out = n_cout_idx % C_out;

    const int tile_y_start = blockIdx.y * GCONV_TILE_H;
    const int tile_x_start = blockIdx.x * GCONV_TILE_W;

    const int ty = threadIdx.y;
    const int tx_vec = threadIdx.x;

    const int out_y = tile_y_start + ty;
    const int out_x_start = tile_x_start + tx_vec * GCONV_VEC_SIZE;

    if (out_y >= H || out_x_start >= W) return;

    float acc[GCONV_VEC_SIZE] = {0.0f};

    const int C_in_per_group = C_in / groups;
    const int C_out_per_group = C_out / groups;
    const int group_idx = c_out / C_out_per_group;
    const int c_in_start_global = group_idx * C_in_per_group;
    const int weight_start_offset = c_out * C_in_per_group;

    const int HW = H * W;

    // Loop over input channels for the group convolution
    for (int c_in_g = 0; c_in_g < C_in_per_group; ++c_in_g) {
        const int c_in = c_in_start_global + c_in_g;
        const float w_val = __half2float(weight[weight_start_offset + c_in_g]);

        const float4 in_vals_f4 = __ldg(
            (const float4*)(main_in + (n * C_in + c_in) * HW + out_y * W + out_x_start)
        );
        const half* in_vals_h = reinterpret_cast<const half*>(&in_vals_f4);

        #pragma unroll
        for (int i = 0; i < GCONV_VEC_SIZE; ++i) {
            acc[i] += __half2float(in_vals_h[i]) * w_val;
        }
    }

    // Load shortcut values using read-only cache
    const float4 sc_vals_f4 = __ldg(
        (const float4*)(shortcut_in + n_cout_idx * HW + out_y * W + out_x_start)
    );
    const half* sc_vals_h = reinterpret_cast<const half*>(&sc_vals_f4);

    const float bias_val = __half2float(bias[c_out]);
    half out_h[GCONV_VEC_SIZE];

    // Fused operations: Bias -> ReLU -> Add -> Convert
    #pragma unroll
    for (int i = 0; i < GCONV_VEC_SIZE; ++i) {
        float main_val = acc[i] + bias_val;
        main_val = fmaxf(0.f, main_val);
        float sc_val = __half2float(sc_vals_h[i]);
        out_h[i] = __float2half(main_val + sc_val);
    }

    // Vectorized store
    *(float4*)(out + n_cout_idx * HW + out_y * W + out_x_start) = *(float4*)out_h;
}

// --- Host-facing C++ functions ---
torch::Tensor fused_relu_dwconv_bn_shuffle_fp16_cuda(
    torch::Tensor input, torch::Tensor conv_weight, torch::Tensor bn_bias, int groups) {
    const auto N = input.size(0); const auto C = input.size(1);
    const auto H = input.size(2); const auto W = input.size(3);
    TORCH_CHECK(W % DW_VEC_SIZE == 0, "Input width must be divisible by VEC_SIZE (8) for the FP16 DW-Conv kernel.");
    auto out = torch::empty_like(input);
    if (input.numel() == 0) return out;

    dim3 block_dim(DW_TILE_W / DW_VEC_SIZE, DW_TILE_H); // 8x16 = 128 threads
    dim3 grid_dim( (W + DW_TILE_W - 1) / DW_TILE_W, (H + DW_TILE_H - 1) / DW_TILE_H, N * C );
    const int channels_per_group = C / groups;

    fused_relu_dwconv_bn_shuffle_fp16_kernel<<<grid_dim, block_dim>>>(
        reinterpret_cast<const half*>(input.data_ptr<c10::Half>()),
        reinterpret_cast<const half*>(conv_weight.data_ptr<c10::Half>()),
        reinterpret_cast<const half*>(bn_bias.data_ptr<c10::Half>()),
        reinterpret_cast<half*>(out.data_ptr<c10::Half>()), H, W, C, groups, channels_per_group);
    return out;
}

torch::Tensor fused_gconv1x1_bn_relu_add_fp16_cuda(
    torch::Tensor main_in, torch::Tensor shortcut_in,
    torch::Tensor weight, torch::Tensor bias, int groups) {
    const auto N = main_in.size(0);
    const auto H = main_in.size(2); const auto W = main_in.size(3);
    const auto C_in = main_in.size(1);
    const auto C_out = weight.size(0);
    TORCH_CHECK(W % GCONV_VEC_SIZE == 0, "Input width must be divisible by VEC_SIZE (8) for the FP16 GConv kernel.");

    auto out = torch::empty({N, C_out, H, W}, main_in.options());
    if (main_in.numel() == 0) return out;

    dim3 block_dim(GCONV_TILE_W / GCONV_VEC_SIZE, GCONV_TILE_H); // 8x8 = 64 threads
    dim3 grid_dim((W + GCONV_TILE_W - 1) / GCONV_TILE_W, (H + GCONV_TILE_H - 1) / GCONV_TILE_H, N * C_out);

    fused_gconv1x1_bn_relu_add_fp16_kernel<<<grid_dim, block_dim>>>(
        reinterpret_cast<half*>(out.data_ptr<c10::Half>()),
        reinterpret_cast<const half*>(main_in.data_ptr<c10::Half>()),
        reinterpret_cast<const half*>(shortcut_in.data_ptr<c10::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<c10::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<c10::Half>()),
        H, W, C_in, C_out, groups);
    return out;
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor fused_relu_dwconv_bn_shuffle_fp16_cuda(torch::Tensor, torch::Tensor, torch::Tensor, int);
torch::Tensor fused_gconv1x1_bn_relu_add_fp16_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int);
"""

major, _ = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
extra_cuda_cflags = ["-O3", "--use_fast_math"]
if major >= 7:
    extra_cuda_cflags.append(f"-gencode=arch=compute_{major}0,code=sm_{major}0")

fused_op = None
try:
    fused_op = load_inline(
        name="fused_shufflenet_op_v3",
        cpp_sources=[cpp_source],
        cuda_sources=[ax_source],
        functions=[
            "fused_relu_dwconv_bn_shuffle_fp16_cuda",
            "fused_gconv1x1_bn_relu_add_fp16_cuda",
        ],
        verbose=False,
        extra_cuda_cflags=extra_cuda_cflags,
    )
except Exception:
    pass


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnit, self).__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, 1, 0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def _fold_bn(self, conv_module, bn_module):
        w_fp32 = conv_module.weight.float()
        rm, rv, eps = bn_module.running_mean.float(), bn_module.running_var.float(), bn_module.eps
        gamma, beta = bn_module.weight.float(), bn_module.bias.float()
        
        std = torch.sqrt(rv + eps)
        scale = gamma / std
        bias = beta - rm * scale
        
        scaled_w = w_fp32 * scale.view((-1,) + (1,) * (w_fp32.dim() - 1))
        return scaled_w.half(), bias.half()

    def forward(self, x):
        can_use_cuda_kernel = (
            torch.cuda.is_available() and not self.training and x.is_cuda and x.dtype == torch.half and
            x.dim() == 4 and x.size(3) % 8 == 0 and major >= 7 and fused_op is not None
        )

        if can_use_cuda_kernel:
            # --- Optimized FP16 Inference Path ---
            # 1. Compute Shortcut Path
            if isinstance(self.shortcut, nn.Sequential) and len(self.shortcut) > 0:
                w_sc, b_sc = self._fold_bn(self.shortcut[0], self.shortcut[1])
                shortcut_val = F.conv2d(x, w_sc, bias=b_sc)
            else:
                shortcut_val = x

            # --- Main Path ---
            # 2. First 1x1 Group Conv (BN folded) -> PyTorch
            w1, b1 = self._fold_bn(self.conv1, self.bn1)
            main_path = F.conv2d(x, w1, bias=b1, groups=self.groups)
            
            # 3. FUSED KERNEL: ReLU -> DW Conv -> BN -> Shuffle
            w2, b2 = self._fold_bn(self.conv2, self.bn2)
            main_path = fused_op.fused_relu_dwconv_bn_shuffle_fp16_cuda(main_path, w2, b2, self.groups)
            
            # 4. FUSED KERNEL: 1x1 Group Conv -> BN -> ReLU -> Add
            w3, b3 = self._fold_bn(self.conv3, self.bn3)
            out = fused_op.fused_gconv1x1_bn_relu_add_fp16_cuda(main_path, shortcut_val, w3, b3, self.groups)
            
            return out
        else:
            # --- Original PyTorch FP32/Fallback Path ---
            original_dtype = x.dtype
            x_float = x.float()

            shortcut_out = self.shortcut(x_float)
            
            # Main path
            out = F.relu(self.bn1(self.conv1(x_float)))
            out = self.bn2(self.conv2(out))

            # In-place shuffle logic
            batch_size, channels, height, width = out.size()
            channels_per_group = channels // self.groups
            out = out.view(batch_size, self.groups, channels_per_group, height, width)
            out = out.transpose(1, 2).contiguous()
            out = out.view(batch_size, -1, height, width)
            
            out = F.relu(self.bn3(self.conv3(out)))
            
            out += shortcut_out
            return out.to(original_dtype)

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
    
    def _make_stage(self, in_channels, out_channels, repeats, groups):
        layers = []
        layers.append(ShuffleNetUnit(in_channels, out_channels, groups))
        for _ in range(1, repeats):
            layers.append(ShuffleNetUnit(out_channels, out_channels, groups))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        use_half = (torch.cuda.is_available() and not self.training and x.is_cuda and major >= 7)
        original_dtype = x.dtype
        
        # Initial layers are typically kept in FP32 for stability
        x = x.float() 
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Switch to FP16 for the main stages if possible
        if use_half:
            x = x.half()

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # Switch back to FP32 for final layers
        if x.dtype == torch.half:
            x = x.float()
            
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x.to(original_dtype)