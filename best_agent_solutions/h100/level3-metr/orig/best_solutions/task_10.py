import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------- Custom CUDA Kernel Definition (Vectorized FP16) -----------------

# C++ string containing the forward declarations of the CUDA functions.
# The user-facing function signatures do not change.
cpp_source = """
#include <torch/extension.h>

// Fused Bias + ReLU
torch::Tensor fused_bias_relu_fp16_cuda(torch::Tensor conv_out, torch::Tensor bias);

// Fused Bias + Add (residual) + ReLU
torch::Tensor fused_bias_add_relu_fp16_cuda(torch::Tensor conv_out, torch::Tensor bias, torch::Tensor identity);

// Fused Bias + ReLU + MaxPool(3x3, s2, p1)
torch::Tensor fused_bias_relu_maxpool_fp16_cuda(torch::Tensor conv_out, torch::Tensor bias);

// Fused (Conv+Bias) + (DownsampleConv+Bias) + ReLU
torch::Tensor fused_dual_bias_add_relu_fp16_cuda(
    torch::Tensor main_path_conv,
    torch::Tensor main_path_bias,
    torch::Tensor shortcut_path_conv,
    torch::Tensor shortcut_path_bias
);
"""

# CUDA source string containing both scalar (fallback) and new vectorized kernels.
# The vectorized kernels process two 'half' elements at a time using 'half2'.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Half.h>

// --- Scalar Fallback Kernels (for odd-sized dimensions) ---
__global__ void scalar_fused_bias_relu_kernel_fp16(
    const half* conv_out, const half* bias, half* relu_out,
    int total_elements, int spatial_dim, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int channel_idx = (idx / spatial_dim) % C;
        half bias_val = bias[channel_idx];
        half conv_val = conv_out[idx];
        half added_val = __hadd(conv_val, bias_val);
        relu_out[idx] = __hmax(added_val, __float2half(0.0f));
    }
}

__global__ void scalar_fused_bias_add_relu_kernel_fp16(
    const half* conv_out, const half* bias, const half* identity, half* final_out,
    int total_elements, int spatial_dim, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int channel_idx = (idx / spatial_dim) % C;
        half bias_val = bias[channel_idx];
        half conv_val = conv_out[idx];
        half identity_val = identity[idx];
        half added_val = __hadd(__hadd(conv_val, bias_val), identity_val);
        final_out[idx] = __hmax(added_val, __float2half(0.0f));
    }
}

__global__ void scalar_fused_dual_bias_add_relu_kernel_fp16(
    const half* main_path_conv, const half* main_path_bias,
    const half* shortcut_path_conv, const half* shortcut_path_bias,
    half* final_out, int total_elements, int spatial_dim, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int channel_idx = (idx / spatial_dim) % C;
        half main_bias_val = main_path_bias[channel_idx];
        half shortcut_bias_val = shortcut_path_bias[channel_idx];
        half main_conv_val = main_path_conv[idx];
        half shortcut_conv_val = shortcut_path_conv[idx];
        half main_val = __hadd(main_conv_val, main_bias_val);
        half shortcut_val = __hadd(shortcut_conv_val, shortcut_bias_val);
        half added_val = __hadd(main_val, shortcut_val);
        final_out[idx] = __hmax(added_val, __float2half(0.0f));
    }
}

// --- IMPROVED: Vectorized Kernels using half2 ---
// These kernels use a grid-stride loop for robustness.
__global__ void vectorized_fused_bias_relu_kernel_fp16(
    const half2* conv_out, const half* bias, half2* relu_out,
    int total_half2_elements, int spatial_dim, int C)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_half2_elements;
         i += gridDim.x * blockDim.x)
    {
        int element_idx = i * 2;
        int channel_idx = (element_idx / spatial_dim) % C;
        half2 bias_h2 = __half2half2(bias[channel_idx]);
        half2 conv_val = conv_out[i];
        half2 added = __hadd2(conv_val, bias_h2);
        relu_out[i] = __hmax2(added, __float2half2_rn(0.0f));
    }
}

__global__ void vectorized_fused_bias_add_relu_kernel_fp16(
    const half2* conv_out, const half* bias, const half2* identity, half2* final_out,
    int total_half2_elements, int spatial_dim, int C)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_half2_elements;
         i += gridDim.x * blockDim.x)
    {
        int element_idx = i * 2;
        int channel_idx = (element_idx / spatial_dim) % C;
        half2 bias_h2 = __half2half2(bias[channel_idx]);
        half2 conv_val = conv_out[i];
        half2 identity_val = identity[i];
        half2 added = __hadd2(__hadd2(conv_val, bias_h2), identity_val);
        final_out[i] = __hmax2(added, __float2half2_rn(0.0f));
    }
}

__global__ void vectorized_fused_dual_bias_add_relu_kernel_fp16(
    const half2* main_path_conv, const half* main_path_bias,
    const half2* shortcut_path_conv, const half* shortcut_path_bias,
    half2* final_out, int total_half2_elements, int spatial_dim, int C)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < total_half2_elements;
         i += gridDim.x * blockDim.x)
    {
        int element_idx = i * 2;
        int channel_idx = (element_idx / spatial_dim) % C;
        half2 main_bias_h2 = __half2half2(main_path_bias[channel_idx]);
        half2 shortcut_bias_h2 = __half2half2(shortcut_path_bias[channel_idx]);
        half2 main_conv_val = main_path_conv[i];
        half2 shortcut_conv_val = shortcut_path_conv[i];
        half2 main_val = __hadd2(main_conv_val, main_bias_h2);
        half2 shortcut_val = __hadd2(shortcut_conv_val, shortcut_bias_h2);
        half2 added = __hadd2(main_val, shortcut_val);
        final_out[i] = __hmax2(added, __float2half2_rn(0.0f));
    }
}


// --- UNCHANGED: Fused Bias + ReLU + MaxPool Kernel (Shared Memory) ---
#define TILE_DIM 16
#define POOL_KERNEL_SIZE 3
#define POOL_STRIDE 2
#define POOL_PADDING 1
#define INPUT_TILE_WIDTH ( (TILE_DIM - 1) * POOL_STRIDE + POOL_KERNEL_SIZE )

__global__ void fused_bias_relu_maxpool_shared_mem_kernel_fp16(
    const half* conv_out, const half* bias, half* out,
    int H_in, int W_in, int H_out, int W_out, int C)
{
    __shared__ half s_tile[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH];
    const half neg_inf = __float2half(-INFINITY);

    int channel_batch_idx = blockIdx.z;
    int c = channel_batch_idx % C;
    int n = channel_batch_idx / C;
    half current_bias = bias[c];

    int out_tile_x_start = blockIdx.x * TILE_DIM;
    int out_tile_y_start = blockIdx.y * TILE_DIM;
    int in_tile_x_start = out_tile_x_start * POOL_STRIDE - POOL_PADDING;
    int in_tile_y_start = out_tile_y_start * POOL_STRIDE - POOL_PADDING;

    for (int i = threadIdx.y; i < INPUT_TILE_WIDTH; i += blockDim.y) {
        for (int j = threadIdx.x; j < INPUT_TILE_WIDTH; j += blockDim.x) {
            int in_x = in_tile_x_start + j;
            int in_y = in_tile_y_start + i;
            if (in_x >= 0 && in_x < W_in && in_y >= 0 && in_y < H_in) {
                int input_idx = n * C * H_in * W_in + c * H_in * W_in + in_y * W_in + in_x;
                half val = __hadd(conv_out[input_idx], current_bias);
                s_tile[i][j] = __hmax(val, __float2half(0.0f));
            } else {
                s_tile[i][j] = neg_inf;
            }
        }
    }
    __syncthreads();
    
    int out_x = out_tile_x_start + threadIdx.x;
    int out_y = out_tile_y_start + threadIdx.y;
    if (out_x < W_out && out_y < H_out) {
        int s_x_start = threadIdx.x * POOL_STRIDE;
        int s_y_start = threadIdx.y * POOL_STRIDE;
        half max_val = neg_inf;
        for (int kh = 0; kh < POOL_KERNEL_SIZE; ++kh) {
            for (int kw = 0; kw < POOL_KERNEL_SIZE; ++kw) {
                max_val = __hmax(max_val, s_tile[s_y_start + kh][s_x_start + kw]);
            }
        }
        int output_idx = n * C * H_out * W_out + c * H_out * W_out + out_y * W_out + out_x;
        out[output_idx] = max_val;
    }
}


// --- C++ Wrapper Functions with Vectorization Dispatch Logic ---
#define CHECK_FP16(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kHalf, #x " must be a contiguous CUDA Half tensor")
#define BLOCK_SIZE 256

torch::Tensor fused_bias_relu_fp16_cuda(torch::Tensor conv_out, torch::Tensor bias) {
    CHECK_FP16(conv_out); CHECK_FP16(bias);
    auto out = torch::empty_like(conv_out);
    const int W = conv_out.size(3);
    const int C = conv_out.size(1);
    const int H = conv_out.size(2);
    const int spatial_dim = H * W;

    if (W > 0 && W % 2 == 0) { // Vectorized path for even width
        const int total_half2_elements = conv_out.numel() / 2;
        const int num_blocks = (total_half2_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vectorized_fused_bias_relu_kernel_fp16<<<num_blocks, BLOCK_SIZE>>>(
            reinterpret_cast<const half2*>(conv_out.data_ptr<c10::Half>()),
            (const half*)bias.data_ptr<c10::Half>(),
            reinterpret_cast<half2*>(out.data_ptr<c10::Half>()),
            total_half2_elements, spatial_dim, C);
    } else { // Scalar fallback for odd width
        const int total_elements = conv_out.numel();
        const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scalar_fused_bias_relu_kernel_fp16<<<num_blocks, BLOCK_SIZE>>>(
            (const half*)conv_out.data_ptr<c10::Half>(), (const half*)bias.data_ptr<c10::Half>(),
            (half*)out.data_ptr<c10::Half>(), total_elements, spatial_dim, C);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_bias_add_relu_fp16_cuda(torch::Tensor conv_out, torch::Tensor bias, torch::Tensor identity) {
    CHECK_FP16(conv_out); CHECK_FP16(bias); CHECK_FP16(identity);
    auto out = torch::empty_like(conv_out);
    const int W = conv_out.size(3);
    const int C = conv_out.size(1);
    const int H = conv_out.size(2);
    const int spatial_dim = H * W;

    if (W > 0 && W % 2 == 0) { // Vectorized path
        const int total_half2_elements = conv_out.numel() / 2;
        const int num_blocks = (total_half2_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vectorized_fused_bias_add_relu_kernel_fp16<<<num_blocks, BLOCK_SIZE>>>(
            reinterpret_cast<const half2*>(conv_out.data_ptr<c10::Half>()),
            (const half*)bias.data_ptr<c10::Half>(),
            reinterpret_cast<const half2*>(identity.data_ptr<c10::Half>()),
            reinterpret_cast<half2*>(out.data_ptr<c10::Half>()),
            total_half2_elements, spatial_dim, C);
    } else { // Scalar fallback
        const int total_elements = conv_out.numel();
        const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scalar_fused_bias_add_relu_kernel_fp16<<<num_blocks, BLOCK_SIZE>>>(
            (const half*)conv_out.data_ptr<c10::Half>(), (const half*)bias.data_ptr<c10::Half>(),
            (const half*)identity.data_ptr<c10::Half>(), (half*)out.data_ptr<c10::Half>(),
            total_elements, spatial_dim, C);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_dual_bias_add_relu_fp16_cuda(torch::Tensor main_conv, torch::Tensor main_bias, torch::Tensor shortcut_conv, torch::Tensor shortcut_bias) {
    CHECK_FP16(main_conv); CHECK_FP16(main_bias); CHECK_FP16(shortcut_conv); CHECK_FP16(shortcut_bias);
    TORCH_CHECK(main_conv.sizes() == shortcut_conv.sizes(), "Feature map tensors must have the same shape");
    auto out = torch::empty_like(main_conv);
    const int W = main_conv.size(3);
    const int C = main_conv.size(1);
    const int H = main_conv.size(2);
    const int spatial_dim = H * W;

    if (W > 0 && W % 2 == 0) { // Vectorized path
        const int total_half2_elements = main_conv.numel() / 2;
        const int num_blocks = (total_half2_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vectorized_fused_dual_bias_add_relu_kernel_fp16<<<num_blocks, BLOCK_SIZE>>>(
            reinterpret_cast<const half2*>(main_conv.data_ptr<c10::Half>()), (const half*)main_bias.data_ptr<c10::Half>(),
            reinterpret_cast<const half2*>(shortcut_conv.data_ptr<c10::Half>()), (const half*)shortcut_bias.data_ptr<c10::Half>(),
            reinterpret_cast<half2*>(out.data_ptr<c10::Half>()),
            total_half2_elements, spatial_dim, C);
    } else { // Scalar fallback
        const int total_elements = main_conv.numel();
        const int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scalar_fused_dual_bias_add_relu_kernel_fp16<<<num_blocks, BLOCK_SIZE>>>(
             (const half*)main_conv.data_ptr<c10::Half>(), (const half*)main_bias.data_ptr<c10::Half>(),
             (const half*)shortcut_conv.data_ptr<c10::Half>(), (const half*)shortcut_bias.data_ptr<c10::Half>(),
             (half*)out.data_ptr<c10::Half>(), total_elements, spatial_dim, C);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_bias_relu_maxpool_fp16_cuda(torch::Tensor conv_out, torch::Tensor bias) {
    CHECK_FP16(conv_out); CHECK_FP16(bias);
    const auto N = conv_out.size(0), C = conv_out.size(1), H_in = conv_out.size(2), W_in = conv_out.size(3);
    const int H_out = (H_in + 2 * POOL_PADDING - POOL_KERNEL_SIZE) / POOL_STRIDE + 1;
    const int W_out = (W_in + 2 * POOL_PADDING - POOL_KERNEL_SIZE) / POOL_STRIDE + 1;
    auto out = torch::empty({N, C, H_out, W_out}, conv_out.options());

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid( (W_out + TILE_DIM - 1) / TILE_DIM, (H_out + TILE_DIM - 1) / TILE_DIM, N * C );

    fused_bias_relu_maxpool_shared_mem_kernel_fp16<<<grid, block>>>(
        (const half*)conv_out.data_ptr<c10::Half>(), (const half*)bias.data_ptr<c10::Half>(),
        (half*)out.data_ptr<c10::Half>(), H_in, W_in, H_out, W_out, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

# JIT compile the vectorized FP16 CUDA kernels.
resnet_kernels_fp16 = load_inline(
    name="resnet_kernels_fp16_vectorized", # Changed name to avoid build conflicts
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=[
        "fused_bias_relu_fp16_cuda",
        "fused_bias_add_relu_fp16_cuda",
        "fused_bias_relu_maxpool_fp16_cuda",
        "fused_dual_bias_add_relu_fp16_cuda"
    ],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

# ----------------- Optimized Model Architecture (FP16) -----------------
# The Python model code remains unchanged as the kernel improvements are
# abstracted away behind the same C++ function interface.

class BottleneckNew(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.downsample_conv = None # Will be populated during fusion

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = resnet_kernels_fp16.fused_bias_relu_fp16_cuda(out, self.bias1)

        out = self.conv2(out)
        out = resnet_kernels_fp16.fused_bias_relu_fp16_cuda(out, self.bias2)

        conv3_out = self.conv3(out)

        if self.downsample_conv is not None:
            downsample_conv_out = self.downsample_conv(identity)
            out = resnet_kernels_fp16.fused_dual_bias_add_relu_fp16_cuda(
                conv3_out, self.bias3, downsample_conv_out, self.downsample_bias
            )
        else:
            out = resnet_kernels_fp16.fused_bias_add_relu_fp16_cuda(conv3_out, self.bias3, identity)

        return out


class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super(ModelNew, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        block = BottleneckNew

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.eval()
        self._fuse_model()
        self.half()
        self.fc.float()


    def _fuse_conv_bn(self, conv, bn):
        assert(not (conv.training or bn.training)), "Fusion only for eval mode."
        
        orig_dtype = conv.weight.dtype
        w_conv = conv.weight.detach().clone().float()
        gamma = bn.weight.detach().clone().float()
        beta = bn.bias.detach().clone().float()
        mean = bn.running_mean.detach().clone().float()
        var = bn.running_var.detach().clone().float()
        eps = bn.eps

        bn_inv_std = torch.rsqrt(var + eps)
        bn_factor = gamma * bn_inv_std
        b_fused = beta - bn_factor * mean
        w_fused = w_conv * bn_factor.reshape(-1, 1, 1, 1)

        conv.weight.data.copy_(w_fused.to(orig_dtype))
        return b_fused.to(orig_dtype)

    def _fuse_model(self):
        stem_bias = self._fuse_conv_bn(self.conv1, self.bn1)
        self.register_buffer('stem_bias', stem_bias)
        self.bn1 = nn.Identity()
        
        for layer_module in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer_module:
                if isinstance(block, BottleneckNew):
                    bias1 = self._fuse_conv_bn(block.conv1, block.bn1)
                    bias2 = self._fuse_conv_bn(block.conv2, block.bn2)
                    bias3 = self._fuse_conv_bn(block.conv3, block.bn3)
                    
                    block.register_buffer('bias1', bias1)
                    block.register_buffer('bias2', bias2)
                    block.register_buffer('bias3', bias3)
                    
                    block.bn1 = nn.Identity()
                    block.bn2 = nn.Identity()
                    block.bn3 = nn.Identity()

                    if block.downsample is not None:
                        downsample_conv = block.downsample[0]
                        downsample_bn = block.downsample[1]
                        downsample_bias = self._fuse_conv_bn(downsample_conv, downsample_bn)
                        block.register_buffer('downsample_bias', downsample_bias)
                        
                        block.downsample_conv = downsample_conv
                        block.downsample = nn.Identity()


    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.half()

        x = self.conv1(x)
        x = resnet_kernels_fp16.fused_bias_relu_maxpool_fp16_cuda(x, self.stem_bias)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x.float())

        return x.float()