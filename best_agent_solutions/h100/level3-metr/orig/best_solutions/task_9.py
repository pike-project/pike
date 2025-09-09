import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- Step 1: Define and Compile Fused & Vectorized FP16 CUDA Kernels ---

# This source string contains an expanded set of kernels:
# 1. Scalar Kernels: The original, robust kernels for any input size.
#    - bias_add_relu_kernel_scalar: Fuses Bias + ReLU.
#    - bias_add_add_relu_kernel_scalar: Fuses Bias + Residual Add + ReLU.
# 2. Vectorized Kernels: Faster versions using half2 for even-sized inputs.
#    - bias_add_relu_kernel_vec: Vectorized version of (1).
#    - bias_add_add_relu_kernel_vec: Vectorized version of (2).
# 3. Specialized Kernels: The highly optimized kernels for the model stem and tail.
#    - bias_add_relu_maxpool_kernel_fp16: Fuses Bias + ReLU + MaxPool.
#    - spatial_mean_warp_reduce_kernel_fp16: Optimized AdaptiveAvgPool2d.
fused_kernels_fp16_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath> // For isinf, INFINITY

#define BLOCK_SIZE 256
constexpr int WARP_SIZE = 32;

// --- SCALAR KERNELS (for odd-sized feature maps) ---

__global__ void bias_add_relu_kernel_scalar(
    const half* __restrict__ input, const half* __restrict__ bias, half* __restrict__ output,
    const int size, const int C, const int HW)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
        const int c_idx = (idx / HW) % C;
        float val = __half2float(input[idx]) + __half2float(bias[c_idx]);
        output[idx] = __float2half(val > 0.0f ? val : 0.0f);
    }
}

__global__ void bias_add_add_relu_kernel_scalar(
    const half* __restrict__ input, const half* __restrict__ bias, const half* __restrict__ residual, half* __restrict__ output,
    const int size, const int C, const int HW)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
        const int c_idx = (idx / HW) % C;
        float val = __half2float(input[idx]) + __half2float(bias[c_idx]) + __half2float(residual[idx]);
        output[idx] = __float2half(val > 0.0f ? val : 0.0f);
    }
}

// --- VECTORIZED KERNELS (for even-sized feature maps) ---

__global__ void bias_add_relu_kernel_vec(
    const half2* __restrict__ input, const half* __restrict__ bias, half2* __restrict__ output,
    const int size_div_2, const int C, const int HW)
{
    for (int idx_div_2 = blockIdx.x * blockDim.x + threadIdx.x; idx_div_2 < size_div_2; idx_div_2 += blockDim.x * gridDim.x) {
        const int c_idx = ((idx_div_2 * 2) / HW) % C;
        float bias_val = __half2float(bias[c_idx]);
        half2 in_vec = input[idx_div_2];
        
        float val1 = __half2float(in_vec.x) + bias_val;
        float val2 = __half2float(in_vec.y) + bias_val;

        output[idx_div_2] = __floats2half2_rn(
            val1 > 0.0f ? val1 : 0.0f,
            val2 > 0.0f ? val2 : 0.0f
        );
    }
}

__global__ void bias_add_add_relu_kernel_vec(
    const half2* __restrict__ input, const half* __restrict__ bias, const half2* __restrict__ residual, half2* __restrict__ output,
    const int size_div_2, const int C, const int HW)
{
    for (int idx_div_2 = blockIdx.x * blockDim.x + threadIdx.x; idx_div_2 < size_div_2; idx_div_2 += blockDim.x * gridDim.x) {
        const int c_idx = ((idx_div_2 * 2) / HW) % C;
        float bias_val = __half2float(bias[c_idx]);
        half2 in_vec = input[idx_div_2];
        half2 res_vec = residual[idx_div_2];

        float val1 = __half2float(in_vec.x) + __half2float(res_vec.x) + bias_val;
        float val2 = __half2float(in_vec.y) + __half2float(res_vec.y) + bias_val;

        output[idx_div_2] = __floats2half2_rn(
            val1 > 0.0f ? val1 : 0.0f,
            val2 > 0.0f ? val2 : 0.0f
        );
    }
}

// --- SPECIALIZED KERNELS (Stem and AvgPool) ---
// (These are unchanged from the previous solution as they are already highly optimized)

__global__ void bias_add_relu_maxpool_kernel_fp16(
    const half* __restrict__ input, const half* __restrict__ bias, half* __restrict__ output,
    const int output_size, const int C, const int H_in, const int W_in, const int H_out, const int W_out)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < output_size; idx += blockDim.x * gridDim.x) {
        const int w_out = idx % W_out; const int h_out = (idx / W_out) % H_out;
        const int c = (idx / (W_out * H_out)) % C; const int n = idx / (C * W_out * H_out);
        const int h_in_start = h_out * 2 - 1; const int w_in_start = w_out * 2 - 1;
        float max_val = -INFINITY;
        const float bias_val = __half2float(bias[c]);
        const half* input_channel = input + (n * C + c) * H_in * W_in;
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                const int h_in = h_in_start + kh; const int w_in = w_in_start + kw;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    float val = __half2float(input_channel[h_in * W_in + w_in]) + bias_val;
                    if (val < 0.0f) val = 0.0f; // Inlined ReLU
                    if (val > max_val) max_val = val;
                }
            }
        }
        output[idx] = __float2half(max_val == -INFINITY ? 0.0f : max_val);
    }
}

__global__ void spatial_mean_warp_reduce_kernel_fp16(
    const half* __restrict__ input, half* __restrict__ output,
    const int N, const int C, const int HW)
{
    const unsigned int tid = threadIdx.x;
    const int n = blockIdx.x; const int c = blockIdx.y;
    float local_sum = 0.0f;
    const half* channel_data = input + (n * C + c) * HW;
    for (int i = tid; i < HW; i += blockDim.x) { local_sum += __half2float(channel_data[i]); }
    
    float warp_sum = local_sum;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
    }

    extern __shared__ float sdata[];
    const unsigned int warp_id = tid / WARP_SIZE; const unsigned int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) { sdata[warp_id] = warp_sum; }
    __syncthreads();

    float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? sdata[tid] : 0.0f;
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
        }
    }
    if (tid == 0) { output[n * C + c] = __float2half(HW > 0 ? (block_sum / (float)HW) : 0.0f); }
}


// --- C++ Wrappers with Dynamic Dispatch ---
#define CHECK_CUDA_HALF(x) TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf, #x " must be a contiguous half CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor bias_add_relu_dispatch(torch::Tensor input, torch::Tensor bias) {
    CHECK_CUDA_HALF(input); CHECK_CONTIGUOUS(input);
    CHECK_CUDA_HALF(bias); CHECK_CONTIGUOUS(bias);
    auto output = torch::empty_like(input);
    const int size = input.numel();
    if (size == 0) return output;
    const int C = input.size(1), H = input.size(2), W = input.size(3), HW = H * W;
    
    if (HW % 2 == 0) { // Dispatch to faster vectorized kernel
        const int size_div_2 = size / 2;
        const int num_blocks = (size_div_2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bias_add_relu_kernel_vec<<<num_blocks, BLOCK_SIZE, 0, c10::cuda::getCurrentCUDAStream()>>>(
            (const half2*)input.data_ptr(), (const half*)bias.data_ptr(), (half2*)output.data_ptr(),
            size_div_2, C, HW);
    } else { // Fallback to scalar kernel
        const int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bias_add_relu_kernel_scalar<<<num_blocks, BLOCK_SIZE, 0, c10::cuda::getCurrentCUDAStream()>>>(
            (const half*)input.data_ptr(), (const half*)bias.data_ptr(), (half*)output.data_ptr(),
            size, C, HW);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor bias_add_add_relu_dispatch(torch::Tensor input, torch::Tensor bias, torch::Tensor residual) {
    CHECK_CUDA_HALF(input); CHECK_CONTIGUOUS(input);
    CHECK_CUDA_HALF(bias); CHECK_CONTIGUOUS(bias);
    CHECK_CUDA_HALF(residual); CHECK_CONTIGUOUS(residual);
    auto output = torch::empty_like(input);
    const int size = input.numel();
    if (size == 0) return output;
    const int C = input.size(1), H = input.size(2), W = input.size(3), HW = H * W;

    if (HW % 2 == 0) { // Dispatch to faster vectorized kernel
        const int size_div_2 = size / 2;
        const int num_blocks = (size_div_2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bias_add_add_relu_kernel_vec<<<num_blocks, BLOCK_SIZE, 0, c10::cuda::getCurrentCUDAStream()>>>(
            (const half2*)input.data_ptr(), (const half*)bias.data_ptr(), 
            (const half2*)residual.data_ptr(), (half2*)output.data_ptr(),
            size_div_2, C, HW);
    } else { // Fallback to scalar kernel
        const int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bias_add_add_relu_kernel_scalar<<<num_blocks, BLOCK_SIZE, 0, c10::cuda::getCurrentCUDAStream()>>>(
            (const half*)input.data_ptr(), (const half*)bias.data_ptr(),
            (const half*)residual.data_ptr(), (half*)output.data_ptr(),
            size, C, HW);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor bias_add_relu_maxpool_cuda_fp16(torch::Tensor input, torch::Tensor bias) {
    CHECK_CUDA_HALF(input); CHECK_CONTIGUOUS(input);
    CHECK_CUDA_HALF(bias); CHECK_CONTIGUOUS(bias);
    const int N = input.size(0), C = input.size(1), H_in = input.size(2), W_in = input.size(3);
    const int H_out = (H_in - 1) / 2 + 1; const int W_out = (W_in - 1) / 2 + 1;
    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    const auto output_size = output.numel();
    if (output_size == 0) return output;
    const int num_blocks = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bias_add_relu_maxpool_kernel_fp16<<<num_blocks, BLOCK_SIZE, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (const half*)input.data_ptr(), (const half*)bias.data_ptr(), (half*)output.data_ptr(),
        output_size, C, H_in, W_in, H_out, W_out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor spatial_mean_cuda_fp16(torch::Tensor input) {
    CHECK_CUDA_HALF(input); CHECK_CONTIGUOUS(input);
    const auto N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    auto output = torch::empty({N, C}, input.options());
    if (input.numel() == 0) return output;
    dim3 grid_dim(N, C);
    dim3 block_dim(BLOCK_SIZE);
    const int num_warps = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    size_t smem_size = num_warps * sizeof(float);
    spatial_mean_warp_reduce_kernel_fp16<<<grid_dim, block_dim, smem_size, c10::cuda::getCurrentCUDAStream()>>>(
        (const half*)input.data_ptr(), (half*)output.data_ptr(), N, C, H * W);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_kernels_fp16_cpp_source = """
#include <torch/extension.h>
torch::Tensor bias_add_relu_dispatch(torch::Tensor input, torch::Tensor bias);
torch::Tensor bias_add_add_relu_dispatch(torch::Tensor input, torch::Tensor bias, torch::Tensor residual);
torch::Tensor bias_add_relu_maxpool_cuda_fp16(torch::Tensor input, torch::Tensor bias);
torch::Tensor spatial_mean_cuda_fp16(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bias_add_relu_dispatch", &bias_add_relu_dispatch, "Dispatching Bias+Add+ReLU (FP16)");
    m.def("bias_add_add_relu_dispatch", &bias_add_add_relu_dispatch, "Dispatching Bias+Add+Add+ReLU (FP16)");
    m.def("bias_add_relu_maxpool_cuda_fp16", &bias_add_relu_maxpool_cuda_fp16, "Bias+ReLU+MaxPool(k=3,s=2,p=1) (FP16)");
    m.def("spatial_mean_cuda_fp16", &spatial_mean_cuda_fp16, "Warp-Optimized Spatial Mean (FP16)");
}
"""

# JIT compile all CUDA kernels into a single Python module.
fused_ops_fp16 = load_inline(
    name="resnet_fused_ops_fp16_v3_graph",
    cpp_sources=fused_kernels_fp16_cpp_source,
    cuda_sources=fused_kernels_fp16_cuda_source,
    verbose=False,
    extra_cuda_cflags=["-O3"],
)

# --- Step 2: Define DType-Robust Host-Side Optimization Utility ---

def fold_bn_into_conv(conv_layer: nn.Conv2d, bn_layer: nn.BatchNorm2d) -> nn.Conv2d:
    device, dtype = conv_layer.weight.device, conv_layer.weight.dtype
    bn_layer.to(device).eval()
    w_conv = conv_layer.weight.clone().detach().float()
    b_conv = torch.zeros(conv_layer.out_channels, device=device, dtype=torch.float)
    if conv_layer.bias is not None: b_conv = conv_layer.bias.clone().detach().float()
    scale = bn_layer.weight.detach().float() / torch.sqrt(bn_layer.running_var.detach() + bn_layer.eps)
    w_fused = w_conv * scale.view(-1, 1, 1, 1)
    b_fused = (b_conv - bn_layer.running_mean.detach()) * scale + bn_layer.bias.detach().float()
    fused_conv = nn.Conv2d(conv_layer.in_channels, conv_layer.out_channels, conv_layer.kernel_size,
                           conv_layer.stride, conv_layer.padding, bias=True, device=device, dtype=dtype)
    fused_conv.weight = nn.Parameter(w_fused.to(dtype))
    fused_conv.bias = nn.Parameter(b_fused.to(dtype))
    return fused_conv

# --- Step 3: Define Original Model Architecture for Inheritance ---

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64); self.relu = nn.ReLU(inplace=True); self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)); self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, block, out_channels, B, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, B): layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x

# --- Step 4: Define New, Graph-Captured & Vectorized FP16 Model ---

class BasicBlockFusedFP16(BasicBlock):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, fused_ops_module=None):
        super().__init__(in_channels, out_channels, stride, downsample)
        if fused_ops_module is None: raise ValueError("fused_ops_module must be provided")
        self.bias_add_relu_op = fused_ops_module.bias_add_relu_dispatch
        self.bias_add_add_relu_op = fused_ops_module.bias_add_add_relu_dispatch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x) if self.downsample is not None else x
        out_conv1 = F.conv2d(x, self.conv1.weight, None, self.conv1.stride, self.conv1.padding)
        out = self.bias_add_relu_op(out_conv1, self.conv1.bias)
        out_conv2 = F.conv2d(out, self.conv2.weight, None, self.conv2.stride, self.conv2.padding)
        return self.bias_add_add_relu_op(out_conv2, self.conv2.bias, identity)

class ModelNew(Model):
    def __init__(self, num_classes=1000):
        self.fused_ops = fused_ops_fp16
        super().__init__(num_classes=num_classes)
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.half().eval()
        self._fuse_layers()

    def _make_layer(self, _, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))
        layers = [BasicBlockFusedFP16(self.in_channels, out_channels, stride, downsample, self.fused_ops)]
        self.in_channels = out_channels
        for _ in range(1, blocks): layers.append(BasicBlockFusedFP16(self.in_channels, out_channels, fused_ops_module=self.fused_ops))
        return nn.Sequential(*layers)

    def _fuse_layers(self):
        self.conv1 = fold_bn_into_conv(self.conv1, self.bn1)
        self.bn1 = nn.Identity()
        for module in self.modules():
            if isinstance(module, BasicBlockFusedFP16):
                module.conv1 = fold_bn_into_conv(module.conv1, module.bn1)
                module.bn1 = nn.Identity()
                module.conv2 = fold_bn_into_conv(module.conv2, module.bn2)
                module.bn2 = nn.Identity()
                if isinstance(getattr(module, 'downsample', None), nn.Sequential):
                    conv, bn = module.downsample[0], module.downsample[1]
                    module.downsample = fold_bn_into_conv(conv, bn)

    def _graph_capture(self, x: torch.Tensor):
        # --- Warmup Run ---
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self._forward_impl(x.half())
        torch.cuda.current_stream().wait_stream(s)
        
        # --- Graph Capture ---
        self.static_input = x.half().clone()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = self._forward_impl(self.static_input)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """The actual forward pass logic, used for both warmup and capture."""
        x_conv = F.conv2d(x, self.conv1.weight, None, self.conv1.stride, self.conv1.padding)
        y = self.fused_ops.bias_add_relu_maxpool_cuda_fp16(x_conv, self.conv1.bias)
        y = self.layer1(y); y = self.layer2(y); y = self.layer3(y); y = self.layer4(y)
        y = self.fused_ops.spatial_mean_cuda_fp16(y)
        return self.fc(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The main forward pass, implementing the CUDA Graph cache and replay."""
        if self.training:
            # Fallback to eager mode for training
            return self._forward_impl(x.half()).float()
        
        if self.graph is None:
            self._graph_capture(x)

        # For this and all subsequent runs:
        self.static_input.copy_(x) # Use copy_ to avoid breaking graph's memory pointer
        self.graph.replay()
        return self.static_output.float().clone()