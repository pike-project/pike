# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This solution builds upon the top-performing program by incorporating grid-stride loops
# for enhanced kernel robustness, while retaining the key optimizations:
# 1. Advanced Fusion: A `dual_bn_add_relu` kernel fuses BatchNorm from both the main
#    and residual paths, eliminating a kernel launch.
# 2. Pre-computation: Effective BatchNorm scale/shift parameters are calculated on the host.
# 3. Vectorization: `float4` is used to maximize memory bandwidth.
# 4. Launch Tuning: `__launch_bounds__` helps the compiler optimize register usage.
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// --- KERNEL 1: BatchNorm -> ReLU (Vectorized, Precomputed, Grid-Stride) ---
__global__ void __launch_bounds__(512, 2)
bn_relu_kernel_vec4(
    const float4* __restrict__ input, float4* __restrict__ output,
    const float* __restrict__ scale, const float* __restrict__ shift,
    int total_vec_elements, int C, int spatial_dim) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_vec_elements;
         idx += gridDim.x * blockDim.x) {
        int c = ((idx * 4) / spatial_dim) % C;
        const float A = scale[c];
        const float B = shift[c];
        const float4 in_vec = input[idx];
        float4 out_vec;
        out_vec.x = fmaxf(0.0f, in_vec.x * A + B);
        out_vec.y = fmaxf(0.0f, in_vec.y * A + B);
        out_vec.z = fmaxf(0.0f, in_vec.z * A + B);
        out_vec.w = fmaxf(0.0f, in_vec.w * A + B);
        output[idx] = out_vec;
    }
}

// --- KERNEL 2: BatchNorm -> Add -> ReLU (Vectorized, Precomputed, Grid-Stride) ---
__global__ void __launch_bounds__(512, 2)
bn_add_relu_kernel_vec4(
    const float4* __restrict__ input, const float4* __restrict__ residual, float4* __restrict__ output,
    const float* __restrict__ scale, const float* __restrict__ shift,
    int total_vec_elements, int C, int spatial_dim) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_vec_elements;
         idx += gridDim.x * blockDim.x) {
        int c = ((idx * 4) / spatial_dim) % C;
        const float A = scale[c];
        const float B = shift[c];
        const float4 in_vec = input[idx];
        const float4 res_vec = residual[idx];
        float4 out_vec;
        out_vec.x = fmaxf(0.0f, (in_vec.x * A + B) + res_vec.x);
        out_vec.y = fmaxf(0.0f, (in_vec.y * A + B) + res_vec.y);
        out_vec.z = fmaxf(0.0f, (in_vec.z * A + B) + res_vec.z);
        out_vec.w = fmaxf(0.0f, (in_vec.w * A + B) + res_vec.w);
        output[idx] = out_vec;
    }
}

// --- KERNEL 3: Dual BatchNorm -> Add -> ReLU (Vectorized, Precomputed, Grid-Stride) ---
__global__ void __launch_bounds__(512, 2)
dual_bn_add_relu_kernel_vec4(
    const float4* __restrict__ main_path_in, const float4* __restrict__ residual_path_in,
    float4* __restrict__ output,
    const float* __restrict__ scale1, const float* __restrict__ shift1, // main path BN params
    const float* __restrict__ scale2, const float* __restrict__ shift2, // residual path BN params
    int total_vec_elements, int C, int spatial_dim) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_vec_elements;
         idx += gridDim.x * blockDim.x) {
        int c = ((idx * 4) / spatial_dim) % C;
        const float A1 = scale1[c], B1 = shift1[c];
        const float A2 = scale2[c], B2 = shift2[c];
        const float4 main_vec = main_path_in[idx];
        const float4 res_vec = residual_path_in[idx];
        float4 out_vec;
        out_vec.x = fmaxf(0.0f, (main_vec.x * A1 + B1) + (res_vec.x * A2 + B2));
        out_vec.y = fmaxf(0.0f, (main_vec.y * A1 + B1) + (res_vec.y * A2 + B2));
        out_vec.z = fmaxf(0.0f, (main_vec.z * A1 + B1) + (res_vec.z * A2 + B2));
        out_vec.w = fmaxf(0.0f, (main_vec.w * A1 + B1) + (res_vec.w * A2 + B2));
        output[idx] = out_vec;
    }
}

// --- C++ WRAPPERS ---
void launch_kernel(int total_elements, auto kernel, auto... args) {
    TORCH_CHECK(total_elements % 4 == 0, "Input tensor size must be divisible by 4 for vec4 optimization");
    const int total_vec_elements = total_elements / 4;
    const int block_size = 512;
    // A modest grid size is sufficient with grid-stride loops
    const int num_blocks = std::min((total_vec_elements + block_size - 1) / block_size, 4096);
    kernel<<<num_blocks, block_size>>>(args...);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor bn_relu_precomputed_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor shift) {
    auto output = torch::empty_like(input);
    const int total_elements = input.numel();
    const int C = input.size(1);
    const int spatial_dim = input.size(2) * input.size(3);
    launch_kernel(total_elements, bn_relu_kernel_vec4,
        (const float4*)input.data_ptr<float>(), (float4*)output.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        total_elements / 4, C, spatial_dim);
    return output;
}

torch::Tensor bn_add_relu_precomputed_cuda(torch::Tensor input, torch::Tensor residual, torch::Tensor scale, torch::Tensor shift) {
    auto output = torch::empty_like(input);
    const int total_elements = input.numel();
    const int C = input.size(1);
    const int spatial_dim = input.size(2) * input.size(3);
    launch_kernel(total_elements, bn_add_relu_kernel_vec4,
        (const float4*)input.data_ptr<float>(), (const float4*)residual.data_ptr<float>(), (float4*)output.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        total_elements / 4, C, spatial_dim);
    return output;
}

torch::Tensor dual_bn_add_relu_precomputed_cuda(torch::Tensor main_in, torch::Tensor res_in,
                                                torch::Tensor scale1, torch::Tensor shift1,
                                                torch::Tensor scale2, torch::Tensor shift2) {
    auto output = torch::empty_like(main_in);
    const int total_elements = main_in.numel();
    const int C = main_in.size(1);
    const int spatial_dim = main_in.size(2) * main_in.size(3);
    launch_kernel(total_elements, dual_bn_add_relu_kernel_vec4,
        (const float4*)main_in.data_ptr<float>(), (const float4*)res_in.data_ptr<float>(), (float4*)output.data_ptr<float>(),
        scale1.data_ptr<float>(), shift1.data_ptr<float>(), scale2.data_ptr<float>(), shift2.data_ptr<float>(),
        total_elements / 4, C, spatial_dim);
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor bn_relu_precomputed_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor shift);
torch::Tensor bn_add_relu_precomputed_cuda(torch::Tensor input, torch::Tensor residual, torch::Tensor scale, torch::Tensor shift);
torch::Tensor dual_bn_add_relu_precomputed_cuda(torch::Tensor main_in, torch::Tensor res_in, torch::Tensor scale1, torch::Tensor shift1, torch::Tensor scale2, torch::Tensor shift2);
"""

fused_ops = load_inline(
    name="resnet_fused_ops_v_final",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["bn_relu_precomputed_cuda", "bn_add_relu_precomputed_cuda", "dual_bn_add_relu_precomputed_cuda"],
    verbose=False,
)

class Model(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Correctly define the downsample layer only when needed
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        else:
            self.downsample = None
        self.stride = stride
        # The nn.ReLU module is no longer needed for the forward pass but is kept for state_dict compatibility
        self.relu = nn.ReLU(inplace=True)

    def _precompute_bn_params(self, bn_layer):
        """Calculates effective scale and shift for BatchNorm."""
        inv_std = torch.rsqrt(bn_layer.running_var + bn_layer.eps)
        scale = bn_layer.weight * inv_std
        shift = bn_layer.bias - bn_layer.running_mean * scale
        return scale, shift

    def forward(self, x):
        # --- First fused block: conv1 -> bn1 -> relu ---
        out = self.conv1(x)
        scale1, shift1 = self._precompute_bn_params(self.bn1)
        out = fused_ops.bn_relu_precomputed_cuda(out, scale1, shift1)
        
        conv2_out = self.conv2(out)

        # --- Second fused block: (conv2 -> bn2) + (downsample) -> relu ---
        if self.downsample is not None:
            identity_conv_out = self.downsample[0](x)
            bn_down = self.downsample[1]
            
            # Pre-compute parameters for both batchnorm layers
            scale2, shift2 = self._precompute_bn_params(self.bn2)
            scale_down, shift_down = self._precompute_bn_params(bn_down)
            
            # Call the advanced dual-bn fusion kernel
            out = fused_ops.dual_bn_add_relu_precomputed_cuda(
                conv2_out, identity_conv_out, scale2, shift2, scale_down, shift_down
            )
        else:
            identity = x
            scale2, shift2 = self._precompute_bn_params(self.bn2)
            
            # Call the standard bn+add+relu fusion kernel
            out = fused_ops.bn_add_relu_precomputed_cuda(
                conv2_out, identity, scale2, shift2
            )
        return out

# Test code
in_channels = 3
out_channels = 64
stride = 1
batch_size = 10
num_classes = 1000

def get_inputs():
    # Input tensor must be on CUDA and contiguous for vectorized memory access.
    return [torch.randn(batch_size, in_channels, 224, 224).cuda().contiguous()]

def get_init_inputs():
    return [in_channels, out_channels, stride]
# EVOLVE-BLOCK-END