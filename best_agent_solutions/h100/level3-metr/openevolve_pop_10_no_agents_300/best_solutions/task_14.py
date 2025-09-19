# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define custom CUDA kernels for fused BatchNorm2d + ReLU for inference,
# with support for both float32 and float16 (half) precision.
# This combines the best 2D-grid launch strategy with mixed-precision execution.
fused_bn_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// --- KERNEL FOR FP32 (FLOAT) PRECISION ---
// This is the top-performing kernel from previous attempts, using a 2D grid.
__global__ void fused_bn_relu_strided_vec4_fp32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output, // Assumed to be contiguous
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int C_in, int H, int W,
    long stride_n, long stride_c, long stride_h) {

    // Each block in the y-dimension processes one (N, C) plane.
    const int nc_idx = blockIdx.y;
    const int n = nc_idx / C_in;
    const int c = nc_idx % C_in;

    // Pre-calculate scale and shift parameters once per block for cache efficiency.
    const float inv_std = rsqrtf(running_var[c] + eps);
    const float scale = weight[c] * inv_std;
    const float shift = bias[c] - running_mean[c] * scale;
    
    // Calculate base pointers using strides for non-contiguous input.
    const char* input_plane_base = (const char*)input + n * stride_n * sizeof(float) + c * stride_c * sizeof(float);
    float* output_plane_base = output + (long long)nc_idx * H * W;

    const int W_vec = W / 4;
    const int HW_vec = H * W_vec;

    // Grid-stride loop over the spatial elements of the plane.
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < HW_vec; idx += gridDim.x * blockDim.x) {
        const int h = idx / W_vec;
        const int w_vec = idx % W_vec;

        const float4 val4 = *reinterpret_cast<const float4*>(input_plane_base + h * stride_h * sizeof(float) + w_vec * 4 * sizeof(float));

        float4 out_val4;
        out_val4.x = fmaxf(val4.x * scale + shift, 0.0f);
        out_val4.y = fmaxf(val4.y * scale + shift, 0.0f);
        out_val4.z = fmaxf(val4.z * scale + shift, 0.0f);
        out_val4.w = fmaxf(val4.w * scale + shift, 0.0f);
        
        *reinterpret_cast<float4*>(output_plane_base + (long long)h * W + w_vec * 4) = out_val4;
    }
}


// --- KERNEL FOR FP16 (HALF) PRECISION ---
// Processes 8 half-precision elements per thread for maximum memory bandwidth.
__global__ void fused_bn_relu_strided_vec8_fp16_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int C_in, int H, int W,
    long stride_n, long stride_c, long stride_h) {

    const int nc_idx = blockIdx.y;
    const int n = nc_idx / C_in;
    const int c = nc_idx % C_in;

    // BN parameters are kept in FP32 for numerical stability.
    const float inv_std = rsqrtf(running_var[c] + eps);
    const float scale = weight[c] * inv_std;
    const float shift = bias[c] - running_mean[c] * scale;
    
    const char* input_plane_base = (const char*)input + n * stride_n * sizeof(__half) + c * stride_c * sizeof(__half);
    __half* output_plane_base = output + (long long)nc_idx * H * W;

    const int W_vec = W / 8; // Vectorize by 8 for half precision
    const int HW_vec = H * W_vec;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < HW_vec; idx += gridDim.x * blockDim.x) {
        const int h = idx / W_vec;
        const int w_vec = idx % W_vec;

        // Load 128 bits (8 halfs) at once.
        const float4 val_loaded = *reinterpret_cast<const float4*>(input_plane_base + h * stride_h * sizeof(__half) + w_vec * 8 * sizeof(__half));
        const __half2* val_h2 = reinterpret_cast<const __half2*>(&val_loaded);

        // Unpack, compute in FP32, and repack to FP16.
        float2 f01 = __half22float2(val_h2[0]);
        float2 f23 = __half22float2(val_h2[1]);
        float2 f45 = __half22float2(val_h2[2]);
        float2 f67 = __half22float2(val_h2[3]);

        f01.x = fmaxf(f01.x * scale + shift, 0.0f); f01.y = fmaxf(f01.y * scale + shift, 0.0f);
        f23.x = fmaxf(f23.x * scale + shift, 0.0f); f23.y = fmaxf(f23.y * scale + shift, 0.0f);
        f45.x = fmaxf(f45.x * scale + shift, 0.0f); f45.y = fmaxf(f45.y * scale + shift, 0.0f);
        f67.x = fmaxf(f67.x * scale + shift, 0.0f); f67.y = fmaxf(f67.y * scale + shift, 0.0f);

        __half2 out_h01 = __float22half2_rn(f01);
        __half2 out_h23 = __float22half2_rn(f23);
        __half2 out_h45 = __float22half2_rn(f45);
        __half2 out_h67 = __float22half2_rn(f67);
        
        // Store 128 bits (8 halfs) at once.
        float4* out_ptr = reinterpret_cast<float4*>(output_plane_base + (long long)h * W + w_vec * 8);
        __half2* out_h2_ptr = reinterpret_cast<__half2*>(out_ptr);
        out_h2_ptr[0] = out_h01; out_h2_ptr[1] = out_h23;
        out_h2_ptr[2] = out_h45; out_h2_ptr[3] = out_h67;
    }
}

// --- C++ DISPATCHER ---
// Chooses the correct kernel based on the input tensor's data type.
torch::Tensor fused_bn_relu_inference_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    
    auto output = torch::empty_like(input, torch::MemoryFormat::Contiguous);
    if (input.numel() == 0) return output;

    const auto N = input.size(0);
    const auto C = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    
    const auto input_strides = input.strides();
    const long stride_n = input_strides[0];
    const long stride_c = input_strides[1];
    const long stride_h = input_strides[2];

    const int block_size_x = 256;
    dim3 threads(block_size_x);
    
    if (input.scalar_type() == torch::kFloat) {
        TORCH_CHECK(W % 4 == 0, "Vectorized kernel requires Width to be divisible by 4 for float32.");
        const int W_vec = W / 4;
        const int HW_vec = H * W_vec;
        const int grid_size_x = std::min(1024, (HW_vec + block_size_x - 1) / block_size_x);
        dim3 blocks(grid_size_x, N * C);

        fused_bn_relu_strided_vec4_fp32_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), weight.data_ptr<float>(),
            bias.data_ptr<float>(), running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
            (float)eps, C, H, W, stride_n, stride_c, stride_h);
    } else if (input.scalar_type() == torch::kHalf) {
        TORCH_CHECK(W % 8 == 0, "Vectorized kernel requires Width to be divisible by 8 for float16.");
        const int W_vec = W / 8;
        const int HW_vec = H * W_vec;
        const int grid_size_x = std::min(1024, (HW_vec + block_size_x - 1) / block_size_x);
        dim3 blocks(grid_size_x, N * C);

        fused_bn_relu_strided_vec8_fp16_kernel<<<blocks, threads>>>(
            (const __half*)input.data_ptr<at::Half>(), (__half*)output.data_ptr<at::Half>(),
            weight.data_ptr<float>(), bias.data_ptr<float>(), running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(), (float)eps, C, H, W, stride_n, stride_c, stride_h);
    } else {
        TORCH_CHECK(false, "Unsupported input dtype for fused_bn_relu. Only float and half are supported.");
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return output;
}
"""

fused_bn_relu_cpp_source = (
    "torch::Tensor fused_bn_relu_inference_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor running_mean, torch::Tensor running_var, double eps);"
)

# JIT compile the CUDA kernels.
fused_ops = load_inline(
    name="fused_bn_relu_fp32_fp16",
    cpp_sources=fused_bn_relu_cpp_source,
    cuda_sources=fused_bn_relu_source,
    functions=["fused_bn_relu_inference_cuda"],
    verbose=False,
)

class FusedBatchNormReLU(nn.Module):
    """
    Custom module that fuses BatchNorm2d and ReLU for efficient inference.
    It supports both float32 and float16 inputs.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(FusedBatchNormReLU, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Fallback for training mode for correct statistics updates.
            bn_out = F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                                  True, self.momentum, self.eps)
            return F.relu(bn_out, inplace=True)
        else:
            # Custom kernel for inference mode, handles non-contiguous inputs and mixed precision.
            return fused_ops.fused_bn_relu_inference_cuda(
                x, self.weight, self.bias, self.running_mean, self.running_var, self.eps
            )

class Model(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        
        layers = []
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            layers.append(self._make_layer(in_features, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            FusedBatchNormReLU(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )
    
    def forward(self, x):
        # Use AMP to leverage Tensor Cores in Conv2d and our custom FP16 kernel.
        with torch.cuda.amp.autocast():
            total_channels = self.num_input_features + self.num_layers * self.growth_rate
            B, _, H, W = x.shape
            
            # Pre-allocate a single large tensor for all features.
            # The dtype is inherited from the input 'x', which may be cast to half by autocast.
            features = torch.empty(B, total_channels, H, W, dtype=x.dtype, device=x.device)
            features[:, :self.num_input_features, :, :] = x

            write_start_channel = self.num_input_features
            for i, layer in enumerate(self.layers):
                in_channels = self.num_input_features + i * self.growth_rate
                
                current_input = features.narrow(1, 0, in_channels)
                new_feature = layer(current_input)
                
                features[:, write_start_channel : write_start_channel + self.growth_rate, :, :] = new_feature
                write_start_channel += self.growth_rate
                
            return features

batch_size = 10
num_layers = 6
num_input_features = 32
growth_rate = 32
height, width = 224, 224

def get_inputs():
    # Input remains float32; autocast will handle the conversion.
    return [torch.randn(batch_size, num_input_features, height, width).cuda()]

def get_init_inputs():
    return [num_layers, num_input_features , growth_rate]
# EVOLVE-BLOCK-END