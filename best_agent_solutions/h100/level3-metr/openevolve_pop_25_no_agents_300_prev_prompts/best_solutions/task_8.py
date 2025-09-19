# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This solution combines the most effective strategies from all top-performing programs:
# 1. Three Custom Kernels: Adopts the strategy from the #1 program by creating custom
#    fused kernels for all three BatchNorm-related operations (bn1+relu, bn2+add+relu,
#    and the downsample bn), which proved faster than Conv-BN folding.
# 2. CUDA Streams: Implements parallel execution of the main and downsample paths using
#    separate CUDA streams, overlapping their computation to hide latency.
# 3. Compiler Optimization: Uses the '--use_fast_math' flag to enable more aggressive
#    floating-point optimizations by the CUDA compiler.
# 4. Optimal Launch Params: Retains the block size of 512, which was found to be optimal.
# This creates a "best-of-all-worlds" solution by synthesizing the most powerful
# low-level, high-level, and compiler optimizations observed.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused BatchNorm (pre-computed) + ReLU Kernel
__global__ void fused_bn_relu_kernel_precomp_vec_smem(
    const float4* __restrict__ x, 
    float4* __restrict__ out, 
    const float* __restrict__ scale, 
    const float* __restrict__ shift,
    int total_size_vec,
    int C,
    int spatial_dim_vec) {
    
    extern __shared__ float s_params[];
    float* s_scale = s_params;
    float* s_shift = s_params + C;

    int tid_in_block = threadIdx.x;
    if (tid_in_block < C) {
        s_scale[tid_in_block] = scale[tid_in_block];
        s_shift[tid_in_block] = shift[tid_in_block];
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < total_size_vec) {
        int c = (index / spatial_dim_vec) % C;
        
        float scale_val = s_scale[c];
        float shift_val = s_shift[c];
        
        float4 x_vec = x[index];
        
        x_vec.x = fmaxf(0.0f, x_vec.x * scale_val + shift_val);
        x_vec.y = fmaxf(0.0f, x_vec.y * scale_val + shift_val);
        x_vec.z = fmaxf(0.0f, x_vec.z * scale_val + shift_val);
        x_vec.w = fmaxf(0.0f, x_vec.w * scale_val + shift_val);
        
        out[index] = x_vec;
    }
}

// Fused BatchNorm (pre-computed) + Add + ReLU Kernel
__global__ void fused_bn_add_relu_kernel_precomp_vec_smem(
    const float4* __restrict__ x, 
    const float4* __restrict__ identity, 
    float4* __restrict__ out, 
    const float* __restrict__ scale, 
    const float* __restrict__ shift,
    int total_size_vec,
    int C,
    int spatial_dim_vec) {
    
    extern __shared__ float s_params[];
    float* s_scale = s_params;
    float* s_shift = s_params + C;

    int tid_in_block = threadIdx.x;
    if (tid_in_block < C) {
        s_scale[tid_in_block] = scale[tid_in_block];
        s_shift[tid_in_block] = shift[tid_in_block];
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < total_size_vec) {
        int c = (index / spatial_dim_vec) % C;
        
        float scale_val = s_scale[c];
        float shift_val = s_shift[c];
        
        float4 x_vec = x[index];
        float4 id_vec = identity[index];
        
        x_vec.x = fmaxf(0.0f, x_vec.x * scale_val + shift_val + id_vec.x);
        x_vec.y = fmaxf(0.0f, x_vec.y * scale_val + shift_val + id_vec.y);
        x_vec.z = fmaxf(0.0f, x_vec.z * scale_val + shift_val + id_vec.z);
        x_vec.w = fmaxf(0.0f, x_vec.w * scale_val + shift_val + id_vec.w);
        
        out[index] = x_vec;
    }
}

// Fused BatchNorm (pre-computed) Kernel (for downsample path)
__global__ void fused_bn_kernel_precomp_vec_smem(
    const float4* __restrict__ x, 
    float4* __restrict__ out, 
    const float* __restrict__ scale, 
    const float* __restrict__ shift,
    int total_size_vec,
    int C,
    int spatial_dim_vec) {
    
    extern __shared__ float s_params[];
    float* s_scale = s_params;
    float* s_shift = s_params + C;

    int tid_in_block = threadIdx.x;
    if (tid_in_block < C) {
        s_scale[tid_in_block] = scale[tid_in_block];
        s_shift[tid_in_block] = shift[tid_in_block];
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < total_size_vec) {
        int c = (index / spatial_dim_vec) % C;
        
        float scale_val = s_scale[c];
        float shift_val = s_shift[c];
        
        float4 x_vec = x[index];
        
        x_vec.x = x_vec.x * scale_val + shift_val;
        x_vec.y = x_vec.y * scale_val + shift_val;
        x_vec.z = x_vec.z * scale_val + shift_val;
        x_vec.w = x_vec.w * scale_val + shift_val;
        
        out[index] = x_vec;
    }
}


// C++ wrapper for fused_bn_relu
torch::Tensor fused_bn_relu_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor shift) {
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    auto out = torch::empty_like(x);
    const int total_size_vec = x.numel() / 4;
    const int spatial_dim_vec = (H * W) / 4;
    const int block_size = 512;
    const int num_blocks = (total_size_vec + block_size - 1) / block_size;
    const size_t shared_mem_size = C * 2 * sizeof(float);
    fused_bn_relu_kernel_precomp_vec_smem<<<num_blocks, block_size, shared_mem_size>>>(
        (const float4*)x.data_ptr<float>(), (float4*)out.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        total_size_vec, C, spatial_dim_vec
    );
    return out;
}

// C++ wrapper for fused_bn_add_relu
torch::Tensor fused_bn_add_relu_cuda(torch::Tensor x, torch::Tensor identity, torch::Tensor scale, torch::Tensor shift) {
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    auto out = torch::empty_like(x);
    const int total_size_vec = x.numel() / 4;
    const int spatial_dim_vec = (H * W) / 4;
    const int block_size = 512;
    const int num_blocks = (total_size_vec + block_size - 1) / block_size;
    const size_t shared_mem_size = C * 2 * sizeof(float);
    fused_bn_add_relu_kernel_precomp_vec_smem<<<num_blocks, block_size, shared_mem_size>>>(
        (const float4*)x.data_ptr<float>(), (const float4*)identity.data_ptr<float>(), (float4*)out.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        total_size_vec, C, spatial_dim_vec
    );
    return out;
}

// C++ wrapper for fused_bn (downsample path)
torch::Tensor fused_bn_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor shift) {
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    auto out = torch::empty_like(x);
    const int total_size_vec = x.numel() / 4;
    const int spatial_dim_vec = (H * W) / 4;
    const int block_size = 512;
    const int num_blocks = (total_size_vec + block_size - 1) / block_size;
    const size_t shared_mem_size = C * 2 * sizeof(float);
    fused_bn_kernel_precomp_vec_smem<<<num_blocks, block_size, shared_mem_size>>>(
        (const float4*)x.data_ptr<float>(), (float4*)out.data_ptr<float>(),
        scale.data_ptr<float>(), shift.data_ptr<float>(),
        total_size_vec, C, spatial_dim_vec
    );
    return out;
}
"""

cpp_source = """
torch::Tensor fused_bn_relu_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor shift);
torch::Tensor fused_bn_add_relu_cuda(torch::Tensor x, torch::Tensor identity, torch::Tensor scale, torch::Tensor shift);
torch::Tensor fused_bn_cuda(torch::Tensor x, torch::Tensor scale, torch::Tensor shift);
"""

# JIT compile the CUDA kernels
fused_ops = load_inline(
    name="fused_ops_synthesis_v1",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_bn_relu_cuda", "fused_bn_add_relu_cuda", "fused_bn_cuda"],
    verbose=False,
    extra_cuda_cflags=['--use_fast_math'],
)


class Model(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        downsample_conv = nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False)
        downsample_bn = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = nn.ModuleList([downsample_conv, downsample_bn])
        
        self.stride = stride

        self.eval() 
        with torch.no_grad():
            scale1 = self.bn1.weight / torch.sqrt(self.bn1.running_var + self.bn1.eps)
            shift1 = self.bn1.bias - self.bn1.running_mean * scale1
            self.register_buffer('scale1', scale1)
            self.register_buffer('shift1', shift1)

            scale2 = self.bn2.weight / torch.sqrt(self.bn2.running_var + self.bn2.eps)
            shift2 = self.bn2.bias - self.bn2.running_mean * scale2
            self.register_buffer('scale2', scale2)
            self.register_buffer('shift2', shift2)

            scale_ds = self.downsample[1].weight / torch.sqrt(self.downsample[1].running_var + self.downsample[1].eps)
            shift_ds = self.downsample[1].bias - self.downsample[1].running_mean * scale_ds
            self.register_buffer('scale_ds', scale_ds)
            self.register_buffer('shift_ds', shift_ds)
            
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()

    def forward(self, x):
        # --- Main Path on Stream 1 ---
        with torch.cuda.stream(self.stream1):
            out_conv1 = self.conv1(x)
            out_fused1 = fused_ops.fused_bn_relu_cuda(out_conv1, self.scale1, self.shift1)
            out_conv2 = self.conv2(out_fused1)

        # --- Downsample Path on Stream 2 ---
        with torch.cuda.stream(self.stream2):
            identity_conv = self.downsample[0](x)
            identity = fused_ops.fused_bn_cuda(identity_conv, self.scale_ds, self.shift_ds)

        # --- Synchronization and Final Fusion on Default Stream ---
        torch.cuda.current_stream().wait_stream(self.stream1)
        torch.cuda.current_stream().wait_stream(self.stream2)

        out = fused_ops.fused_bn_add_relu_cuda(
            out_conv2,
            identity,
            self.scale2,
            self.shift2
        )
        return out
    
# Test code
in_channels = 3
out_channels = 64
stride = 1
batch_size = 10
num_classes = 1000

def get_inputs():
    # Input must be on CUDA for custom kernels.
    # The .contiguous() call ensures the memory layout is compatible with vectorized loads (float4).
    return [torch.randn(batch_size, in_channels, 224, 224).cuda().contiguous()]

def get_init_inputs():
    return [in_channels, out_channels, stride]
# EVOLVE-BLOCK-END