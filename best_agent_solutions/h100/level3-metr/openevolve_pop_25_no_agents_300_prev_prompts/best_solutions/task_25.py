# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define and compile custom CUDA kernels for fused operations.
# This version synthesizes the best features from previous top programs:
# 1. CUDA Streams for concurrency.
# 2. float4 vectorization for memory bandwidth.
# 3. fmaf (fused multiply-add) instructions for arithmetic throughput.
# 4. A gather-read/coalesced-write pattern for the channel shuffle, which was empirically the fastest.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// Kernel for Fused BatchNorm2d + ReLU (Vectorized with float4 and FMA)
__global__ void batchnorm_relu_kernel_fma(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int total_elements_v4,
    int C, int HW) {

    for (int index_v4 = blockIdx.x * blockDim.x + threadIdx.x;
         index_v4 < total_elements_v4;
         index_v4 += blockDim.x * gridDim.x) {

        int c = ((index_v4 * 4) / HW) % C;

        const float inv_std = rsqrtf(running_var[c] + eps);
        const float scale = weight[c] * inv_std;
        const float shift = bias[c] - running_mean[c] * scale;

        const float4 in_val = input[index_v4];
        float4 out_val;

        out_val.x = fmaxf(0.0f, fmaf(in_val.x, scale, shift));
        out_val.y = fmaxf(0.0f, fmaf(in_val.y, scale, shift));
        out_val.z = fmaxf(0.0f, fmaf(in_val.z, scale, shift));
        out_val.w = fmaxf(0.0f, fmaf(in_val.w, scale, shift));

        output[index_v4] = out_val;
    }
}

// Kernel for Fused BatchNorm2d + ChannelShuffle (Gather Read / Coalesced Write with FMA)
// This pattern was empirically faster than the scatter-write alternative.
__global__ void batchnorm_channel_shuffle_kernel_gather_fma(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int total_elements_v4,
    int C, int HW, int groups) {

    const int C_per_group = C / groups;
    const int HW_v4 = HW / 4;
    const int CHW_v4 = C * HW_v4;

    for (int index_v4 = blockIdx.x * blockDim.x + threadIdx.x;
         index_v4 < total_elements_v4;
         index_v4 += blockDim.x * gridDim.x) {

        // Current thread is responsible for an element in the *output* tensor.
        // We calculate which element from the *input* tensor maps to here.
        const int hw_v4 = index_v4 % HW_v4;
        const int c_out = (index_v4 / HW_v4) % C;
        const int n = index_v4 / CHW_v4;

        // Inverse shuffle logic to find the source channel (c_in)
        const int g_in = c_out % groups;
        const int cpg_in = c_out / groups;
        const int c_in = g_in * C_per_group + cpg_in;
        
        const int src_index_v4 = n * CHW_v4 + c_in * HW_v4 + hw_v4;

        // BN parameters are indexed by the source channel
        const float inv_std = rsqrtf(running_var[c_in] + eps);
        const float scale = weight[c_in] * inv_std;
        const float shift = bias[c_in] - running_mean[c_in] * scale;

        // Gather read from input tensor
        const float4 in_val = input[src_index_v4];
        float4 out_val;

        out_val.x = fmaf(in_val.x, scale, shift);
        out_val.y = fmaf(in_val.y, scale, shift);
        out_val.z = fmaf(in_val.z, scale, shift);
        out_val.w = fmaf(in_val.w, scale, shift);
        
        // Coalesced write to output tensor
        output[index_v4] = out_val;
    }
}


// Kernel for Fused BatchNorm2d then ReLU + Add (Vectorized with float4 and FMA)
__global__ void batchnorm_then_relu_add_kernel_fma(
    const float4* __restrict__ in1,
    const float4* __restrict__ in2,
    float4* __restrict__ out,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int size_v4,
    int C, int HW) {

    for (int idx_v4 = blockIdx.x * blockDim.x + threadIdx.x;
         idx_v4 < size_v4;
         idx_v4 += blockDim.x * gridDim.x) {

        int c = ((idx_v4 * 4) / HW) % C;

        const float inv_std = rsqrtf(running_var[c] + eps);
        const float scale = weight[c] * inv_std;
        const float shift = bias[c] - running_mean[c] * scale;

        const float4 in1_val = in1[idx_v4];
        const float4 in2_val = in2[idx_v4];
        float4 out_val;

        out_val.x = fmaxf(0.0f, fmaf(in1_val.x, scale, shift)) + in2_val.x;
        out_val.y = fmaxf(0.0f, fmaf(in1_val.y, scale, shift)) + in2_val.y;
        out_val.z = fmaxf(0.0f, fmaf(in1_val.z, scale, shift)) + in2_val.z;
        out_val.w = fmaxf(0.0f, fmaf(in1_val.w, scale, shift)) + in2_val.w;

        out[idx_v4] = out_val;
    }
}


constexpr int THREADS_PER_BLOCK = 512;

// Wrapper for BN+ReLU
torch::Tensor batchnorm_relu_cuda(
    torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight, torch::Tensor bias, double eps) {
    
    auto x_contig = x.contiguous();
    const auto C = x_contig.size(1);
    const auto H = x_contig.size(2);
    const auto W = x_contig.size(3);
    const auto HW = H * W;
    const auto total_elements = x_contig.numel();
    auto output = torch::empty_like(x_contig);

    if (total_elements == 0) return output;
    TORCH_CHECK(W % 4 == 0, "Width must be divisible by 4 for float4 vectorization.");

    const int total_elements_v4 = total_elements / 4;
    const int num_blocks = (total_elements_v4 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    batchnorm_relu_kernel_fma<<<num_blocks, THREADS_PER_BLOCK>>>(
        (const float4*)x_contig.data_ptr<float>(), (float4*)output.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        static_cast<float>(eps), total_elements_v4, C, HW);
        
    return output;
}

// Wrapper for BN+Shuffle
torch::Tensor batchnorm_channel_shuffle_cuda(
    torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight, torch::Tensor bias, double eps, int64_t groups) {
    
    auto x_contig = x.contiguous();
    const auto C = x_contig.size(1);
    const auto H = x_contig.size(2);
    const auto W = x_contig.size(3);
    const auto HW = H * W;
    const auto total_elements = x_contig.numel();
    auto output = torch::empty_like(x_contig);

    if (total_elements == 0) return output;
    TORCH_CHECK(W % 4 == 0, "Width must be divisible by 4 for float4 vectorization.");

    const int total_elements_v4 = total_elements / 4;
    const int num_blocks = (total_elements_v4 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    batchnorm_channel_shuffle_kernel_gather_fma<<<num_blocks, THREADS_PER_BLOCK>>>(
        (const float4*)x_contig.data_ptr<float>(), (float4*)output.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        static_cast<float>(eps), total_elements_v4, C, HW, static_cast<int>(groups));

    return output;
}

// Wrapper for BN then ReLU+Add
torch::Tensor batchnorm_then_relu_add_cuda(
    torch::Tensor in1, torch::Tensor in2, 
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight, torch::Tensor bias, double eps) {
    
    auto in1_contig = in1.contiguous();
    auto in2_contig = in2.contiguous();
    
    const auto C = in1_contig.size(1);
    const auto H = in1_contig.size(2);
    const auto W = in1_contig.size(3);
    const auto HW = H * W;
    const auto size = in1_contig.numel();
    auto output = torch::empty_like(in1_contig);

    if (size == 0) return output;
    TORCH_CHECK(W % 4 == 0, "Width must be divisible by 4 for float4 vectorization.");

    const int size_v4 = size / 4;
    const int num_blocks = (size_v4 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    batchnorm_then_relu_add_kernel_fma<<<num_blocks, THREADS_PER_BLOCK>>>(
        (const float4*)in1_contig.data_ptr<float>(), (const float4*)in2_contig.data_ptr<float>(),
        (float4*)output.data_ptr<float>(),
        running_mean.data_ptr<float>(), running_var.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        static_cast<float>(eps), size_v4, C, HW);
    
    return output;
}
"""

cpp_source = """
torch::Tensor batchnorm_relu_cuda(
    torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight, torch::Tensor bias, double eps);

torch::Tensor batchnorm_channel_shuffle_cuda(
    torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight, torch::Tensor bias, double eps, int64_t groups);

torch::Tensor batchnorm_then_relu_add_cuda(
    torch::Tensor in1, torch::Tensor in2, 
    torch::Tensor running_mean, torch::Tensor running_var,
    torch::Tensor weight, torch::Tensor bias, double eps);
"""

custom_ops = load_inline(
    name="custom_shufflenet_ops_hybrid_shuffle",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["batchnorm_relu_cuda", "batchnorm_channel_shuffle_cuda", "batchnorm_then_relu_add_cuda"],
    verbose=False,
)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        """
        ShuffleNet unit implementation with fused CUDA kernels and concurrent streams.
        """
        super(Model, self).__init__()
        
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.groups = groups
        
        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.main_stream = torch.cuda.Stream()
        self.shortcut_stream = torch.cuda.Stream()
    
    def forward(self, x):
        """
        Forward pass using streams to overlap shortcut and main path execution.
        """
        # --- Main Path ---
        with torch.cuda.stream(self.main_stream):
            main_out = self.conv1(x)
            main_out = custom_ops.batchnorm_relu_cuda(
                main_out, self.bn1.running_mean, self.bn1.running_var, 
                self.bn1.weight, self.bn1.bias, self.bn1.eps
            )
            main_out = self.conv2(main_out)
            main_out = custom_ops.batchnorm_channel_shuffle_cuda(
                main_out, self.bn2.running_mean, self.bn2.running_var,
                self.bn2.weight, self.bn2.bias, self.bn2.eps, self.groups
            )
            main_out = self.conv3(main_out)
        
        # --- Shortcut Path ---
        with torch.cuda.stream(self.shortcut_stream):
            shortcut_out = self.shortcut(x)

        # --- Synchronization and Final Operation ---
        torch.cuda.current_stream().wait_stream(self.main_stream)
        torch.cuda.current_stream().wait_stream(self.shortcut_stream)
        
        out = custom_ops.batchnorm_then_relu_add_cuda(
            main_out, shortcut_out,
            self.bn3.running_mean, self.bn3.running_var,
            self.bn3.weight, self.bn3.bias, self.bn3.eps
        )
        
        return out

batch_size = 10
input_channels = 240
out_channels = 480
groups = 3
height = 224
width = 224
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width).cuda()]

def get_init_inputs():
    return [input_channels, out_channels, groups]
# EVOLVE-BLOCK-END