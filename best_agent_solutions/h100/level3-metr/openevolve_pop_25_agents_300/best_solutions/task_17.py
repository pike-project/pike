# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# This solution introduces output channel tiling to the fused expand kernel.
# Each CUDA thread now computes two output channels simultaneously. This allows the thread
# to load input values from global memory once and reuse them for both output channels,
# effectively halving the memory bandwidth required for the input tensor and significantly
# improving the arithmetic intensity of the kernel. This builds upon the best features of
# previous solutions (fusion, vectorization, unrolling, FMA) to push performance further.
#
# FIX: The original code had a compilation error because `output_size(1)` is not a valid
# function/variable inside the CUDA kernel. The total number of output channels, C_out_total,
# was calculated in the host C++ code but not passed to the kernel. This fix adds C_out_total
# as a kernel argument and passes it during the kernel launch. The erroneous line in the
# kernel is removed.
fused_expand_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__global__ void fused_expand_tiled_channels_kernel(
    const float* __restrict__ input,
    const float* __restrict__ w_1x1,
    const float* __restrict__ b_1x1,
    const float* __restrict__ w_3x3,
    const float* __restrict__ b_3x3,
    float* __restrict__ output,
    const int N, const int C_in, const int H, const int W,
    const int C_out1,
    const int C_out_total,
    const int W_vec,
    const long C_pair_stride_vec, const long N_pair_stride_vec
) {
    const int C_out_pairs = C_out_total / 2;
    const long total_output_pairs_vec = (long)N * C_out_pairs * H * W_vec;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_stride = gridDim.x * blockDim.x;

    for (long idx_pair_vec = i; idx_pair_vec < total_output_pairs_vec; idx_pair_vec += grid_stride) {
        // --- Decompose index to get (n, c_pair, h, w_vec) ---
        const int n = idx_pair_vec / N_pair_stride_vec;
        long temp = idx_pair_vec % N_pair_stride_vec;
        const int c_out_pair_idx = temp / C_pair_stride_vec;
        temp %= C_pair_stride_vec;
        const int h_out = temp / W_vec;
        const int w_out_vec = temp % W_vec;

        const int c_out_total1 = c_out_pair_idx * 2;
        const int c_out_total2 = c_out_total1 + 1;

        const int w_out_base = w_out_vec * 4;

        float4 acc1, acc2;

        if (c_out_total1 < C_out1) {
            // --- 1x1 Convolution Path (Tiled) ---
            acc1 = make_float4(b_1x1[c_out_total1], b_1x1[c_out_total1], b_1x1[c_out_total1], b_1x1[c_out_total1]);
            acc2 = make_float4(b_1x1[c_out_total2], b_1x1[c_out_total2], b_1x1[c_out_total2], b_1x1[c_out_total2]);
            
            const float* input_ptr = &input[n * C_in * H * W + h_out * W + w_out_base];
            const float* w_1x1_ptr1 = &w_1x1[c_out_total1 * C_in];
            const float* w_1x1_ptr2 = &w_1x1[c_out_total2 * C_in];

            for (int c_in = 0; c_in < C_in; ++c_in) {
                // Load input ONCE, reuse for two output channels
                const float4 in_val = *reinterpret_cast<const float4*>(&input_ptr[c_in * H * W]);
                
                const float weight1 = w_1x1_ptr1[c_in];
                const float weight2 = w_1x1_ptr2[c_in];
                
                acc1.x += in_val.x * weight1; acc1.y += in_val.y * weight1; acc1.z += in_val.z * weight1; acc1.w += in_val.w * weight1;
                acc2.x += in_val.x * weight2; acc2.y += in_val.y * weight2; acc2.z += in_val.z * weight2; acc2.w += in_val.w * weight2;
            }
        } else {
            // --- 3x3 Convolution Path (Tiled) ---
            const int c_out2_idx1 = c_out_total1 - C_out1;
            const int c_out2_idx2 = c_out_total2 - C_out1;
            acc1 = make_float4(b_3x3[c_out2_idx1], b_3x3[c_out2_idx1], b_3x3[c_out2_idx1], b_3x3[c_out2_idx1]);
            acc2 = make_float4(b_3x3[c_out2_idx2], b_3x3[c_out2_idx2], b_3x3[c_out2_idx2], b_3x3[c_out2_idx2]);
            
            const float* w_3x3_ptr1 = &w_3x3[c_out2_idx1 * C_in * 9];
            const float* w_3x3_ptr2 = &w_3x3[c_out2_idx2 * C_in * 9];

            for (int c_in = 0; c_in < C_in; ++c_in) {
                const float* input_channel_ptr = &input[n * C_in * H * W + c_in * H * W];

                #pragma unroll
                for (int kh = 0; kh < 3; ++kh) {
                    const int h_in = h_out + kh - 1;
                    if (h_in < 0 || h_in >= H) continue;

                    const float* input_row_ptr = input_channel_ptr + h_in * W;
                    const float* w_row_ptr1 = w_3x3_ptr1 + c_in * 9 + kh * 3;
                    const float* w_row_ptr2 = w_3x3_ptr2 + c_in * 9 + kh * 3;

                    // Load input data ONCE, reuse for two output channels
                    const float4 in_kw1 = *reinterpret_cast<const float4*>(&input_row_ptr[w_out_base]);
                    const int w_in_kw0_base = w_out_base - 1;
                    const float in_m1 = (w_in_kw0_base >= 0) ? input_row_ptr[w_in_kw0_base] : 0.f;
                    const int w_in_kw2_base = w_out_base + 4;
                    const float in_p4 = (w_in_kw2_base < W) ? input_row_ptr[w_in_kw2_base] : 0.f;

                    const float w0_1 = w_row_ptr1[0], w1_1 = w_row_ptr1[1], w2_1 = w_row_ptr1[2];
                    const float w0_2 = w_row_ptr2[0], w1_2 = w_row_ptr2[1], w2_2 = w_row_ptr2[2];
                    
                    acc1.x += in_m1   * w0_1 + in_kw1.x * w1_1 + in_kw1.y * w2_1;
                    acc1.y += in_kw1.x * w0_1 + in_kw1.y * w1_1 + in_kw1.z * w2_1;
                    acc1.z += in_kw1.y * w0_1 + in_kw1.z * w1_1 + in_kw1.w * w2_1;
                    acc1.w += in_kw1.z * w0_1 + in_kw1.w * w1_1 + in_p4   * w2_1;
                    
                    acc2.x += in_m1   * w0_2 + in_kw1.x * w1_2 + in_kw1.y * w2_2;
                    acc2.y += in_kw1.x * w0_2 + in_kw1.y * w1_2 + in_kw1.z * w2_2;
                    acc2.z += in_kw1.y * w0_2 + in_kw1.z * w1_2 + in_kw1.w * w2_2;
                    acc2.w += in_kw1.z * w0_2 + in_kw1.w * w1_2 + in_p4   * w2_2;
                }
            }
        }

        acc1.x = max(0.0f, acc1.x); acc1.y = max(0.0f, acc1.y); acc1.z = max(0.0f, acc1.z); acc1.w = max(0.0f, acc1.w);
        acc2.x = max(0.0f, acc2.x); acc2.y = max(0.0f, acc2.y); acc2.z = max(0.0f, acc2.z); acc2.w = max(0.0f, acc2.w);
        
        const long C_stride_vec_out = (long)H * W_vec;
        const long N_stride_vec_out = (long)C_out_total * C_stride_vec_out;
        const long out_idx1_vec = n * N_stride_vec_out + c_out_total1 * C_stride_vec_out + h_out * W_vec + w_out_vec;
        const long out_idx2_vec = out_idx1_vec + C_stride_vec_out;

        reinterpret_cast<float4*>(output)[out_idx1_vec] = acc1;
        reinterpret_cast<float4*>(output)[out_idx2_vec] = acc2;
    }
}

torch::Tensor fused_expand_cuda(
    torch::Tensor input,
    torch::Tensor w_1x1,
    torch::Tensor b_1x1,
    torch::Tensor w_3x3,
    torch::Tensor b_3x3
) {
    const auto N = input.size(0);
    const auto C_in = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);
    const auto C_out1 = w_1x1.size(0);
    const auto C_out2 = w_3x3.size(0);
    const auto C_out_total = C_out1 + C_out2;
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(W % 4 == 0, "Width must be a multiple of 4 for vectorization");
    TORCH_CHECK(C_out_total % 2 == 0, "Total output channels must be even for tiling");
    TORCH_CHECK(C_out1 % 2 == 0, "1x1 output channels must be even for tiling boundary");

    auto output = torch::empty({N, C_out_total, H, W}, input.options());

    const int W_vec = W / 4;
    const int C_out_pairs = C_out_total / 2;
    const long total_output_pairs_vec = (long)N * C_out_pairs * H * W_vec;
    const int block_size = 256;
    const int num_blocks = (total_output_pairs_vec + block_size - 1) / block_size;

    const long C_pair_stride_vec = (long)H * W_vec;
    const long N_pair_stride_vec = (long)C_out_pairs * C_pair_stride_vec;

    fused_expand_tiled_channels_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        w_1x1.data_ptr<float>(),
        b_1x1.data_ptr<float>(),
        w_3x3.data_ptr<float>(),
        b_3x3.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H, W,
        C_out1,
        C_out_total,
        W_vec, C_pair_stride_vec, N_pair_stride_vec
    );
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

fused_expand_cpp_source = "torch::Tensor fused_expand_cuda(torch::Tensor input, torch::Tensor w_1x1, torch::Tensor b_1x1, torch::Tensor w_3x3, torch::Tensor b_3x3);"

# Compile the inline CUDA code
fused_expand_op = load_inline(
    name="fused_expand_op_v6_tiled_fixed",
    cpp_sources=fused_expand_cpp_source,
    cuda_sources=fused_expand_source,
    functions=["fused_expand_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--use_fast_math"]
)

class Model(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Model, self).__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        # Expand layers are kept to hold the parameters (weights and biases) for our kernel
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        squeezed_x = self.squeeze_activation(self.squeeze(x))

        return fused_expand_op.fused_expand_cuda(
            squeezed_x,
            self.expand1x1.weight,
            self.expand1x1.bias,
            self.expand3x3.weight,
            self.expand3x3.bias
        )

# Test code
batch_size = 10
num_input_features = 3
num_output_features = 64
height, width = 224, 224
squeeze_channels = 6
expand1x1_channels = 64
expand3x3_channels = 64

def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width).cuda()]

def get_init_inputs():
    return [num_input_features, squeeze_channels, expand1x1_channels, expand3x3_channels]
# EVOLVE-BLOCK-END