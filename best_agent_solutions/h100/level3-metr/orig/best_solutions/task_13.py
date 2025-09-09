import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# CUDA and C++ source for the fused kernels.
# This version includes an improved vectorized kernel for performance and
# retains the original scalar kernel as a fallback for general compatibility.
bn_relu_avgpool_fused_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// --- KERNEL 1: Original Scalar Implementation (Fallback) ---
// This kernel is used for input dimensions not compatible with the vectorized version.
__global__ void bn_relu_avgpool_fused_scalar_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    const float bn_eps,
    float* __restrict__ output,
    const int N, const int C, const int H_in, const int W_in,
    const int H_out, const int W_out)
{
    const int total_output_elements = N * C * H_out * W_out;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_output_elements; i += gridDim.x * blockDim.x) {
        int remaining = i;
        const int w_out = remaining % W_out; remaining /= W_out;
        const int h_out = remaining % H_out; remaining /= H_out;
        const int c = remaining % C;
        const int n = remaining / C;

        const int h_in_start = h_out * 2;
        const int w_in_start = w_out * 2;

        const float mean = bn_mean[c];
        const float var = bn_var[c];
        const float inv_std = rsqrtf(var + bn_eps);
        const float bn_scale = bn_weight[c] * inv_std;
        const float bn_shift = bn_bias[c] - mean * bn_scale;

        const int in_batch_stride = C * H_in * W_in;
        const int in_channel_stride = H_in * W_in;
        const float* in_ptr_channel_base = input + n * in_batch_stride + c * in_channel_stride;

        const float val00 = in_ptr_channel_base[h_in_start * W_in + w_in_start];
        const float val01 = in_ptr_channel_base[h_in_start * W_in + w_in_start + 1];
        const float val10 = in_ptr_channel_base[(h_in_start + 1) * W_in + w_in_start];
        const float val11 = in_ptr_channel_base[(h_in_start + 1) * W_in + w_in_start + 1];

        const float bn_relu00 = fmaxf(0.0f, val00 * bn_scale + bn_shift);
        const float bn_relu01 = fmaxf(0.0f, val01 * bn_scale + bn_shift);
        const float bn_relu10 = fmaxf(0.0f, val10 * bn_scale + bn_shift);
        const float bn_relu11 = fmaxf(0.0f, val11 * bn_scale + bn_shift);

        output[i] = (bn_relu00 + bn_relu01 + bn_relu10 + bn_relu11) * 0.25f;
    }
}

// --- KERNEL 2: High-Performance Vectorized Implementation ---

// Helper device function to apply BatchNorm and ReLU to a float4 vector.
__device__ inline float4 bn_relu_f4(const float4& v, const float scale, const float shift) {
    float4 result;
    result.x = fmaxf(0.0f, v.x * scale + shift);
    result.y = fmaxf(0.0f, v.y * scale + shift);
    result.z = fmaxf(0.0f, v.z * scale + shift);
    result.w = fmaxf(0.0f, v.w * scale + shift);
    return result;
}

__global__ void bn_relu_avgpool_fused_vectorized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bn_weight,
    const float* __restrict__ bn_bias,
    const float* __restrict__ bn_mean,
    const float* __restrict__ bn_var,
    const float bn_eps,
    float* __restrict__ output,
    const int N, const int C, const int H_in, const int W_in,
    const int H_out, const int W_out)
{
    // Each thread processes a vector of 4 output elements, which corresponds to a 1x4 horizontal strip.
    const int total_output_elements = N * C * H_out * W_out;
    const int num_vecs = total_output_elements / 4;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vecs; i += gridDim.x * blockDim.x) {
        // Decompose the linear vector index `i` into 4D coordinates (n, c, h_out, w_out_vec).
        int remaining = i;
        const int w_out_vec_dim = W_out / 4;
        const int w_out_vec = remaining % w_out_vec_dim; remaining /= w_out_vec_dim;
        const int h_out = remaining % H_out; remaining /= H_out;
        const int c = remaining % C;
        const int n = remaining / C;

        const float mean = bn_mean[c];
        const float var = bn_var[c];
        const float inv_std = rsqrtf(var + bn_eps);
        const float bn_scale = bn_weight[c] * inv_std;
        const float bn_shift = bn_bias[c] - mean * bn_scale;
        
        // Calculate input pointers. Each thread needs a 2x8 patch of input data to compute 4 output pixels.
        const int h_in = h_out * 2;
        const int w_in_start = w_out_vec * 8; // 4 output pixels correspond to an 8-wide strip in the input.

        const int in_batch_stride = C * H_in * W_in;
        const int in_channel_stride = H_in * W_in;
        const float* in_ptr_base = input + n * in_batch_stride + c * in_channel_stride;
        const float* in_ptr_row0 = in_ptr_base + h_in * W_in + w_in_start;
        const float* in_ptr_row1 = in_ptr_row0 + W_in;

        // Load the 2x8 input patch using four float4 loads for maximum memory bandwidth.
        const float4 in00 = reinterpret_cast<const float4*>(in_ptr_row0)[0];
        const float4 in01 = reinterpret_cast<const float4*>(in_ptr_row0)[1];
        const float4 in10 = reinterpret_cast<const float4*>(in_ptr_row1)[0];
        const float4 in11 = reinterpret_cast<const float4*>(in_ptr_row1)[1];

        // Apply BatchNorm and ReLU on the loaded vectors using the helper function.
        const float4 bn_relu00 = bn_relu_f4(in00, bn_scale, bn_shift);
        const float4 bn_relu01 = bn_relu_f4(in01, bn_scale, bn_shift);
        const float4 bn_relu10 = bn_relu_f4(in10, bn_scale, bn_shift);
        const float4 bn_relu11 = bn_relu_f4(in11, bn_scale, bn_shift);

        // Perform 4 separate 2x2 average pools on the 16 processed values.
        const float out0 = (bn_relu00.x + bn_relu00.y + bn_relu10.x + bn_relu10.y) * 0.25f;
        const float out1 = (bn_relu00.z + bn_relu00.w + bn_relu10.z + bn_relu10.w) * 0.25f;
        const float out2 = (bn_relu01.x + bn_relu01.y + bn_relu11.x + bn_relu11.y) * 0.25f;
        const float out3 = (bn_relu01.z + bn_relu01.w + bn_relu11.z + bn_relu11.w) * 0.25f;

        // Pack the 4 scalar results into a float4 vector and store to global memory.
        float4 out_vec = {out0, out1, out2, out3};
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }
}

// --- C++ Wrapper and Dispatcher ---
torch::Tensor bn_relu_avgpool_fused_cuda(
    torch::Tensor input,
    torch::Tensor bn_weight,
    torch::Tensor bn_bias,
    torch::Tensor bn_mean,
    torch::Tensor bn_var,
    double bn_eps)
{
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(c10::MemoryFormat::Contiguous), "Input must be a contiguous CUDA tensor");
    TORCH_CHECK(bn_weight.is_cuda() && bn_weight.is_contiguous(), "bn_weight must be a contiguous CUDA tensor");
    TORCH_CHECK(bn_bias.is_cuda() && bn_bias.is_contiguous(), "bn_bias must be a contiguous CUDA tensor");
    TORCH_CHECK(bn_mean.is_cuda() && bn_mean.is_contiguous(), "bn_mean must be a contiguous CUDA tensor");
    TORCH_CHECK(bn_var.is_cuda() && bn_var.is_contiguous(), "bn_var must be a contiguous CUDA tensor");

    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    const int N = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    TORCH_CHECK(C == bn_weight.size(0), "Channel dimensions of input and batchnorm parameters must match");
    TORCH_CHECK(H_in % 2 == 0 && W_in % 2 == 0, "Input height and width must be even for 2x2 pooling");
    const int H_out = H_in / 2;
    const int W_out = W_in / 2;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    const int total_output_elements = N * C * H_out * W_out;
    if (total_output_elements == 0) {
        return output;
    }

    // --- Kernel Dispatch Logic ---
    // The vectorized kernel is faster but requires W_in to be a multiple of 8,
    // which ensures W_out is a multiple of 4 and that all memory accesses are safe.
    const bool use_vectorized_kernel = (W_in % 8 == 0);
    const int block_size = 256;

    if (use_vectorized_kernel) {
        const int num_vecs = total_output_elements / 4;
        const int num_blocks = (num_vecs + block_size - 1) / block_size;
        bn_relu_avgpool_fused_vectorized_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
            bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(), static_cast<float>(bn_eps),
            output.data_ptr<float>(), N, C, H_in, W_in, H_out, W_out);
    } else {
        const int num_blocks = (total_output_elements + block_size - 1) / block_size;
        bn_relu_avgpool_fused_scalar_kernel<<<num_blocks, block_size>>>(
            input.data_ptr<float>(), bn_weight.data_ptr<float>(), bn_bias.data_ptr<float>(),
            bn_mean.data_ptr<float>(), bn_var.data_ptr<float>(), static_cast<float>(bn_eps),
            output.data_ptr<float>(), N, C, H_in, W_in, H_out, W_out);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

bn_relu_avgpool_fused_cpp_source = """
torch::Tensor bn_relu_avgpool_fused_cuda(
    torch::Tensor input, torch::Tensor bn_weight,
    torch::Tensor bn_bias, torch::Tensor bn_mean,
    torch::Tensor bn_var, double bn_eps);
"""

# JIT compile the fused op. Using a new name to avoid caching issues.
bn_relu_avgpool_op = load_inline(
    name="bn_relu_avgpool_op_v2",
    cpp_sources=bn_relu_avgpool_fused_cpp_source,
    cuda_sources=bn_relu_avgpool_fused_cuda_source,
    functions=["bn_relu_avgpool_fused_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(ModelNew, self).__init__()
        # Instantiate original layers to hold parameters and buffers
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        # Store the compiled custom operator
        self.fused_bn_relu_pool_op = bn_relu_avgpool_op

    def forward(self, x):
        """
        Executes the forward pass using a two-stage optimized approach.
        Stage 1: Custom fused kernel for BatchNorm -> ReLU -> AvgPool.
        Stage 2: PyTorch's optimized 1x1 Convolution.
        """
        # Call the custom fused kernel. The C++ dispatcher inside handles
        # contiguity checks and selects the best kernel (vectorized/scalar).
        processed_x = self.fused_bn_relu_pool_op.bn_relu_avgpool_fused_cuda(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps
        )

        # Call the standard Conv2d layer for the compute-bound operation.
        output = self.conv(processed_x)
        return output


# --- Boilerplate for testing ---
batch_size = 10
num_input_features = 32
num_output_features = 64
height, width = 224, 224

def get_inputs():
    # Input tensor for the forward pass
    return [torch.randn(batch_size, num_input_features, height, width).cuda()]

def get_init_inputs():
    # Arguments for the model's __init__ method
    return [num_input_features, num_output_features]