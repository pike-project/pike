import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# --- Fused CUDA/C++ Kernels ---
# This version adds a custom 3x3 depthwise convolution kernel.

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>
#include <map>
#include <cublas_v2.h>

// --- cuBLAS Handle Management ---
static cublasHandle_t get_cublas_handle() {
    static bool initialized = false;
    static cublasHandle_t handle;
    if (!initialized) {
        TORCH_CHECK(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cuBLAS handle creation failed");
        initialized = true;
    }
    return handle;
}

// --- KERNEL 1: Fused Bias + Activation (In-place) ---
enum class ActivationType { NONE = 0, RELU = 1, RELU6 = 2 };

template <typename T, ActivationType activation>
__global__ void bias_activation_kernel_inplace(
    T* __restrict__ tensor, const T* __restrict__ bias,
    const int total_elements, const int elements_per_channel, const int num_channels) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements; i += blockDim.x * gridDim.x) {
        const int channel_idx = (i / elements_per_channel) % num_channels;
        float val_f = static_cast<float>(tensor[i]) + static_cast<float>(bias[channel_idx]);
        if constexpr (activation == ActivationType::RELU) { val_f = fmaxf(0.0f, val_f); }
        else if constexpr (activation == ActivationType::RELU6) { val_f = fminf(fmaxf(0.0f, val_f), 6.0f); }
        tensor[i] = static_cast<T>(val_f);
    }
}

template <ActivationType activation>
void launch_bias_activation(torch::Tensor& output, const torch::Tensor& bias) {
    const int total_elements = output.numel();
    if (total_elements == 0) return;
    const int num_channels = output.size(1);
    const int elements_per_channel = output.numel() / (output.size(0) * output.size(1));
    const int block_size = 256;
    const int num_blocks = std::min((total_elements + block_size - 1) / block_size, 4096);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "bias_activation_launcher", [&] {
        bias_activation_kernel_inplace<scalar_t, activation><<<num_blocks, block_size>>>(
            output.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
            total_elements, elements_per_channel, num_channels);
    });
}

// --- KERNEL 2: Custom 3x3 Depthwise Convolution (FP16 only) ---
template <int STRIDE>
__global__ void depthwise_conv3x3_p1_fp16_kernel(
    const __half* __restrict__ input, const __half* __restrict__ weight, __half* __restrict__ output,
    const int C, const int H_in, const int W_in, const int H_out, const int W_out) {

    constexpr int TILE_W = 16;
    constexpr int TILE_H = 16;
    constexpr int SH_TILE_W = (TILE_W - 1) * STRIDE + 3;
    constexpr int SH_TILE_H = (TILE_H - 1) * STRIDE + 3;
    __shared__ __half s_input[SH_TILE_H * SH_TILE_W];
    __shared__ __half s_weight[9];

    const int plane_idx = blockIdx.z;
    const int n = plane_idx / C;
    const int c = plane_idx % C;

    const __half* input_plane = input + n * C * H_in * W_in + c * H_in * W_in;
    const __half* weight_ch = weight + c * 9;
    __half* output_plane = output + n * C * H_out * W_out + c * H_out * W_out;

    if (threadIdx.x < 9 && threadIdx.y == 0) { s_weight[threadIdx.x] = weight_ch[threadIdx.x]; }
    
    const int tile_x_in_start = (blockIdx.x * TILE_W) * STRIDE - 1; // -1 for padding=1
    const int tile_y_in_start = (blockIdx.y * TILE_H) * STRIDE - 1; // -1 for padding=1
    
    __syncthreads();

    for (int i = threadIdx.y * TILE_W + threadIdx.x; i < SH_TILE_H * SH_TILE_W; i += TILE_W * TILE_H) {
        int y = i / SH_TILE_W;
        int x = i % SH_TILE_W;
        int in_y = tile_y_in_start + y;
        int in_x = tile_x_in_start + x;
        s_input[i] = (in_y >= 0 && in_y < H_in && in_x >= 0 && in_x < W_in) ? input_plane[in_y * W_in + in_x] : __float2half(0.0f);
    }
    __syncthreads();

    const int x_out = blockIdx.x * TILE_W + threadIdx.x;
    const int y_out = blockIdx.y * TILE_H + threadIdx.y;

    if (x_out < W_out && y_out < H_out) {
        float acc = 0.0f;
        const int sh_y_start = threadIdx.y * STRIDE;
        const int sh_x_start = threadIdx.x * STRIDE;
        
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                acc += __half2float(s_input[(sh_y_start + ky) * SH_TILE_W + (sh_x_start + kx)]) * __half2float(s_weight[ky * 3 + kx]);
            }
        }
        output_plane[y_out * W_out + x_out] = __float2half(acc);
    }
}

torch::Tensor depthwise_conv3x3_cuda(
    const torch::Tensor& input, const torch::Tensor& weight, int stride) {
    
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "Input must be a contiguous CUDA tensor");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous(), "Weight must be a contiguous CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kHalf, "Custom depthwise kernel only supports FP16");

    const int N = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int H_out = (H_in + 2 * 1 - 3) / stride + 1;
    const int W_out = (W_in + 2 * 1 - 3) / stride + 1;
    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    constexpr int TILE_W = 16, TILE_H = 16;
    const dim3 block_dim(TILE_W, TILE_H, 1);
    const dim3 grid_dim((W_out + TILE_W - 1) / TILE_W, (H_out + TILE_H - 1) / TILE_H, N * C);

    if (stride == 1) {
        depthwise_conv3x3_p1_fp16_kernel<1><<<grid_dim, block_dim>>>(
            (const __half*)input.data_ptr<at::Half>(), (const __half*)weight.data_ptr<at::Half>(), (__half*)output.data_ptr<at::Half>(),
            C, H_in, W_in, H_out, W_out);
    } else if (stride == 2) {
        depthwise_conv3x3_p1_fp16_kernel<2><<<grid_dim, block_dim>>>(
            (const __half*)input.data_ptr<at::Half>(), (const __half*)weight.data_ptr<at::Half>(), (__half*)output.data_ptr<at::Half>(),
            C, H_in, W_in, H_out, W_out);
    } else {
        TORCH_CHECK(false, "Custom depthwise kernel only supports stride 1 or 2.");
    }
    return output;
}

// --- KERNEL 3: Fused Global Average Pooling ---
template <typename T>
__global__ void global_avg_pool_kernel(
    const T* __restrict__ input, T* __restrict__ output,
    const int total_planes, const int plane_size, const int planes_per_block) {

    extern __shared__ float sdata[];
    const float inv_plane_size = 1.0f / static_cast<float>(plane_size);
    const int block_plane_start_idx = blockIdx.x * planes_per_block;

    for (int p_offset = 0; p_offset < planes_per_block; ++p_offset) {
        int plane_idx = block_plane_start_idx + p_offset;
        if (plane_idx >= total_planes) break;

        const T* plane_start = input + plane_idx * plane_size;
        float thread_sum = 0.0f;

        for (int i = threadIdx.x; i < plane_size; i += blockDim.x) {
            thread_sum += static_cast<float>(plane_start[i]);
        }
        
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
        }

        if (threadIdx.x % 32 == 0) { sdata[threadIdx.x / 32] = thread_sum; }
        __syncthreads();

        if (threadIdx.x < (blockDim.x / 32)) { thread_sum = sdata[threadIdx.x]; } else { thread_sum = 0.0f; }

        if (threadIdx.x < 32) {
             for (int offset = 16; offset > 0; offset /= 2) {
                thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);
            }
        }

        if (threadIdx.x == 0) { output[plane_idx] = static_cast<T>(thread_sum * inv_plane_size); }
        __syncthreads();
    }
}

torch::Tensor global_avg_pool_cuda(const torch::Tensor& input) {
    auto input_contig = input.contiguous();
    const int N = input_contig.size(0), C = input_contig.size(1), H = input_contig.size(2), W = input_contig.size(3);
    const int total_planes = N * C, plane_size = H * W;
    auto output = torch::empty({N, C}, input_contig.options());
    if (plane_size == 0 || total_planes == 0) { output.zero_(); return output.view({N, C, 1, 1}); }

    const int block_size = 512, warps_per_block = block_size / 32;
    const int planes_per_block = 4, num_blocks = (total_planes + planes_per_block - 1) / planes_per_block;
    const size_t shared_mem_size = warps_per_block * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_contig.scalar_type(), "global_avg_pool_launcher", [&] {
        global_avg_pool_kernel<scalar_t><<<num_blocks, block_size, shared_mem_size>>>(
            input_contig.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
            total_planes, plane_size, planes_per_block);
    });
    return output;
}

// --- Main Dispatcher: Fused Conv + Bias + Activation ---
torch::Tensor fused_conv_bias_activation(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation,
    int64_t groups, const std::string& activation_name) {
    
    auto input_cont = input.contiguous();
    auto weight_cont = weight.contiguous();

    const int N = input_cont.size(0);
    const int C_in = input_cont.size(1);
    const int H = input_cont.size(2);
    const int W = input_cont.size(3);
    const int C_out = weight_cont.size(0);

    const bool use_cublas_for_1x1 = (
        input_cont.scalar_type() == torch::kHalf && weight_cont.size(2) == 1 && weight_cont.size(3) == 1 &&
        stride[0] == 1 && stride[1] == 1 && padding[0] == 0 && padding[1] == 0 &&
        dilation[0] == 1 && dilation[1] == 1 && groups == 1
    );

    const bool use_custom_depthwise = (
        input_cont.scalar_type() == torch::kHalf && groups > 1 && groups == C_in &&
        weight_cont.size(2) == 3 && weight_cont.size(3) == 3 &&
        padding[0] == 1 && padding[1] == 1 && dilation[0] == 1 && dilation[1] == 1
    );
    
    torch::Tensor conv_output;

    if (use_cublas_for_1x1) {
        const int H_out = (H + 2 * padding[0] - dilation[0] * (weight_cont.size(2) - 1) - 1) / stride[0] + 1;
        const int W_out = (W + 2 * padding[1] - dilation[1] * (weight_cont.size(3) - 1) - 1) / stride[1] + 1;
        conv_output = torch::empty({N, C_out, H_out, W_out}, input_cont.options());
        const __half alpha = __float2half(1.0f); const __half beta = __float2half(0.0f);
        cublasHandle_t handle = get_cublas_handle();
        cublasStatus_t status = cublasHgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, H * W, C_out, C_in, &alpha,
            (const __half*)input_cont.data_ptr<at::Half>(), H * W, (long long)C_in * H * W,
            (const __half*)weight_cont.data_ptr<at::Half>(), C_in, 0,
            &beta, (__half*)conv_output.data_ptr<at::Half>(), H * W, (long long)C_out * H * W, N );
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS Hgemm Strided Batched for 1x1 Conv failed. Error: ", status);
    } else if (use_custom_depthwise) {
        conv_output = depthwise_conv3x3_cuda(input_cont, weight_cont, stride[0]);
    } else {
        conv_output = at::conv2d(input_cont, weight_cont, {}, stride, padding, dilation, groups);
    }
    
    static const std::map<std::string, ActivationType> activation_map = {
        {"none", ActivationType::NONE}, {"relu", ActivationType::RELU}, {"relu6", ActivationType::RELU6}
    };
    ActivationType activation_type = activation_map.at(activation_name);

    if (activation_type != ActivationType::NONE) {
        if (activation_type == ActivationType::RELU) { launch_bias_activation<ActivationType::RELU>(conv_output, bias); }
        else if (activation_type == ActivationType::RELU6) { launch_bias_activation<ActivationType::RELU6>(conv_output, bias); }
    } else {
        launch_bias_activation<ActivationType::NONE>(conv_output, bias);
    }
    
    return conv_output;
}
"""

fused_ops_cpp_source = """
#include <string>
torch::Tensor fused_conv_bias_activation(
    const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& bias,
    torch::IntArrayRef stride, torch::IntArrayRef padding, torch::IntArrayRef dilation,
    int64_t groups, const std::string& activation_name);
torch::Tensor depthwise_conv3x3_cuda(const torch::Tensor& input, const torch::Tensor& weight, int stride);
torch::Tensor global_avg_pool_cuda(const torch::Tensor& input);
"""

fused_ops = load_inline(
    name="fused_efficientnet_ops_v2",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_conv_bias_activation", "global_avg_pool_cuda", "depthwise_conv3x3_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_70"],
    extra_ldflags=["-lcublas"]
)

def fuse_conv_bn(conv, bn):
    assert not (conv.training or bn.training), "Fusion only works in eval mode."
    w_conv = conv.weight.clone()
    b_conv = conv.bias.clone() if conv.bias is not None else torch.zeros(conv.out_channels, device=w_conv.device)
    gamma, beta, mean, var, eps = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
    std = torch.sqrt(var + eps)
    scale_factor = gamma / std
    w_fused = w_conv * scale_factor.view(-1, *([1] * (w_conv.dim() - 1)))
    b_fused = scale_factor * (b_conv - mean) + beta
    return w_fused, b_fused

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        
        original_model = Model(num_classes)
        original_model.eval()

        self.params = nn.ParameterDict()
        
        def process_block(name, conv, bn):
            w_fused, b_fused = fuse_conv_bn(conv, bn)
            self.params[f"{name}_w"] = nn.Parameter(w_fused.half())
            self.params[f"{name}_b"] = nn.Parameter(b_fused.half())

        process_block('stem_conv', original_model.conv1, original_model.bn1)
        
        self.sequential_mbconv_configs = [
            {'p': 'mbconv1', 'in_c': 32, 'out_c': 16, 's': 1, 't': 1},
            {'p': 'mbconv2', 'in_c': 16, 'out_c': 24, 's': 2, 't': 6},
            {'p': 'mbconv3', 'in_c': 24, 'out_c': 40, 's': 2, 't': 6},
            {'p': 'mbconv4', 'in_c': 40, 'out_c': 80, 's': 2, 't': 6},
            {'p': 'mbconv5', 'in_c': 80, 'out_c': 112, 's': 1, 't': 6},
            {'p': 'mbconv6', 'in_c': 112, 'out_c': 192, 's': 2, 't': 6},
            {'p': 'mbconv7', 'in_c': 192, 'out_c': 320, 's': 1, 't': 6},
        ]
        
        original_mb_blocks = [m for m in original_model.children() if isinstance(m, nn.Sequential)][:7]
        
        for i, config in enumerate(self.sequential_mbconv_configs):
            prefix = config['p']
            block_module = original_mb_blocks[i]
            process_block(f'{prefix}_e', block_module[0], block_module[1])
            process_block(f'{prefix}_d', block_module[3], block_module[4])
            process_block(f'{prefix}_p', block_module[6], block_module[7])

        process_block('head_conv', original_model.conv2, original_model.bn2)
        fc = original_model.fc
        self.params["fc_w"] = nn.Parameter(fc.weight.half())
        self.params["fc_b"] = nn.Parameter(fc.bias.half())
        
        self.graph = None
        self.static_input = None
        self.static_output = None

    def _call_fused_conv(self, x, name_prefix, stride, padding, groups, activation):
        return fused_ops.fused_conv_bias_activation(
            x, self.params[f"{name_prefix}_w"], self.params[f"{name_prefix}_b"],
            stride, padding, [1, 1], groups, activation
        )

    def _forward_mbconv(self, x, prefix, stride, in_c, out_c, t):
        hidden_dim = round(in_c * t)
        
        out = self._call_fused_conv(x, f"{prefix}_e", (1, 1), (0, 0), 1, 'relu6')
        out = self._call_fused_conv(out, f"{prefix}_d", (stride, stride), (1, 1), hidden_dim, 'relu6')
        out = self._call_fused_conv(out, f"{prefix}_p", (1, 1), (0, 0), 1, 'none')
        
        return out
    
    def _forward_impl(self, x):
        x = x.cuda().half()
        
        x = self._call_fused_conv(x, 'stem_conv', (2, 2), (1, 1), 1, 'relu')
        for config in self.sequential_mbconv_configs:
            x = self._forward_mbconv(x, config['p'], config['s'], config['in_c'], config['out_c'], config['t'])
        
        x = self._call_fused_conv(x, 'head_conv', (1, 1), (0, 0), 1, 'relu')
        x = fused_ops.global_avg_pool_cuda(x)
        x = torch.flatten(x, 1)
        x = F.linear(x, self.params["fc_w"], self.params["fc_b"])
        
        return x.float()

    def forward(self, x):
        if self.training:
            return self._forward_impl(x)

        if self.graph is None:
            self.static_input = x.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.static_output = self._forward_impl(self.static_input)
            torch.cuda.current_stream().wait_stream(s)
            
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self._forward_impl(self.static_input)
        
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.mbconv1 = self._make_mbconv_block(32, 16, 1, 1)
        self.mbconv2 = self._make_mbconv_block(16, 24, 2, 6)
        self.mbconv3 = self._make_mbconv_block(24, 40, 2, 6)
        self.mbconv4 = self._make_mbconv_block(40, 80, 2, 6)
        self.mbconv5 = self._make_mbconv_block(80, 112, 1, 6)
        self.mbconv6 = self._make_mbconv_block(112, 192, 2, 6)
        self.mbconv7 = self._make_mbconv_block(192, 320, 1, 6)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, num_classes)
    def _make_mbconv_block(self, in_channels, out_channels, stride, expand_ratio):
        hidden_dim = round(in_channels * expand_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        pass

batch_size = 10
input_shape = (3, 240, 240)
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [num_classes]