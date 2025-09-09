# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import os

# Set a persistent cache directory to avoid recompilation on every run
py_file_path = os.path.abspath(__file__)
py_file_dir = os.path.dirname(py_file_path)
cache_dir = os.path.join(py_file_dir, "cuda_cache_hybrid_final_v2")
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_EXTENSIONS_DIR'] = cache_dir

# Define the custom CUDA kernels for fused operations.
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// --- Fused Element-wise Operations (Folded BN + Activation) using float4 ---

enum class ActivationType { NONE, RELU, RELU6 };

template <ActivationType act_type>
__global__ void fused_bn_act_vec4_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    const int total_elements,
    const int C,
    const int HW)
{
    // This kernel assumes total_elements is a multiple of 4.
    const int total_vectors = total_elements / 4;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < total_vectors) {
        const int base_idx = i * 4;
        // Assumes a float4 does not cross a channel boundary (HW is multiple of 4)
        const int c = (base_idx / HW) % C;

        const float s = scale[c];
        const float b = bias[c];

        // Load one float4 vector
        const float4 in_val = reinterpret_cast<const float4*>(input)[i];
        float4 out_val;

        // Apply folded BN
        out_val.x = in_val.x * s + b;
        out_val.y = in_val.y * s + b;
        out_val.z = in_val.z * s + b;
        out_val.w = in_val.w * s + b;

        // Apply activation
        if (act_type == ActivationType::RELU) {
            out_val.x = max(0.0f, out_val.x);
            out_val.y = max(0.0f, out_val.y);
            out_val.z = max(0.0f, out_val.z);
            out_val.w = max(0.0f, out_val.w);
        } else if (act_type == ActivationType::RELU6) {
            out_val.x = min(max(0.0f, out_val.x), 6.0f);
            out_val.y = min(max(0.0f, out_val.y), 6.0f);
            out_val.z = min(max(0.0f, out_val.z), 6.0f);
            out_val.w = min(max(0.0f, out_val.w), 6.0f);
        }

        // Store result
        reinterpret_cast<float4*>(output)[i] = out_val;
    }
}

// --- Fused Depthwise Convolution + Folded BN + ReLU6 using Shared Memory ---
__global__ void fused_depthwise_bn_relu6_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weights, // flattened weights
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int stride, int padding) {

    constexpr int TILE_DIM = 16;
    constexpr int KERNEL_SIZE = 3;
    // Max input tile dim needed is for stride=2: (16-1)*2+3 = 33
    constexpr int MAX_IN_TILE_DIM = (TILE_DIM - 1) * 2 + KERNEL_SIZE;

    __shared__ float s_input[MAX_IN_TILE_DIM][MAX_IN_TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int w_out = blockIdx.x * TILE_DIM + tx;
    const int h_out = blockIdx.y * TILE_DIM + ty;

    const int n = blockIdx.z / C;
    const int c = blockIdx.z % C;

    if (w_out >= W_out || h_out >= H_out) {
        return;
    }

    const int h_start = (blockIdx.y * TILE_DIM) * stride - padding;
    const int w_start = (blockIdx.x * TILE_DIM) * stride - padding;

    const int in_tile_h = (TILE_DIM - 1) * stride + KERNEL_SIZE;
    const int in_tile_w = (TILE_DIM - 1) * stride + KERNEL_SIZE;

    const int threads_per_block = TILE_DIM * TILE_DIM;
    const int items_to_load = in_tile_h * in_tile_w;
    const long input_base_idx = (long)n * C * H * W + (long)c * H * W;

    for (int i = ty * TILE_DIM + tx; i < items_to_load; i += threads_per_block) {
        const int h = i / in_tile_w;
        const int w = i % in_tile_w;
        const int h_in = h_start + h;
        const int w_in = w_start + w;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            s_input[h][w] = input[input_base_idx + (long)h_in * W + w_in];
        } else {
            s_input[h][w] = 0.0f;
        }
    }
    __syncthreads();

    float acc = 0.0f;
    const int h_in_tile_start = ty * stride;
    const int w_in_tile_start = tx * stride;
    const float* weights_c = weights + c * 9;

    #pragma unroll
    for (int kh = 0; kh < 3; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < 3; ++kw) {
            acc += s_input[h_in_tile_start + kh][w_in_tile_start + kw] * weights_c[kh * 3 + kw];
        }
    }

    float bn_val = acc * scale[c] + bias[c];
    const long output_idx = (long)n * C * H_out * W_out + (long)c * H_out * W_out + (long)h_out * W_out + w_out;
    output[output_idx] = fminf(fmaxf(0.0f, bn_val), 6.0f);
}

// --- Fused Adaptive Average Pool + Flatten ---
__global__ void avg_pool_flatten_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    const int nc = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_channels = N * C;

    if (nc >= total_channels) return;
    
    const int HW = H * W;
    const float* input_ptr = input + (long)nc * HW;
    
    float sum = 0.0f;
    for (int i = 0; i < HW; ++i) {
        sum += input_ptr[i];
    }
    
    output[nc] = sum / (float)HW;
}

// --- C++ Wrapper Functions ---

template <ActivationType act_type>
torch::Tensor launch_fused_bn_act_vec4(
    torch::Tensor input, torch::Tensor scale, torch::Tensor bias)
{
    const auto total_elements = input.numel();
    TORCH_CHECK(total_elements % 4 == 0, "Input elements must be divisible by 4 for vec4 kernel");
    
    const auto C = input.size(1);
    const auto HW = input.size(2) * input.size(3);
    auto output = torch::empty_like(input);

    const int block_size = 256;
    const int num_blocks = (total_elements / 4 + block_size - 1) / block_size;

    fused_bn_act_vec4_kernel<act_type><<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        scale.data_ptr<float>(), bias.data_ptr<float>(),
        total_elements, C, HW
    );
    return output;
}

torch::Tensor fused_bn_vec4(torch::Tensor i, torch::Tensor s, torch::Tensor b) {
    return launch_fused_bn_act_vec4<ActivationType::NONE>(i, s, b);
}
torch::Tensor fused_bn_relu_vec4(torch::Tensor i, torch::Tensor s, torch::Tensor b) {
    return launch_fused_bn_act_vec4<ActivationType::RELU>(i, s, b);
}
torch::Tensor fused_bn_relu6_vec4(torch::Tensor i, torch::Tensor s, torch::Tensor b) {
    return launch_fused_bn_act_vec4<ActivationType::RELU6>(i, s, b);
}

torch::Tensor fused_depthwise_bn_relu6(
    torch::Tensor input, torch::Tensor weights,
    torch::Tensor scale, torch::Tensor bias,
    int stride, int padding) {
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int H_out = (H + 2 * padding - 3) / stride + 1;
    const int W_out = (W + 2 * padding - 3) / stride + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    if (output.numel() == 0) return output;

    constexpr int TILE_DIM = 16;
    dim3 block_dim(TILE_DIM, TILE_DIM, 1);
    dim3 grid_dim(
        (W_out + TILE_DIM - 1) / TILE_DIM,
        (H_out + TILE_DIM - 1) / TILE_DIM,
        (long)N * C
    );

    fused_depthwise_bn_relu6_shared_kernel<<<grid_dim, block_dim>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        weights.data_ptr<float>(), scale.data_ptr<float>(), bias.data_ptr<float>(),
        N, C, H, W, H_out, W_out, stride, padding);
    return output;
}

torch::Tensor avg_pool_flatten(torch::Tensor input) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty({N, C}, input.options());
    const int total_channels = N * C;
    if (total_channels == 0) return output;

    const int block_size = 256;
    const int num_blocks = (total_channels + block_size - 1) / block_size;

    avg_pool_flatten_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H, W
    );
    return output;
}

"""

cpp_source = """
torch::Tensor fused_bn_vec4(torch::Tensor i, torch::Tensor s, torch::Tensor b);
torch::Tensor fused_bn_relu_vec4(torch::Tensor i, torch::Tensor s, torch::Tensor b);
torch::Tensor fused_bn_relu6_vec4(torch::Tensor i, torch::Tensor s, torch::Tensor b);
torch::Tensor fused_depthwise_bn_relu6(torch::Tensor input, torch::Tensor weights, torch::Tensor scale, torch::Tensor bias, int stride, int padding);
torch::Tensor avg_pool_flatten(torch::Tensor input);
"""

# JIT Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops_hybrid_v3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_bn_vec4", "fused_bn_relu_vec4", "fused_bn_relu6_vec4", "fused_depthwise_bn_relu6", "avg_pool_flatten"],
    verbose=False,
)

def _fold_bn(bn_module):
    """Folds BatchNorm parameters for inference."""
    bn_module.eval()
    scale = bn_module.weight / torch.sqrt(bn_module.running_var + bn_module.eps)
    bias = bn_module.bias - bn_module.running_mean * scale
    return scale.detach(), bias.detach()

class CudaMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = round(in_channels * expand_ratio)
        
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False)
        bn1 = nn.BatchNorm2d(hidden_dim)
        
        conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False)
        bn3 = nn.BatchNorm2d(out_channels)

        scale1, bias1 = _fold_bn(bn1)
        scale2, bias2 = _fold_bn(bn2)
        scale3, bias3 = _fold_bn(bn3)
        
        self.register_buffer('scale1', scale1)
        self.register_buffer('bias1', bias1)
        self.register_buffer('scale2', scale2)
        self.register_buffer('bias2', bias2)
        self.register_buffer('scale3', scale3)
        self.register_buffer('bias3', bias3)

        conv2_weight_flat = conv2.weight.detach().clone().view(hidden_dim, -1)
        self.register_buffer('conv2_weight_flat_buffer', conv2_weight_flat)
        self.stride = conv2.stride[0]
        self.padding = conv2.padding[0]
        
    def forward(self, x):
        x = self.conv1(x)
        x = fused_ops.fused_bn_relu6_vec4(x, self.scale1, self.bias1)
        
        x = fused_ops.fused_depthwise_bn_relu6(
            x, self.conv2_weight_flat_buffer, self.scale2, self.bias2, 
            self.stride, self.padding
        )
        
        x = self.conv3(x)
        x = fused_ops.fused_bn_vec4(x, self.scale3, self.bias3)
        return x

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(32)
        
        self.mbconv1 = CudaMBConvBlock(32, 16, 1, 1)
        self.mbconv2 = CudaMBConvBlock(16, 24, 2, 6)
        self.mbconv3 = CudaMBConvBlock(24, 40, 2, 6)
        self.mbconv4 = CudaMBConvBlock(40, 80, 2, 6)
        self.mbconv5 = CudaMBConvBlock(80, 112, 1, 6)
        self.mbconv6 = CudaMBConvBlock(112, 192, 2, 6)
        self.mbconv7 = CudaMBConvBlock(192, 320, 1, 6)
        
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        bn2 = nn.BatchNorm2d(1280)
        
        self.fc = nn.Linear(1280, num_classes)
        
        scale1, bias1 = _fold_bn(bn1)
        scale2, bias2 = _fold_bn(bn2)
        self.register_buffer('scale1', scale1)
        self.register_buffer('bias1', bias1)
        self.register_buffer('scale2', scale2)
        self.register_buffer('bias2', bias2)
        
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = fused_ops.fused_bn_relu_vec4(x, self.scale1, self.bias1)
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        x = self.conv2(x)
        x = fused_ops.fused_bn_relu_vec4(x, self.scale2, self.bias2)
        
        x = fused_ops.avg_pool_flatten(x)
        x = self.fc(x)
        
        return x

    def forward(self, x):
        if self.training:
            return self._forward_impl(x)

        if self.graph is None:
            self.static_input = x.clone()
            
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._forward_impl(self.static_input) 
            torch.cuda.current_stream().wait_stream(s)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self._forward_impl(self.static_input)
        
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

# Test code
batch_size = 10
input_shape = (3, 240, 240)
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END