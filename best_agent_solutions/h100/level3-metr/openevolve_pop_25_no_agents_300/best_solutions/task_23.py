# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define and compile the custom CUDA kernel for Fused (BatchNorm -> Activation)
# This version combines the best features of previous attempts:
# 1. Pre-computation of BatchNorm scale/bias for inference (from the top performer).
# 2. Vectorized memory access using float4 for higher bandwidth (inspired by other attempts).
# 3. C++ templates to eliminate runtime branching for different activations.
# 4. Correct per-element channel indexing for float4, making it robust.
fused_bn_act_vec4_inference_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h> // Header for C10_CUDA_KERNEL_LAUNCH_CHECK

template <bool is_relu6>
__global__ void fused_bn_act_inference_vec4_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ y,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    const int C,
    const int HW,
    const int total_float4_elements)
{
    // Grid-stride loop for robustly handling any number of elements.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_float4_elements; i += blockDim.x * gridDim.x) {
        const int base_idx = i * 4;
        
        const float4 val_in = x[i];
        float4 val_out;

        // Correctly calculate channel index for each element in the float4 vector
        // This is necessary as a float4 read can cross channel boundaries.
        const int c0 = (base_idx     / HW) % C;
        const int c1 = ((base_idx + 1) / HW) % C;
        const int c2 = ((base_idx + 2) / HW) % C;
        const int c3 = ((base_idx + 3) / HW) % C;
        
        // Fused calculation: y = x * scale + bias for each element
        const float val0 = val_in.x * scale[c0] + bias[c0];
        const float val1 = val_in.y * scale[c1] + bias[c1];
        const float val2 = val_in.z * scale[c2] + bias[c2];
        const float val3 = val_in.w * scale[c3] + bias[c3];

        // Compile-time selection of activation function
        if constexpr (is_relu6) {
            val_out.x = fminf(fmaxf(0.0f, val0), 6.0f);
            val_out.y = fminf(fmaxf(0.0f, val1), 6.0f);
            val_out.z = fminf(fmaxf(0.0f, val2), 6.0f);
            val_out.w = fminf(fmaxf(0.0f, val3), 6.0f);
        } else { // ReLU
            val_out.x = fmaxf(0.0f, val0);
            val_out.y = fmaxf(0.0f, val1);
            val_out.z = fmaxf(0.0f, val2);
            val_out.w = fmaxf(0.0f, val3);
        }
        y[i] = val_out;
    }
}

torch::Tensor fused_bn_act_inference_cuda(
    torch::Tensor x,
    torch::Tensor scale,
    torch::Tensor bias,
    int activation_type) // 0 for ReLU, 1 for ReLU6
{
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be a 4D tensor");
    TORCH_CHECK(x.numel() % 4 == 0, "Input tensor size must be divisible by 4 for vectorization");
    
    x = x.contiguous();
    scale = scale.contiguous();
    bias = bias.contiguous();

    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);
    const auto HW = H * W;

    auto y = torch::empty_like(x);

    const int total_elements = x.numel();
    if (total_elements == 0) {
        return y;
    }
    const int total_float4_elements = total_elements / 4;
    
    // Using a larger block size can sometimes be beneficial for vectorized kernels
    const int block_size = 512;
    const int num_blocks = std::min((total_float4_elements + block_size - 1) / block_size, 65535);

    if (activation_type == 1) { // ReLU6
        fused_bn_act_inference_vec4_kernel<true><<<num_blocks, block_size>>>(
            (const float4*)x.data_ptr<float>(), (float4*)y.data_ptr<float>(),
            scale.data_ptr<float>(), bias.data_ptr<float>(),
            C, HW, total_float4_elements
        );
    } else { // ReLU
        fused_bn_act_inference_vec4_kernel<false><<<num_blocks, block_size>>>(
            (const float4*)x.data_ptr<float>(), (float4*)y.data_ptr<float>(),
            scale.data_ptr<float>(), bias.data_ptr<float>(),
            C, HW, total_float4_elements
        );
    }
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

fused_bn_act_inference_cpp_source = """
torch::Tensor fused_bn_act_inference_cuda(
    torch::Tensor x, torch::Tensor scale, torch::Tensor bias,
    int activation_type);
"""

# Give the library a unique name to prevent JIT compilation conflicts
fused_bn_act_lib = load_inline(
    name="fused_bn_act_lib_vec4_inference_v2",
    cpp_sources=fused_bn_act_inference_cpp_source,
    cuda_sources=fused_bn_act_vec4_inference_source,
    functions=["fused_bn_act_inference_cuda"],
    verbose=False,
)

class MBConvBlock(nn.Module):
    """
    MBConv block with fused BatchNorm-Activation for inference.
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = round(in_channels * expand_ratio)
        
        # Define layers
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        bn1 = nn.BatchNorm2d(hidden_dim)
        
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        bn2 = nn.BatchNorm2d(hidden_dim)
        
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels) # No activation at the end

        # Pre-compute and register buffers for fused operations
        # y = gamma * (x - mean) / sqrt(var + eps) + beta 
        #   = x * [gamma / sqrt(var + eps)] + [beta - mean * gamma / sqrt(var + eps)]
        #   = x * scale + new_bias
        
        # Fuse first BN-ReLU6
        scale1 = bn1.weight.data / torch.sqrt(bn1.running_var + bn1.eps)
        bias1 = bn1.bias.data - bn1.running_mean.data * scale1
        self.register_buffer('scale1', scale1)
        self.register_buffer('bias1', bias1)
        
        # Fuse second BN-ReLU6
        scale2 = bn2.weight.data / torch.sqrt(bn2.running_var + bn2.eps)
        bias2 = bn2.bias.data - bn2.running_mean.data * scale2
        self.register_buffer('scale2', scale2)
        self.register_buffer('bias2', bias2)

    def forward(self, x):
        x = self.expand_conv(x)
        x = fused_bn_act_lib.fused_bn_act_inference_cuda(x, self.scale1, self.bias1, 1) # ReLU6
        
        x = self.depthwise_conv(x)
        x = fused_bn_act_lib.fused_bn_act_inference_cuda(x, self.scale2, self.bias2, 1) # ReLU6
        
        x = self.project_conv(x)
        x = self.bn3(x)
        return x

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB1 architecture with custom fused & vectorized CUDA kernels for inference.
        """
        super(Model, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.mbconv1 = MBConvBlock(32, 16, 1, 1)
        self.mbconv2 = MBConvBlock(16, 24, 2, 6)
        self.mbconv3 = MBConvBlock(24, 40, 2, 6)
        self.mbconv4 = MBConvBlock(40, 80, 2, 6)
        self.mbconv5 = MBConvBlock(80, 112, 1, 6)
        self.mbconv6 = MBConvBlock(112, 192, 2, 6)
        self.mbconv7 = MBConvBlock(192, 320, 1, 6)
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
        
        # Pre-compute scale and bias for the standalone BN layers
        # Fuse initial BN-ReLU
        scale_initial = bn1.weight.data / torch.sqrt(bn1.running_var + bn1.eps)
        bias_initial = bn1.bias.data - bn1.running_mean.data * scale_initial
        self.register_buffer('scale_initial', scale_initial)
        self.register_buffer('bias_initial', bias_initial)

        # Fuse final BN-ReLU
        scale_final = bn2.weight.data / torch.sqrt(bn2.running_var + bn2.eps)
        bias_final = bn2.bias.data - bn2.running_mean.data * scale_final
        self.register_buffer('scale_final', scale_final)
        self.register_buffer('bias_final', bias_final)

    def forward(self, x):
        """
        Forward pass of the EfficientNetB1 model.
        """
        x = self.conv1(x)
        x = fused_bn_act_lib.fused_bn_act_inference_cuda(x, self.scale_initial, self.bias_initial, 0) # ReLU
        
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)
        
        x = self.conv2(x)
        x = fused_bn_act_lib.fused_bn_act_inference_cuda(x, self.scale_final, self.bias_final, 0) # ReLU
        
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Test code
batch_size = 10
input_shape = (3, 240, 240)
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, *input_shape).cuda()]

def get_init_inputs():
    return [num_classes]
# EVOLVE-BLOCK-END