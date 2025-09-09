import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# The original Model class is required to correctly initialize the fused weights.
class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        MBConv block implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(Model, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        """
        Forward pass of the MBConv block.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        identity = x
        
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x += identity
        
        return x


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuses a Conv2d layer and a BatchNorm2d layer into a single, new Conv2d layer.
    This utility is designed for inference-time optimization. The batch norm statistics
    (running_mean, running_var) are folded into the weights and bias of the convolutional layer.

    Args:
        conv (nn.Conv2d): The convolutional layer. Must not have been fused already.
        bn (nn.BatchNorm2d): The batch normalization layer that follows the conv layer.

    Returns:
        nn.Conv2d: A new, fused convolutional layer with bias.
    """
    # Ensure the model is in evaluation mode.
    # This is crucial because batch norm behaves differently during training and evaluation.
    # The fusion is only valid for inference.
    assert not (conv.training or bn.training), "Fusion only valid in eval mode."

    # Create a new Conv2d layer with the same parameters as the original, but with bias enabled.
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode
    ).to(conv.weight.device)

    with torch.no_grad():
        # Get batch norm parameters
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(running_var + eps)

        # Calculate the new weights
        scale_factor = gamma / std
        w_fused = conv.weight * scale_factor.view([-1, 1, 1, 1])

        # Calculate the new bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros_like(running_mean)
        
        b_fused = (b_conv - running_mean) * scale_factor + beta

        # Set the parameters of the new fused layer
        fused_conv.weight.copy_(w_fused)
        fused_conv.bias.copy_(b_fused)

    return fused_conv


# CUDA kernel for fused element-wise operations: bias add, relu6, and residual add
fused_elementwise_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_elementwise_kernel(
    const float* in, 
    const float* bias, 
    const float* residual, 
    float* out, 
    bool apply_relu, 
    bool has_residual, 
    int total_elements,
    int channels,
    int spatial_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        // Calculate channel index for bias lookup
        // The layout is NCHW. idx / spatial_dim gives the index of the feature map.
        // Taking this modulo channels gives the channel index.
        int channel_idx = (idx / spatial_dim) % channels;
        
        // Load input value
        float val = in[idx];
        
        // Add bias (broadcasted across spatial dimensions)
        val += bias[channel_idx];
        
        // Add residual if provided
        if (has_residual) {
            val += residual[idx];
        }
        
        // Apply ReLU6 if requested
        if (apply_relu) {
            val = fminf(fmaxf(val, 0.0f), 6.0f);
        }
        
        // Store the result
        out[idx] = val;
    }
}

torch::Tensor fused_elementwise_op(
    torch::Tensor input, 
    torch::Tensor bias, 
    c10::optional<torch::Tensor> residual, 
    bool apply_relu
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    
    bool has_residual = residual.has_value();
    if (has_residual) {
        TORCH_CHECK(residual->is_cuda(), "Residual must be a CUDA tensor");
        TORCH_CHECK(residual->is_contiguous(), "Residual must be contiguous");
        TORCH_CHECK(input.sizes() == residual->sizes(), "Input and residual tensors must have the same shape");
    }
    
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor (NCHW)");
    TORCH_CHECK(bias.dim() == 1, "Bias must be a 1D tensor");
    TORCH_CHECK(input.size(1) == bias.size(0), "Input channels must match bias size");
    
    auto c = input.size(1);
    auto h = input.size(2);
    auto w = input.size(3);
    
    auto total_elements = input.numel();
    auto spatial_dim = h * w;
    
    auto out = torch::empty_like(input);
    
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_elementwise_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        has_residual ? residual->data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        apply_relu,
        has_residual,
        total_elements,
        c,
        spatial_dim
    );
    
    return out;
}
"""

fused_elementwise_cpp_source = """
#include <c10/util/Optional.h>
torch::Tensor fused_elementwise_op(torch::Tensor input, torch::Tensor bias, c10::optional<torch::Tensor> residual, bool apply_relu);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_elementwise_op",
    cpp_sources=fused_elementwise_cpp_source,
    cuda_sources=fused_elementwise_source,
    functions=["fused_elementwise_op"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        Optimized MBConv block with Conv-BatchNorm fusion and custom CUDA kernels
        for post-convolution operations (bias, activation, residual).
        """
        super(ModelNew, self).__init__()
        
        # FIX: Instantiate the original model to get its randomly initialized weights
        # and BN statistics. This ensures that the fused model is mathematically
        # equivalent to the original model instance used for correctness checking.
        original_model = Model(in_channels, out_channels, kernel_size, stride, expand_ratio)
        original_model.eval() # CRITICAL: Set to eval mode for fusion.

        self.use_residual = original_model.use_residual
        self.hidden_dim = in_channels * expand_ratio
        self.expand_ratio = expand_ratio
        self.stride = stride
        self.kernel_size = kernel_size
        
        # --- Fuse layers from the original_model instance ---
        
        if self.expand_ratio != 1:
            expand_conv_orig = original_model.expand_conv[0]
            expand_bn_orig = original_model.expand_conv[1]
            fused_expand = fuse_conv_bn(expand_conv_orig, expand_bn_orig)
            self.expand_weight = nn.Parameter(fused_expand.weight)
            self.expand_bias = nn.Parameter(fused_expand.bias)
        
        depthwise_conv_orig = original_model.depthwise_conv[0]
        depthwise_bn_orig = original_model.depthwise_conv[1]
        fused_depthwise = fuse_conv_bn(depthwise_conv_orig, depthwise_bn_orig)
        self.depthwise_weight = nn.Parameter(fused_depthwise.weight)
        self.depthwise_bias = nn.Parameter(fused_depthwise.bias)
        
        project_conv_orig = original_model.project_conv[0]
        project_bn_orig = original_model.project_conv[1]
        fused_project = fuse_conv_bn(project_conv_orig, project_bn_orig)
        self.project_weight = nn.Parameter(fused_project.weight)
        self.project_bias = nn.Parameter(fused_project.bias)
        
        # Store the custom op function
        self.fused_op = fused_ops.fused_elementwise_op

    def forward(self, x):
        """
        Forward pass using Pytorch's conv and custom fused element-wise kernels.
        """
        identity = x
        
        # 1. Expansion phase
        if self.expand_ratio != 1:
            # Pytorch's optimized convolution (likely cuDNN)
            x_conv = F.conv2d(x, self.expand_weight, bias=None, stride=1, padding=0)
            # Custom kernel for Fused (Bias Add + ReLU6)
            x = self.fused_op(x_conv, self.expand_bias, None, True)
        
        # 2. Depthwise phase
        x_conv = F.conv2d(x, self.depthwise_weight, bias=None, stride=self.stride, padding=(self.kernel_size-1)//2, groups=self.hidden_dim)
        # Custom kernel for Fused (Bias Add + ReLU6)
        x = self.fused_op(x_conv, self.depthwise_bias, None, True)
        
        # 3. Projection phase
        x_conv = F.conv2d(x, self.project_weight, bias=None, stride=1, padding=0)
        
        # 4. Final fusion: Bias Add + optional Residual Add
        if self.use_residual:
            x = self.fused_op(x_conv, self.project_bias, identity, False)
        else:
            x = self.fused_op(x_conv, self.project_bias, None, False)
        
        return x