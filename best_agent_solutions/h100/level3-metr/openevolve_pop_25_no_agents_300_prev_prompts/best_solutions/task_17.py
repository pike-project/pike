# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F

# The original high-performing model implementation.
# We will use torch.compile on this model to achieve SOTA performance by
# leveraging compiler-based operator fusion.
class _Model(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        """
        :param in_channels: Number of input channels
        :param squeeze_channels: Number of output channels for the squeeze layer
        :param expand1x1_channels: Number of output channels for the 1x1 expand layer
        :param expand3x3_channels: Number of output channels for the 3x3 expand layer
        """
        super(_Model, self).__init__()
        
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, expand1x1_channels + expand3x3_channels, height, width)
        """
        # This sequence of operations is ideal for kernel fusion by a compiler.
        x = self.squeeze_activation(self.squeeze(x))
        out1 = self.expand1x1_activation(self.expand1x1(x))
        out2 = self.expand3x3_activation(self.expand3x3(x))
        return torch.cat([out1, out2], 1)

# A wrapper class is used to apply torch.compile. The evaluation framework
# instantiates this 'Model' class, which in turn creates and compiles the
# underlying '_Model'. This pattern allows injecting compilation without
# changing the evaluation harness.
class Model(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Model, self).__init__()
        # Instantiate the model that contains the actual architecture.
        model_to_compile = _Model(in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels)
        
        # Compile the model using torch.compile. This will fuse the operations
        # (Conv2d, ReLU, cat) into highly efficient kernels using backends like Triton.
        # The 'max-autotune' mode enables extra features and heuristics to
        # find the fastest kernel configuration, which is ideal for maximizing
        # performance, at the cost of a longer initial compile time.
        self.compiled_model = torch.compile(model_to_compile, mode="max-autotune")

    def forward(self, x):
        # Execute the forward pass using the compiled model.
        return self.compiled_model(x)

# Test code
batch_size = 10
num_input_features = 3
num_output_features = 64
height, width = 224, 224
squeeze_channels = 6
expand1x1_channels = 64
expand3x3_channels = 64

def get_inputs():
    # The input tensor must be on a CUDA device for torch.compile to generate
    # and run optimized CUDA kernels.
    return [torch.randn(batch_size, num_input_features, height, width).cuda()]

def get_init_inputs():
    return [num_input_features, squeeze_channels, expand1x1_channels, expand3x3_channels]
# EVOLVE-BLOCK-END