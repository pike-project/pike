import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Original Model Components (required for model initialization) ---

class BasicBlock(nn.Module):
    """
    Standard BasicBlock from ResNet, used for initial model construction
    before applying inference-time optimizations.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# --- Fusion Logic ---

def fuse_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuses a convolutional and a batch normalization layer into a single,
    new convolutional layer. This is a standard optimization for inference.
    
    Args:
        conv (nn.Conv2d): The convolutional layer.
        bn (nn.BatchNorm2d): The batch normalization layer.
        
    Returns:
        nn.Conv2d: The fused convolutional layer.
    """
    assert not (conv.training or bn.training), "Fusion is only valid in evaluation mode."
    
    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,  # Bias is now required to hold the BN parameters
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)

    # Fuse weights
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    bn_std = torch.sqrt(bn.running_var + bn.eps)
    scale_factor = bn.weight / bn_std
    w_bn = torch.diag(scale_factor)
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view_as(conv.weight))
    
    # Fuse bias
    b_conv = torch.zeros(conv.out_channels, device=conv.weight.device) if conv.bias is None else conv.bias.clone()
    b_bn = bn.bias - bn.running_mean * scale_factor
    fused_conv.bias.copy_(b_conv * scale_factor + b_bn)
    
    return fused_conv


class FusedBasicBlock(nn.Module):
    """
    An optimized BasicBlock where Conv-BN layers are fused. The forward
    pass is expressed using standard PyTorch ops to allow torch.compile
    to perform optimal fusions (e.g., Conv -> Add -> ReLU).
    """
    expansion = 1
    def __init__(self, basic_block: BasicBlock):
        super().__init__()
        assert not basic_block.training, "basic_block must be in eval mode for fusion"
        
        self.fused_conv1 = fuse_conv_bn_eval(basic_block.conv1, basic_block.bn1)
        self.relu1 = nn.ReLU(inplace=False) # Non-inplace is safer for compilers
        self.fused_conv2 = fuse_conv_bn_eval(basic_block.conv2, basic_block.bn2)
        self.relu2 = nn.ReLU(inplace=False)

        if basic_block.downsample is not None:
            # The downsample block is a Sequential(Conv2d, BatchNorm2d)
            self.downsample = fuse_conv_bn_eval(basic_block.downsample[0], basic_block.downsample[1])
        else:
            self.downsample = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu1(self.fused_conv1(x))
        out = self.fused_conv2(out)
        
        # This sequence is a prime target for torch.compile's fusion capabilities.
        # It will fuse the addition and ReLU into the epilogue of the conv kernel.
        out = self.relu2(out + identity)
        return out


class ModelNew(nn.Module):
    """
    An optimized ResNet-style model for high-performance inference.
    This model applies Conv-BN fusion and then leverages torch.compile
    for comprehensive graph-level optimizations.
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        
        # Define model with standard, unfused blocks to allow loading pretrained weights.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Apply optimizations after the model is fully constructed.
        self._prepare_for_inference()

    def _prepare_for_inference(self):
        """Applies a sequence of optimizations for fast inference."""
        self.to(torch.device("cuda"))
        self.eval()

        # Statically fuse modules before compilation
        self._fuse_model()
        
        # Convert to half-precision and channels-last memory format for Tensor Core usage
        self.to(dtype=torch.float16, memory_format=torch.channels_last)
        
        # Compile the core forward pass logic for maximum performance.
        # TorchInductor (the backend) will generate optimized Triton kernels.
        self._forward_impl = torch.compile(self._forward_impl, mode="max-autotune", fullgraph=True)

    def _fuse_model(self):
        """Replaces standard modules with their fused/optimized counterparts."""
        with torch.no_grad():
            self.conv1 = fuse_conv_bn_eval(self.conv1, self.bn1)
            self.bn1 = nn.Identity() # bn1 is now fused into conv1

            for name, module in self.named_children():
                if isinstance(module, nn.Sequential) and len(module) > 0 and isinstance(module[0], BasicBlock):
                    fused_layers = [FusedBasicBlock(block) for block in module]
                    setattr(self, name, nn.Sequential(*fused_layers))

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """Helper to build layers of BasicBlocks."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """The core forward logic that gets compiled."""
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Public forward method that handles input preparation and calls the
        compiled implementation.
        """
        # Ensure input is on the correct device and in the optimal format/dtype
        x_opt = x.cuda(non_blocking=True).to(memory_format=torch.channels_last, dtype=torch.float16)
        output = self._forward_impl(x_opt)
        # Cast output back to the original float32 dtype for consistency
        return output.to(torch.float32)

# --- Test and Input Generation Functions ---
batch_size = 2
num_classes = 1000
input_shape = (batch_size, 3, 224, 224)

def get_inputs():
    # Return a tensor on CPU; the model's forward pass will move it to CUDA
    return [torch.randn(input_shape)]

def get_init_inputs():
    return [num_classes]