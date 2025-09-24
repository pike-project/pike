import torch
import torch.nn as nn
import torch.nn.functional as F

class _OriginalModel(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Original MobileNetV2 architecture implementation in PyTorch.
        This class is used as a reference to build the fused model.
        """
        super(_OriginalModel, self).__init__()
        
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _inverted_residual_block(inp, oup, stride, expand_ratio):
            hidden_dim = int(inp * expand_ratio)
            use_res_connect = stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # Pointwise convolution
                layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.ReLU6(inplace=True))

            layers.extend([
                # Depthwise convolution
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # Pointwise linear convolution
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])

            # Note: The original implementation in the prompt does not use the residual connection.
            # We will replicate that behavior.
            return nn.Sequential(*layers), use_res_connect

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU6(inplace=True)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                block, _ = _inverted_residual_block(input_channel, output_channel, stride, expand_ratio=t)
                features.append(block)
                input_channel = output_channel

        features.append(nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(last_channel))
        features.append(nn.ReLU6(inplace=True))
        features.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        
        # Instantiate the original model to get its architecture
        ref_model = _OriginalModel(num_classes=num_classes)
        # Set to eval mode to use running mean/var for fusion
        ref_model.eval()

        # Recursively fuse all Conv-BN modules in the model
        self.features = self._fuse_modules(ref_model.features)
        
        # The classifier section is unchanged
        self.classifier = ref_model.classifier

    def _fuse_conv_bn(self, conv, bn):
        """
        Fuses a Conv2d and BatchNorm2d layer into a single Conv2d layer.
        """
        # Detach parameters to avoid modifying the original model's graph
        w_conv = conv.weight.detach().clone()
        running_mean = bn.running_mean.detach().clone()
        running_var = bn.running_var.detach().clone()
        gamma = bn.weight.detach().clone()
        beta = bn.bias.detach().clone()
        eps = bn.eps

        # Mathematical fusion of parameters
        std = torch.sqrt(running_var + eps)
        scale_factor = gamma / std
        
        # Fuse weights and bias
        w_fused = w_conv * scale_factor.view([-1, 1, 1, 1])
        b_fused = beta - running_mean * scale_factor
        
        # Create a new Conv2d layer with the fused parameters
        fused_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True  # The fused layer will have a bias
        )
        fused_conv.weight = nn.Parameter(w_fused)
        fused_conv.bias = nn.Parameter(b_fused)
        
        return fused_conv

    def _fuse_modules(self, module):
        """
        Recursively traverses a module and fuses Conv-BN sequences.
        """
        new_children = []
        children_list = list(module.children())
        
        i = 0
        while i < len(children_list):
            current_layer = children_list[i]

            # Primary fusion pattern: Conv2d -> BatchNorm2d -> [ReLU6]
            if isinstance(current_layer, nn.Conv2d) and \
               i + 1 < len(children_list) and \
               isinstance(children_list[i+1], nn.BatchNorm2d):
                
                conv = current_layer
                bn = children_list[i+1]
                fused_conv = self._fuse_conv_bn(conv, bn)
                
                # Check for an optional activation layer (ReLU6) to keep
                if i + 2 < len(children_list) and isinstance(children_list[i+2], nn.ReLU6):
                    # Combine fused_conv and ReLU6 into a new Sequential block
                    new_children.append(nn.Sequential(fused_conv, children_list[i+2]))
                    i += 3  # Advance index past Conv, BN, and ReLU
                else:
                    # Only Conv and BN were found
                    new_children.append(fused_conv)
                    i += 2  # Advance index past Conv and BN
            
            # Recursive step for nested nn.Sequential modules
            elif isinstance(current_layer, nn.Sequential):
                fused_block = self._fuse_modules(current_layer)
                new_children.append(fused_block)
                i += 1
            
            # Layer is not part of a fusion pattern, keep it as is
            else:
                new_children.append(current_layer)
                i += 1
                
        return nn.Sequential(*new_children)

    def forward(self, x):
        """
        Forward pass of the optimized MobileNetV2 model.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]