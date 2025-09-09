# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        This version of the model abandons the manual CUDA streams and custom kernel
        approach in favor of `torch.compile` to optimize the entire Inception module's 
        forward pass as a single computational graph. The previous approach with streams
        and a custom `cat` kernel was slower than `torch.compile` because it still required
        writing four intermediate tensors to global memory before the custom kernel could
        read them back for concatenation.

        By compiling the entire graph, `torch.compile`'s backend (Inductor) can perform
        more advanced, holistic optimizations:

        1.  **Vertical Fusion**: It can fuse sequential operations within each branch 
            (e.g., Conv -> Conv or MaxPool -> Conv) into a single, more efficient kernel. 
            This reduces kernel launch overhead and memory access, as intermediate results
            are kept in registers or shared memory.

        2.  **Horizontal Fusion**: This is the most critical optimization for this architecture.
            The compiler can analyze the parallel branches and the final `torch.cat` operation
            and fuse them. This allows the kernels for each branch to write their results 
            *directly* into the correct slices of the final, pre-allocated output tensor. This
            completely eliminates the memory traffic associated with writing and reading the
            four intermediate output tensors, providing a significant performance boost over
            the manual stream-based approach.

        3.  **Optimal Scheduling**: The compiler analyzes the data dependencies and resource 
            requirements of the four parallel branches and can schedule their execution more 
            efficiently than manual CUDA streams, which may not achieve perfect overlap and
            incur synchronization overhead.

        The `mode="reduce-overhead"` is chosen as it's particularly effective for models 
        with many small operations, like this Inception module. `fullgraph=True` ensures 
        the entire method is compiled as a single unit without graph breaks, maximizing 
        the potential for these advanced fusions.
        """
        super(Model, self).__init__()
        
        # Branch 1: 1x1 convolution
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # Branch 2: 1x1 followed by 3x3 convolution
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # Branch 3: 1x1 followed by 5x5 convolution
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Branch 4: Max pooling followed by 1x1 convolution
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

        # Overwrite the forward method with its compiled version. This is the core optimization.
        self.forward = torch.compile(self._forward, mode="reduce-overhead", fullgraph=True)
    
    def _forward(self, x):
        """
        The core forward pass logic that will be compiled by `torch.compile`.
        The compiler will optimize the execution of these parallel branches and
        the final concatenation.
        """
        branch1x1_out = self.branch1x1(x)
        branch3x3_out = self.branch3x3(x)
        branch5x5_out = self.branch5x5(x)
        branch_pool_out = self.branch_pool(x)
        
        # This list creation and concatenation will be horizontally fused by the compiler.
        outputs = [branch1x1_out, branch3x3_out, branch5x5_out, branch_pool_out]
        return torch.cat(outputs, 1)

# Test code
in_channels = 480
out_1x1 = 192
reduce_3x3 = 96
out_3x3 = 208
reduce_5x5 = 16
out_5x5 = 48
pool_proj = 64
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]
# EVOLVE-BLOCK-END