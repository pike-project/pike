# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        :param input_size: The number of expected features in the input x
        :param hidden_size: The number of features in the hidden state h
        :param num_layers: Number of recurrent layers (default: 1)
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh (default: True)
        :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
        """
        super(Model, self).__init__()
        # The native torch.nn.GRU is backed by cuDNN, which is extremely fast.
        # The primary remaining optimization opportunity is to reduce the CPU overhead
        # of launching the cuDNN kernels for each forward pass.
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=0,
            bidirectional=True
        )
        
        # Attributes for CUDA graph. This will store the captured computation graph.
        self.graph = None
        # Static tensors to hold input/output data for the graph. The graph captures
        # operations on these specific tensors (by memory address).
        self.static_input = None
        self.static_h0 = None
        self.static_output = None

    def forward(self, x, h0):
        # The benchmark runs the forward pass repeatedly on inputs of the same shape.
        # This is the ideal scenario for CUDA Graphs, which eliminate kernel launch overhead.
        if self.graph is None:
            # --- First Run: Warmup and Graph Capture ---
            
            # 1. Create static tensors on the same device and with the same dtype as the input.
            #    These will be the tensors whose memory addresses are recorded by the graph.
            self.static_input = torch.empty_like(x)
            self.static_h0 = torch.empty_like(h0)
            
            # 2. Copy the first batch of data into the static tensors.
            self.static_input.copy_(x)
            self.static_h0.copy_(h0)

            # 3. Warmup: Run the operations once before capturing. This ensures that any
            #    one-time setup costs for the kernels are paid, and the graph captures
            #    the steady-state execution.
            # Use a separate stream to isolate warmup from the main stream.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.gru(self.static_input, self.static_h0)
            torch.cuda.current_stream().wait_stream(s)

            # 4. Capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # The operations here are recorded into the graph. They operate on our
                # static tensors. The output tensor is also captured.
                self.static_output, _ = self.gru(self.static_input, self.static_h0)

        # --- Subsequent Runs: Graph Replay ---
        
        # For every run (including the first, after capture is complete),
        # we copy the new data into the static input tensors.
        # non_blocking=True allows the CPU to continue working without waiting for the copy to finish.
        self.static_input.copy_(x, non_blocking=True)
        self.static_h0.copy_(h0, non_blocking=True)
        
        # Replay the captured graph. This executes the recorded sequence of kernels
        # on the GPU with minimal CPU overhead. The results are written into self.static_output.
        self.graph.replay()
        
        # Return a clone of the output. Cloning is important to avoid downstream
        # code from modifying the graph's static output buffer.
        return self.static_output.clone()

# Test code
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    # Placing tensors on the CUDA device is crucial for GPU execution.
    x = torch.randn(seq_len, batch_size, input_size).cuda()
    h0 = torch.randn((num_layers * 2, batch_size, hidden_size)).cuda()
    return [x, h0]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]
# EVOLVE-BLOCK-END