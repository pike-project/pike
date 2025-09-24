import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_rnn_cell_kernel(
    # Pointers to matrices
    X_ptr, H_ptr, Wx_ptr, Wh_ptr, B_ptr, NewH_ptr,
    # Matrix dimensions
    batch_size, input_size, hidden_size,
    # Strides
    stride_x_batch, stride_x_in,
    stride_h_batch, stride_h_hidden,
    stride_wx_hidden, stride_wx_in,
    stride_wh_hidden, stride_wh_hidden_in,
    stride_newh_batch, stride_newh_hidden,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K_IN: tl.constexpr, BLOCK_SIZE_K_HID: tl.constexpr,
):
    """
    Computes NewH = tanh(X @ Wx.T + H @ Wh.T + B) in a fused manner.
    X: Input tensor (batch_size, input_size)
    H: Hidden state (batch_size, hidden_size)
    Wx: Input-to-hidden weights (hidden_size, input_size)
    Wh: Hidden-to-hidden weights (hidden_size, hidden_size)
    B: Bias (hidden_size,)
    NewH: Output new hidden state (batch_size, hidden_size)
    """
    # Grid and program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(batch_size, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(hidden_size, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Create pointers for the first loop (X @ Wx.T)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k_in = tl.arange(0, BLOCK_SIZE_K_IN)
    
    x_ptrs = X_ptr + (offs_m[:, None] * stride_x_batch + offs_k_in[None, :] * stride_x_in)
    wx_ptrs = Wx_ptr + (offs_n[:, None] * stride_wx_hidden + offs_k_in[None, :] * stride_wx_in)

    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K for X @ Wx.T
    for k in range(0, tl.cdiv(input_size, BLOCK_SIZE_K_IN)):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < batch_size) & (offs_k_in[None, :] < input_size), other=0.0)
        wx = tl.load(wx_ptrs, mask=(offs_n[:, None] < hidden_size) & (offs_k_in[None, :] < input_size), other=0.0)
        accumulator += tl.dot(x, tl.trans(wx))
        x_ptrs += BLOCK_SIZE_K_IN * stride_x_in
        wx_ptrs += BLOCK_SIZE_K_IN * stride_wx_in
        
    # Create pointers for the second loop (H @ Wh.T)
    offs_k_hid = tl.arange(0, BLOCK_SIZE_K_HID)
    h_ptrs = H_ptr + (offs_m[:, None] * stride_h_batch + offs_k_hid[None, :] * stride_h_hidden)
    wh_ptrs = Wh_ptr + (offs_n[:, None] * stride_wh_hidden + offs_k_hid[None, :] * stride_wh_hidden_in)

    # Loop over K for H @ Wh.T, accumulating into the same accumulator
    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K_HID)):
        h = tl.load(h_ptrs, mask=(offs_m[:, None] < batch_size) & (offs_k_hid[None, :] < hidden_size), other=0.0)
        wh = tl.load(wh_ptrs, mask=(offs_n[:, None] < hidden_size) & (offs_k_hid[None, :] < hidden_size), other=0.0)
        accumulator += tl.dot(h, tl.trans(wh))
        h_ptrs += BLOCK_SIZE_K_HID * stride_h_hidden
        wh_ptrs += BLOCK_SIZE_K_HID * stride_wh_hidden_in

    # Add bias
    b_ptrs = B_ptr + offs_n
    bias = tl.load(b_ptrs, mask=offs_n < hidden_size, other=0.0)
    accumulator += bias[None, :]

    # Apply Tanh activation
    # The triton.language package does not have a 'tanh' function.
    # We implement it manually using a numerically stable formula:
    # tanh(x) = 2 * sigmoid(2*x) - 1 = 2 / (1 + exp(-2x)) - 1
    result = 2.0 / (1.0 + tl.exp(-2.0 * accumulator)) - 1.0

    # Write back result
    offs_newh_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_newh_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    newh_ptrs = NewH_ptr + stride_newh_batch * offs_newh_m[:, None] + stride_newh_hidden * offs_newh_n[None, :]
    mask = (offs_newh_m[:, None] < batch_size) & (offs_newh_n[None, :] < hidden_size)
    tl.store(newh_ptrs, result, mask=mask)

class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Use register_buffer for the hidden state so it's moved to the correct device
        self.register_buffer('hidden', torch.randn(batch_size, hidden_size))
        
        # Create a temporary standard layer to get initialized weights
        i2h = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Split weights for the fused kernel
        self.Wx = nn.Parameter(i2h.weight[:, :input_size].clone())
        self.Wh = nn.Parameter(i2h.weight[:, input_size:].clone())
        self.B = nn.Parameter(i2h.bias.clone())
        
        # The hidden-to-output layer remains the same
        self.h2o = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor, initial_hidden=None) -> torch.Tensor:
        if initial_hidden is not None:
            self.hidden.copy_(initial_hidden)
        
        self.hidden = self.hidden.to(x.device, x.dtype)
        
        batch_size_dyn, _ = x.shape
        
        # Allocate output tensor for the new hidden state
        new_hidden = torch.empty_like(self.hidden)
        
        # Define grid for launching the kernel
        grid = lambda META: (
            triton.cdiv(batch_size_dyn, META['BLOCK_SIZE_M']) * triton.cdiv(self.hidden_size, META['BLOCK_SIZE_N']),
        )
        
        # Launch the fused kernel
        fused_rnn_cell_kernel[grid](
            x, self.hidden, self.Wx, self.Wh, self.B, new_hidden,
            batch_size_dyn, self.input_size, self.hidden_size,
            x.stride(0), x.stride(1),
            self.hidden.stride(0), self.hidden.stride(1),
            self.Wx.stride(0), self.Wx.stride(1),
            self.Wh.stride(0), self.Wh.stride(1),
            new_hidden.stride(0), new_hidden.stride(1),
            # Tuning parameters for the kernel
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K_IN=64,
            BLOCK_SIZE_K_HID=64,
        )
        
        self.hidden = new_hidden
        output = self.h2o(self.hidden)
        return output

# --- Global variables and helper functions required by the evaluation script ---
batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256

def get_inputs():
    return [torch.randn(batch_size, input_size), torch.randn(batch_size, hidden_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]