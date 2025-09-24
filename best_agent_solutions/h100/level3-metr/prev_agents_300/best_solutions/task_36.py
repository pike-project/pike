import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def tanh(x):
    """
    Numerically stable implementation of tanh using Triton primitives.
    The identity tanh(x) = 2 * sigmoid(2x) - 1 is used.
    """
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_K_HID': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_H': 128, 'BLOCK_SIZE_K_HID': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_H': 64, 'BLOCK_SIZE_K_HID': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_H': 128, 'BLOCK_SIZE_K_HID': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_K_HID': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_H': 256, 'BLOCK_SIZE_K_HID': 64}, num_warps=8, num_stages=3),
    ],
    key=['D_hid'],
)
@triton.jit
def lstm_step_recurrent_only_kernel(
    # Pointers to tensors
    x_proj_t_ptr, h_prev_ptr, c_prev_ptr,
    w_hh_ptr, b_ih_ptr, b_hh_ptr,
    h_next_ptr, c_next_ptr,
    # Dimensions
    B, D_hid,
    # Strides
    sxp_b, sxp_d,
    sh_b, sh_d,
    sc_b, sc_d,
    swh_n, swh_k,
    sbi, sbh,
    shn_b, shn_d,
    scn_b, scn_d,
    # Meta-parameters
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_K_HID: tl.constexpr,
):
    """
    This kernel computes one recurrent step of an LSTM cell.
    It assumes the input projection (X @ W_ih.T) has been pre-computed.
    """
    # Parallelize over batch and hidden dimension
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    # This program's block of hidden units
    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    h_mask = offs_h < D_hid

    # Load pre-computed projection from x for the current batch item and hidden block
    x_proj_ptr_b = x_proj_t_ptr + pid_b * sxp_b
    x_proj_i = tl.load(x_proj_ptr_b + (offs_h) * sxp_d, mask=h_mask, other=0.0)
    x_proj_f = tl.load(x_proj_ptr_b + (offs_h + D_hid) * sxp_d, mask=h_mask, other=0.0)
    x_proj_g = tl.load(x_proj_ptr_b + (offs_h + 2 * D_hid) * sxp_d, mask=h_mask, other=0.0)
    x_proj_o = tl.load(x_proj_ptr_b + (offs_h + 3 * D_hid) * sxp_d, mask=h_mask, other=0.0)

    # --- Gate Computation Part 1: h_prev @ W_hh.T ---
    # Accumulators for the 4 gates for our block of hidden units
    acc_i = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    acc_f = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    acc_g = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    
    # Pointer to the current batch item's previous hidden state
    h_prev_b_ptr = h_prev_ptr + pid_b * sh_b
    for k_start in range(0, D_hid, BLOCK_SIZE_K_HID):
        offs_k_hid = k_start + tl.arange(0, BLOCK_SIZE_K_HID)
        k_hid_mask = offs_k_hid < D_hid

        # Load a block of h_prev
        h_k = tl.load(h_prev_b_ptr + offs_k_hid * sh_d, mask=k_hid_mask, other=0.0)

        # Load blocks of weights for all 4 gates
        w_hh_i_ptr = w_hh_ptr + (offs_h[:, None] * swh_n + offs_k_hid[None, :] * swh_k)
        w_hh_f_ptr = w_hh_ptr + ((offs_h[:, None] + D_hid) * swh_n + offs_k_hid[None, :] * swh_k)
        w_hh_g_ptr = w_hh_ptr + ((offs_h[:, None] + 2 * D_hid) * swh_n + offs_k_hid[None, :] * swh_k)
        w_hh_o_ptr = w_hh_ptr + ((offs_h[:, None] + 3 * D_hid) * swh_n + offs_k_hid[None, :] * swh_k)
        
        w_mask = h_mask[:, None] & k_hid_mask[None, :]
        
        w_i_k = tl.load(w_hh_i_ptr, mask=w_mask, other=0.0)
        w_f_k = tl.load(w_hh_f_ptr, mask=w_mask, other=0.0)
        w_g_k = tl.load(w_hh_g_ptr, mask=w_mask, other=0.0)
        w_o_k = tl.load(w_hh_o_ptr, mask=w_mask, other=0.0)

        # Accumulate matrix-vector products
        acc_i += tl.sum(w_i_k * h_k[None, :], axis=1)
        acc_f += tl.sum(w_f_k * h_k[None, :], axis=1)
        acc_g += tl.sum(w_g_k * h_k[None, :], axis=1)
        acc_o += tl.sum(w_o_k * h_k[None, :], axis=1)
    
    # --- Gate Computation Part 2: Add biases and x_proj ---
    b_ih_i = tl.load(b_ih_ptr + offs_h * sbi, mask=h_mask, other=0.0)
    b_ih_f = tl.load(b_ih_ptr + (offs_h + D_hid) * sbi, mask=h_mask, other=0.0)
    b_ih_g = tl.load(b_ih_ptr + (offs_h + 2 * D_hid) * sbi, mask=h_mask, other=0.0)
    b_ih_o = tl.load(b_ih_ptr + (offs_h + 3 * D_hid) * sbi, mask=h_mask, other=0.0)
    
    b_hh_i = tl.load(b_hh_ptr + offs_h * sbh, mask=h_mask, other=0.0)
    b_hh_f = tl.load(b_hh_ptr + (offs_h + D_hid) * sbh, mask=h_mask, other=0.0)
    b_hh_g = tl.load(b_hh_ptr + (offs_h + 2 * D_hid) * sbh, mask=h_mask, other=0.0)
    b_hh_o = tl.load(b_hh_ptr + (offs_h + 3 * D_hid) * sbh, mask=h_mask, other=0.0)

    gate_i = acc_i + x_proj_i + b_ih_i + b_hh_i
    gate_f = acc_f + x_proj_f + b_ih_f + b_hh_f
    gate_g = acc_g + x_proj_g + b_ih_g + b_hh_g
    gate_o = acc_o + x_proj_o + b_ih_o + b_hh_o

    # --- Element-wise part: update c and h ---
    c_prev = tl.load(c_prev_ptr + pid_b * sc_b + offs_h * sc_d, mask=h_mask, other=0.0)
    
    ingate = tl.sigmoid(gate_i)
    forgetgate = tl.sigmoid(gate_f)
    cellgate = tanh(gate_g)
    outgate = tl.sigmoid(gate_o)
    
    c_next = forgetgate * c_prev + ingate * cellgate
    h_next = outgate * tanh(c_next)
    
    # --- Store results ---
    tl.store(h_next_ptr + pid_b * shn_b + offs_h * shn_d, h_next, mask=h_mask)
    tl.store(c_next_ptr + pid_b * scn_b + offs_h * scn_d, c_next, mask=h_mask)

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Model, self).__init__()
        
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # For multi-layer LSTMs, we use the standard PyTorch implementation for the first N-1 layers
        # and our custom kernel for the last layer.
        if num_layers > 1:
            self.first_layers = nn.LSTM(input_size, hidden_size, num_layers - 1, batch_first=True, dropout=dropout)
            last_layer_input_size = hidden_size
        else:
            self.first_layers = None
            last_layer_input_size = input_size
            
        # Create a dummy LSTM layer to easily initialize weights in a compatible format
        dummy_lstm = nn.LSTM(last_layer_input_size, hidden_size, 1, batch_first=True)
        self.weight_ih_l_last = nn.Parameter(dummy_lstm.weight_ih_l0.clone())
        self.weight_hh_l_last = nn.Parameter(dummy_lstm.weight_hh_l0.clone())
        self.bias_ih_l_last = nn.Parameter(dummy_lstm.bias_ih_l0.clone())
        self.bias_hh_l_last = nn.Parameter(dummy_lstm.bias_hh_l0.clone())
        
        self.fc = nn.Linear(hidden_size, output_size)

        # CUDA Graph related attributes
        self.graph = None
        self.static_inputs = {}
        self.static_outputs = {}

    def _triton_lstm_step(self, x_proj_t, h_prev, c_prev, h_next, c_next):
        """Wrapper for the Triton kernel for a single time step."""
        B = x_proj_t.shape[0]
        D_hid = self.hidden_size
        
        grid = lambda META: (B, triton.cdiv(D_hid, META['BLOCK_SIZE_H']))
        
        lstm_step_recurrent_only_kernel[grid](
            x_proj_t, h_prev, c_prev,
            self.weight_hh_l_last, self.bias_ih_l_last, self.bias_hh_l_last,
            h_next, c_next,
            B, D_hid,
            x_proj_t.stride(0), x_proj_t.stride(1),
            h_prev.stride(0), h_prev.stride(1),
            c_prev.stride(0), c_prev.stride(1),
            self.weight_hh_l_last.stride(0), self.weight_hh_l_last.stride(1),
            self.bias_ih_l_last.stride(0), self.bias_hh_l_last.stride(0),
            h_next.stride(0), h_next.stride(1),
            c_next.stride(0), c_next.stride(1),
        )

    def _graphable_lstm_loop(self, x_proj, h0_last, c0_last, h_buffer, c_buffer):
        """A graphable function that executes the LSTM loop over the time sequence."""
        h_t, c_t = h0_last, c0_last
        seq_len = x_proj.size(1)
        # Use a ping-pong buffer scheme for h and c to avoid in-place modifications within the graph
        buffers = [(h_buffer, c_buffer), (h0_last, c0_last)]
        
        for t in range(seq_len):
            x_proj_t = x_proj[:, t, :].contiguous()
            h_in, c_in = h_t, c_t
            
            # Select the output buffer for the current step
            h_out, c_out = buffers[t % 2]
            
            # Execute the custom Triton kernel for one step
            self._triton_lstm_step(x_proj_t, h_in, c_in, h_out, c_out)
            
            # Update the pointers for the next iteration
            h_t, c_t = h_out, c_out
            
        return h_t, c_t

    def forward(self, x, h0, c0):
        # Handle the first N-1 layers using the standard nn.LSTM
        if self.first_layers:
            h0_first, h0_last = h0[:-1, ...].contiguous(), h0[-1, ...].contiguous().squeeze(0)
            c0_first, c0_last = c0[:-1, ...].contiguous(), c0[-1, ...].contiguous().squeeze(0)
            x_last_layer, (h_n_first, _) = self.first_layers(x, (h0_first, c0_first))
        else:
            h0_last = h0[0, ...].contiguous()
            c0_last = c0[0, ...].contiguous()
            x_last_layer = x
            # Create an empty tensor for concatenation later if there are no first layers
            h_n_first = torch.empty((0, *h0.shape[1:]), device=h0.device, dtype=h0.dtype)

        # --- Major Optimization: Pre-compute the input projection ---
        # The X @ W_ih part of the LSTM equation is independent of the time step.
        # We compute it for all time steps at once using a single, efficient matmul.
        B, S, D_in_last = x_last_layer.shape
        x_proj = torch.matmul(x_last_layer.reshape(B * S, D_in_last), self.weight_ih_l_last.t())
        x_proj = x_proj.view(B, S, 4 * self.hidden_size)
            
        # --- CUDA Graph Execution ---
        # On the first forward pass, we capture the recurrent loop into a CUDA graph.
        # Subsequent passes will replay this graph, significantly reducing CPU overhead.
        if self.graph is None:
            # Create persistent buffers for graph inputs/outputs
            h_buffer = torch.empty_like(h0_last)
            c_buffer = torch.empty_like(c0_last)
            
            self.static_inputs['x_proj'] = torch.zeros_like(x_proj)
            self.static_inputs['h0_last'] = torch.zeros_like(h0_last)
            self.static_inputs['c0_last'] = torch.zeros_like(c0_last)
            self.static_inputs['h_buffer'] = h_buffer
            self.static_inputs['c_buffer'] = c_buffer

            # Warmup run to ensure everything is initialized
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._graphable_lstm_loop(
                    x_proj, h0_last, c0_last, h_buffer, c_buffer
                )
            torch.cuda.current_stream().wait_stream(s)
            
            # Capture the graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                h_n_last, c_n_last = self._graphable_lstm_loop(**self.static_inputs)
                self.static_outputs['h_n_last'] = h_n_last
                self.static_outputs['c_n_last'] = c_n_last
        
        # Copy the current batch's data into the static tensors used by the graph
        self.static_inputs['x_proj'].copy_(x_proj)
        self.static_inputs['h0_last'].copy_(h0_last)
        self.static_inputs['c0_last'].copy_(c0_last)
        
        # Replay the captured graph
        self.graph.replay()
        
        # Original model only returns the last time step output from the fc layer
        # Let's match the output format.
        # The original model returns state[0], which is h_n
        
        # The output of the fc layer is not used in the original model's return value
        # out = self.fc(self.static_outputs['h_n_last'])

        # Combine the results from the standard LSTM layers and our custom layer
        h_n_last_out = self.static_outputs['h_n_last'].unsqueeze(0)
        h_n_final = torch.cat((h_n_first, h_n_last_out), dim=0)
        
        return h_n_final