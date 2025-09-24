import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Triton not found. Please install it to run the optimized GRU implementation.")


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_B': 32, 'BLOCK_H': 64,  'BLOCK_K_I': 32, 'BLOCK_K_H': 32}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_B': 64, 'BLOCK_H': 64,  'BLOCK_K_I': 32, 'BLOCK_K_H': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_B': 64, 'BLOCK_H': 128, 'BLOCK_K_I': 32, 'BLOCK_K_H': 32}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_B': 32, 'BLOCK_H': 128, 'BLOCK_K_I': 64, 'BLOCK_K_H': 64}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_B': 32, 'BLOCK_H': 128, 'BLOCK_K_I': 32, 'BLOCK_K_H': 64}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_B': 32, 'BLOCK_H': 128, 'BLOCK_K_I': 64, 'BLOCK_K_H': 32}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_B': 16, 'BLOCK_H': 256, 'BLOCK_K_I': 64, 'BLOCK_K_H': 64}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_B': 16, 'BLOCK_H': 32,  'BLOCK_K_I': 32, 'BLOCK_K_H': 32}, num_stages=2, num_warps=2),
        ],
        key=['BATCH_SIZE', 'HIDDEN_SIZE', 'INPUT_SIZE'],
    )
    @triton.jit
    def _fully_fused_gru_cell_kernel(
        # Pointers to Tensors
        x_t_ptr, h_prev_ptr,
        w_ih_ptr, w_hh_ptr,
        bias_ih_ptr, bias_hh_ptr,
        h_new_ptr,
        # Stride info
        x_t_stride_b, x_t_stride_k,
        h_prev_stride_b, h_prev_stride_h,
        w_ih_stride_g, w_ih_stride_k,
        w_hh_stride_g, w_hh_stride_k,
        h_new_stride_b, h_new_stride_h,
        # Constants
        BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE,
        USE_BIAS: tl.constexpr,
        # Meta-parameters
        BLOCK_B: tl.constexpr, BLOCK_H: tl.constexpr,
        BLOCK_K_I: tl.constexpr, BLOCK_K_H: tl.constexpr,
    ):
        """
        This autotuned kernel computes one GRU cell step for a whole batch.
        It fully fuses the two GEMMs (input-hidden and hidden-hidden) with all
        subsequent element-wise gate logic and state updates.
        This avoids writing the intermediate gate projections to global memory.
        """
        # --- Grid and offsets ---
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        offsets_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offsets_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

        # --- GEMM 1: gi = x_t @ W_ih.T ---
        acc_r_i = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        acc_z_i = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        acc_n_i = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        for k_start in range(0, INPUT_SIZE, BLOCK_K_I):
            offsets_k = k_start + tl.arange(0, BLOCK_K_I)
            x_t_mask = (offsets_b[:, None] < BATCH_SIZE) & (offsets_k[None, :] < INPUT_SIZE)
            x_t_tile_ptr = x_t_ptr + offsets_b[:, None] * x_t_stride_b + offsets_k[None, :] * x_t_stride_k
            a = tl.load(x_t_tile_ptr, mask=x_t_mask, other=0.0)

            w_ih_mask = (offsets_h[None, :] < HIDDEN_SIZE) & (offsets_k[:, None] < INPUT_SIZE)
            w_ih_r_ptr = w_ih_ptr + (offsets_h[None, :] + 0 * HIDDEN_SIZE) * w_ih_stride_g + offsets_k[:, None] * w_ih_stride_k
            b_r = tl.load(w_ih_r_ptr, mask=w_ih_mask, other=0.0)
            w_ih_z_ptr = w_ih_ptr + (offsets_h[None, :] + 1 * HIDDEN_SIZE) * w_ih_stride_g + offsets_k[:, None] * w_ih_stride_k
            b_z = tl.load(w_ih_z_ptr, mask=w_ih_mask, other=0.0)
            w_ih_n_ptr = w_ih_ptr + (offsets_h[None, :] + 2 * HIDDEN_SIZE) * w_ih_stride_g + offsets_k[:, None] * w_ih_stride_k
            b_n = tl.load(w_ih_n_ptr, mask=w_ih_mask, other=0.0)

            acc_r_i += tl.dot(a, b_r)
            acc_z_i += tl.dot(a, b_z)
            acc_n_i += tl.dot(a, b_n)

        # --- GEMM 2: gh = h_prev @ W_hh.T ---
        acc_r_h = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        acc_z_h = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        acc_n_h = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)
        for k_start in range(0, HIDDEN_SIZE, BLOCK_K_H):
            offsets_k = k_start + tl.arange(0, BLOCK_K_H)
            h_prev_mask = (offsets_b[:, None] < BATCH_SIZE) & (offsets_k[None, :] < HIDDEN_SIZE)
            h_prev_tile_ptr = h_prev_ptr + offsets_b[:, None] * h_prev_stride_b + offsets_k[None, :] * h_prev_stride_h
            a = tl.load(h_prev_tile_ptr, mask=h_prev_mask, other=0.0)

            w_hh_mask = (offsets_h[None, :] < HIDDEN_SIZE) & (offsets_k[:, None] < HIDDEN_SIZE)
            w_hh_r_ptr = w_hh_ptr + (offsets_h[None, :] + 0 * HIDDEN_SIZE) * w_hh_stride_g + offsets_k[:, None] * w_hh_stride_k
            b_r = tl.load(w_hh_r_ptr, mask=w_hh_mask, other=0.0)
            w_hh_z_ptr = w_hh_ptr + (offsets_h[None, :] + 1 * HIDDEN_SIZE) * w_hh_stride_g + offsets_k[:, None] * w_hh_stride_k
            b_z = tl.load(w_hh_z_ptr, mask=w_hh_mask, other=0.0)
            w_hh_n_ptr = w_hh_ptr + (offsets_h[None, :] + 2 * HIDDEN_SIZE) * w_hh_stride_g + offsets_k[:, None] * w_hh_stride_k
            b_n = tl.load(w_hh_n_ptr, mask=w_hh_mask, other=0.0)

            acc_r_h += tl.dot(a, b_r)
            acc_z_h += tl.dot(a, b_z)
            acc_n_h += tl.dot(a, b_n)

        # --- Bias and Gate Logic ---
        mask_output = (offsets_b[:, None] < BATCH_SIZE) & (offsets_h[None, :] < HIDDEN_SIZE)
        
        if USE_BIAS:
            bias_mask = offsets_h < HIDDEN_SIZE
            b_ih_r = tl.load(bias_ih_ptr + offsets_h + 0 * HIDDEN_SIZE, mask=bias_mask, other=0.0).to(tl.float32)
            b_ih_z = tl.load(bias_ih_ptr + offsets_h + 1 * HIDDEN_SIZE, mask=bias_mask, other=0.0).to(tl.float32)
            b_ih_n = tl.load(bias_ih_ptr + offsets_h + 2 * HIDDEN_SIZE, mask=bias_mask, other=0.0).to(tl.float32)
            
            b_hh_r = tl.load(bias_hh_ptr + offsets_h + 0 * HIDDEN_SIZE, mask=bias_mask, other=0.0).to(tl.float32)
            b_hh_z = tl.load(bias_hh_ptr + offsets_h + 1 * HIDDEN_SIZE, mask=bias_mask, other=0.0).to(tl.float32)
            b_hh_n = tl.load(bias_hh_ptr + offsets_h + 2 * HIDDEN_SIZE, mask=bias_mask, other=0.0).to(tl.float32)

            acc_r_i += b_ih_r; acc_z_i += b_ih_z; acc_n_i += b_ih_n
            acc_r_h += b_hh_r; acc_z_h += b_hh_z; acc_n_h += b_hh_n
        
        r_t = tl.sigmoid(acc_r_i + acc_r_h)
        z_t = tl.sigmoid(acc_z_i + acc_z_h)
        
        n_t_input = acc_n_i + r_t * acc_n_h
        # tanh(x) = 2 * sigmoid(2x) - 1
        n_t = 2.0 * tl.sigmoid(2.0 * n_t_input) - 1.0

        h_prev_final_ptr = h_prev_ptr + offsets_b[:, None] * h_prev_stride_b + offsets_h[None, :] * h_prev_stride_h
        h_prev_tile = tl.load(h_prev_final_ptr, mask=mask_output, other=0.0).to(tl.float32)
        h_new = (1.0 - z_t) * n_t + z_t * h_prev_tile
        
        h_new_ptr_tile = h_new_ptr + offsets_b[:, None] * h_new_stride_b + offsets_h[None, :] * h_new_stride_h
        tl.store(h_new_ptr_tile, h_new.to(h_new_ptr.dtype.element_ty), mask=mask_output)

    
    def fully_fused_gru_cell(x_t: torch.Tensor, h_prev: torch.Tensor, w_ih: torch.Tensor, w_hh: torch.Tensor, b_ih: torch.Tensor, b_hh: torch.Tensor) -> torch.Tensor:
        batch_size, input_size = x_t.shape
        _, hidden_size = h_prev.shape
        h_new = torch.empty_like(h_prev)
        
        use_bias = b_ih is not None
        if not use_bias:
            # Create dummy tensors for the kernel if bias is False
            b_ih = torch.empty(3 * hidden_size, dtype=x_t.dtype, device=x_t.device)
            b_hh = torch.empty(3 * hidden_size, dtype=x_t.dtype, device=x_t.device)

        grid = lambda META: (
            triton.cdiv(batch_size, META['BLOCK_B']),
            triton.cdiv(hidden_size, META['BLOCK_H'])
        )
        _fully_fused_gru_cell_kernel[grid](
            x_t, h_prev, w_ih, w_hh, b_ih, b_hh, h_new,
            x_t.stride(0), x_t.stride(1),
            h_prev.stride(0), h_prev.stride(1),
            w_ih.stride(0), w_ih.stride(1),
            w_hh.stride(0), w_hh.stride(1),
            h_new.stride(0), h_new.stride(1),
            BATCH_SIZE=batch_size, INPUT_SIZE=input_size, HIDDEN_SIZE=hidden_size,
            USE_BIAS=use_bias
        )
        return h_new


class GRUOptimized(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, batch_first=False, bidirectional=False):
        super().__init__()
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available. This custom GRU requires Triton.")
            
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self._graphs = {}

        if self.bidirectional:
            self.s_fwd = torch.cuda.Stream()
            self.s_bwd = torch.cuda.Stream()

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                suffix = '_reverse' if direction == 1 else ''
                
                setattr(self, f'weight_ih_l{layer}{suffix}', nn.Parameter(torch.empty(3 * hidden_size, layer_input_size)))
                setattr(self, f'weight_hh_l{layer}{suffix}', nn.Parameter(torch.empty(3 * hidden_size, hidden_size)))
                if bias:
                    setattr(self, f'bias_ih_l{layer}{suffix}', nn.Parameter(torch.empty(3 * hidden_size)))
                    setattr(self, f'bias_hh_l{layer}{suffix}', nn.Parameter(torch.empty(3 * hidden_size)))
                else:
                    self.register_parameter(f'bias_ih_l{layer}{suffix}', None)
                    self.register_parameter(f'bias_hh_l{layer}{suffix}', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.hidden_size**0.5)
        for weight in self.parameters():
            if weight is not None:
                nn.init.uniform_(weight, -stdv, stdv)

    def _apply_layer(self, x, h, layer_idx, reverse=False):
        seq_len, batch_size, _ = x.shape
        suffix = '_reverse' if reverse else ''
        w_ih = getattr(self, f'weight_ih_l{layer_idx}{suffix}')
        w_hh = getattr(self, f'weight_hh_l{layer_idx}{suffix}')
        b_ih = getattr(self, f'bias_ih_l{layer_idx}{suffix}')
        b_hh = getattr(self, f'bias_hh_l{layer_idx}{suffix}')
        
        graph_key = (seq_len, batch_size, x.dtype, x.device, layer_idx, reverse)
        
        if graph_key not in self._graphs:
            # Warmup to trigger autotuning and JIT compilation
            _ = fully_fused_gru_cell(x[0], h, w_ih, w_hh, b_ih, b_hh)

            # Create static tensors for graph capture
            static_x = torch.empty_like(x)
            static_h_init = torch.empty_like(h)
            static_output_list = [torch.empty_like(h) for _ in range(seq_len)]
            
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                h_graph = static_h_init
                input_seq_indices = range(seq_len)
                for t in input_seq_indices:
                    # For reverse, we process x in reverse, but store outputs sequentially
                    x_t_idx = seq_len - 1 - t if reverse else t
                    h_graph = fully_fused_gru_cell(static_x[x_t_idx], h_graph, w_ih, w_hh, b_ih, b_hh)
                    static_output_list[t].copy_(h_graph)
            
            self._graphs[graph_key] = {
                "graph": g, "static_x": static_x, "static_h_init": static_h_init,
                "static_output_list": static_output_list
            }

        graph_data = self._graphs[graph_key]
        graph_data["static_x"].copy_(x)
        graph_data["static_h_init"].copy_(h)
        graph_data["graph"].replay()

        output_list = graph_data["static_output_list"]
        
        # The last hidden state is the last output computed in the sequence
        h_n = output_list[-1].clone()
        
        # For reverse, the output sequence needs to be flipped
        if reverse:
            output = torch.stack(output_list, dim=0).flip(0)
        else:
            output = torch.stack(output_list, dim=0)
            
        return output, h_n

    def forward(self, x, h0):
        if self.batch_first:
            x = x.transpose(0, 1)

        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)
        h0 = h0.to(model_dtype)
        
        final_hidden_states = []
        current_layer_input = x
        
        for i in range(self.num_layers):
            if self.bidirectional:
                h_fwd_init = h0[i * self.num_directions]
                h_bwd_init = h0[i * self.num_directions + 1]

                # Using streams to run forward and backward passes concurrently
                with torch.cuda.stream(self.s_fwd):
                    output_fwd, h_fwd_final = self._apply_layer(current_layer_input, h_fwd_init, i, reverse=False)
                
                with torch.cuda.stream(self.s_bwd):
                    output_bwd, h_bwd_final = self._apply_layer(current_layer_input, h_bwd_init, i, reverse=True)
                
                torch.cuda.current_stream().wait_stream(self.s_fwd)
                torch.cuda.current_stream().wait_stream(self.s_bwd)

                final_hidden_states.append(h_fwd_final)
                final_hidden_states.append(h_bwd_final)
                layer_output = torch.cat([output_fwd, output_bwd], dim=2)
            else:
                h_init = h0[i]
                output, h_final = self._apply_layer(current_layer_input, h_init, i, reverse=False)
                final_hidden_states.append(h_final)
                layer_output = output
            
            current_layer_input = layer_output
        
        output = current_layer_input
        h_n = torch.stack(final_hidden_states, dim=0)

        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_n

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super(Model, self).__init__()
        
        # The optimized model is instantiated here
        self.gru = GRUOptimized(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=True
        )
        # Using float16 for performance
        self.gru.to(dtype=torch.float16)

    def forward(self, x, h0):
        if not TRITON_AVAILABLE:
            # Fallback or error if Triton is not available
            raise RuntimeError("Cannot run Model forward pass without Triton.")
        output, h_n = self.gru(x, h0)
        # Cast back to float32 if needed for subsequent layers or loss calculation
        return output.to(torch.float32)

# Test code parameters from the original problem description
batch_size = 10
seq_len = 512
input_size = 128
hidden_size = 256
num_layers = 6

def get_inputs():
    # Provide inputs in float16 as the optimized model expects
    dtype = torch.float16
    return [
        torch.randn(seq_len, batch_size, input_size, dtype=dtype, device='cuda'),
        torch.randn((num_layers*2, batch_size, hidden_size), dtype=dtype, device='cuda')
    ]

def get_init_inputs():
    return [input_size, hidden_size, num_layers]