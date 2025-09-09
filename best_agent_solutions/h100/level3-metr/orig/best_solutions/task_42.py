import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False, bidirectional=True):
        """
        A custom GRU implementation with parameter names matching torch.nn.GRU
        to ensure identical initialization for correctness testing.
        """
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Register parameters with names matching nn.GRU
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size * self.num_directions
            
            # Forward direction parameters
            setattr(self, f'weight_ih_l{i}', nn.Parameter(torch.Tensor(3 * hidden_size, layer_input_size)))
            setattr(self, f'weight_hh_l{i}', nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size)))
            if bias:
                setattr(self, f'bias_ih_l{i}', nn.Parameter(torch.Tensor(3 * hidden_size)))
                setattr(self, f'bias_hh_l{i}', nn.Parameter(torch.Tensor(3 * hidden_size)))
            
            if self.bidirectional:
                # Backward direction parameters
                setattr(self, f'weight_ih_l{i}_reverse', nn.Parameter(torch.Tensor(3 * hidden_size, layer_input_size)))
                setattr(self, f'weight_hh_l{i}_reverse', nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size)))
                if bias:
                    setattr(self, f'bias_ih_l{i}_reverse', nn.Parameter(torch.Tensor(3 * hidden_size)))
                    setattr(self, f'bias_hh_l{i}_reverse', nn.Parameter(torch.Tensor(3 * hidden_size)))
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights uniformly, same as the default PyTorch implementation.
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    @staticmethod
    def gru_cell(x_t, h_prev, weight_ih, weight_hh, bias_ih, bias_hh):
        """
        Performs a single time-step of GRU computation using F.linear.
        This is equivalent to the previous logic but uses raw parameters instead of nn.Linear modules.
        """
        # Calculate input-hidden and hidden-hidden transformations.
        gi = F.linear(x_t, weight_ih, bias_ih)
        gh = F.linear(h_prev, weight_hh, bias_hh)

        # Split the transformations into the three gates: reset, update, and new.
        r_i, z_i, n_i = gi.chunk(3, 1)
        r_h, z_h, n_h = gh.chunk(3, 1)

        # Calculate the gates.
        resetgate = torch.sigmoid(r_i + r_h)
        updategate = torch.sigmoid(z_i + z_h)
        
        # Calculate the new gate's candidate hidden state.
        # n_t = tanh(W_in*x_t + b_in + r_t * (W_hn*h_{t-1} + b_hn))
        newgate = torch.tanh(n_i + resetgate * n_h)

        # Compute the next hidden state.
        # h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        h_next = (1 - updategate) * newgate + updategate * h_prev
        
        return h_next

    def forward(self, x, h0):
        if self.batch_first:
            x = x.transpose(0, 1)
        
        seq_len, batch_size, _ = x.shape
        
        # Unpack h0 for each layer and direction
        h0_unpacked = h0.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

        layer_input = x
        final_hiddens = []

        for i in range(self.num_layers):
            # --- Forward Pass ---
            h_fwd = h0_unpacked[i, 0]
            w_ih_fwd = getattr(self, f'weight_ih_l{i}')
            w_hh_fwd = getattr(self, f'weight_hh_l{i}')
            b_ih_fwd = getattr(self, f'bias_ih_l{i}') if self.bias else None
            b_hh_fwd = getattr(self, f'bias_hh_l{i}') if self.bias else None
            
            outputs_fwd = []
            for t in range(seq_len):
                h_fwd = self.gru_cell(layer_input[t], h_fwd, w_ih_fwd, w_hh_fwd, b_ih_fwd, b_hh_fwd)
                outputs_fwd.append(h_fwd)
            output_fwd_tensor = torch.stack(outputs_fwd)
            final_hiddens.append(h_fwd)

            if self.bidirectional:
                # --- Backward Pass ---
                h_bwd = h0_unpacked[i, 1]
                w_ih_bwd = getattr(self, f'weight_ih_l{i}_reverse')
                w_hh_bwd = getattr(self, f'weight_hh_l{i}_reverse')
                b_ih_bwd = getattr(self, f'bias_ih_l{i}_reverse') if self.bias else None
                b_hh_bwd = getattr(self, f'bias_hh_l{i}_reverse') if self.bias else None

                outputs_bwd = []
                for t in range(seq_len - 1, -1, -1):
                    h_bwd = self.gru_cell(layer_input[t], h_bwd, w_ih_bwd, w_hh_bwd, b_ih_bwd, b_hh_bwd)
                    outputs_bwd.append(h_bwd)
                final_hiddens.append(h_bwd)
                
                outputs_bwd.reverse()
                output_bwd_tensor = torch.stack(outputs_bwd)
                
                layer_input = torch.cat([output_fwd_tensor, output_bwd_tensor], dim=2)
            else:
                layer_input = output_fwd_tensor

        output = layer_input
        # Stack the final hidden states from each layer and direction
        h_n = torch.stack(final_hiddens, dim=0)
        
        if self.batch_first:
            output = output.transpose(0, 1)
            
        return output, h_n

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
        
        # Replace the standard nn.GRU with our custom implementation.
        # The original model's test configuration uses bidirectional=True.
        self.gru = CustomGRU(input_size, hidden_size, num_layers, bias, batch_first, bidirectional=True)
    
    def forward(self, x, h0):
        """
        :param x: The input tensor, shape (seq_len, batch_size, input_size) if batch_first=False, otherwise (batch_size, seq_len, input_size)
        :param h_0: The initial hidden state for the input sequence, shape (num_layers * num_directions, batch_size, hidden_size) (default: None)
        :return: h_n: The hidden state for t = seq_len, shape (num_layers * num_directions, batch_size, hidden_size)
        """
        # The original model's forward pass returns only the final hidden state.
        _output, h_n = self.gru(x, h0)
        return h_n