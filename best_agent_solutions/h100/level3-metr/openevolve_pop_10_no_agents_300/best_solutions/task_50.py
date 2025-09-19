# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused scale, mask, and relu
fused_attention_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>

__global__ void scale_mask_relu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const float scale,
    const int H,
    const int T,
    const int S) { // S is max_seqlen for bias stride
    
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Each block in the z dimension handles one (batch, head) pair
    const int bh_idx = blockIdx.z;

    if (i < T && j < T) {
        const int b = bh_idx / H;
        const int h = bh_idx % H;

        // Index for input and output tensors (B, H, T, T)
        const long long int idx = (long long int)b * H * T * T + 
                                  (long long int)h * T * T + 
                                  (long long int)i * T + j;
        
        // Index for bias tensor (1, 1, S, S)
        const long long int bias_idx = (long long int)i * S + j;

        float val = input[idx] * scale;

        if (bias[bias_idx] == 0.0f) {
            val = -std::numeric_limits<float>::infinity();
        }
        
        output[idx] = fmaxf(val, 0.0f);
    }
}

torch::Tensor scale_mask_relu_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    const float scale) {
    
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    const auto B = input.size(0);
    const auto H = input.size(1);
    const auto T = input.size(2);
    TORCH_CHECK(input.size(3) == T, "Input must be a square matrix in last two dimensions");
    
    // Bias shape is (1, 1, max_seqlen, max_seqlen)
    const auto S = bias.size(2);
    TORCH_CHECK(bias.size(3) == S, "Bias must be a square matrix in last two dimensions");
    TORCH_CHECK(T <= S, "Sequence length T cannot be greater than max_seqlen S");

    auto output = torch::empty_like(input);

    const dim3 block_size(16, 16, 1); // 256 threads per block
    const dim3 num_blocks(
        (T + block_size.x - 1) / block_size.x,
        (T + block_size.y - 1) / block_size.y,
        B * H
    );

    scale_mask_relu_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        scale,
        H, T, S
    );
    
    return output;
}
"""

fused_attention_cpp_source = (
    "torch::Tensor scale_mask_relu_cuda(torch::Tensor input, torch::Tensor bias, const float scale);"
)

# JIT compile the CUDA kernel
fused_attention = load_inline(
    name="fused_attention_kernel",
    cpp_sources=fused_attention_cpp_source,
    cuda_sources=fused_attention_source,
    functions=["scale_mask_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
)


# From https://github.com/karpathy/minGPT/blob/master/mingpt/model.py

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super(NewGELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class Model(nn.Module):
    """
    A multi-head masked self-attention layer with a projection at the end that uses ReLU instead of Softmax.
    This version uses a custom CUDA kernel to fuse the scale, mask, and relu operations.
    """

    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        # Load our custom fused kernel
        self.fused_op = fused_attention

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1))
        
        # Fused scale, mask, and relu operation
        scale = 1.0 / math.sqrt(k.size(-1))
        att = self.fused_op.scale_mask_relu_cuda(att, self.bias, scale)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        return y

batch_size = 16
max_seqlen = 1024
n_embd = 768  # Hidden dimension, typical for BERT-base size
n_head = 12   # Number of attention heads, typical for BERT-base size

def get_inputs():
    return [torch.randn(batch_size, max_seqlen, n_embd).cuda()]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]
# EVOLVE-BLOCK-END