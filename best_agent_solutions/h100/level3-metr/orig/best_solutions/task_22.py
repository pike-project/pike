import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Original Model and MBConv definitions are needed for initializing the new model
class Model(nn.Module):
    def __init__(self, num_classes=1000):
        """
        EfficientNetB0 architecture implementation in PyTorch.

        :param num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(Model, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            # MBConv1 (32, 16, 1, 1)
            MBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            MBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            MBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            MBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            MBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            MBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            MBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            MBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            MBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            MBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            MBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        )
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        if self.use_residual:
            x += identity
        return x

# --- Optimized Custom FP16 CUDA Kernels ---
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <c10/cuda/CUDAStream.h>

// --- Part 1: BatchNorm Folding Kernel (FP16) [Unchanged from previous] ---
__global__ void fold_bn_kernel_fp16(
    const half* __restrict__ conv_w, const half* __restrict__ conv_b,
    const half* __restrict__ bn_gamma, const half* __restrict__ bn_beta, 
    const half* __restrict__ bn_mean, const half* __restrict__ bn_var,
    const float bn_eps, const int C_out, const int C_in_per_group, const int K_h, const int K_w,
    half* __restrict__ new_conv_w, half* __restrict__ new_conv_b)
{
    int c_out = blockIdx.x;
    if (c_out >= C_out) return;

    float scale = __half2float(bn_gamma[c_out]) / sqrtf(__half2float(bn_var[c_out]) + bn_eps);
    float conv_b_float = conv_b ? __half2float(conv_b[c_out]) : 0.0f;
    
    new_conv_b[c_out] = __float2half((conv_b_float - __half2float(bn_mean[c_out])) * scale + __half2float(bn_beta[c_out]));

    const int filter_size = C_in_per_group * K_h * K_w;
    const int w_base_idx = c_out * filter_size;
    for (int i = threadIdx.x; i < filter_size; i += blockDim.x) {
        new_conv_w[w_base_idx + i] = __float2half(__half2float(conv_w[w_base_idx + i]) * scale);
    }
}

// --- Part 2: Vectorized Pointwise Fusion Kernel (FP16) [Unchanged from previous] ---
enum class Activation { NONE = 0, RELU = 1, RELU6 = 2 };

__device__ inline half2 apply_activation_vec(half2 val, Activation act) {
    if (act == Activation::NONE) return val;
    float f1 = __low2float(val), f2 = __high2float(val);
    switch (act) {
        case Activation::RELU: f1 = fmaxf(0.0f, f1); f2 = fmaxf(0.0f, f2); break;
        case Activation::RELU6: f1 = fminf(fmaxf(0.0f, f1), 6.0f); f2 = fminf(fmaxf(0.0f, f2), 6.0f); break;
        default: break;
    }
    return __floats2half2_rn(f1, f2);
}

__global__ void pointwise_fusion_kernel_fp16_vec(
    const half* __restrict__ in, const half* __restrict__ bias, const half* __restrict__ residual, 
    half* __restrict__ out, int N, int C, int H, int W, Activation activation_type)
{
    const int total_elements_div_2 = (N * C * H * W) / 2;
    int idx_div_2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_div_2 >= total_elements_div_2) return;

    half2 in_vec = ((const half2*)in)[idx_div_2];
    int c = ((idx_div_2 * 2) / (H * W)) % C;
    half2 bias_vec = __half2half2(bias[c]);
    half2 result_vec = __hadd2(in_vec, bias_vec);
    if (residual != nullptr) {
        result_vec = __hadd2(result_vec, ((const half2*)residual)[idx_div_2]);
    }
    ((half2*)out)[idx_div_2] = apply_activation_vec(result_vec, activation_type);
}

// --- Part 3: Optimized Fused Adaptive Average Pool + Flatten Kernel (FP16) [Unchanged from previous] ---
__global__ void fused_adaptive_avg_pool_flatten_kernel_fp16_2d(
    const half* __restrict__ in, half* __restrict__ out, int N, int C, int H, int W)
{
    extern __shared__ float sdata[];
    const float inv_HW = 1.0f / (float)(H * W);
    int n = blockIdx.y;
    int c = blockIdx.x * blockDim.y + threadIdx.y;
    if (c >= C) return;

    int s_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const half* in_ptr = in + (n * C + c) * H * W;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < H * W; i += blockDim.x) {
        sum += __half2float(in_ptr[i]);
    }
    sdata[s_idx] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[s_idx] += sdata[s_idx + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[n * C + c] = __float2half(sdata[s_idx] * inv_HW);
}

// --- Part 4: NEW - Fused DepthwiseConv+Bias+ReLU6 Kernel (FP16) ---
// This kernel is templated on Kernel Size (K) and Stride (S) to allow the compiler
// to unroll loops and optimize memory access patterns. It computes a full
// depthwise convolution, adds a fused bias, and applies a ReLU6 activation in a single pass.
// This avoids separate kernel launches and intermediate memory transfers.
template <int K_h, int K_w, int S_h, int S_w>
__global__ void fused_dw_conv_bias_relu6_fp16_kernel(
    const half* __restrict__ in, const half* __restrict__ weight, const half* __restrict__ bias,
    half* __restrict__ out, const int N, const int C, const int H, const int W,
    const int P_h, const int P_w)
{
    const int H_out = (H + 2 * P_h - K_h) / S_h + 1;
    const int W_out = (W + 2 * P_w - K_w) / S_w + 1;
    const int out_plane_size = H_out * W_out;
    const int in_plane_size = H * W;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * out_plane_size) return;

    // Decompose global index to (n, c, y_out, x_out)
    const int x_out = idx % W_out;
    const int y_out = (idx / W_out) % H_out;
    const int c = (idx / out_plane_size) % C;
    const int n = idx / (out_plane_size * C);

    float acc = 0.0f;
    const int y_in_start = y_out * S_h - P_h;
    const int x_in_start = x_out * S_w - P_w;

    const half* in_ptr = in + (n * C + c) * in_plane_size;
    const half* w_ptr = weight + c * K_h * K_w;
    
    #pragma unroll
    for (int kh = 0; kh < K_h; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < K_w; ++kw) {
            const int y_in = y_in_start + kh;
            const int x_in = x_in_start + kw;
            if (y_in >= 0 && y_in < H && x_in >= 0 && x_in < W) {
                acc += __half2float(in_ptr[y_in * W + x_in]) * __half2float(w_ptr[kh * K_w + kw]);
            }
        }
    }

    acc += __half2float(bias[c]);
    acc = fminf(fmaxf(0.0f, acc), 6.0f);
    out[idx] = __float2half(acc);
}

// --- C++ Wrapper Functions ---
std::vector<torch::Tensor> fold_bn_fp16_cuda(
    torch::Tensor conv_w, c10::optional<torch::Tensor> conv_b,
    torch::Tensor bn_gamma, torch::Tensor bn_beta, torch::Tensor bn_mean, torch::Tensor bn_var, double bn_eps)
{
    const int C_out = conv_w.size(0);
    const int C_in_per_group = conv_w.dim() > 1 ? conv_w.size(1) : 1;
    const int K_h = conv_w.dim() > 2 ? conv_w.size(2) : 1;
    const int K_w = conv_w.dim() > 3 ? conv_w.size(3) : 1;
    auto new_conv_w = torch::empty_like(conv_w);
    auto new_conv_b = torch::empty({C_out}, conv_w.options());
    fold_bn_kernel_fp16<<<C_out, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (const half*)conv_w.data_ptr(), conv_b.has_value() ? (const half*)conv_b.value().data_ptr() : nullptr,
        (const half*)bn_gamma.data_ptr(), (const half*)bn_beta.data_ptr(), (const half*)bn_mean.data_ptr(), (const half*)bn_var.data_ptr(),
        (float)bn_eps, C_out, C_in_per_group, K_h, K_w, (half*)new_conv_w.data_ptr(), (half*)new_conv_b.data_ptr());
    return {new_conv_w, new_conv_b};
}

torch::Tensor pointwise_fusion_fp16_vec_cuda(
    torch::Tensor in, torch::Tensor bias, c10::optional<torch::Tensor> residual, int activation_type)
{
    auto out = torch::empty_like(in);
    TORCH_CHECK(in.numel() % 2 == 0, "Vectorized kernel requires even number of elements.");
    const int block_size = 256;
    const int num_blocks = (in.numel() / 2 + block_size - 1) / block_size;
    pointwise_fusion_kernel_fp16_vec<<<num_blocks, block_size, 0, c10::cuda::getCurrentCUDAStream()>>>(
        (const half*)in.data_ptr(), (const half*)bias.data_ptr(), residual.has_value() ? (const half*)residual.value().data_ptr() : nullptr,
        (half*)out.data_ptr(), in.size(0), in.size(1), in.size(2), in.size(3), static_cast<Activation>(activation_type));
    return out;
}
    
torch::Tensor fused_adaptive_avg_pool_flatten_fp16_2d_cuda(torch::Tensor in) {
    const auto N = in.size(0), C = in.size(1), H = in.size(2), W = in.size(3);
    auto out = torch::empty({N, C}, in.options());
    if (in.numel() == 0) return out;
    const int CHANNELS_PER_BLOCK = 4, REDUCE_THREADS = 256;
    dim3 block_dim(REDUCE_THREADS, CHANNELS_PER_BLOCK);
    dim3 grid_dim((C + CHANNELS_PER_BLOCK - 1) / CHANNELS_PER_BLOCK, N);
    size_t shared_mem_size = block_dim.x * block_dim.y * sizeof(float);
    fused_adaptive_avg_pool_flatten_kernel_fp16_2d<<<grid_dim, block_dim, shared_mem_size, c10::cuda::getCurrentCUDAStream()>>>(
        (const half*)in.data_ptr(), (half*)out.data_ptr(), N, C, H, W);
    return out;
}

// NEW: C++ dispatcher for the fused depthwise convolution kernel
torch::Tensor fused_dw_conv_fp16_cuda(
    torch::Tensor in, torch::Tensor weight, torch::Tensor bias,
    long K_h, long K_w, long S_h, long S_w, long P_h, long P_w)
{
    const long N = in.size(0), C = in.size(1), H = in.size(2), W = in.size(3);
    const long H_out = (H + 2 * P_h - K_h) / S_h + 1;
    const long W_out = (W + 2 * P_w - K_w) / S_w + 1;
    auto out = torch::empty({N, C, H_out, W_out}, in.options());
    if (in.numel() == 0) return out;
    
    const int total_outputs = N * C * H_out * W_out;
    const int block_size = 256;
    const int num_blocks = (total_outputs + block_size - 1) / block_size;
    
    // Dispatch to the correct template specialization based on K and S
    #define LAUNCH_DW_KERNEL(KH, KW, SH, SW) \\
        fused_dw_conv_bias_relu6_fp16_kernel<KH, KW, SH, SW><<<num_blocks, block_size, 0, c10::cuda::getCurrentCUDAStream()>>>( \\
            (const half*)in.data_ptr(), (const half*)weight.data_ptr(), (const half*)bias.data_ptr(), \\
            (half*)out.data_ptr(), N, C, H, W, P_h, P_w)

    if (K_h == 3 && K_w == 3) {
        if (S_h == 1 && S_w == 1) { LAUNCH_DW_KERNEL(3, 3, 1, 1); } 
        else if (S_h == 2 && S_w == 2) { LAUNCH_DW_KERNEL(3, 3, 2, 2); } 
        else { TORCH_CHECK(false, "Unsupported Stride for K=3"); }
    } else if (K_h == 5 && K_w == 5) {
        if (S_h == 1 && S_w == 1) { LAUNCH_DW_KERNEL(5, 5, 1, 1); } 
        else if (S_h == 2 && S_w == 2) { LAUNCH_DW_KERNEL(5, 5, 2, 2); }
        else { TORCH_CHECK(false, "Unsupported Stride for K=5"); }
    } else {
        TORCH_CHECK(false, "Unsupported Kernel Size for Fused DW Conv");
    }
    #undef LAUNCH_DW_KERNEL
    return out;
}
"""

fused_ops_cpp_source = """
#include <vector>
std::vector<torch::Tensor> fold_bn_fp16_cuda(
    torch::Tensor conv_w, c10::optional<torch::Tensor> conv_b,
    torch::Tensor bn_gamma, torch::Tensor bn_beta, torch::Tensor bn_mean, torch::Tensor bn_var,
    double bn_eps);

torch::Tensor pointwise_fusion_fp16_vec_cuda(
    torch::Tensor in, torch::Tensor bias, c10::optional<torch::Tensor> residual, int activation_type);
    
torch::Tensor fused_adaptive_avg_pool_flatten_fp16_2d_cuda(torch::Tensor in);

torch::Tensor fused_dw_conv_fp16_cuda(
    torch::Tensor in, torch::Tensor weight, torch::Tensor bias,
    long K_h, long K_w, long S_h, long S_w, long P_h, long P_w);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_efficientnet_ops_fp16_v4",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fold_bn_fp16_cuda", "pointwise_fusion_fp16_vec_cuda", "fused_adaptive_avg_pool_flatten_fp16_2d_cuda", "fused_dw_conv_fp16_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)

def _fuse_conv_bn_fp16(conv, bn):
    """Helper to call the FP16 BN folding CUDA kernel."""
    device = conv.weight.device
    dtype = torch.float16
    
    return fused_ops.fold_bn_fp16_cuda(
        conv.weight.to(device=device, dtype=dtype), 
        conv.bias.to(device=device, dtype=dtype) if conv.bias is not None else None,
        bn.weight.to(device=device, dtype=dtype), 
        bn.bias.to(device=device, dtype=dtype), 
        bn.running_mean.to(device=device, dtype=dtype), 
        bn.running_var.to(device=device, dtype=dtype), 
        bn.eps
    )

class MBConvNew(nn.Module):
    """A new MBConv block that uses the highly fused FP16 operations, including a custom depthwise conv kernel."""
    def __init__(self, orig_mbconv: MBConv):
        super().__init__()
        self.use_residual = orig_mbconv.use_residual
        self.has_expand_conv = hasattr(orig_mbconv, 'expand_conv')

        if self.has_expand_conv:
            expand_conv, expand_bn = orig_mbconv.expand_conv[0], orig_mbconv.expand_conv[1]
            w, b = _fuse_conv_bn_fp16(expand_conv, expand_bn)
            self.register_buffer('fused_expand_w', w)
            self.register_buffer('fused_expand_b', b)
            self.expand_params = {'stride': expand_conv.stride, 'padding': expand_conv.padding, 'groups': expand_conv.groups}
        
        dw_conv, dw_bn = orig_mbconv.depthwise_conv[0], orig_mbconv.depthwise_conv[1]
        w, b = _fuse_conv_bn_fp16(dw_conv, dw_bn)
        self.register_buffer('fused_dw_w', w)
        self.register_buffer('fused_dw_b', b)
        self.dw_params = {'stride': dw_conv.stride, 'padding': dw_conv.padding, 'groups': dw_conv.groups}
        self.dw_k_size = dw_conv.kernel_size

        proj_conv, proj_bn = orig_mbconv.project_conv[0], orig_mbconv.project_conv[1]
        w, b = _fuse_conv_bn_fp16(proj_conv, proj_bn)
        self.register_buffer('fused_proj_w', w)
        self.register_buffer('fused_proj_b', b)
        self.proj_params = {'stride': proj_conv.stride, 'padding': proj_conv.padding, 'groups': proj_conv.groups}

    def forward(self, x):
        identity = x
        
        # Expansion phase (using PyTorch conv + custom pointwise)
        if self.has_expand_conv:
            x_conv = F.conv2d(x, self.fused_expand_w, None, **self.expand_params)
            x = fused_ops.pointwise_fusion_fp16_vec_cuda(x_conv, self.fused_expand_b, None, 2)  # 2=ReLU6
        
        # Depthwise phase (using fully fused custom kernel)
        kh, kw = self.dw_k_size
        sh, sw = self.dw_params['stride']
        ph, pw = self.dw_params['padding']
        x = fused_ops.fused_dw_conv_fp16_cuda(x, self.fused_dw_w, self.fused_dw_b, kh, kw, sh, sw, ph, pw)
        
        # Projection phase (using PyTorch conv + custom pointwise)
        x_proj_conv = F.conv2d(x, self.fused_proj_w, None, **self.proj_params)
        x = fused_ops.pointwise_fusion_fp16_vec_cuda(
            x_proj_conv, self.fused_proj_b, 
            identity if self.use_residual else None, 
            0 # 0=None activation
        )
        return x

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        device = torch.cuda.current_device()
        with torch.no_grad():
            orig_model = Model(num_classes=num_classes).to(device).eval()
        
        # Fuse first Conv+BN+ReLU layer
        w, b = _fuse_conv_bn_fp16(orig_model.conv1, orig_model.bn1)
        self.register_buffer('fused_conv1_w', w)
        self.register_buffer('fused_conv1_b', b)
        self.conv1_params = {'stride': orig_model.conv1.stride, 'padding': orig_model.conv1.padding}

        # Replace MBConv blocks with our new fused version
        self.blocks = nn.Sequential(*[MBConvNew(block) for block in orig_model.blocks])

        # Fuse final Conv+BN+ReLU layer
        w, b = _fuse_conv_bn_fp16(orig_model.conv2, orig_model.bn2)
        self.register_buffer('fused_conv2_w', w)
        self.register_buffer('fused_conv2_b', b)
        self.conv2_params = {'stride': orig_model.conv2.stride, 'padding': orig_model.conv2.padding}
        
        # Use FP16 for FC layer as well
        self.fc = orig_model.fc.half()

    def forward(self, x):
        # Input must be converted to FP16 for the kernels
        x = x.half()

        # Initial Conv -> Fused BiasAdd -> ReLU
        x_conv1 = F.conv2d(x, self.fused_conv1_w, None, **self.conv1_params)
        x = fused_ops.pointwise_fusion_fp16_vec_cuda(x_conv1, self.fused_conv1_b, None, 1) # 1=ReLU

        # Fused MBConv blocks
        x = self.blocks(x)

        # Final Conv -> Fused BiasAdd -> ReLU
        x_conv2 = F.conv2d(x, self.fused_conv2_w, None, **self.conv2_params)
        x = fused_ops.pointwise_fusion_fp16_vec_cuda(x_conv2, self.fused_conv2_b, None, 1) # 1=ReLU

        # Fused Pooling and Flatten
        x = fused_ops.fused_adaptive_avg_pool_flatten_fp16_2d_cuda(x)

        # Final classifier
        x = self.fc(x)
        
        # Convert back to FP32 for output
        return x.float()

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]