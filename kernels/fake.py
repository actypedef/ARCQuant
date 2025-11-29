import torch
import torch.nn.functional as F
import numpy as np


def quantize_e2m1(tensor):

    representable_vals = torch.tensor([
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
        0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    ], device=tensor.device, dtype=tensor.dtype)
    
    diff = torch.abs(tensor.unsqueeze(-1) - representable_vals)
    indices = torch.argmin(diff, dim=-1)
    return representable_vals[indices]

def dequantize_e2m1(tensor):
    return tensor

def quantize_ue4m3(tensor):
    tensor = torch.clamp(tensor, min=1/512, max=448.0)
    
    exponent = torch.floor(torch.log2(tensor))
    mantissa_val = tensor / (2**exponent) - 1.0 

    quantized_mantissa_val = torch.round(mantissa_val * 8) / 8
    
    reconstructed_val = (1 + quantized_mantissa_val) * (2**exponent)
    return reconstructed_val

def dequantize_ue4m3(tensor):
    return tensor

def quantize_nvfp4_tensor(tensor, group_size=16):

    original_shape = tensor.shape
    
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))
        
    reshaped_tensor = tensor.view(-1, group_size)
    
    max_abs_val = torch.max(torch.abs(reshaped_tensor), dim=1, keepdim=True)[0]
    scale = max_abs_val / 6.0
    scale[scale == 0] = 1e-9 
    
    quantized_scale = quantize_ue4m3(scale)
    dequantized_scale = dequantize_ue4m3(quantized_scale)
    
    normalized_tensor = reshaped_tensor / dequantized_scale
    
    quantized_e2m1_tensor = quantize_e2m1(normalized_tensor)
    
    dequantized_tensor_groups = dequantize_e2m1(quantized_e2m1_tensor) * dequantized_scale
    
    dequantized_tensor = dequantized_tensor_groups.view(tensor.shape)
    
    if padding != 0:
        dequantized_tensor = dequantized_tensor[..., :-padding]
        
    return dequantized_tensor.view(original_shape)

class Quantizer:
    def __init__(self, **kwargs):
        pass

    def quantize_linear_layer(self, x, w):
        raise NotImplementedError

    def get_name(self):
        return self.__class__.__name__

class FP32Baseline(Quantizer):
    def quantize_linear_layer(self, x, w):
        return F.linear(x, w)

class Nvfp4AugmentedQuantizer(Quantizer):
    def __init__(self, k, group_size=16):
        super().__init__()
        assert k >= 0, "k < 0 !"
        self.k = k
        self.group_size = group_size

    def get_name(self):
        if self.k == 0:
            return "Nvfp4Base"
        return f"Nvfp4Augmented(k={self.k})"

    def quantize_linear_layer(self, x, w):
        scale_x = x.abs().max(dim=1, keepdim=True)[0] / 63.0
        scale_x = torch.ones(scale_x.shape, dtype=torch.bfloat16, device='cuda')
        scale_x[scale_x == 0] = 1e-9
        scale_w = w.abs().max(dim=1, keepdim=True)[0] / 63.0
        scale_w = torch.ones(scale_w.shape, dtype=torch.bfloat16, device='cuda')
        scale_w[scale_w == 0] = 1e-9
        
        x_scaled = x / scale_x
        w_scaled = w / scale_w

        if self.k > 0:

            q_x_trial = quantize_nvfp4_tensor(x_scaled, self.group_size)
            error_e = x_scaled - q_x_trial
            norm_e = torch.linalg.norm(error_e, ord=2, dim=0)
            norm_w = torch.linalg.norm(w_scaled, ord=2, dim=0)
            importance_scores = norm_e * norm_w
            _, top_k_indices = torch.topk(importance_scores, self.k)
            
            q_x = quantize_nvfp4_tensor(x_scaled, self.group_size)
            q_w = quantize_nvfp4_tensor(w_scaled, self.group_size)
            
            error_k = error_e[:, top_k_indices]
            q_error_k = quantize_nvfp4_tensor(error_k, self.group_size)
            
            w_k = w_scaled[:, top_k_indices]
            q_w_k = quantize_nvfp4_tensor(w_k, self.group_size)

            x_gemm = torch.cat([q_x, q_error_k], dim=1)
            w_gemm = torch.cat([q_w, q_w_k], dim=1)

        else: # self.k == 0
            q_x = quantize_nvfp4_tensor(x_scaled, self.group_size)
            q_w = quantize_nvfp4_tensor(w_scaled, self.group_size)
            
            x_gemm = q_x
            w_gemm = q_w

        y_quantized_scaled = F.linear(x_gemm, w_gemm)

        final_scale = scale_x @ scale_w.T
        y_final = y_quantized_scaled * final_scale
        
        return y_final

class INT8Quantizer(Quantizer):
    def quantize_linear_layer(self, x, w):
        scale_x = x.abs().max() / 127.0
        scale_w = w.abs().max() / 127.0
        
        q_x = torch.round(x / scale_x).clamp(-128, 127)
        q_w = torch.round(w / scale_w).clamp(-128, 127)
        
        dq_x = q_x * scale_x
        dq_w = q_w * scale_w
        
        return F.linear(dq_x, dq_w)

class W4A16Quantizer(Quantizer):
    def quantize_linear_layer(self, x, w):
        x_a16 = x.to(torch.float16).to(torch.float32) 
        
        scale_w = w.abs().max(dim=1, keepdim=True)[0] / 7.0 # 4-bit range [-8, 7]
        scale_w[scale_w == 0] = 1e-9
        
        q_w = torch.round(w / scale_w).clamp(-8, 7)
        dq_w = q_w * scale_w
        
        return F.linear(x_a16, dq_w)

class INT4Quantizer(Quantizer):
    def quantize_linear_layer(self, x, w):
        scale_x = x.abs().max() / 7.0
        scale_w = w.abs().max() / 7.0
        
        q_x = torch.round(x / scale_x).clamp(-8, 7)
        q_w = torch.round(w / scale_w).clamp(-8, 7)
        
        dq_x = q_x * scale_x
        dq_w = q_w * scale_w
        
        return F.linear(dq_x, dq_w)

def run_experiment(seq_len=128, in_features=512, out_features=1024):
    print(f"\n--- Running Experiment ---")
    print(f"Matrix shapes: X({seq_len}, {in_features}), W({out_features}, {in_features})")

    M, N, K = seq_len, out_features, in_features
    w = torch.rand(N, K, dtype=torch.bfloat16, device='cuda') * 3
    
    KN, KS, KO = 4096-512, 512-128, 128
    signs = (torch.randint(0, 2, (M, K), dtype=torch.bfloat16, device='cuda') * 2 - 1)
    x = torch.rand(M, K, dtype=torch.bfloat16, device='cuda') * 3
    x[:, -KS:] = torch.rand(M, KS, dtype=torch.bfloat16, device='cuda') * 3 + 3
    x[:, -KO:] = torch.rand(M, KO, dtype=torch.bfloat16, device='cuda') * 8 + 8
    x[:, -16:] = torch.rand(M, 16, dtype=torch.bfloat16, device='cuda') * 16 + 16
    x = x * signs
    
    quantizers = [
        FP32Baseline(),
        # INT8Quantizer(),
        # W4A16Quantizer(),
        # INT4Quantizer(),
        Nvfp4AugmentedQuantizer(k=0),
        Nvfp4AugmentedQuantizer(k=512),
        Nvfp4AugmentedQuantizer(k=1024),
        Nvfp4AugmentedQuantizer(k=1536),
        Nvfp4AugmentedQuantizer(k=2048),
        Nvfp4AugmentedQuantizer(k=2560),
        Nvfp4AugmentedQuantizer(k=3072),
        Nvfp4AugmentedQuantizer(k=3584),
        Nvfp4AugmentedQuantizer(k=4096),
    ]
    
    y_true = quantizers[0].quantize_linear_layer(x, w)
    
    print("\n--- MSE Analysis ---")
    print(f"{'Quantization Method':<25} | {'MSE':<15}")
    print("-" * 45)

    for quantizer in quantizers[1:]: 
        y_quant = quantizer.quantize_linear_layer(x, w)
        
        mse = F.mse_loss(y_true, y_quant).item()
        
        print(f"{quantizer.get_name():<25} | {mse:<15.8e}")

if __name__ == '__main__':
    torch.manual_seed(45510)
    
    run_experiment(seq_len=1, in_features=4096, out_features=4096)