import torch
import torch.nn.functional as F
import numpy as np
import gc

import sys
sys.path.append('kernels/build/')
import agemm 

import math
import random


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

def quantize_int4(tensor):
    representable_vals = torch.tensor([
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
        0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0
    ], device=tensor.device, dtype=tensor.dtype)
    
    diff = torch.abs(tensor.unsqueeze(-1) - representable_vals)
    indices = torch.argmin(diff, dim=-1)
    return representable_vals[indices]

def dequantize_int4(tensor):
    return tensor

def quantize_ue4m3(tensor):
    tensor = torch.clamp(tensor, min=2e-3, max=448.0)
    
    exponent = torch.floor(torch.log2(tensor + 1e-9))
    mantissa_val = tensor / (2**exponent) - 1.0 
    
    quantized_mantissa_val = torch.round(mantissa_val * 8) / 8
    
    reconstructed_val = (1 + quantized_mantissa_val) * (2**exponent)
    return reconstructed_val

def dequantize_ue4m3(tensor):
    return tensor

def quantize_ue8m0(tensor):
    
    exponent = torch.ceil(torch.log2(tensor + 1e-9))
    exponent = torch.clamp(exponent, min=-127, max=127)
    
    reconstructed_val = (2**exponent)
    return reconstructed_val

def dequantize_ue8m0(tensor):
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

def quantize_mxfp4_tensor(tensor, group_size=32):

    original_shape = tensor.shape
    
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))
        
    reshaped_tensor = tensor.view(-1, group_size)
    
    max_abs_val = torch.max(torch.abs(reshaped_tensor), dim=1, keepdim=True)[0]
    scale = max_abs_val / 6.0
    scale[scale == 0] = 1e-9 
    
    quantized_scale = quantize_ue8m0(scale)
    dequantized_scale = dequantize_ue8m0(quantized_scale)
    
    normalized_tensor = reshaped_tensor / dequantized_scale
    
    quantized_e2m1_tensor = quantize_e2m1(normalized_tensor)
    
    dequantized_tensor_groups = dequantize_e2m1(quantized_e2m1_tensor) * dequantized_scale
    
    dequantized_tensor = dequantized_tensor_groups.view(tensor.shape)
    
    if padding != 0:
        dequantized_tensor = dequantized_tensor[..., :-padding]
        
    return dequantized_tensor.view(original_shape)

def quantize_int4_tensor(tensor, group_size=128):

    original_shape = tensor.shape
    
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))
        
    reshaped_tensor = tensor.view(-1, group_size)
    
    max_abs_val = torch.max(torch.abs(reshaped_tensor), dim=1, keepdim=True)[0]
    scale = max_abs_val / 7
    scale[scale == 0] = 1e-9 
    
    dequantized_scale = scale
    
    normalized_tensor = reshaped_tensor / dequantized_scale
    
    quantized_int4_tensor = quantize_int4(normalized_tensor)
    
    dequantized_tensor_groups = dequantize_int4(quantized_int4_tensor) * dequantized_scale
    
    dequantized_tensor = dequantized_tensor_groups.view(tensor.shape)
    
    if padding != 0:
        dequantized_tensor = dequantized_tensor[..., :-padding]
        
    return dequantized_tensor.view(original_shape)

def get_e3m2_values(device, dtype):
    
    vals = [0.0]
    
    vals.extend([0.0625, 0.125, 0.1875])
    
    mantissas = [1.0, 1.25, 1.5, 1.75]
    for E in range(1, 8): # E from 1 to 7
        exponent_val = 2 ** (E - 3)
        for m in mantissas:
            vals.append(m * exponent_val)
            
    pos_vals = torch.tensor(vals, device=device, dtype=dtype)
    all_vals = torch.cat([-pos_vals, pos_vals]).unique()
    return torch.sort(all_vals)[0]

def quantize_e3m2(tensor):
    representable_vals = get_e3m2_values(tensor.device, tensor.dtype)
    
    diff = torch.abs(tensor.unsqueeze(-1) - representable_vals)
    indices = torch.argmin(diff, dim=-1)
    
    return representable_vals[indices]

def dequantize_e3m2(tensor):
    return tensor

def quantize_mxfp6_tensor(tensor, group_size=32):

    original_shape = tensor.shape
    
    # 1. Padding to align with group_size
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))
        
    reshaped_tensor = tensor.view(-1, group_size)
    
    # 2. Calculate Scale
    max_abs_val = torch.max(torch.abs(reshaped_tensor), dim=1, keepdim=True)[0]
    
    scale = max_abs_val / 28.0 
    scale[scale == 0] = 1e-9 
    
    # 3. Quantize Scale (Shared Exponent)
    quantized_scale = quantize_ue8m0(scale)
    dequantized_scale = dequantize_ue8m0(quantized_scale)
    
    # 4. Normalize Tensor
    normalized_tensor = reshaped_tensor / dequantized_scale
    
    # 5. Quantize Mantissa/Element (E3M2)
    quantized_e3m2_tensor = quantize_e3m2(normalized_tensor)
    
    # 6. Dequantize (Restore Scale)
    dequantized_tensor_groups = dequantize_e3m2(quantized_e3m2_tensor) * dequantized_scale
    
    # 7. Reshape & Remove Padding
    dequantized_tensor = dequantized_tensor_groups.view(tensor.shape)
    
    if padding != 0:
        dequantized_tensor = dequantized_tensor[..., :-padding]
        
    return dequantized_tensor.view(original_shape)


def fake_reorder_quantize_w(w, reorder_index, select_num, dtype='NVFP4'):
    
    scale = torch.max(w).float() / (448.0*6.0)
    quantize_func = quantize_nvfp4_tensor
    
    if dtype == "NVFP4":
        scale = torch.max(w).float() / (448.0*6.0)
        quantize_func = quantize_nvfp4_tensor
    elif dtype == "MXFP4":
        scale = 1.0
        quantize_func = quantize_mxfp4_tensor
    else:
        scale = 1.0
        quantize_func = quantize_int4_tensor
    
    w = w / scale

    scale_w = w.abs().max(dim=1, keepdim=True)[0]
    if select_num == 0:
        return quantize_func(w), scale_w, scale
    else:
        topk_index = reorder_index[-select_num:]
        return torch.cat([quantize_func(w), quantize_func(w[:, topk_index])], dim=1), scale_w, scale

def fake_reorder_quantize_x(x, reorder_index, select_num, dtype='NVFP4'):
    
    scale = torch.max(x).float() / (448.0*6.0)
    quantize_func = quantize_nvfp4_tensor
    
    if dtype == "NVFP4":
        scale = torch.max(x).float() / (448.0*6.0)
        quantize_func = quantize_nvfp4_tensor
    elif dtype == "MXFP4":
        scale = 1.0
        quantize_func = quantize_mxfp4_tensor
    else:
        scale = 1.0
        quantize_func = quantize_int4_tensor
    
    x = x / scale
    
    scale_x = x.abs().max(dim=1, keepdim=True)[0]
    if select_num == 0:
        return quantize_func(x), scale_x, scale
    else:
        topk_index = reorder_index[-select_num:]
        q_x = quantize_func(x)
        error_e = x - q_x
        q_error_k = quantize_func(error_e[:, topk_index])
        return torch.cat([q_x, q_error_k], dim=1) * scale, scale_x, scale

def hadamard_transform(x, normalize=True, block_size=-1):
    n = x.shape[-1]
    if block_size == -1:
        if n <= 0 or (n & (n - 1)) != 0:
            return x
    else:
        if block_size <= 0 or (block_size & (block_size - 1)) != 0:
            raise ValueError(f"block_size {block_size}")
        if n % block_size != 0:
            raise ValueError(f" {n} block_size {block_size}")
    
    original_shape = x.shape
    
    if block_size != -1:
        num_blocks = n // block_size
        x = x.view(-1, num_blocks, block_size)
        batch_dim = x.shape[0]
        current_n = block_size
    else:
        x = x.view(-1, n)
        batch_dim = x.shape[0]
        current_n = n
    
    h = x.clone()
    num_stages = int(torch.log2(torch.tensor(current_n, dtype=torch.float32)).item())
    
    for stage in range(num_stages):
        stage_block_size = 2 ** (stage + 1)
        half_block_size = stage_block_size // 2
        if block_size != -1:
            temp = h.view(batch_dim, -1, stage_block_size)
        else:
            temp = h.view(batch_dim, -1, stage_block_size)
        front_half = temp[:, :, :half_block_size]
        back_half = temp[:, :, half_block_size:]
        new_front = front_half + back_half
        new_back = front_half - back_half
        h = torch.cat([new_front, new_back], dim=2)
        if block_size != -1:
            h = h.view(batch_dim, -1, current_n)
        else:
            h = h.view(batch_dim, current_n)
    
    if normalize:
        h = h / torch.sqrt(torch.tensor(current_n, dtype=torch.float32))
    if block_size != -1:
        h = h.view(-1, num_blocks * block_size)
    h = h.view(original_shape)
    return h

