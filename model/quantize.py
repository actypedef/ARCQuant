import torch
import torch.nn.functional as F
import numpy as np
import gc

import sys
sys.path.append('kernels/build/')
import agemm 


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

def reorder_quantize_w(w, reorder_index, select_num):
    scale = torch.max(w) / (448.0*6.0)
    # scale = 1.0
    w = w / scale
    scale_w = w.abs().max(dim=1, keepdim=True)[0] / 63.0
    scale_w[scale_w == 0] = 1e-9
    scale_w[scale_w != 0] = 1.0
    scaled_w = w / scale_w
    if select_num == 0:
        return quantize_nvfp4_tensor(scaled_w), scale_w, scale
        # return quantize_mxfp4_tensor(scaled_w), scale_w, scale
        # return quantize_int4_tensor(scaled_w), scale_w, scale
    else:
        topk_index = reorder_index[:select_num]
        return torch.cat([quantize_nvfp4_tensor(scaled_w), quantize_nvfp4_tensor(scaled_w[:, topk_index])], dim=1), scale_w, scale
        # return torch.cat([quantize_mxfp4_tensor(scaled_w), quantize_mxfp4_tensor(scaled_w[:, topk_index])], dim=1), scale_w, scale
        # return torch.cat([quantize_int4_tensor(scaled_w), quantize_int4_tensor(scaled_w[:, topk_index])], dim=1), scale_w, scale

def reorder_quantize_x(x, reorder_index, select_num):
    scale = torch.max(x) / (448.0*6.0)
    # scale = 1.0
    x = x / scale
    scale_x = x.abs().max(dim=1, keepdim=True)[0] / 63.0
    scale_x[scale_x == 0] = 1e-9
    scale_x[scale_x != 0] = 1.0
    scaled_x = x / scale_x
    if select_num == 0:
        return quantize_nvfp4_tensor(scaled_x), scale_x, scale
        # return quantize_mxfp4_tensor(scaled_x), scale_x, scale
        # return quantize_int4_tensor(scaled_x), scale_x, scale
    else:
        topk_index = reorder_index[:select_num]
        q_x = quantize_nvfp4_tensor(scaled_x)
        # q_x = quantize_mxfp4_tensor(scaled_x)
        # q_x = quantize_int4_tensor(scaled_x)
        error_e = scaled_x - q_x
        q_error_k = quantize_nvfp4_tensor(error_e[:, topk_index])
        # q_error_k = quantize_mxfp4_tensor(error_e[:, topk_index])
        # q_error_k = quantize_int4_tensor(error_e[:, topk_index])
        return torch.cat([q_x, q_error_k], dim=1), scale_x, scale

def reorder_quantize_w(w, reorder_index, select_num):
    scale = torch.max(w) / (448.0*6.0)
    # scale = 1.0
    w = w / scale
    qw, scale_w = agemm.reorder_quantize_w(w, reorder_index, select_num)
    return qw, scale_w, scale

def reorder_quantize_x(x, reorder_index, select_num):
    # scale = torch.max(x) / (448.0*6.0)
    scale = 1.0
    # x = x / scale
    qx, scale_x = agemm.reorder_quantize_x(x, reorder_index, select_num)
    return qx, scale_x, scale
