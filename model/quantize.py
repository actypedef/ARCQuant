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

def get_e3m2_values(device, dtype):
    
    # 构建所有可能的正数
    vals = [0.0]
    
    # Subnormals (E=0): 0.M * 2^-2 -> M * 0.25 * 0.25 -> M * 0.0625
    # M values: 1, 2, 3 (0已包含)
    vals.extend([0.0625, 0.125, 0.1875])
    
    # Normals (E=1 to 7): 1.M * 2^(E-3)
    # Mantissa steps: 0.00, 0.25, 0.50, 0.75 -> Val: 1.0, 1.25, 1.5, 1.75
    mantissas = [1.0, 1.25, 1.5, 1.75]
    for E in range(1, 8): # E from 1 to 7
        exponent_val = 2 ** (E - 3)
        for m in mantissas:
            vals.append(m * exponent_val)
            
    # 转为 tensor
    pos_vals = torch.tensor(vals, device=device, dtype=dtype)
    # 添加负数部分并去重排序
    all_vals = torch.cat([-pos_vals, pos_vals]).unique()
    return torch.sort(all_vals)[0]

def quantize_e3m2(tensor):
    # 获取 E3M2 的量化码本
    representable_vals = get_e3m2_values(tensor.device, tensor.dtype)
    
    # 寻找最近邻
    # diff shape: (tensor_flat, num_representable)
    diff = torch.abs(tensor.unsqueeze(-1) - representable_vals)
    indices = torch.argmin(diff, dim=-1)
    
    return representable_vals[indices]

def dequantize_e3m2(tensor):
    # 伪量化函数直接返回的是浮点值，不需要额外反量化操作
    return tensor

def quantize_mxfp6_tensor(tensor, group_size=32):
    """
    MXFP6 伪量化主函数
    Block-wise scaling + FP6 (E3M2) quantization
    """
    original_shape = tensor.shape
    
    # 1. Padding to align with group_size
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))
        
    reshaped_tensor = tensor.view(-1, group_size)
    
    # 2. Calculate Scale
    # 找到 block 内最大绝对值
    max_abs_val = torch.max(torch.abs(reshaped_tensor), dim=1, keepdim=True)[0]
    
    # [关键修改] E3M2 的最大可表示值为 28.0 (E=7, M=3 => 1.75 * 2^4 = 28)
    # 将最大值归一化到 range [-1, 1] 之外的 [-28, 28] 空间，或者理解为
    # 我们希望 x / scale 能够落在 E3M2 的覆盖范围内。
    scale = max_abs_val / 28.0 
    scale[scale == 0] = 1e-9 
    
    # 3. Quantize Scale (Shared Exponent)
    quantized_scale = quantize_ue8m0(scale)
    dequantized_scale = dequantize_ue8m0(quantized_scale)
    
    # 4. Normalize Tensor
    normalized_tensor = reshaped_tensor / dequantized_scale
    
    # 5. Quantize Mantissa/Element (E3M2)
    # 这里 normalized_tensor 的值域应该在 [-28, 28] 之间 (理想情况下)，
    # 但由于 scale 量化的精度损失，可能会轻微溢出，quantize_e3m2 会自动 clamp 到最近值(即最大值)。
    quantized_e3m2_tensor = quantize_e3m2(normalized_tensor)
    
    # 6. Dequantize (Restore Scale)
    dequantized_tensor_groups = dequantize_e3m2(quantized_e3m2_tensor) * dequantized_scale
    
    # 7. Reshape & Remove Padding
    dequantized_tensor = dequantized_tensor_groups.view(tensor.shape)
    
    if padding != 0:
        dequantized_tensor = dequantized_tensor[..., :-padding]
        
    return dequantized_tensor.view(original_shape)


# def reorder_quantize_w(w, reorder_index, select_num):
#     scale = torch.max(w) / (448.0*6.0)
#     # scale = 1.0
#     w = w / scale
#     scale_w = w.abs().max(dim=1, keepdim=True)[0] / 63.0
#     scale_w[scale_w == 0] = 1e-9
#     scale_w[scale_w != 0] = 1.0
#     scaled_w = w / scale_w
#     if select_num == 0:
#         return quantize_nvfp4_tensor(scaled_w), scale_w, scale
#         # return quantize_mxfp4_tensor(scaled_w), scale_w, scale
#         # return quantize_int4_tensor(scaled_w), scale_w, scale
#     else:
#         topk_index = reorder_index[:select_num]
#         return torch.cat([quantize_nvfp4_tensor(scaled_w), quantize_nvfp4_tensor(scaled_w[:, topk_index])], dim=1), scale_w, scale
#         # return torch.cat([quantize_mxfp4_tensor(scaled_w), quantize_mxfp4_tensor(scaled_w[:, topk_index])], dim=1), scale_w, scale
#         # return torch.cat([quantize_int4_tensor(scaled_w), quantize_int4_tensor(scaled_w[:, topk_index])], dim=1), scale_w, scale

# def reorder_quantize_x(x, reorder_index, select_num):
#     scale = torch.max(x) / (448.0*6.0)
#     # scale = 1.0
#     x = x / scale
#     scale_x = x.abs().max(dim=1, keepdim=True)[0] / 63.0
#     scale_x[scale_x == 0] = 1e-9
#     scale_x[scale_x != 0] = 1.0
#     scaled_x = x / scale_x
#     if select_num == 0:
#         return quantize_nvfp4_tensor(scaled_x), scale_x, scale
#         # return quantize_mxfp4_tensor(scaled_x), scale_x, scale
#         # return quantize_int4_tensor(scaled_x), scale_x, scale
#     else:
#         topk_index = reorder_index[:select_num]
#         q_x = quantize_nvfp4_tensor(scaled_x)
#         # q_x = quantize_mxfp4_tensor(scaled_x)
#         # q_x = quantize_int4_tensor(scaled_x)
#         error_e = scaled_x - q_x
#         q_error_k = quantize_nvfp4_tensor(error_e[:, topk_index])
#         # q_error_k = quantize_mxfp4_tensor(error_e[:, topk_index])
#         # q_error_k = quantize_int4_tensor(error_e[:, topk_index])
#         return torch.cat([q_x, q_error_k], dim=1), scale_x, scale

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
