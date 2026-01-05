import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from quantize import *
import sys
sys.path.append('kernels/build/')
import agemm 

import math
import random


def find_qlinear_layers(module, name=''):
    if type(module) == QLinearLayer:
        if module.enable_quant:
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlinear_layers(
            child, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def NVFP4_reorder_quantize_w(w, reorder_index, select_num):
    scale = torch.max(w).float() / (448.0*6.0)
    qw, scale_w = agemm.reorder_quantize_w(w/scale, reorder_index, select_num)
    return qw, scale_w, scale
    
class QLinearLayer(nn.Module):
    def __init__(
        self,
        originalLayer: nn.Linear,
        select_num, 
        reorder_index,
        out_reorder_index=None,
        quant_type='NVFP4',
    ):
        super().__init__()
      
        self.in_features = originalLayer.in_features
        self.out_features = originalLayer.out_features
    
        
        if originalLayer.bias is not None:
            self.register_buffer('bias', originalLayer.bias)
        else:
            self.bias = None
        
        self.select_num = select_num
        self.quant_type = quant_type

        if self.quant_type == 'NVFP4':
            # self.W, self.scale_w, self.scale = NVFP4_reorder_quantize_w((originalLayer.weight.data), torch.arange(self.in_features).to(torch.int16).cuda(), 0)
            self.W, self.scale_w, self.scale = NVFP4_reorder_quantize_w((originalLayer.weight.data), reorder_index.to(torch.int16).cuda(), select_num)
        else:
            self.W, self.scale_w, self.scale = fake_reorder_quantize_w(originalLayer.weight.data, torch.arange(self.in_features), 0, dtype=quant_type)
            # self.W, self.scale_w, self.scale = fake_reorder_quantize_w(torch.index_select(originalLayer.weight.data, 1, reorder_index.to(torch.int32).cuda()), torch.arange(self.in_features), select_num, dtype=quant_type)
        
        reorder_index.cpu()
        del reorder_index
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, x):
        qx, scale_x, scale, bsz, q_len = x

        if self.quant_type == 'NVFP4':
            y = agemm.matmul(qx, self.W, scale_x, self.scale_w, scale * self.scale)
        else:
            y = F.linear(qx, self.W) * scale * self.scale
        
        torch.cuda.synchronize()
        if self.bias is not None:
            y = y + self.bias

        y = y.reshape(bsz, q_len, -1)
        return y

    