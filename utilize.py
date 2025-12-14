from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForCausalLM, Qwen2ForCausalLM
from datasets import load_dataset
import torch.nn as nn
import gc
import torch
from collections import defaultdict
import functools
from typing import List
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import torch.nn.functional as F
from sklearn.cluster import KMeans
import sys
from model.quantize import *
from model.kv_cache import *


@torch.no_grad()
def get_reorder_index(model, act_scales, metric='mean'):
    act_orders = {}
    def is_permutation(x: torch.Tensor) -> bool:
        if not torch.is_tensor(x) or x.dim() != 1:
            return False
            
        if x.dtype.is_floating_point:
            return False
    
        n = len(x)
    
        if n == 0:
            return True
    
        expected = torch.arange(n, device=x.device, dtype=x.dtype)
        
        return torch.equal(torch.sort(x).values, expected)
    def reorder_tensor(tensor):
        # assert dimension == 1
        assert tensor.dim() == 1, "Choosing outliers must be 1 dimensional"
        sorted_tensor, sorted_index = torch.sort(tensor, descending=False) # For putting outliers at last
        # _, sorted_index = torch.sort(tensor, descending=True) # For putting outliers at first
        assert is_permutation(sorted_index)
        return sorted_index
        # return torch.arange(tensor.shape[0])
        
    for name, m in model.model.named_modules():
        if isinstance(m, nn.Linear):
            m.name = name
            # Reorder Index of each layer's input
            # Used to reorder the weight and previous layer's output
            inputName = name + ".input"
            # act_orders[inputName] = reorder_tensor(act_scales[inputName])
            # if metric == 'frobenius': 
            #     importance = torch.linalg.norm(m.weight.data, ord=2, dim=0) * act_scales[inputName]
            # else: 
            #     importance = act_scales[inputName]
            act_orders[inputName] = reorder_tensor(act_scales[inputName])
            # act_orders[inputName] = reorder_tensor(importance)

            assert act_orders[inputName].dim() == 1, "Return Index must be 1 dimensional"

    return act_orders



def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    kwargs = {"torch_dtype": "auto", "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True, **kwargs)
    model.eval()
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=False)
    return model, enc



@torch.no_grad()
def get_act_stats(model, dataloader, device_, metric='mean', seqlen=2048, reorder_index=None):
    nsamples = len(dataloader)
    device = device_
    act_scales = {}

    def stat_tensor(name, tensor, weight=None, reorder_index=None):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()

        if metric == 'hessian':
            tensorH = math.sqrt(2 / nsamples) * tensor.float().t()
            comming_H = tensorH.matmul(tensorH.t())
            comming_scales = torch.diag(comming_H)
        elif metric == 'score':
            if reorder_index is not None:
                tensor = torch.index_select(tensor, 1, reorder_index)
                    
            tensorE = tensor - quantize_nvfp4_tensor(tensor, group_size=16)
            # if weight is not None:
            #     if reorder_index is not None:
            #         weight = torch.index_select(weight.to(tensor.device, non_blocking=True), 1, reorder_index)
            #     weight_norm = torch.linalg.norm(weight.to(tensor.device, non_blocking=True), ord=2, dim=0).float()
            #     tensor_norm = torch.linalg.norm(tensorE, ord=2, dim=0).float()
            #     comming_scales = (tensor_norm * weight_norm).cpu()
            # else:
            comming_scales = torch.linalg.norm(tensorE, ord=2, dim=0).float().cpu()
        else:
            # comming_scales = torch.mean(tensor.abs(), dim=0).float().cpu()
            comming_scales = torch.linalg.norm(tensor.abs(), ord=float('inf'), dim=0).float().cpu()

        if name in act_scales:
            if metric == 'hessian':
                act_scales[name] += comming_scales
            else:
                act_scales[name] = torch.max(act_scales[name], comming_scales)
        else:
            act_scales[name] = comming_scales

    def stat_input_hook(m, x, y, name, weight_for_input_stat=None, reorder_index=None):
        if isinstance(x, tuple):
            x = x[0]
            assert isinstance(x, torch.Tensor)
        if isinstance(y, tuple):
            y = y[0]
            assert isinstance(y, torch.Tensor)

        inputName = name + ".input"
        outputName = name + ".output"
        if reorder_index is not None:
            # stat_tensor(inputName, x[:, reorder_index[inputName].to(torch.int32)], weight=weight_for_input_stat[:, reorder_index[inputName].to(torch.int32)])
            stat_tensor(inputName, x, weight=weight_for_input_stat, reorder_index=reorder_index)
        else:
            stat_tensor(inputName, x, weight=weight_for_input_stat)
        stat_tensor(outputName, y)

    hooks = []
    nameTemplate = 'layers.{}.{}.{}.{}'
    
    for layer_idx, layer in enumerate(model.model.layers):
        
        attn_block = layer.self_attn
        
        qkv_weight_combined = torch.cat([
            attn_block.q_proj.weight.data,
            attn_block.k_proj.weight.data,
            attn_block.v_proj.weight.data
        ], dim=0).to(device=device, non_blocking=True)
        
        for proj_name, proj_module in [('q_proj', attn_block.q_proj), ('k_proj', attn_block.k_proj), ('v_proj', attn_block.v_proj)]:
            name = f'layers.{layer_idx}.self_attn.{proj_name}'
            index = reorder_index[nameTemplate.format(layer_idx, 'self_attn', proj_name, 'input')].cuda().to(torch.int32) if reorder_index is not None else None
            hooks.append(
                proj_module.register_forward_hook(
                    functools.partial(stat_input_hook, name=name, weight_for_input_stat=qkv_weight_combined, reorder_index=index)
                )
            )
            
        o_proj_name = f'layers.{layer_idx}.self_attn.o_proj'
        o_proj_weight_for_hook = attn_block.o_proj.weight.data if 'o_proj' in o_proj_name and metric == 'frobenius' else None
        index = reorder_index[nameTemplate.format(layer_idx, 'self_attn', 'o_proj', 'input')].cuda().to(torch.int32) if reorder_index is not None else None
        hooks.append(
            attn_block.o_proj.register_forward_hook(
                functools.partial(stat_input_hook, name=o_proj_name, weight_for_input_stat=o_proj_weight_for_hook, reorder_index=index)
            )
        )
        
        mlp_block = layer.mlp
        
        gate_up_weight_combined = torch.cat([
            mlp_block.gate_proj.weight.data, 
            mlp_block.up_proj.weight.data
        ], dim=0).to(device=device, non_blocking=True)
        
        for proj_name, proj_module in [('gate_proj', mlp_block.gate_proj), ('up_proj', mlp_block.up_proj)]:
            name = f'layers.{layer_idx}.mlp.{proj_name}'
            nameTemplate = 'layers.{}.{}.{}.{}'
            index = reorder_index[nameTemplate.format(layer_idx, 'mlp', proj_name, 'input')].cuda().to(torch.int32) if reorder_index is not None else None
            hooks.append(
                proj_module.register_forward_hook(
                    functools.partial(stat_input_hook, name=name, weight_for_input_stat=gate_up_weight_combined, reorder_index=index)
                )
            )
        
        down_proj_name = f'layers.{layer_idx}.mlp.down_proj'
        down_proj_weight_for_hook = mlp_block.down_proj.weight.data if 'down_proj' in down_proj_name and metric == 'frobenius' else None
        index = reorder_index[nameTemplate.format(layer_idx, 'mlp', 'down_proj', 'input')].cuda().to(torch.int32) if reorder_index is not None else None
        hooks.append(
            mlp_block.down_proj.register_forward_hook(
                functools.partial(stat_input_hook, name=down_proj_name, weight_for_input_stat=down_proj_weight_for_hook, reorder_index=index)
            )
        )

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, 'norm') and not model.model.norm.weight.is_meta:
        model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            hidden_states = inp[0] if isinstance(inp, tuple) else inp
            inps[cache['i']] = hidden_states.squeeze(0)
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError

    layers[0] = Catcher(layers[0])
    
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    assert cache['i'] == nsamples, "Captured samples should be equal to nsamples"
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, 'norm') and not model.model.norm.weight.is_meta:
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i].to(device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        inps, outs = outs, inps
        torch.cuda.empty_cache()
        gc.collect()

    for h in hooks:
        h.remove()

    return act_scales

    

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
  
    import random
    random.seed(seed)
    trainloader = []
    inps = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)
    return trainloader, inps 

def get_c4(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    import random
    import torch

    traindata = load_dataset(
        'allenai/c4', 'en', 
        split='validation', 
        trust_remote_code=True
    )
    
    random.seed(seed)
    trainloader = []
    inps = []
    
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = traindata[i]['text']
            
            encoded = tokenizer(text, return_tensors='pt')
            
            if encoded.input_ids.shape[1] >= seqlen:
                i = random.randint(0, encoded.input_ids.shape[1] - seqlen - 1)
                inp = encoded.input_ids[:, i : i + seqlen]
                break
        
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)
        
    return trainloader, inps

def get_pile(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    import random
    
    try:
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    except:
        print("Falling back to pile-10k")
        dataset = load_dataset("NeelNanda/pile-10k", split="train")

    dataset = dataset.shuffle(seed=seed)

    trainloader = []
    inps = []
    
    for data in dataset:
        if len(trainloader) == nsamples:
            break
            
        text = data['text']
        enc = tokenizer(text, return_tensors='pt')
        
        if enc.input_ids.shape[1] >= seqlen:
            i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = enc.input_ids[:, i:j]
            
            tar = inp.clone()
            tar[:, :-1] = -100 # Mask out context
            
            trainloader.append((inp, tar))
            inps.append(inp)
            
    return trainloader, inps

def get_humaneval(nsamples, seed, seqlen, tokenizer):
    import random
    
    try:
        from human_eval.data import read_problems
        problems = read_problems()  
        dataset = list(problems.values())
    except ImportError:
        print("=" * 80)
        print("run 'pip install humaneval'")
        print("=" * 80)
        return [], []
    except Exception as e:
        print(f" 'humaneval' loading error: {e}")
        return [], []

    text_corpus = "\n\n".join([sample['prompt'] for sample in dataset])
    trainenc = tokenizer(text_corpus, return_tensors='pt')

    random.seed(seed)
    trainloader = []
    inps = []
    for _ in range(nsamples):
        if trainenc.input_ids.shape[1] <= seqlen:
            print(f"warning: HumanEval total length ({trainenc.input_ids.shape[1]}) <= seqlen ({seqlen}).")
            inp = trainenc.input_ids
        else:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]

        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        inps.append(inp)
        
        if trainenc.input_ids.shape[1] <= seqlen:
            break 

    return trainloader, inps


# @torch.no_grad()
# def search_select_proportions(
#     model, 
#     act_scores, 
#     select_ratio=0.05,
#     group_size=64,  # 硬件对齐要求
#     epsilon=1e-8
# ):
#     select_nums = {}
#     average_bits = {}
    
#     all_linear_layers = []
#     total_model_channels = 0
#     for name, m in model.model.named_modules():
#         if isinstance(m, nn.Linear):
#             all_linear_layers.append((name, m))
#             total_model_channels += m.in_features
    
#     # 1. 计算全局预算（以 Block 为单位）
#     # 向下取整，确保严格不超预算
#     total_budget_channels = int(total_model_channels * select_ratio)
#     total_budget_blocks = total_budget_channels // group_size
    
#     global_block_pool = []
    
#     print(f"Global Budget: {total_budget_channels} channels (~{total_budget_blocks} blocks of {group_size})")

#     for layer_name, m in all_linear_layers:
#         dict_key = layer_name + ".input"
#         in_features = m.in_features
        
#         # 如果某层连一个 group 都凑不齐，直接跳过（很少见，但为了代码健壮性）
#         if in_features < group_size:
#             select_nums[dict_key] = 0
#             average_bits[dict_key] = 4.5 # 假设全低比特
#             continue
            
#         if dict_key in act_scores:
#             scales = act_scores[dict_key]
            
#             if not isinstance(scales, torch.Tensor):
#                 scales = torch.tensor(scales)
            
#             # --- 归一化 ---
#             # 依然除以中位数，保持你的相对误差逻辑
#             if scales.numel() > 1:
#                 threshold = torch.quantile(scales ** 2, 0.50)
#                 # threshold = 1.0
#             else: 
#                 threshold = scales.item() + epsilon
            
#             if threshold == 0: threshold = epsilon
            
#             normalized_scores = (scales ** 2) / threshold

#             # --- 关键修改：从后往前切片 ---
#             # 我们不需要排序，因为你说了顺序已经固定，后面的是重要的。
#             # 我们只需要评估“倒数第1个组”、“倒数第2个组”...的价值。
            
#             # 该层最多能切出多少个完整的 64 通道组
#             max_blocks = in_features // group_size
            
#             for i in range(max_blocks):
#                 # i=0: 最后64个 (in_features-64 : in_features)
#                 # i=1: 倒数65-128个 (in_features-128 : in_features-64)
                
#                 end_idx = in_features - i * group_size
#                 start_idx = end_idx - group_size
                
#                 # 获取这一段的分数
#                 block_scores = normalized_scores[start_idx : end_idx]
                
#                 # 计算该 Block 的总价值 (Sum of relative errors)
#                 # 这代表了“如果我投入64个通道的显存预算，我能挽回多少相对误差”
#                 block_value = block_scores.sum().item()
                
#                 # 加入全局竞争池
#                 # 只需要记录分数和层名。不需要记录是第几组，因为最后我们只统计每层中了几个组。
#                 global_block_pool.append((block_value, layer_name))

#     # --- 2. 全局排序 & 截断 ---
#     # 按照 Block 价值从大到小排序
#     global_block_pool.sort(key=lambda x: x[0], reverse=True)
    
#     # 选出前 N 个最有价值的 Block
#     selected_blocks = global_block_pool[:total_budget_blocks]

#     # --- 3. 统计结果 ---
#     layer_win_counts = defaultdict(int)
#     for _, layer_name in selected_blocks:
#         layer_win_counts[layer_name] += 1
        
#     total_selected_actual = 0
    
#     for layer_name, m in all_linear_layers:
#         dict_key = layer_name + ".input"
#         in_features = m.in_features
        
#         # 该层赢得了多少个 Block
#         wins = layer_win_counts.get(layer_name, 0)
        
#         # 转换为通道数 (必然是 64 的倍数)
#         final_select_num = wins * group_size
        
#         # 记录结果
#         select_nums[dict_key] = final_select_num
#         total_selected_actual += final_select_num
        
#         # 计算平均 bit
#         actual_ratio = final_select_num / in_features if in_features > 0 else 0
#         # 假设选中部分占用相当于 9 bit (或你的实际开销)，未选中是 4.5 bit
#         average_bits[dict_key] = 9 * actual_ratio + 4.5 * (1.0 - actual_ratio)
        
#         print(f"{layer_name}: {final_select_num} ({actual_ratio*100:.2f}%)")

#     # 打印最终统计
#     if len(global_block_pool) > total_budget_blocks:
#         print(f"\nThreshold Score: {global_block_pool[total_budget_blocks][0]:.4f}")
    
#     print(f"Target Ratio: {select_ratio*100:.2f}%")
#     print(f"Actual Ratio: {(total_selected_actual/total_model_channels)*100:.2f}%")

#     return select_nums, average_bits


# def search_select_proportions(
#     model, 
#     act_scales, 
#     select_ratio=1.0,
#     epsilon=1e-8  # 用于防止标准差为0时除法错误
# ):
#     select_nums = {}
#     average_bits = {}
#     group_size = 16
#     align_size = 64
#     select_ratio = 1.0

#     for name, m in model.model.named_modules():
#         if 'output' in name:
#                 continue
#         if isinstance(m, nn.Linear):
#             in_features = m.in_features
            
#             # select_num = round(in_features * select_ratio / 64) * 64
            
#             dict_key = name + ".input"
            
#             scales = act_scales[dict_key]
#             scales, sorted_index = torch.sort(scales, descending=False)
#             if not isinstance(scales, torch.Tensor):
#                 scales = torch.tensor(scales)

#             print(f"max value is {scales[-1]}")
#             threshold = scales[-1] / 4

#             group_max_values = scales[group_size-1::group_size]
#             count = (group_max_values > threshold).sum().item()
#             select_num = round(count / 4) * align_size 

#             actual_ratio = select_num / in_features
#             average_bits[dict_key] = 9 * actual_ratio + 4.5 * (1.0 - actual_ratio)
#             select_nums[dict_key] = select_num
            
#             print(f"{name}: {select_num} ({actual_ratio*100:.2f}%)")


#     return select_nums, average_bits
    

@torch.no_grad()
def search_select_proportions(model, dataloader, device_, seqlen, reorder_index):
    nsamples = len(dataloader)
    device = device_
    
    select_nums = {}
    average_bits = {}
   
    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
            assert isinstance(x, torch.Tensor)
        if isinstance(y, tuple):
            y = y[0]
            assert isinstance(y, torch.Tensor)
        act_scales[name+".input"] = x
        act_scales[name+".output"] = y
     
    hooks = []
    for name, m in model.model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)
                )
            )

    layers = model.model.layers
    
    model.to(device)
    
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.self_attn = module.self_attn
        def forward(self, inp, **kwargs):
            cache['inps'] = inp
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])

    dataloader = torch.stack(dataloader, dim=0).squeeze(1)
    
    try:
        model(torch.tensor(dataloader).to(device))
    except ValueError:
        pass
    
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.cpu()

    torch.cuda.empty_cache()

    inps = cache['inps']
  
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    total_elements = 0
    total_bits = 0
  
    for i in tqdm(range(len(layers))):
        act_scales = {}
        layer = layers[i].to(device)
  
        inps = layer(inps, attention_mask=attention_mask, position_ids=position_ids)[0]
       
        for name, keys in act_scales.items():
            if 'output' in name:
                continue
                
            keys = keys.reshape(-1, keys.shape[-1]).contiguous()
            seqlen, in_features = keys.shape 
            keys = keys[:, reorder_index[name].to(torch.int32)]
       
            threshold = keys.max(dim=-1, keepdim=True)[0] * 0.125
            
     
            select_ratio = (keys > threshold).sum() / keys.numel()
            select_num = math.ceil(in_features * select_ratio / 64) * 64
            select_ratio = select_num / in_features
            average_bits[name] = 4.5 * (in_features + select_num) / in_features
            total_elements += in_features
            total_bits += 4.5 * (in_features + select_num)
            print(f'{name}: {select_ratio*100:.2f}%, avg:{average_bits[name]:.2f}')
            select_nums[name] = select_num
            del keys
            
            gc.collect()
            torch.cuda.empty_cache()
        
        
        layer.cpu()
        del act_scales
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()
        
    print(f'average bits is {(total_bits / total_elements):.2f}')
    return select_nums, average_bits

