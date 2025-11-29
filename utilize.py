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
        elif metric == 'frobenius':
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

    dataset = load_dataset(
        'allenai/c4', 'en', 
        split='train', 
        streaming=True, 
        trust_remote_code=True
    )
    
    shuffled_dataset = dataset.shuffle(buffer_size=10000, seed=seed)

    trainloader = []
    inps = []
    for data in shuffled_dataset:
        if len(inps) == nsamples:
            break

        text = data.get('text', '')
        if not text:
            continue

        enc = tokenizer(text, return_tensors='pt')

        if enc.input_ids.shape[1] >= seqlen:
            i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = enc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            
            trainloader.append((inp, tar))
            inps.append(inp)
            
    if len(inps) < nsamples:
        print(f"warning: only get {len(inps)} samples >= {seqlen} ")

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
# def search_select_proportions(model, act_scales, select_ratio=0.06):
#     """
#     根据激活值大小，在同类层之间全局选择重要通道，并为每个层分配选择数量。

#     Args:
#         model (nn.Module): The transformer model.
#         act_scales (dict): 字典，key为 "layer_name.input"，value为对应激活值的统计张量。
#         select_ratio (float): 在每类层的通道池中，要选择的重要通道的全局比例。

#     Returns:
#         tuple: (select_nums, average_bits)
#             - select_nums (dict): 每个线性层输入应选择的通道数量。
#             - average_bits (dict): 每个线性层输入对应的预估平均比特。
#     """
#     # 最终返回的结果字典
#     select_nums = {}
#     average_bits = {}

#     # 1. 按类型对所有线性层进行分组
#     # key: 'q_proj', 'down_proj', etc.
#     # value: list of (layer_name, module) tuples
#     layer_groups = defaultdict(list)
#     for name, m in model.model.named_modules():
#         if isinstance(m, nn.Linear):
#             # 获取层的类型，例如 'q_proj'
#             layer_type = name.split('.')[-1]
#             layer_groups[layer_type].append((name, m))

#     # 2. 对每个分组进行处理
#     for layer_type, layers_in_group in layer_groups.items():
#         # 2.1. 建立当前分组的通道池
#         # channel_pool 是一个列表，每个元素是 (scale_value, layer_name, channel_index)
#         channel_pool = []
#         total_channels_in_group = 0

#         for layer_name, m in layers_in_group:
#             dict_key = layer_name + ".input"
#             if dict_key not in act_scales:
#                 print(f"警告: 在 act_scales 中找不到 {dict_key} 的键，跳过此层。")
#                 continue
            
#             scales = act_scales[dict_key]
#             in_features = m.in_features
#             assert len(scales) == in_features, f"{layer_name} 的特征数量与 scale 长度不匹配!"

#             total_channels_in_group += in_features
            
#             # 将每个通道的 scale 值、来源层和索引存入池中
#             for i in range(in_features):
#                 channel_pool.append((scales[i].item(), layer_name, i))
        
#         if not channel_pool:
#             continue

#         # 2.2. 全局排序与选择
#         # 按 scale 值从大到小排序
#         channel_pool.sort(key=lambda x: x[0], reverse=True)

#         # 确定要选择的通道总数
#         num_to_select = int(total_channels_in_group * select_ratio)
        
#         # 选出最重要的通道
#         selected_channels = channel_pool[:num_to_select]

#         # 2.3. 统计每个层的被选中通道数量
#         layer_selection_counts = defaultdict(int)
#         for scale, layer_name, channel_idx in selected_channels:
#             layer_selection_counts[layer_name] += 1
            
#         # 2.4. 为组内的每个层计算最终的 select_num 和 average_bits
#         for layer_name, m in layers_in_group:
#             in_features = m.in_features
#             dict_key = layer_name + ".input"

#             # 获取该层被选中的原始通道数
#             raw_select_count = layer_selection_counts[layer_name] # 如果没有被选中的，默认为0

#             # 向上取整到64的倍数
#             final_select_num = math.ceil(raw_select_count / 64) * 64
#             # 确保选择数量不超过总特征数
#             final_select_num = min(final_select_num, in_features)

#             select_nums[dict_key] = final_select_num

#             # 基于最终确定的选择数量，计算该层的实际选择比例和平均比特
#             if in_features > 0:
#                 actual_ratio_for_layer = final_select_num / in_features
#             else:
#                 actual_ratio_for_layer = 0.0

#             average_bits[dict_key] = 9 * actual_ratio_for_layer + 4.5 * (1.0 - actual_ratio_for_layer)

#             print(f"{layer_name}: {(actual_ratio_for_layer*100):.2f}%")
    
#     return select_nums, average_bits

# @torch.no_grad()
# def search_select_proportions(
#     model, 
#     act_scales, 
#     select_ratio=0.06,
#     epsilon=1e-8  # 用于防止标准差为0时除法错误
# ):
#     """
#     基于“Z-score局部归一化，全局贪心”的策略来自适应分配重要通道。
#     1. 计算每个层类型（池）中激活值的均值和标准差。
#     2. 对每个通道计算其Z-score = (value - mean) / (std + epsilon)。
#     3. 将所有通道的Z-score放入一个全局池中。
#     4. 贪心选择Z-score最高的通道，直到达到总预算。
#     """
#     select_nums = {}
#     average_bits = {}

#     # --- 阶段1：分组，计算统计量，并构建全局竞争池 ---

#     layer_groups = defaultdict(list)
#     total_model_channels = 0
    
#     # 1.1. 分组，同时将所有scales收集起来，方便后续计算统计量
#     pool_scales_tensors = defaultdict(list)
#     for name, m in model.model.named_modules():
#         if isinstance(m, nn.Linear):
#             layer_type = name.split('.')[-1]
#             layer_groups[layer_type].append((name, m))
#             total_model_channels += m.in_features
            
#             dict_key = name + ".input"
#             if dict_key in act_scales:
#                 # 收集tensor，而不是展平的list，方便用torch计算
#                 pool_scales_tensors[layer_type].append(act_scales[dict_key])
#             else:
#                 print(f"警告: 在 act_scales 中找不到 {dict_key} 的键。")

#     # 1.2. 计算每个池的均值和标准差
#     pool_stats = {}
#     for layer_type, tensor_list in pool_scales_tensors.items():
#         if not tensor_list:
#             continue
#         # 将一个池中所有层的scales拼接成一个大的一维张量
#         full_pool_tensor = torch.cat(tensor_list)
#         mean = torch.mean(full_pool_tensor).item()
#         std = torch.std(full_pool_tensor).item()
#         pool_stats[layer_type] = {'mean': mean, 'std': std}

#     # 1.3. 构建全局竞争池，存储 (z_score, layer_name, channel_index)
#     global_competition_pool = []
#     for layer_type, layers_in_group in layer_groups.items():
#         if layer_type not in pool_stats:
#             continue
        
#         stats = pool_stats[layer_type]
#         mean = stats['mean']
#         std = stats['std']
            
#         for layer_name, m in layers_in_group:
#             dict_key = layer_name + ".input"
#             if dict_key in act_scales:
#                 scales = act_scales[dict_key]
#                 for i in range(len(scales)):
#                     original_scale = scales[i].item()
#                     # 计算Z-score
#                     z_score = (original_scale - mean) / (std + epsilon)
#                     global_competition_pool.append((z_score, layer_name, i))

#     # --- 阶段2：全局排序与选择 ---

#     # 2.1. 按Z-score从大到小排序
#     global_competition_pool.sort(key=lambda x: x[0], reverse=True)
    
#     # 2.2. 确定要选择的总通道数
#     total_budget_to_select = int(total_model_channels * select_ratio)
    
#     # 2.3. 选出Z-score最高的通道
#     selected_channels = global_competition_pool[:total_budget_to_select]

#     print(f"threshold is {global_competition_pool[-total_budget_to_selsect]}, max is {global_competition_pool[-1]}")
    
#     # --- 阶段3：统计与收尾 ---
    
#     # 3.1. 统计每个层的被选中通道数量
#     layer_selection_counts = defaultdict(int)
#     for _, layer_name, _ in selected_channels:
#         layer_selection_counts[layer_name] += 1
        
#     # 3.2. 遍历所有层，计算最终的 select_num 和 average_bits
#     for layer_type, layers_in_group in layer_groups.items():
#         for layer_name, m in layers_in_group:
#             in_features = m.in_features
#             dict_key = layer_name + ".input"
            
#             raw_select_count = layer_selection_counts.get(layer_name, 0) # 使用.get确保安全
#             final_select_num = math.ceil(raw_select_count / 64) * 64
#             final_select_num = min(final_select_num, in_features)
            
#             select_nums[dict_key] = final_select_num
            
#             actual_ratio_for_layer = final_select_num / in_features if in_features > 0 else 0
#             average_bits[dict_key] = 9 * actual_ratio_for_layer + 4.5 * (1.0 - actual_ratio_for_layer)
#             print(f"{layer_name}: {(actual_ratio_for_layer*100):.2f}%")

#     return select_nums, average_bits

@torch.no_grad()
def search_select_proportions(
    model, 
    act_scores, 
    select_ratio=0.06,
    epsilon=1e-8  # 用于防止标准差为0时除法错误
):
    """
    最终版：基于“逐层Z-score归一化，全局贪心”的策略。
    1. 为每一个独立的线性层（如 layer.0.q_proj, layer.31.q_proj）计算其私有的均值和标准差。
    2. 对每个通道计算其相对于自身所在层的Z-score。
    3. 将所有通道的局部Z-score放入一个全局池中进行排序和选择。
    """
    select_nums = {}
    average_bits = {}
    
    all_linear_layers = []
    total_model_channels = 0
    for name, m in model.model.named_modules():
        if isinstance(m, nn.Linear):
            all_linear_layers.append((name, m))
            total_model_channels += m.in_features

    # --- 阶段1：逐层计算Z-score并构建全局竞争池 ---
    
    global_competition_pool = []
    
    for layer_name, m in all_linear_layers:
        dict_key = layer_name + ".input"
        
        if dict_key in act_scores:
            scales = act_scores[dict_key]
            
            # 确保scales是tensor
            if not isinstance(scales, torch.Tensor):
                scales = torch.tensor(scales)
                
            if scales.numel() > 1:
                # 为当前这一个层计算其私有的均值和标准差
                # mean = torch.mean(scales).item()
                mean = torch.quantile(scales, 0.95)
                std = torch.std(scales).item()
            else: # 如果只有一个通道，无法计算标准差
                mean = scales.item()
                std = 0

            for i in range(len(scales)):
                original_scale = scales[i].item()
                # 计算相对于本层的Z-score
                # z_score = (original_scale - mean) / (std + epsilon)
                z_score = original_scale / mean
                global_competition_pool.append((z_score, layer_name, i))

    # --- 阶段2：全局排序与选择 ---

    global_competition_pool.sort(key=lambda x: x[0], reverse=True)
    
    total_budget_to_select = int(total_model_channels * select_ratio)
    selected_channels = global_competition_pool[:total_budget_to_select]

    print(f"threshold is {global_competition_pool[total_budget_to_select]}, max is {global_competition_pool[1]}")
    
    # --- 阶段3：统计与收尾 ---
    
    layer_selection_counts = defaultdict(int)
    for _, layer_name, _ in selected_channels:
        layer_selection_counts[layer_name] += 1
        
    for layer_name, m in all_linear_layers:
        in_features = m.in_features
        dict_key = layer_name + ".input"
        
        raw_select_count = layer_selection_counts.get(layer_name, 0)
        final_select_num = math.ceil(raw_select_count / 64) * 64
        final_select_num = min(final_select_num, in_features)
        
        select_nums[dict_key] = final_select_num
        
        actual_ratio_for_layer = final_select_num / in_features if in_features > 0 else 0
        average_bits[dict_key] = 9 * actual_ratio_for_layer + 4.5 * (1.0 - actual_ratio_for_layer)
        print(f"{layer_name}: {(actual_ratio_for_layer*100):.2f}%")

    return select_nums, average_bits

# def search_select_proportions(
#     model, 
#     act_scales, 
#     select_ratio=0.06,
#     epsilon=1e-8  # 用于防止标准差为0时除法错误
# ):
#     select_nums = {}
#     average_bits = {}

#     for name, m in model.model.named_modules():
#         if 'output' in name:
#                 continue
#         if isinstance(m, nn.Linear):
#             in_features = m.in_features
            
#             select_num = math.ceil(in_features * select_ratio / 64) * 64
            
#             dict_key = name + ".input"
            
#             average_bits[dict_key] = 9 * select_ratio + 4.5 * (1.0 - select_ratio)
#             select_nums[dict_key] = select_num


#     return select_nums, average_bits