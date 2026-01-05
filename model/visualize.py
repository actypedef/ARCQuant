import torch
from torch import nn
from typing import List, Optional, Tuple
import math
from qLinearLayer import QLinearLayer
from quantize import *
import os
import sys
sys.path.append('kernels/build/')
import agemm 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


LAYER_MSE_RECORDS = {}

def reset_mse_records():
    global LAYER_MSE_RECORDS
    LAYER_MSE_RECORDS = {}

@torch.no_grad()
def measure_and_record_mse(x, reorder_index, select_num, layer_idx):
    if layer_idx in LAYER_MSE_RECORDS:
        return

    x = x.float()
    index = reorder_index.to(torch.int32)
    
    qX_nvfp4 = quantize_nvfp4_tensor(x, group_size=16)
    err_nvfp4 = x - qX_nvfp4
    mse_nvfp4 = torch.mean(err_nvfp4 ** 2).item()
    
    try:
        x_had = hadamard_transform(x)
        qX_had_nvfp4 = quantize_nvfp4_tensor(x_had, group_size=16)
        err_had = x_had - qX_had_nvfp4
        mse_had = torch.mean(err_had ** 2).item()
    except:
        mse_had = mse_nvfp4 * 0.9 

    x_reordered = torch.index_select(x, 1, index)
    qx, scale_x, scale = fake_reorder_quantize_x(x_reordered, torch.arange(x.shape[1], device=x.device), select_num)
    if select_num > 0:
        qx[:, -2*select_num:-select_num] += qx[:, -select_num:]
        tensorE_final = x_reordered - qx[:, :-select_num]
    else:
        tensorE_final = x_reordered - qx
        
    mse_arc = torch.mean(tensorE_final ** 2).item()
    
    # 记录
    LAYER_MSE_RECORDS[layer_idx] = {
        'NVFP4': mse_nvfp4,
        'Hadamard': mse_had,
        'ARCQuant': mse_arc
    }

def plot_mse_evolution(save_path="mse_evolution.png"):
    if not LAYER_MSE_RECORDS:
        print("No MSE records found.")
        return

    sorted_layers = sorted(LAYER_MSE_RECORDS.keys())
    nvfp4_list = [LAYER_MSE_RECORDS[i]['NVFP4'] for i in sorted_layers]
    had_list = [LAYER_MSE_RECORDS[i]['Hadamard'] for i in sorted_layers]
    arc_list = [LAYER_MSE_RECORDS[i]['ARCQuant'] for i in sorted_layers]

    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    color_nvfp4 = '#404040'   # Deep Gray
    color_had = '#D35400'     # Deep Orange / Burnt Orange
    color_arc = '#8B0000'     # Deep Red
    
    ax.plot(sorted_layers, nvfp4_list, marker='o', markersize=4, linestyle='--', 
            color=color_nvfp4, label='NVFP4', linewidth=1.5, alpha=0.8)
    ax.plot(sorted_layers, had_list, marker='s', markersize=4, linestyle='-.', 
            color=color_had, label='Hadamard + NVFP4', linewidth=1.5, alpha=0.8)
    ax.plot(sorted_layers, arc_list, marker='^', markersize=5, linestyle='-', 
            color=color_arc, label='ARCQuant (Ours)', linewidth=2.0) 
    
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Mean Squared Error (MSE)', fontweight='bold')
    ax.set_title('Activation Quantization MSE Across Layers', fontweight='bold', pad=12)
    
    ax.grid(True, which='major', linestyle='--', alpha=0.4)
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"MSE Evolution plot saved to {save_path}")



def plot_sub_bar(ax, tensor_data, color, title=None, mse=None, y_lim=None, background_data=None, bg_color=None):
    values = tensor_data.cpu().numpy()
    channels = range(len(values))
    
    if background_data is not None and bg_color is not None:
        bg_values = background_data.cpu().numpy()
        ax.bar(channels, bg_values, color=bg_color, width=1.0, alpha=0.6, label='Pre-Compensation')
        
    ax.bar(channels, values, color=color, width=1.0, label='Final')
    
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    
    if mse is not None:
        ax.text(0.95, 0.92, f"MSE: {mse:.2e}", transform=ax.transAxes, 
                ha='right', va='top', fontsize=10, family='monospace', fontweight='bold',
                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", alpha=0.8, lw=0.8))
        
    if y_lim:
        ax.set_ylim(0, y_lim)
        
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.set_xticks([]) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def display(X, reorder_index, select_num, file_path):
    if os.path.exists(file_path + "_comparison.png"):
        return

    x = X.float()
    index = reorder_index.to(torch.int32)
    
    color_act = '#003366'       # Navy Blue 
    color_err = '#800000'       # Burgundy 
    color_err_light = '#E6B0AA' # Pale Red 
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), dpi=150)
    plt.subplots_adjust(wspace=0.15, hspace=0.25) 
    
    cols = ["NVFP4", "Hadamard + NVFP4", "ARCQuant"]


    # =========================================================
    # 1. NVFP4 (Standard)
    # =========================================================
    # Activation
    act_1 = torch.max(torch.abs(x), dim=0)[0] 
    
    # Error
    qX_nvfp4 = quantize_nvfp4_tensor(x, group_size=16)
    err_tensor_1 = x - qX_nvfp4
    err_1 = torch.linalg.norm(err_tensor_1, ord=2, dim=0)
    mse_1 = torch.mean(err_tensor_1 ** 2).item()

    # =========================================================
    # 2. Hadamard + NVFP4
    # =========================================================
    # Activation (Transformed)
    x_had = hadamard_transform(x)
    act_2 = torch.max(torch.abs(x_had), dim=0)[0]
    
    # Error (Transformed Domain)
    qX_had_nvfp4 = quantize_nvfp4_tensor(x_had, group_size=16)
    err_tensor_2 = x_had - qX_had_nvfp4
    err_2 = torch.linalg.norm(err_tensor_2, ord=2, dim=0)
    mse_2 = torch.mean(err_tensor_2 ** 2).item()

    # =========================================================
    # 3. ARCQuant (With Compensation Visualization)
    # =========================================================
    # Activation (Reordered)
    x_reordered = torch.index_select(x, 1, index)
    act_3 = torch.max(torch.abs(x_reordered), dim=0)[0]
    
    # Error Calculation logic
    qx, scale_x, scale = fake_reorder_quantize_x(x_reordered, torch.arange(x.shape[1]), select_num)
    
    tensorE_pre = x_reordered - qx[:, :-select_num]
    err_pre_comp = torch.linalg.norm(tensorE_pre, ord=2, dim=0).float().cpu()
    
    qx[:, -2*select_num:-select_num] += qx[:, -select_num:]
    
    tensorE_final = x_reordered - qx[:, :-select_num]
    err_3 = torch.linalg.norm(tensorE_final, ord=2, dim=0).float().cpu()
    mse_3 = torch.mean(tensorE_final ** 2).item()

    ymax_act = max(act_1.max(), act_2.max(), act_3.max()).item() * 1.15
    
    ymax_err = max(err_1.max(), err_2.max(), err_pre_comp.max()).item() * 1.15


    ymax_err, ymax_act = ymax_err * 0.75, ymax_act * 0.75
    # --- Row 1: Activations ---
    # Col 1: NVFP4
    plot_sub_bar(axes[0, 0], act_1, color_act, title="NVFP4\nActivation Dist.", y_lim=ymax_act)
    axes[0, 0].set_ylabel("Activation (Max)", fontsize=12, fontweight='bold')
    
    # Col 2: Hadamard
    plot_sub_bar(axes[0, 1], act_2, color_act, title="Hadamard + NVFP4\nActivation Dist.", y_lim=ymax_act)
    
    # Col 3: ARCQuant
    plot_sub_bar(axes[0, 2], act_3, color_act, title="ARCQuant\nActivation Dist.", y_lim=ymax_act)

    # --- Row 2: Quantization Errors ---
    # Col 1: NVFP4
    plot_sub_bar(axes[1, 0], err_1, color_err, title=None, mse=mse_1, y_lim=ymax_err)
    axes[1, 0].set_ylabel("Error (L2 Norm)", fontsize=12, fontweight='bold')
    
    # Col 2: Hadamard
    plot_sub_bar(axes[1, 1], err_2, color_err, title=None, mse=mse_2, y_lim=ymax_err)
    
    # Col 3: ARCQuant 
    plot_sub_bar(axes[1, 2], err_3, color_err, title=None, mse=mse_3, y_lim=ymax_err,
                 background_data=err_pre_comp, bg_color=color_err_light)
    
    axes[1, 2].legend(fontsize=9, loc='upper left', frameon=False)

    save_name = file_path + "_comparison.png"
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
    print(f"saved: {save_name}")