import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, List
import re
import time
import gc
import scipy.stats as stats

from reconstruct_moe_modeling import *
from gptq_utils import GPTQ, Quantizer, find_layers


DEV = torch.device('cuda:0')

def dp_pareto_step_fit(x, n, step_size=32):
    ori_len = len(x)
    x = [np.mean(x[i:min(i+step_size, len(x))]) for i in range(0, len(x), step_size)]
    n_points = len(x)
    if n_points <= n:
        return x, [i for i in range(n)], x
    
    squared_errors = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(i, n_points):
            segment = x[i:j+1]
            mean_val = np.mean(segment)
            squared_errors[i, j] = np.sum((segment - mean_val) ** 2)
    
    dp = np.full((n_points + 1, n + 1), float('inf'))
    split_points = np.full((n_points + 1, n + 1), -1, dtype=int)
    
    dp[0][0] = 0
    
    for i in range(1, n_points + 1):
        for k in range(1, min(i, n) + 1):
            for j in range(k - 1, i):
                error = dp[j][k-1] + squared_errors[j][i-1]
                if error < dp[i][k]:
                    dp[i][k] = error
                    split_points[i][k] = j
    
    splits = []
    i, k = n_points, n
    while k > 0:
        j = split_points[i][k]
        splits.append(j)
        i, k = j, k - 1
    
    splits.reverse()

    fit_data = []
    for i in splits:
        fit_data.append(x[i])

    split_points = [i * step_size for i in splits]
    # print(splits, split_points)

    split_points.append(n_points * step_size)
    fit_line = []
    for i in range(len(fit_data)):
        for j in range(split_points[i], split_points[i + 1]):
            fit_line.append(fit_data[i])

    return fit_data, split_points, fit_line[:ori_len]


@torch.no_grad()
def analyze_neuron_activations(act_fn, inps, gate_proj_weights, up_proj_weights, save_path: Optional[str] = None, sparsity = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    h = act_fn(F.linear(F.normalize(inps, p=2, dim=-1), F.normalize(gate_proj_weights, p=2, dim=1)))
    scores = h * F.linear(F.normalize(inps, p=2, dim=-1), F.normalize(up_proj_weights, p=2, dim=1))

    K = max(1, math.ceil(scores.shape[-1] * sparsity))  # Top 10% neurons 
    # print("K (top neurons per sample):", scores.shape, K)

    scores = scores.detach().cpu()
    batch_size, seq_len, inter_size = scores.shape
    total_samples = batch_size * seq_len

    flat_states = scores.reshape(-1, inter_size)
    activation_markers = torch.zeros_like(flat_states)
    # activation_values = torch.zeros_like(flat_states)
    
    for i in range(total_samples):
        sample_values = flat_states[i]
        abs_values = sample_values.abs().float()

        # Get indices of top-k absolute values
        top_values, top_indices = torch.topk(abs_values, k=K)
        activation_markers[i, top_indices] = 1.0
        # for idx in top_indices:
        #     activation_values[i, idx] = abs_values[idx]

    # Sum up activations across all samples
    activation_counts = activation_markers.sum(dim=0)
    # activation_values = activation_values.sum(dim=0)
    activation_rates = activation_counts / total_samples

    if save_path:
        plt.figure(figsize=(10, 10))
        
        # Plot 1: Activation rates histogram
        plt.subplot(3, 1, 1)  
        plt.hist(activation_rates.detach().to(dtype=torch.float32).numpy(), bins=500, edgecolor='black')
        plt.title('Distribution of Neuron Activation Rates')
        plt.xlabel('Activation Rate')
        plt.ylabel('Number of Neurons')
        plt.grid(True, alpha=0.3)
        
        # Add statistics for rates
        mean_rate = activation_rates.mean()
        std_rate = activation_rates.std()
        stats_text = (f'Mean rate: {mean_rate:.3f}\n'
                     f'Std rate: {std_rate:.3f}\n'
                     f'Max rate: {activation_rates.max():.3f}\n'
                     f'Min rate: {activation_rates.min():.3f}')
        
        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 2: Neuron indices vs Activation counts
        plt.subplot(3, 1, 2)  
        neuron_indices = np.arange(inter_size)
        stats_text = (f'Mean count: {activation_counts.mean():.3f}\n'
                     f'Std count: {activation_counts.std():.3f}\n'
                     f'Max count: {activation_counts.max():.3f}\n'
                     f'Min count: {activation_counts.min():.3f}')
        
        activation_counts = activation_counts.detach().to(dtype=torch.float32).numpy()
        activation_counts = sorted(activation_counts, reverse=True)

        plt.plot(neuron_indices, activation_counts, 'b-', alpha=0.6)
        plt.title('Activation Counts per Neuron Index')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Count')

        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: Neuron indices vs Activation values
        plt.subplot(3, 1, 3)  
        neuron_indices = np.arange(inter_size)
        stats_text = (f'Mean value: {activation_values.mean():.3f}\n'
                     f'Std value: {activation_values.std():.3f}\n'
                     f'Max value: {activation_values.max():.3f}\n'
                     f'Min value: {activation_values.min():.3f}')
        
        activation_values = activation_values.detach().to(dtype=torch.float32).numpy()
        activation_values = sorted(activation_values, reverse=True)

        plt.plot(neuron_indices, activation_values, 'b-', alpha=0.6)
        plt.title('Activation Values per Neuron Index')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Value')

        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
    
    return activation_rates
    # return activation_counts, activation_values, activation_markers

@torch.no_grad()
def construct_unequal_by_rates(origin_rates, num_experts):

    rates = origin_rates.detach().cpu().numpy()
    rates = sorted(rates, reverse=True)
    fit_data, quantiles, fit_line = dp_pareto_step_fit(np.log(rates), num_experts * 2)
    print("quantiles:", quantiles)
    quantiles = quantiles[:2] + quantiles[-2:]
    print("quantiles:", quantiles)
    assert quantiles[0] == 0 and quantiles[-1] == origin_rates.shape[0] and len(quantiles) == num_experts + 1

    hidden_size = origin_rates.shape[0]

    expert_groups = []
    rates = origin_rates.float()

    _, top_indices = torch.topk(rates, hidden_size)
    for i in range(num_experts):
        expert_indices = top_indices[quantiles[i]:quantiles[i+1]].tolist()
        expert_groups.append(expert_indices)
    
    return expert_groups, None

@torch.no_grad()
def construct_experts_by_rates(
    origin_rates,
    num_experts,
    num_shared_experts):

    # print("origin_rates:", origin_rates.shape)
    hidden_size = origin_rates.shape[0]
    neurons_per_expert = hidden_size // num_experts

    expert_groups = []
    rates = origin_rates.float()
    # markers = activation_markers.float()

    expert_groups.append([])
    _, top_indices = torch.topk(rates, hidden_size)
    for i in range(num_experts):
        expert_indices = top_indices[i*neurons_per_expert:(i+1)*neurons_per_expert].tolist()
        expert_groups.append(expert_indices)
    
    return expert_groups, None

@torch.no_grad()
def lowrank_compress_svd(weight_matrix, lowrank_sparsity, save_path=None):
    U, S, Vh = torch.linalg.svd(weight_matrix.float(), full_matrices=False)

    if lowrank_sparsity is not None:
        rank = int(weight_matrix.shape[1] * (1 - lowrank_sparsity))
    else:
        S_sum = S.sum()
        cum = torch.cumsum(S, 0)
        rank = torch.searchsorted(cum, 0.99 * S_sum).item() + 1

    ratio = S[:rank].sum() / S.sum()
    # print(f"Rank: {rank}, Ratio: {ratio}")
    
    U_reduced = U[:, :]
    S_reduced = S[:]
    Vh_reduced = Vh[:, :rank]
    # print(U_reduced.shape, S_reduced.shape, Vh_reduced.shape)
    
    low_rank_matrix = torch.mm(U_reduced, torch.mm(torch.diag(S_reduced), Vh_reduced))
    # print(weight_matrix.shape, rank, low_rank_matrix.shape)

    if save_path:
        plt.figure(figsize=(12, 8))
        
        neuron_indices = np.arange(S.shape[0])
        stats_text = ()
        
        S_cpu = S.cpu().numpy()
        plt.plot(neuron_indices, S_cpu, 'b-', alpha=0.6)
        plt.title(f'SVD decomposition of weight matrix, rank={rank}')
        plt.xlabel('Neuron Index')
        plt.ylabel('Weight Value')

        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
       
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
    
    return low_rank_matrix.to(weight_matrix.dtype)

@torch.no_grad()
def analyze_rates(model, layer_idx, up_proj_rates, new_rates, gate_proj_rates, down_proj_rates, new_expert_sizes, rank_indices_tensor, save_path=None):
    
    new_up_proj_rates = up_proj_rates[0][rank_indices_tensor]
    new_gate_proj_rates = gate_proj_rates[0][rank_indices_tensor]
    new_down_proj_rates = down_proj_rates[0][rank_indices_tensor]
    start_id = 0
    ret0 = []
    ret1 = []
    ret2 = []
    for ii, new_expert_size in enumerate(new_expert_sizes):
        new_intermediate_size = int(new_expert_size * up_proj_rates[0].shape[0])
        ret0.append(new_up_proj_rates[start_id:start_id+new_intermediate_size].mean().detach().cpu().numpy().item())
        ret1.append(new_gate_proj_rates[start_id:start_id+new_intermediate_size].mean().detach().cpu().numpy().item())
        ret2.append(new_down_proj_rates[start_id:start_id+new_intermediate_size].mean().detach().cpu().numpy().item())
        start_id += new_intermediate_size

    # 
    if save_path:
        plt.figure(figsize=(10, 10))
        # plt_set = [rates, up_proj_loss, gate_proj_loss]
        plt_set = [new_rates, new_up_proj_rates, new_gate_proj_rates, new_down_proj_rates]
        for i, pp in enumerate(plt_set):
            plt.subplot(len(plt_set), 1, i + 1)
            pps = pp.detach().cpu().to(dtype=torch.float32).numpy()
            # pps = sorted(pps, reverse=True)

            neuron_indices = np.arange(pp.shape[0])
            plt.plot(neuron_indices, pps, 'b-', alpha=0.6)
            plt.title('Distribution of Neuron Loss Rates')
            plt.xlabel('Neuron Index')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            mean_rate = pps.mean()
            std_rate = pps.std()
            stats_text = (f'Mean rate: {mean_rate:.3f}\n'
                        f'Std rate: {std_rate:.3f}\n'
                        f'Max rate: {pps.max():.3f}\n'
                        f'Min rate: {pps.min():.3f}')
            
            plt.text(0.95, 0.95, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
    
    return {'up_proj':ret0, 'gate_proj':ret1, 'down_proj':ret2}
    
@torch.no_grad()
def analyze_quant_outlier(layer, layer_idx, hidden_states, slice_num, n, wbits = 8, if_dense=True, filters = ['up_proj', 'gate_proj', 'down_proj'], save_path=None, args=None):
    nsample = hidden_states.shape[0]
    gptq = {}

    groupsize = 128
    act_order = True
    static_groups = False
    
    loss = {}
    
    for ff in filters:
        qmodule_all = find_layers(layer, filters=[ff])
        qbatch = 64

        for qmi in range(0, len(qmodule_all.keys()), qbatch):
            tick0 = time.time()   

            qmodule = {k: qmodule_all[k] for k in list(qmodule_all.keys())[qmi: qmi + qbatch]}
            if len(qmodule.keys()) == 0:
                continue
            for name in qmodule.keys():
                split_name = name.split('.')[-1]
                gptq[name] = GPTQ(qmodule[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(wbits, perchannel=True, sym=False, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in qmodule.keys():
                handles.append(qmodule[name].register_forward_hook(add_batch(name)))
            
            for isample in range(nsample):
                if split_name in filters:
                    ffn_sample = hidden_states[isample].unsqueeze(0)
                    layer.mlp(ffn_sample)
                else:
                    assert False, f"Not quantize {name}"

            for handle in handles:
                handle.remove()
            
            avg_losses = []
            max_losses = []
            for name in qmodule.keys():
                loss[name] = gptq[name].fasterquant(name=f"layer_idx.{layer_idx}."+name, groupsize=groupsize, actorder=act_order, static_groups=static_groups, update=False)
                avg_losses.append(loss[name].sum(dim=1).mean().detach().cpu().numpy().item())
                max_losses.append(loss[name].sum(dim=1).max().detach().cpu().numpy().item())
                gptq[name].free()
                del gptq[name]
            
            tick1 = time.time()
            # print(f"Simulate quant to find outliers, layer {layer_idx} {ff} {qmi}:{qmi + min(qbatch, len(qmodule.keys()))} bits: {wbits} time: {tick1 - tick0} avg_loss: {avg_losses} max_loss: {max_losses}")
            print(f"Simulate quant to find outliers, layer {layer_idx} {ff} {qmi}:{qmi + min(qbatch, len(qmodule.keys()))} bits: {wbits} time: {tick1 - tick0}")

        del qmodule_all
    
    # print(loss)
    all_rates = []
    if if_dense:
        assert n == 1, "dense model n == 1"
    for expert_idx in range(n):
        if n == 1:
            u = f'mlp.up_proj'
            g = f'mlp.gate_proj'
            d = f'mlp.down_proj'
        else:
            u = f'mlp.experts.{expert_idx}.up_proj'
            g = f'mlp.experts.{expert_idx}.gate_proj'
            d = f'mlp.experts.{expert_idx}.down_proj'
        
        if d.split('.')[-1] in filters:
            return_loss = torch.sum(loss[d], dim=0)
            # return_loss = torch.max(loss[d], dim=0)[0]
        if u.split('.')[-1] in filters:
            return_loss = torch.sum(loss[u], dim=1)
            # return_loss = torch.max(loss[u], dim=1)[0]
        if g.split('.')[-1] in filters:
            return_loss = torch.sum(loss[g], dim=1)
            # return_loss = torch.max(loss[g], dim=1)[0]

        all_rates.append(return_loss)
        # rates = up_proj_loss / up_proj_loss.mean() + gate_proj_loss / gate_proj_loss.mean()
    # print(f"Layer {layer_idx}, neural loss rates: ", all_rates)

    if save_path:
        rates = all_rates[0]
        plt.figure(figsize=(10, 10))
        # plt_set = [rates, up_proj_loss, gate_proj_loss]
        plt_set = [rates]
        for i, pp in enumerate(plt_set):
            plt.subplot(len(plt_set), 1, i + 1)
            pps = pp.detach().cpu().to(dtype=torch.float32).numpy()
            # pps = sorted(pps, reverse=True)
            fit_data, split_points, fit_line = dp_pareto_step_fit(np.log(pps), slice_num)
            fit_line = np.exp(fit_line)

            neuron_indices = np.arange(pp.shape[0])
            plt.plot(neuron_indices, fit_line, 'r-', alpha=0.6)
            plt.plot(neuron_indices, pps, 'b-', alpha=0.6)
            plt.title('Distribution of Neuron Loss Rates')
            plt.xlabel('Neuron Index')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            mean_rate = rates.mean()
            std_rate = rates.std()
            stats_text = (f'Mean rate: {mean_rate:.3f}\n'
                        f'Std rate: {std_rate:.3f}\n'
                        f'Max rate: {rates.max():.3f}\n'
                        f'Min rate: {rates.min():.3f}')
            
            plt.text(0.95, 0.95, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
    
    return all_rates

@torch.no_grad()
def quant_layer_mix_precision(layer, layer_idx, quant_attn, slice_expert_num, 
                attn_hidden_states, ffn_hidden_states, attention_mask, position_ids, position_embeddings, 
                dyn_qscheme,
                args):
    print(f"Quantize layer {layer_idx}, slice_expert_num: {slice_expert_num}")
    nsample = attn_hidden_states.shape[0]
    assert attn_hidden_states.shape[0] == ffn_hidden_states.shape[0], f"attn_hidden_states.shape: {attn_hidden_states.shape}, ffn_hidden_states.shape: {ffn_hidden_states.shape}"

    gptq = {}
    groupsize = 128
    act_order = True
    static_groups = False
    sym = False
    ffn_filters = ['up_proj', 'gate_proj', 'down_proj']
    attn_filters = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'kv_a_proj_with_mqa', 'kv_b_proj']
    if quant_attn:
        filters = attn_filters + ffn_filters
    else:
        filters = ffn_filters

    qscheme_str = args.quant_scheme
    qscheme_attn = [8]
    qscheme_share = [4]
    qscheme_expert = {'up_proj':[2, 2, 2, 2, 2, 2, 2, 2], 
                      'gate_proj':[2, 2, 2, 2, 2, 2, 2, 2], 
                      'down_proj':[2, 2, 2, 2, 2, 2, 2, 2]}
    if qscheme_str is not None:
        try:
            # sample: "a8s4m3221", "a8s4m33222222"
            match = re.search(r'a(\d)s(\d)m(\d+)', qscheme_str)
            aa = match.group(1)
            ss = match.group(2)
            ee = match.group(3)
            qscheme_attn = [int(aa)]
            qscheme_share = {'up_proj': [int(ss)], 'gate_proj': [int(ss)], 'down_proj': [int(ss)]}
            qscheme_expert = {'up_proj':[int(e) for e in ee], 
                            'gate_proj':[int(e) for e in ee], 
                            'down_proj':[int(e) for e in ee]}
        except:
            print(f"Quant scheme {qscheme_str} is not valid.")
    if dyn_qscheme is not None:
        qscheme_share = dyn_qscheme['shared']
        qscheme_attn = dyn_qscheme['attn']
        qscheme_expert = dyn_qscheme['expert']

    loss = {}
    quanted_size = 0

    for ff in filters:
        qmodule_all = find_layers(layer, filters=[ff])
        qbatch = 64

        for qmi in range(0, len(qmodule_all.keys()), qbatch):
            tick0 = time.time()   

            qmodule = {k: qmodule_all[k] for k in list(qmodule_all.keys())[qmi: qmi + qbatch]}
            # print("Quant modules", ff, qmodule.keys())
            # print("Quant modules", ff, qmi)
            if len(qmodule.keys()) == 0:
                continue
            for name in qmodule.keys():
                split_name = name.split('.')[-1]
                gptq[name] = GPTQ(qmodule[name])
                gptq[name].quantizer = Quantizer()

                if split_name in attn_filters:
                    bit = qscheme_attn
                    gptq[name].quantizer.configure(bit[0], perchannel=True, sym=sym, mse=False)
                    quanted_size += qmodule[name].weight.numel() * (bit[0] + 32 / groupsize) ## sym = False is 32, sym = True has no zero point is 16
                elif split_name in ffn_filters:
                    match = re.search(r'mlp\.experts\.(\d+)', name)
                    expert_id = int(match.group(1)) if match else -1  ## shared expert id is -1
                    if expert_id == -1:
                        bit = qscheme_share[split_name]
                        gptq[name].quantizer.configure(bit[0], perchannel=True, sym=sym, mse=False)
                        quanted_size += qmodule[name].weight.numel() * (bit[0] + 32 / groupsize)
                    else:
                        bit = qscheme_expert[split_name]
                        eid = expert_id % slice_expert_num
                        gptq[name].quantizer.configure(bit[eid], perchannel=True, sym=sym, mse=False)
                        quanted_size += qmodule[name].weight.numel() * (bit[eid] + 32 / groupsize)
                else:
                    assert False, f"Some modules are not quantize {name}, check code!"

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in qmodule.keys():
                handles.append(qmodule[name].register_forward_hook(add_batch(name)))
            
            for isample in range(nsample):
                if split_name in attn_filters:
                    attn_sample = attn_hidden_states[isample].unsqueeze(0)
                    try:
                        layer.self_attn(
                            hidden_states=attn_sample, 
                            attention_mask=attention_mask, 
                            position_ids=position_ids,
                            position_embeddings=position_embeddings)
                    except:
                        layer.self_attn(
                            hidden_states=attn_sample, 
                            attention_mask=attention_mask, 
                            position_ids=position_ids)
                elif split_name in ffn_filters:
                    ffn_sample = ffn_hidden_states[isample].unsqueeze(0)
                    layer.mlp(ffn_sample)
                else:
                    assert False, f"Not quantize {name}"

            for handle in handles:
                handle.remove()
            avg_losses = []
            max_losses = []
            for name in qmodule.keys():
                loss[name] = gptq[name].fasterquant(name=f"layer_idx.{layer_idx}."+name, groupsize=groupsize, actorder=act_order, static_groups=static_groups)
                avg_losses.append(loss[name].sum(dim=1).mean().detach().cpu().numpy().item())
                max_losses.append(loss[name].sum(dim=1).max().detach().cpu().numpy().item())
                gptq[name].free()
                del gptq[name]
            
            tick1 = time.time()
            # print(f"Quantize layer {layer_idx} {ff} {qmi}:{qmi + min(qbatch, len(qmodule.keys()))} bits: {bit} time: {tick1 - tick0} avg_loss: {avg_losses} max_loss: {max_losses}")
            print(f"Quantize layer {layer_idx} {ff} {qmi}:{qmi + min(qbatch, len(qmodule.keys()))} bits: {bit} time: {tick1 - tick0}")

        del qmodule_all

    return quanted_size