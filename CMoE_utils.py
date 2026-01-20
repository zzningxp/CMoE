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

from CMoE_model import *

DEV = torch.device('cuda:0')

@torch.no_grad()
def analyze_neuron_activations(scores: torch.Tensor, save_path: Optional[str] = None, sparsity = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    K = max(1, math.ceil(scores.shape[-1] * sparsity))  # Top 10% neurons 
    # print("K (top neurons per sample):", scores.shape, K)

    scores = scores.detach().cpu()
    batch_size, seq_len, inter_size = scores.shape
    total_samples = batch_size * seq_len

    flat_states = scores.reshape(-1, inter_size)
    activation_markers = torch.zeros_like(flat_states)
    activation_values = torch.zeros_like(flat_states)
    
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
    activation_values = activation_values.sum(dim=0)
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
    
    return activation_counts, activation_rates, activation_markers
    # return activation_counts, activation_values, activation_markers

def analyze_expert_profile(layer, hidden_states, topk, save_path: Optional[str] = None):

    g = layer.mlp.gate(hidden_states)
    g = g.squeeze(0)
    g = g.softmax(dim=-1, dtype=torch.float32)
    # print(g.shape)
    # g = g.sort(dim=-1, descending=True)
    # print(g[0])
    # expert_activate_sum = g  # [batch_size * seqlen, num_experts]
    total_samples, expert_num = g.shape

    activation_markers = torch.zeros_like(g)
    for i in range(total_samples):
        top_values, top_indices = torch.topk(g[i], k=topk)
        activation_markers[i, top_indices] = 1.0

    activation_counts = activation_markers.sum(dim=0) 
    activation_rates = activation_counts / total_samples

    sorted_indices = torch.argsort(activation_counts, descending=True)
    ranks = torch.empty_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(expert_num, device=activation_counts.device)
    drop_expert_nums = torch.ones_like(ranks)
    drop_expert_nums[ranks < expert_num // 4] = 0
    drop_expert_nums[ranks >= 3 * expert_num // 4] = 2
    # print(drop_expert_nums, drop_expert_nums.sum())
    assert drop_expert_nums.sum() == expert_num
    # top 1/4 experts dropped none of neurons, bottom 1/4 experts dropped 2 slices neurons, other dropped 1 slice neurons
    # assert all the drop_expert_nums are equal to expert_num

    # 可视化
    if save_path:
        plt.figure(figsize=(12, 10))
        
        neuron_indices = np.arange(inter_size)
        stats_text = (f'Mean count: {activation_counts.mean():.3f}\n'
                     f'Std count: {activation_counts.std():.3f}\n'
                     f'Max count: {activation_counts.max():.3f}\n'
                     f'Min count: {activation_counts.min():.3f}')
        
        activation_counts_sorted = sorted(activation_counts.cpu().numpy(), reverse=True)

        plt.plot(neuron_indices, activation_counts_sorted, 'b-', alpha=0.6)
        plt.title('Activation Counts per Neuron Index')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Count')

        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
       
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()

    # print(activation_counts, ranks, drop_expert_nums)
    return activation_counts, ranks, drop_expert_nums

@torch.no_grad()
def construct_experts_k_means(
    activation_rates,
    activation_markers,
    num_experts,
    num_shared_experts):

    hidden_size = activation_rates.shape[0]
    neurons_per_expert = hidden_size // num_experts

    expert_groups = []
    remaining_indices = set(range(hidden_size))
    # print("activation_rates:", activation_rates, sum(activation_rates), activation_rates.shape)
    rates = activation_rates.float()
    # print("rates:", rates, sum(rates), rates.shape)

    # print("activation_markers:", activation_markers, sum(sum(activation_markers)), activation_markers.shape)
    markers = activation_markers.float()
    # print("markers:", markers, sum(sum(markers)), markers.shape)

    _, top_indices = torch.topk(rates, neurons_per_expert * num_shared_experts)
    # print("Top shared expert indices:", top_indices)
    # print("Top shared expert rates:", rates[top_indices])
    shared_expert_indices = top_indices.tolist()
    expert_groups.append(shared_expert_indices)
    remaining_indices -= set(shared_expert_indices)
    remaining_indices_list = list(remaining_indices)

    # print(hidden_size, neurons_per_expert, num_shared_experts, num_experts)
    # print("Shared expert neurons, Remain expert neurons:", len(shared_expert_indices), len(remaining_indices_list))
    #start_k_means

    n_samples = len(remaining_indices_list)
    k = num_experts - num_shared_experts
    cluster_size = neurons_per_expert

    remaining_rates = rates[remaining_indices_list]
    _, top_indices = torch.topk(remaining_rates, k)
    # print("Remaining expert rates:", remaining_rates)
    # print("Top k Remaining expert rates:", k, remaining_rates[top_indices])

    selected_index = [remaining_indices_list[idx] for idx in top_indices]

    # print(len(selected_index), markers.shape, k)
    centroids = markers[:, selected_index].clone()

    max_iters = 10
    prev_cost = -1

    import lap

    for iteration in range(max_iters):
        
        distances = torch.cdist(markers[:, remaining_indices_list].T, centroids.T, p=1) 
        distances_np = distances.numpy()
        repeated_distances = np.zeros((n_samples, n_samples))
        # print("markers:", markers, sum(sum(markers)), markers.shape, markers[:, 0], sum(markers[:, 0]))
        # print("centroids:", centroids, sum(sum(centroids)), centroids.shape, centroids[:, 0], sum(centroids[:, 0]))
        # print("distances:", distances, distances.shape, sum(sum(distances)))
        # print("LAPJV input shape 0:", distances.shape, repeated_distances.shape, n_samples, torch.argmin(distances))

        for i in range(k):
            repeated_distances[:, i*cluster_size:(i+1)*cluster_size] = distances_np[:, i:i+1]
        cost, row_ind, col_ind = lap.lapjv(repeated_distances)

        assignments = torch.tensor(row_ind // cluster_size)
        # print("Assignments shape, row_ind, col_ind, cluster_size, cost:", assignments.shape, row_ind, col_ind, cluster_size, cost)

        if prev_cost is not None and abs(cost - prev_cost) / cost < 1e-3:
            # print("Converged after", iteration + 1, "iterations")
            break
        
        prev_cost = cost
        
        for i in range(k):
            cluster_points = markers[:, remaining_indices_list][:, assignments == i]
            if cluster_points.size(1) > 0:
                centroids[:, i] = cluster_points.mean(dim=1)
    
    # print("Cluster points shape:", cluster_points.shape)
    representative_indices = []
    
    for i in range(k):
        cluster_mask = assignments == i
        cluster_points = markers[:, remaining_indices_list][:, cluster_mask] 
        cluster_indices = torch.where(cluster_mask)[0]
        
        distances = torch.cdist(centroids[:, i:i+1].T, cluster_points.T, p=1)
        
        closest_idx_in_cluster = torch.argmin(distances)
        original_idx = cluster_indices[closest_idx_in_cluster]
        ori_ori_idx = remaining_indices_list[original_idx]
        
        representative_indices.append(ori_ori_idx)
        group_indices = [remaining_indices_list[idx] for idx in cluster_indices]
        expert_groups.append(group_indices)
    
    return expert_groups, representative_indices


@torch.no_grad()
def construct_moe(model, layer, layer_idx, inp, attention_mask, position_ids, position_embeddings, n_experts, n_activated, n_shared, args):

    device = next(layer.parameters()).device
    
    # Forward attention
    inp = inp.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    if position_ids is not None:
        position_ids = position_ids.to(device)
    
    # if position_embeddings is not None:
    #     position_embeddings = position_embeddings.to(device)

    residual = inp
    hidden_states = layer.input_layernorm(inp)
    hidden_states = layer.self_attn(
        hidden_states=hidden_states, 
        attention_mask=attention_mask, 
        position_ids=position_ids,
        position_embeddings=position_embeddings)[0]
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)

    # Forward FFN, normalize FFN input and weights to compute normalized h
    h_pre = hidden_states
    h = layer.mlp.act_fn(F.linear(F.normalize(hidden_states, p=2, dim=-1), F.normalize(layer.mlp.gate_proj.weight, p=2, dim=1)))
    h = h * F.linear(F.normalize(hidden_states, p=2, dim=-1), F.normalize(layer.mlp.up_proj.weight, p=2, dim=1))

    scores_all = h.to('cpu')

    # Calculate activation markers, activation rates
    # print(scores_all.shape)
    counts, rates, markers = analyze_neuron_activations(scores_all)
    # counts, rates, markers = analyze_neuron_activations(scores_all, save_path=f'./plot/scores_analysis_{layer.self_attn.layer_idx}.png')

    # 可视化激活率
    # plot_activation_rates(counts, rates, save_path=f'./plot/activation_rates_{layer.self_attn.layer_idx}.png')
    
    # Construct shared and routed experts
    expert_groups, representative_indices = construct_experts_k_means(
        rates,
        markers,
        num_experts = n_experts,
        num_shared_experts = n_shared,
    )

    # print(n_experts, n_shared, n_activated, len(expert_groups), len(representative_indices), len(rates))
    # for i, group in enumerate(expert_groups):
    #     print(f"Expert {i}: {len(group)} neurons")
    # for i, idx in enumerate(representative_indices):
    #     print(f"Representative neuron for Expert {i+1}: Index {idx})")

    experts = nn.ModuleList()
    for expert_indices in expert_groups:
        expert_mlp = LlamaMLP(layer.mlp.hidden_size, len(expert_indices)).to('cpu')
        
        # Initialize weights from the corresponding neurons in original projections
        with torch.no_grad():
            expert_mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[expert_indices, :]
            expert_mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[expert_indices, :]
            expert_mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:, expert_indices]
        
        experts.append(expert_mlp)

    
    # Router
    router = Router(layer.mlp.hidden_size, n_experts-n_shared, n_activated, bias_speed = args.bias_speed)

    core_neurons = representative_indices
    core_weights = layer.mlp.up_proj.weight.data[core_neurons, :]
    core_gate_weights = layer.mlp.gate_proj.weight.data[core_neurons, :]
    router.classifier.weight.data = F.normalize(core_weights, p=2, dim=1)
    router.gate.weight.data = F.normalize(core_gate_weights, p=2, dim=1)

    # MoE
    moe = MoE(layer.mlp.hidden_size, layer.mlp.intermediate_size//n_experts, n_experts, n_shared, n_activated)
    moe.gate = router
    moe.experts = experts[1:]
    moe.shared_experts = experts[0]
    moe.cus_training = False
    moe_out = moe(h_pre) + residual

    layer.mlp = moe

    del h, h_pre, scores_all, residual, hidden_states, 

    return moe_out


def construct_experts_by_rates(
    activation_rates,
    activation_markers,
    num_experts,
    num_shared_experts):

    hidden_size = activation_rates.shape[0]
    neurons_per_expert = hidden_size // num_experts

    expert_groups = []
    rates = activation_rates.float()
    markers = activation_markers.float()

    expert_groups.append([])
    _, top_indices = torch.topk(rates, hidden_size)
    for i in range(num_experts):        
        expert_indices = top_indices[i*neurons_per_expert:(i+1)*neurons_per_expert].tolist()
        expert_groups.append(expert_indices)
    
    return expert_groups, None

def reconstruct_moe_harddrop(model, layer, hidden_states, n_experts, n_activated, slice_expert_num, if_quantized, device, args):

    ori_expert_num = len(layer.mlp.experts)
    drop_expert_nums = torch.ones(ori_expert_num, dtype=torch.int)

    drop_expert_nums = drop_expert_nums.to(device)

    avg_drop_expert_num = 1
    new_expert_num = ori_expert_num * (slice_expert_num - avg_drop_expert_num) 
    scaling_factor = slice_expert_num - avg_drop_expert_num
    sparsity = 0.1

    if if_quantized:
        ori_router_gate = layer.mlp.gate.compressor.decompress_module(layer.mlp.gate)
    else:
        ori_router_gate = layer.mlp.gate.weight
    new_router = nn.Linear(model.config.hidden_size, new_expert_num, dtype=ori_router_gate.dtype, bias=False).to(device)
    all_new_experts = nn.ModuleList()

    total_neurons_processed = 0
    gate_start_idx = 0

    for expert_idx, expert in enumerate(layer.mlp.experts):
        if if_quantized:
            ori_gate_proj_weights = expert.gate_proj.compressor.decompress_module(expert.gate_proj)
            ori_up_proj_weights = expert.up_proj.compressor.decompress_module(expert.up_proj)
            ori_down_proj_weights = expert.down_proj.compressor.decompress_module(expert.down_proj)
        else:
            ori_gate_proj_weights = expert.gate_proj.weight
            ori_up_proj_weights = expert.up_proj.weight
            ori_down_proj_weights = expert.down_proj.weight

        # print(f"\nProcessing original expert {expert_idx} / {ori_expert_num}")
        # Get scores for this specific expert
        h = expert.act_fn(F.linear(F.normalize(hidden_states, p=2, dim=-1), F.normalize(ori_gate_proj_weights, p=2, dim=1)))
        h = h * F.linear(F.normalize(hidden_states, p=2, dim=-1), F.normalize(ori_up_proj_weights, p=2, dim=1))
        expert_scores = h.to('cpu')
        
        # print(f"Expert {expert_idx} scores shape: {expert_scores.shape}")
        # Calculate activation markers and rates for this expert's neurons
        counts, rates, markers = analyze_neuron_activations(expert_scores, sparsity=sparsity)
        # counts, rates, markers = analyze_neuron_activations(expert_scores, save_path=f'./plot/scores_analysis_{layer.self_attn.layer_idx}_{expert_idx}.png', sparsity=sparsity)

        expert_groups, representative_indices = construct_experts_by_rates(
            rates,
            markers,
            num_experts = slice_expert_num,
            num_shared_experts = 0,
        )

        drop_expert_num = drop_expert_nums[expert_idx] * avg_drop_expert_num
        remain_expert_num = slice_expert_num - drop_expert_num

        expert_groups = expert_groups[1:remain_expert_num + 1]
        # Create new experts for this original expert
        for ii, group_indices in enumerate(expert_groups):
            expert_mlp = expert.__class__(model.config).to(device)
            
            with torch.no_grad():
                gate_proj_weights = []
                up_proj_weights = []
                down_proj_weights = []

                for global_idx in group_indices:
                    gate_proj_weights.append(ori_gate_proj_weights[global_idx, :])
                    up_proj_weights.append(ori_up_proj_weights[global_idx, :])
                    down_proj_weights.append(ori_down_proj_weights[:, global_idx] * scaling_factor)

                expert_mlp.gate_proj.weight.data = torch.stack(gate_proj_weights)
                expert_mlp.up_proj.weight.data = torch.stack(up_proj_weights)
                expert_mlp.down_proj.weight.data = torch.stack(down_proj_weights, dim=1)

            all_new_experts.append(expert_mlp)
            total_neurons_processed += len(group_indices)

        expanded_gate = ori_router_gate.data[expert_idx, :].unsqueeze(0).repeat(remain_expert_num, 1).to(device)

        new_router.weight.data[gate_start_idx: gate_start_idx + remain_expert_num, :] = expanded_gate
        gate_start_idx += remain_expert_num

    model.config.intermediate_size = all_new_experts[0].up_proj.weight.shape[0]
    model.config.num_experts = new_expert_num
    model.config.num_experts_per_tok = n_activated
    print(model.config.intermediate_size, model.config.num_experts, model.config.num_experts_per_tok)

    moe = layer.mlp
    moe.num_experts = len(all_new_experts)
    moe.top_k = n_activated
    moe.gate = new_router
    moe.experts = all_new_experts

    return moe

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

def reconstruct_moe(model, layer, inps, n_experts, n_activated, slice_expert_num, device, args):

    ori_expert_num = len(layer.mlp.experts)
    new_expert_num = ori_expert_num * slice_expert_num 
    scaling_factor = slice_expert_num

    model.config.intermediate_size = model.config.intermediate_size * slice_expert_num
    model.config.num_experts = new_expert_num
    model.config.num_experts_per_tok = n_activated

    ori_router_gate = layer.mlp.gate.weight
    # new_router = nn.Linear(model.config.hidden_size, new_expert_num, dtype=ori_router_gate.dtype, bias=False).to(device)
    new_router = layer.mlp.gate.__class__(model.config).to(device).to(layer.mlp.gate.weight.dtype)
    # print(new_router)
    all_new_experts = nn.ModuleList()

    total_neurons_processed = 0
    gate_start_idx = 0

    for expert_idx, expert in enumerate(layer.mlp.experts):
        ori_gate_proj_weights = expert.gate_proj.weight
        ori_up_proj_weights = expert.up_proj.weight
        ori_down_proj_weights = expert.down_proj.weight

        # print(f"\nProcessing original expert {expert_idx} / {ori_expert_num}")
        h = expert.act_fn(F.linear(F.normalize(inps, p=2, dim=-1), F.normalize(ori_gate_proj_weights, p=2, dim=1)))
        h = h * F.linear(F.normalize(inps, p=2, dim=-1), F.normalize(ori_up_proj_weights, p=2, dim=1))
        expert_scores = h.to('cpu')
        
        analyze_sparsity = 0.1
        counts, rates, markers = analyze_neuron_activations(expert_scores, sparsity=analyze_sparsity)

        expert_groups, representative_indices = construct_experts_by_rates(
            rates,
            markers,
            num_experts = slice_expert_num,
            num_shared_experts = 0,
        )

        lowrank_sparsity = 0
        expert_groups = expert_groups[1:]
        # Create new experts for this original expert
        for ii, group_indices in enumerate(expert_groups):
            n_neurons = len(group_indices)
            expert_mlp = expert.__class__(model.config).to(device)
            
            with torch.no_grad():
                gate_proj_weights = []
                up_proj_weights = []
                down_proj_weights = []

                for global_idx in group_indices:
                    gate_proj_weights.append(ori_gate_proj_weights[global_idx, :])
                    up_proj_weights.append(ori_up_proj_weights[global_idx, :])
                    down_proj_weights.append(ori_down_proj_weights[:, global_idx] * scaling_factor)

                # gate_proj_weights_t = torch.stack(gate_proj_weights).T
                # expert_mlp.gate_proj.weight.data = lowrank_compress_svd(gate_proj_weights_t, lowrank_sparsity).T
                # up_proj_weights_t = torch.stack(up_proj_weights).T
                # expert_mlp.up_proj.weight.data = lowrank_compress_svd(up_proj_weights_t, lowrank_sparsity).T
                # down_proj_weights = torch.stack(down_proj_weights, dim=1)
                # expert_mlp.down_proj.weight.data = lowrank_compress_svd(down_proj_weights, lowrank_sparsity)

                expert_mlp.gate_proj.weight.data = torch.stack(gate_proj_weights)
                expert_mlp.up_proj.weight.data = torch.stack(up_proj_weights)
                expert_mlp.down_proj.weight.data = torch.stack(down_proj_weights, dim=1)

            all_new_experts.append(expert_mlp)
            new_expert_intermediate_size = expert_mlp.up_proj.weight.shape[0]
            total_neurons_processed += new_expert_intermediate_size

            # print(expert_idx, ii, new_expert_intermediate_size, expert_mlp.gate_proj.weight.shape, expert_mlp.up_proj.weight.shape, expert_mlp.down_proj.weight.shape)
        expanded_gate = ori_router_gate.data[expert_idx, :].unsqueeze(0).repeat(slice_expert_num, 1).to(device)

        new_router.weight.data[gate_start_idx: gate_start_idx + slice_expert_num, :] = expanded_gate
        gate_start_idx += slice_expert_num

    if layer.mlp.shared_experts:
        shared_expert = layer.mlp.shared_experts

    moe = layer.mlp.__class__(model.config).to(device)
    moe.num_experts = len(all_new_experts)
    moe.top_k = n_activated
    moe.gate = new_router
    moe.experts = all_new_experts
    if layer.mlp.shared_experts:
        moe.shared_experts = layer.mlp.shared_experts

    return moe

@torch.no_grad()
def construct_moe_from_existing(model, layer, layer_idx, inp, attention_mask, position_ids, position_embeddings, n_experts, n_activated, n_shared, args):

    device = next(layer.parameters()).device
    # print(layer, device)

    # Forward attention
    inp = inp.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    if position_ids is not None:
        position_ids = position_ids.to(device)
    
    residual = inp
    hidden_states_inorm = layer.input_layernorm(inp)
    try:
        hidden_states = layer.self_attn(
            hidden_states=hidden_states_inorm, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            position_embeddings=position_embeddings)[0]
    except:
        hidden_states = layer.self_attn(
            hidden_states=hidden_states_inorm, 
            attention_mask=attention_mask, 
            position_ids=position_ids)[0]
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    
    if hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts'):
        slice_expert_num = n_experts // len(layer.mlp.experts)
        assert slice_expert_num * len(layer.mlp.experts) == n_experts, "n_experts must be multiple of existing expert num"
        
        moe = reconstruct_moe(model, layer, hidden_states, n_experts, n_activated, slice_expert_num, device, args)
        layer.mlp = moe
        
    else:
        slice_expert_num = 1

    # print(layer)
    quant_layer = False
    quant_attn = True

    if quant_layer:
        print(f"Quantize layer {layer_idx}")
        from gptqutil import GPTQ, Quantizer, find_layers

        gptq = {}
        wbits = 4
        groupsize = 128
        act_order = True
        static_groups = False
        sym = False
        filters = ['up_proj', 'gate_proj', 'down_proj']
        if quant_attn:
            filters.extend(['q_proj', 'k_proj', 'v_proj', 'o_proj', 'kv_a_proj_with_mqa', 'kv_b_proj'])

        for ff in filters:
            qmodule = find_layers(layer, filters=[ff])

            print("Quant modules", ff, qmodule.keys())
            if len(qmodule.keys()) == 0:
                continue
            for name in qmodule.keys():
                gptq[name] = GPTQ(qmodule[name])
                gptq[name].quantizer = Quantizer()
                match = re.search(r'mlp\.experts\.(\d+)', name)
                expert_id = int(match.group(1)) if match else 0  ## shared expert id is 0
                # print(name, expert_id)
                if slice_expert_num == 1:
                    gptq[name].quantizer.configure(4, perchannel=True, sym=sym, mse=False)
                elif slice_expert_num == 2:
                    if expert_id % 2 == 0:
                        gptq[name].quantizer.configure(4, perchannel=True, sym=sym, mse=False)
                    elif expert_id % 2 == 1:
                        gptq[name].quantizer.configure(3, perchannel=True, sym=sym, mse=False)
                elif slice_expert_num == 4:
                    if expert_id % 4 == 0:
                        gptq[name].quantizer.configure(3, perchannel=True, sym=sym, mse=False)
                    elif expert_id % 4 == 1:
                        gptq[name].quantizer.configure(2, perchannel=True, sym=sym, mse=False)
                    elif expert_id % 4 == 2:
                        gptq[name].quantizer.configure(2, perchannel=True, sym=sym, mse=False)
                    elif expert_id % 4 == 3:
                        gptq[name].quantizer.configure(1, perchannel=True, sym=sym, mse=False)
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in qmodule.keys():
                handles.append(qmodule[name].register_forward_hook(add_batch(name)))
            
            try:
                layer.self_attn(
                    hidden_states=hidden_states_inorm, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    position_embeddings=position_embeddings)[0]
            except:
                layer.self_attn(
                    hidden_states=hidden_states, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids)[0]
            
            layer.mlp(hidden_states)

            for handle in handles:
                handle.remove()
            for name in qmodule.keys():
                gptq[name].fasterquant(name=f"layer_idx.{layer_idx}."+name, groupsize=groupsize, actorder=act_order, static_groups=static_groups)
                gptq[name].free()
    
    return_router_info = 'olmoe' in args.model.lower()
    if return_router_info:
        moe_out, _ = layer.mlp(hidden_states)
    else:
        moe_out = layer.mlp(hidden_states)

    moe_out = moe_out + residual
    # print("moe_out")
    return moe_out