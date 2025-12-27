import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

from typing import Optional, Tuple, List

from CMoE_model import *

import json

import lap


@torch.no_grad()
def analyze_neuron_activations(scores: torch.Tensor, plot_results: bool = False,
                             save_path: Optional[str] = None, sparsity = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    K = max(1, math.ceil(scores.shape[-1] * sparsity))  # Top 10% neurons 
    print("K (top neurons per sample):", scores.shape, K)

    scores = scores.detach().cpu()
    batch_size, seq_len, inter_size = scores.shape
    total_samples = batch_size * seq_len
    
    flat_states = scores.reshape(-1, inter_size)
    activation_markers = torch.zeros_like(flat_states)
    
    for i in range(total_samples):
        sample_values = flat_states[i]
        abs_values = sample_values.abs().float()

        # Get indices of top-k absolute values
        top_values, top_indices = torch.topk(abs_values, k=K)
        activation_markers[i, top_indices] = 1.0

    # Sum up activations across all samples
    activation_counts = activation_markers.sum(dim=0)
    activation_rates = activation_counts / total_samples

    # print("activation_rates:", activation_rates, sum(activation_rates), activation_rates.shape)
    # print("activation_counts:", activation_counts, sum(activation_counts), activation_counts.shape)
    # print("activation_markers:", activation_markers, sum(sum(activation_markers)), activation_markers.shape)
    
    if plot_results:
        plt.figure(figsize=(10, 10))
        
        # Plot 1: Activation rates histogram
        plt.subplot(2, 1, 1)  
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
        plt.subplot(2, 1, 2)  
        neuron_indices = np.arange(inter_size)
        plt.plot(neuron_indices, activation_counts.detach().to(dtype=torch.float32).numpy(), 'b-', alpha=0.6)
        plt.title('Activation Counts per Neuron Index')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Count')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    return activation_counts, activation_rates, activation_markers


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
def construct_moe(layer, layer_idx, inp, attention_mask, position_ids, position_embeddings, n_experts, n_activated, n_shared, args):

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
    counts, rates, markers = analyze_neuron_activations(scores_all, plot_results=False, save_path=f'./plot/scores_analysis_{layer.self_attn.layer_idx}.png')

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

def reconstrut_moe(layer, hidden_states, n_experts, n_activated, slice_expert_num, device, args):

    ori_expert_num = len(layer.mlp.experts)
    ori_up_proj_size = layer.mlp.experts[0].up_proj.weight.shape[0]
    ori_hidden_size = hidden_states.shape[2]
    print(f"Original expert num: {ori_expert_num}, up_proj_size per expert: {ori_up_proj_size}, hidden_size: {ori_hidden_size}")

    all_new_experts = nn.ModuleList()
    new_router = Router(ori_hidden_size, n_experts, n_activated, bias_speed=args.bias_speed)
    # new_router.classifier.weight.data = torch.zeros(ori_hidden_size, n_experts)
    new_router.gate.weight.data = torch.zeros(n_experts, ori_hidden_size).to(torch.bfloat16).to(device)
    new_router.classifier = None
    ori_router_gate = layer.mlp.gate.weight
    total_neurons_processed = 0

    scaling_factor = slice_expert_num
    print(f"Original router gate weights shape: {ori_router_gate.shape}")

    # Process each original expert
    for expert_idx, expert in enumerate(layer.mlp.experts):
        print(f"\nProcessing original expert {expert_idx} / {ori_expert_num}")
        
        # Get scores for this specific expert
        h = expert.act_fn(F.linear(F.normalize(hidden_states, p=2, dim=-1), F.normalize(expert.gate_proj.weight, p=2, dim=1)))
        h = h * F.linear(F.normalize(hidden_states, p=2, dim=-1), F.normalize(expert.up_proj.weight, p=2, dim=1))
        expert_scores = h.to('cpu')
        
        # print(f"Expert {expert_idx} scores shape: {expert_scores.shape}")

        # Calculate activation markers and rates for this expert's neurons
        counts, rates, markers = analyze_neuron_activations(expert_scores, plot_results=False, sparsity=0.01)
        
        # Determine number of new experts to create from this original expert
        # We'll split each original expert into split_factor new experts
    
        expert_groups, representative_indices = construct_experts_k_means(
            rates,
            markers,
            num_experts = slice_expert_num,
            num_shared_experts = 0,
        )

        expert_groups = expert_groups[1:]
        # Create new experts for this original expert
        for group_indices in expert_groups:
            expert_mlp = LlamaMLP(ori_hidden_size, len(group_indices)).to(device)
            
            with torch.no_grad():
                gate_proj_weights = []
                up_proj_weights = []
                down_proj_weights = []
                
                # print(f"Creating new expert with {len(group_indices)} neurons from original expert {expert_idx}")
                # group_indices = [i for i in range(1024)]
                for global_idx in group_indices:    
                    gate_proj_weights.append(layer.mlp.experts[expert_idx].gate_proj.weight.data[global_idx, :])
                    up_proj_weights.append(layer.mlp.experts[expert_idx].up_proj.weight.data[global_idx, :])
                    down_proj_weights.append(layer.mlp.experts[expert_idx].down_proj.weight.data[:, global_idx] * scaling_factor)
                
                # print(gate_proj_weights[0].shape, len(gate_proj_weights), expert_mlp.gate_proj.weight.shape)
                expert_mlp.gate_proj.weight.data = torch.stack(gate_proj_weights)
                expert_mlp.up_proj.weight.data = torch.stack(up_proj_weights)
                expert_mlp.down_proj.weight.data = torch.stack(down_proj_weights, dim=1)
            
            all_new_experts.append(expert_mlp)
            total_neurons_processed += len(group_indices)
 
        # Router
       
        core_weights = []
        core_gate_weights = []
        for core_neuron in representative_indices:
            core_weights.append(layer.mlp.experts[expert_idx].up_proj.weight.data[core_neuron, :])
            core_gate_weights.append(layer.mlp.experts[expert_idx].gate_proj.weight.data[core_neuron, :])
      
        core_gate_weights = F.normalize(torch.stack(core_gate_weights).to(dtype=torch.bfloat16).to(device), p=2, dim=1)
        expanded_gate = ori_router_gate.data[expert_idx, :].unsqueeze(0).repeat(slice_expert_num, 1).to(device)
        # new_router_gate = expanded_gate * core_gate_weights

        alpha = 0.1
        new_router_gate = (1 - alpha) * expanded_gate + alpha * core_gate_weights

        new_router.gate.weight.data[expert_idx * slice_expert_num:(expert_idx + 1) * slice_expert_num, :] = new_router_gate
        new_router.gate.weight.to(device)
        print(new_router.gate.weight.shape)

    # Now handle shared experts if needed
    print(f"\nTotal neurons processed: {total_neurons_processed}")
    print(f"Total new experts created from original experts: {len(all_new_experts)}")

    return total_neurons_processed, all_new_experts, new_router

@torch.no_grad()
def construct_moe_from_existing(layer, layer_idx, inp, attention_mask, position_ids, position_embeddings, n_experts, n_activated, n_shared, args):

    device = next(layer.parameters()).device
    
    slice_expert_num = n_experts // len(layer.mlp.experts)
    assert slice_expert_num * len(layer.mlp.experts) == n_experts, "n_experts must be multiple of existing expert num"
    
    print(f"Converting existing {layer_idx} MoE layer with {len(layer.mlp.experts)} experts and {int(hasattr(layer.mlp, 'shared_expert'))} shared experts \
           to {n_experts} experts")
    
    # Forward attention
    inp = inp.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    if position_ids is not None:
        position_ids = position_ids.to(device)
    
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
    
    # print(hidden_states.shape)
    # print(layer.mlp.experts)

    return_router_info = 'olmoe' in args.model.lower()

    reconstruct_start_layer = 0
    if layer_idx >= reconstruct_start_layer:
        total_neurons_processed, all_new_experts, new_router = reconstrut_moe(layer, hidden_states, n_experts, n_activated, slice_expert_num, device, args)

        # MoE
        moe = MoE(hidden_states.shape[2], total_neurons_processed, len(all_new_experts), n_shared, n_activated, return_router_info)
        moe.gate = new_router
        moe.experts = all_new_experts
        moe.shared_experts = None
        moe.cus_training = False

    # Test the new MoE
    if return_router_info:
        if layer_idx >= reconstruct_start_layer:
            moe_out, _ = moe(hidden_states)
        else:
            moe_out, _ = layer.mlp(hidden_states)
    else:
        if layer_idx >= reconstruct_start_layer:
            moe_out = moe(hidden_states)
        else:
            moe_out = layer.mlp(hidden_states)
    
    moe_out = moe_out + residual
    if layer_idx >= reconstruct_start_layer:
        layer.mlp = moe
    
    print("moe_out")
    return moe_out