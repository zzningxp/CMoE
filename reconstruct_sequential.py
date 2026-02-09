import torch
import torch.nn as nn
import copy
import time
from tqdm import tqdm
from reconstruct_utils import *
from datautils import *
from eval_reconstruct import ppl_eval
from mixqdense_modeling import MixQDenseMLP, MixQDenseMoEBlock

try:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP, Qwen3MoeSparseMoeBlock
except:
    pass

DEV = torch.device('cuda:0')

@torch.no_grad()
def dynamic_quant_scheme(model, layer, layer_idx, d_wbit, inps, n_experts, n_activated, slice_expert_num, device, args):
    d_wbit = 3
    thr_l = 5
    thr_h = 50
    if args.dyn_scheme is not None:
        try:
            # sample: "b3l5h50"
            match = re.search(r'b(\d)l(\d+)h(\d+)', args.dyn_scheme)
            d_wbit = int(match.group(1))
            thr_l = int(match.group(2))
            thr_h = int(match.group(3))
        except:
            print(f"Quant scheme {args.dyn_scheme} is not valid.")
    print(f"d_wbit: {d_wbit}, thr_l: {thr_l}, thr_h: {thr_h}")

    if args.rank_mode == "quant_outlier":
        # save_path=f"plot/_{model.config.model_type}_quant_outlier_{layer_idx}.png"
        save_path = None
        up_proj_rates = analyze_quant_outlier(layer, layer_idx, inps, slice_expert_num, 1, wbits=d_wbit, if_dense=True, filters = ['up_proj'], save_path=save_path, args=args)
        gate_proj_rates = analyze_quant_outlier(layer, layer_idx, inps, slice_expert_num, 1, wbits=d_wbit, if_dense=True, filters = ['gate_proj'], save_path=save_path, args=args)
        down_proj_rates = analyze_quant_outlier(layer, layer_idx, inps, slice_expert_num, 1, wbits=d_wbit, if_dense=True, filters = ['down_proj'], save_path=save_path, args=args)
    else:
        assert False, f"Unsupported rank mode in mixqdense: {args.rank_mode}"

    intermediate_size = up_proj_rates[0].shape[0]
    rates = gate_proj_rates[0] + up_proj_rates[0] + down_proj_rates[0]
    new_rates, rank_indices = torch.topk(rates, intermediate_size)
    rank_indices_tensor = torch.tensor(rank_indices, dtype=torch.long, device=device)

    new_expert_sizes = [1]
    assert len(new_expert_sizes) == slice_expert_num

    avg_loss = analyze_rates(model, layer_idx, up_proj_rates, new_rates, gate_proj_rates, down_proj_rates, new_expert_sizes, rank_indices_tensor)                             
                            #  save_path = f"plot/_{model.config.model_type}_rates_{layer_idx}.png")

    dyn_qscheme = {}
    for k in avg_loss.keys():
        dx = []
        for i in avg_loss[k]:
            if i < thr_l:
                dx.append(d_wbit - 1)
            elif i < thr_h:
                dx.append(d_wbit)
            else:
                dx.append(d_wbit + 1)
        dyn_qscheme[k] = dx
    print(avg_loss, "dyn_qscheme", dyn_qscheme)

    return layer.mlp, {'shared': dyn_qscheme, 'attn': [d_wbit], 'expert': None}


@torch.no_grad()
def reconstruct_dense_dynamic_split(model, layer, layer_idx, inps, n_experts, n_activated, slice_expert_num, device, args):
    if args.rank_mode == "quant_outlier":
        # save_path=f"plot/_{model.config.model_type}_quant_outlier_{layer_idx}.png", 
        up_proj_rates = analyze_quant_outlier(layer, layer_idx, inps, slice_expert_num, 1, wbits=d_wbit, if_dense=True, filters = ['up_proj'], args=args)
        gate_proj_rates = analyze_quant_outlier(layer, layer_idx, inps, slice_expert_num, 1, wbits=d_wbit, if_dense=True, filters = ['gate_proj'], args=args)
        down_proj_rates = analyze_quant_outlier(layer, layer_idx, inps, slice_expert_num, 1, wbits=d_wbit, if_dense=True, filters = ['down_proj'], args=args)
    else:
        assert False, f"Unsupported rank mode in mixqdense: {args.rank_mode}"

    intermediate_size = up_proj_rates[0].shape[0]
    rates = gate_proj_rates[0] + up_proj_rates[0] + down_proj_rates[0]
    new_rates, rank_indices = torch.topk(rates, intermediate_size)
    rank_indices_tensor = torch.tensor(rank_indices, dtype=torch.long, device=device)
    new_up_proj_weight = layer.mlp.up_proj.weight[rank_indices_tensor, :]
    new_gate_proj_weight = layer.mlp.gate_proj.weight[rank_indices_tensor, :]
    new_down_proj_weight = layer.mlp.down_proj.weight[:, rank_indices_tensor]

    # layer.mlp.down_proj.weight = nn.Parameter(new_down_proj_weight)
    # analyze_quant_outlier(layer, layer_idx, inps, slice_expert_num, 1, if_dense=True, filters = ['down_proj'], args=args, save_path=f"plot/_{model.config.model_type}_quant_outlier_{layer_idx}.png")

    experts = nn.ModuleList()
    x = 4
    new_expert_sizes = [1/x, (x-2)/x, 1/x]
    assert len(new_expert_sizes) == slice_expert_num

    avg_loss = analyze_rates(model, layer_idx, up_proj_rates, new_rates, gate_proj_rates, down_proj_rates, new_expert_sizes, rank_indices_tensor, 
                             save_path = f"plot/_{model.config.model_type}_rates_{layer_idx}.png")

    expert_qscheme = {}
    for k in avg_loss.keys():
        dx = []
        for i in avg_loss[k]:
            if i < 5:
                dx.append(d_wbit - 1)
            elif i < 50:
                dx.append(d_wbit)
            else:
                dx.append(d_wbit + 1)
        expert_qscheme[k] = dx
    dyn_qscheme = {'shared': None, 'attn': [d_wbit], 'expert': expert_qscheme}
    print(avg_loss)
    print("dyn_qscheme", dyn_qscheme)

    start_id = 0
    for ii, new_expert_size in enumerate(new_expert_sizes):
        new_intermediate_size = int(new_expert_size * intermediate_size)

        expert_mlp = MixQDenseMLP(layer.mlp.hidden_size, new_intermediate_size)

        with torch.no_grad():
            expert_mlp.gate_proj.weight.data = new_gate_proj_weight[start_id:start_id+new_intermediate_size, :]
            expert_mlp.up_proj.weight.data = new_up_proj_weight[start_id:start_id+new_intermediate_size, :]
            expert_mlp.down_proj.weight.data = new_down_proj_weight[:, start_id:start_id+new_intermediate_size]
            
        experts.append(expert_mlp)
        start_id += new_intermediate_size
        print(ii, new_intermediate_size, expert_mlp.gate_proj.weight.shape, expert_mlp.up_proj.weight.shape, expert_mlp.down_proj.weight.shape)
    
    moe = MixQDenseMoEBlock(slice_expert_num).to(device)
    moe.experts = experts

    return moe, dyn_qscheme

@torch.no_grad()
def reconstruct_moe_from_dense(model, layer, layer_idx, inps, n_experts, n_activated, slice_expert_num, device, args):

    if args.rank_mode == "activation":
        analyze_sparsity = 0.1
        rates = analyze_neuron_activations(layer.mlp.act_fn, inps, layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight, sparsity=analyze_sparsity)
    elif args.rank_mode == "quant_outlier":
        rates = analyze_quant_outlier(layer, layer_idx, inps, slice_expert_num, n_experts // slice_expert_num, if_dense=True, args=args) #, save_path=f"plot/_quant_outlier_{layer_idx}.png")
        rates = rates[0]
    elif args.rank_mode == "random":
        rates = torch.randn(layer.mlp.intermediate_size, device=device)
    elif args.rank_mode == "neuron_index":
        rates = torch.arange(layer.mlp.intermediate_size, device=device)
    else:
        assert False, f"Unknown rank mode: {args.rank_mode}"

    expert_groups, representative_indices = construct_experts_by_rates(
        rates,
        num_experts = slice_expert_num,
        num_shared_experts = 0,
    )
    
    experts = nn.ModuleList()
    total_neurons_processed = 0
    scaling_factor = slice_expert_num

    new_intermediate_size = layer.mlp.intermediate_size // slice_expert_num
    expert_groups = expert_groups[1:]
    for ii, group_indices in enumerate(expert_groups):
        if model.config.model_type == 'qwen3':
            expert_mlp = Qwen3MoeMLP(model.config)
        else:
            expert_mlp = LlamaMLP(layer.mlp.hidden_size, new_intermediate_size)

        with torch.no_grad():
            group_indices_tensor = torch.tensor(group_indices, dtype=torch.long, device=device)
            
            expert_mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight[group_indices_tensor, :]
            expert_mlp.up_proj.weight.data = layer.mlp.up_proj.weight[group_indices_tensor, :]
            expert_mlp.down_proj.weight.data = layer.mlp.down_proj.weight[:, group_indices_tensor] * scaling_factor
            
        experts.append(expert_mlp)
        new_expert_intermediate_size = expert_mlp.up_proj.weight.shape[0]
        total_neurons_processed += new_expert_intermediate_size
        print(ii, scaling_factor, new_expert_intermediate_size, expert_mlp.gate_proj.weight.shape, expert_mlp.up_proj.weight.shape, expert_mlp.down_proj.weight.shape)
    
    if model.config.model_type == 'qwen3':
        moe = Qwen3MoeSparseMoeBlock(model.config)
        moe.gate.weight.data = torch.ones(slice_expert_num, layer.mlp.hidden_size).to(torch.bfloat16).to(device)
        moe.experts = experts
    else:
        router = Router(layer.mlp.hidden_size, slice_expert_num, n_activated).to(device)
        router.gate.weight.data = torch.ones(slice_expert_num, layer.mlp.hidden_size).to(torch.bfloat16).to(device)
        router.classifier = None
        # init router gate to 1，and make all experts are activated

        # MoE
        moe = MoE(layer.mlp.hidden_size, new_intermediate_size, slice_expert_num, 0, n_activated).to(device)
        moe.gate = router
        moe.experts = experts
        moe.shared_experts = None

    return moe

@torch.no_grad()
def reconstruct_moe_from_existing(model, layer, layer_idx, inps, n_experts, n_activated, slice_expert_num, device, args):

    ori_expert_num = len(layer.mlp.experts)
    new_expert_num = ori_expert_num * slice_expert_num 
    scaling_factor = slice_expert_num

    ori_router_gate = layer.mlp.gate.weight
    if type(layer.mlp.gate) == nn.Linear:
        new_router = nn.Linear(model.config.hidden_size, new_expert_num, dtype=ori_router_gate.dtype, bias=False).to(device)
    else:
        new_router = layer.mlp.gate.__class__(model.config).to(device).to(layer.mlp.gate.weight.dtype)
        # new_router.training = False ### for moonlight model
    # print(new_router)
    all_new_experts = nn.ModuleList()

    total_neurons_processed = 0
    gate_start_idx = 0

    tick0 = time.time()

    if args.rank_mode == "quant_outlier":
        all_rates = analyze_quant_outlier(layer, layer_idx, inps, slice_expert_num, n_experts // slice_expert_num, if_dense=False, save_path=None, args=args)

    for expert_idx, expert in enumerate(layer.mlp.experts):
        ori_gate_proj_weights = expert.gate_proj.weight
        ori_up_proj_weights = expert.up_proj.weight
        ori_down_proj_weights = expert.down_proj.weight

        # print(f"\nProcessing original expert {expert_idx} / {ori_expert_num}")
        if args.rank_mode == "activation":
            analyze_sparsity = 0.1
            rates = analyze_neuron_activations(expert.act_fn, inps, ori_gate_proj_weights, ori_up_proj_weights, sparsity=analyze_sparsity)
        elif args.rank_mode == "quant_outlier":
            rates = all_rates[expert_idx]
        elif args.rank_mode == "random":
            rates = torch.randn(layer.mlp.intermediate_size, device=device)
        elif args.rank_mode == "neuron_index":
            rates = torch.arange(layer.mlp.intermediate_size, device=device)
        else:
            assert False, f"Unknown rank mode: {args.rank_mode}"
        
        expert_groups, representative_indices = construct_experts_by_rates(
            rates,
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
                group_indices_tensor = torch.tensor(group_indices, dtype=torch.long, device=ori_gate_proj_weights.device)
                
                expert_mlp.gate_proj.weight.data = ori_gate_proj_weights[group_indices_tensor, :]
                expert_mlp.up_proj.weight.data = ori_up_proj_weights[group_indices_tensor, :]
                expert_mlp.down_proj.weight.data = ori_down_proj_weights[:, group_indices_tensor] * scaling_factor
                
            all_new_experts.append(expert_mlp)
            new_expert_intermediate_size = expert_mlp.up_proj.weight.shape[0]
            total_neurons_processed += new_expert_intermediate_size
            # print(expert_idx, ii, new_expert_intermediate_size, expert_mlp.gate_proj.weight.shape, expert_mlp.up_proj.weight.shape, expert_mlp.down_proj.weight.shape)
        
        expanded_gate = ori_router_gate.data[expert_idx, :].unsqueeze(0).repeat(slice_expert_num, 1).to(device)

        # print(f"gate_start_idx, slice_expert_num, expanded_gate.shape: {expert_idx, gate_start_idx, slice_expert_num, expanded_gate.shape}")
        new_router.weight.data[gate_start_idx: gate_start_idx + slice_expert_num, :] = expanded_gate
        gate_start_idx += slice_expert_num

    tick1 = time.time()
    print(f"Layer {layer_idx}, {args.rank_mode} expert re- sort time: {tick1 - tick0}")

    moe = layer.mlp.__class__(model.config).to(device)
    moe.num_experts = len(all_new_experts)
    moe.top_k = n_activated
    moe.gate = new_router
    moe.experts = all_new_experts
    if hasattr(layer.mlp, 'shared_experts'):
        moe.shared_experts = layer.mlp.shared_experts

    return moe

### unused:
@torch.no_grad()
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
    calib_size = 8
    calib_hidden_states = hidden_states[:8]

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

        # print(f"Expert {expert_idx} scores shape: {expert_scores.shape}")
        rates = analyze_neuron_activations(expert.act_fn, hidden_states, ori_gate_proj_weights, ori_up_proj_weights, sparsity=sparsity)

        expert_groups, representative_indices = construct_experts_by_rates(
            rates,
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
    # print(model.config.intermediate_size, model.config.num_experts, model.config.num_experts_per_tok)

    moe = layer.mlp
    moe.num_experts = len(all_new_experts)
    moe.top_k = n_activated
    moe.gate = new_router
    moe.experts = all_new_experts

    return moe

@torch.no_grad()
def construct_moe(model, moe_model_flag, layer, layer_idx, inp, attention_mask, position_ids, position_embeddings, 
                                n_experts, n_activated, slice_expert_num, n_shared, args):
    
    modeltype = model.config.model_type
    batchsize = inp.shape[0]

    device = next(layer.parameters()).device
    # print(layer, device)
    # print(inp.shape)

    # Forward attention
    inp = inp.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    if position_ids is not None:
        position_ids = position_ids.to(device)
    
    residual = inp
    hidden_states_inorm = layer.input_layernorm(inp)

    tick0 = time.time()
    attn_out = torch.zeros_like(hidden_states_inorm)
    for b_i in range(0, batchsize):
        if modeltype == 'olmoe' or modeltype == 'llama' or modeltype == 'qwen3' or modeltype == 'deepseek_v3':
            attn_out[b_i:b_i+1] = layer.self_attn(
                hidden_states=hidden_states_inorm[b_i:b_i+1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings)[0]
        else:
            attn_out[b_i:b_i+1] = layer.self_attn(
                hidden_states=hidden_states_inorm[b_i:b_i+1],
                attention_mask=attention_mask, 
                position_ids=position_ids)[0]
    tick1 = time.time()
    print(f"Inference in origin attention layer {layer_idx} with batch size {batchsize} time: {tick1 - tick0}")

    hidden_states = residual + attn_out
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)

    # print(hidden_states.shape)
    
    dyn_qscheme = None
    base_bit = 3
    if moe_model_flag:
        if hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts'):        
            moe = reconstruct_moe_from_existing(model, layer, layer_idx, hidden_states, n_experts, n_activated, slice_expert_num, device, args)
            layer.mlp = moe
    elif args.mixqdense:
        # moe, dyn_qscheme = reconstruct_dense_dynamic_split_dense(model, layer, layer_idx, hidden_states, n_experts, n_activated, slice_expert_num, device, args)
        _, dyn_qscheme = dynamic_quant_scheme(model, layer, layer_idx, base_bit, hidden_states, n_experts, n_activated, slice_expert_num, device, args)
    else:
        moe = reconstruct_moe_from_dense(model, layer, layer_idx, hidden_states, n_experts, n_activated, slice_expert_num, device, args)
        layer.mlp = moe
    
    gc.collect()
    torch.cuda.empty_cache()

    # print(layer)
    if_quant_layer = True
    if_quant_attn = True

    if if_quant_layer:
        quanted_size = quant_layer_mix_precision(layer, layer_idx, if_quant_attn, slice_expert_num, 
                    hidden_states_inorm, hidden_states, attention_mask, position_ids, position_embeddings, 
                    dyn_qscheme,
                    args)
        print(f"Quantized size of layer {layer_idx}: {quanted_size}")
        gc.collect()
        torch.cuda.empty_cache()
    
    print(hidden_states.shape)
    tick0 = time.time()
    moe_out = torch.zeros_like(hidden_states)
    for b_i in range(0, batchsize):
        out = layer.mlp(hidden_states[b_i:b_i+1])
        if isinstance(out, tuple):
            moe_out[b_i:b_i+1] = out[0]
        else:
            moe_out[b_i:b_i+1] = out
    tick1 = time.time()
    print(f"Inference in new moe layer {layer_idx} with batch size {batchsize} time: {tick1 - tick0}")

    moe_out = moe_out + residual
    # print("moe_out")
    return moe_out, quanted_size

def sequential(model, tokenizer, dataloader, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    bsz = args.carve_bsz
    
    inps = torch.zeros(
        (args.nsamples//bsz, bsz, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    print(inps.shape)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):

            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    layers[0] = Catcher(layers[0])
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch[0].to(DEV))
            except ValueError:
                pass

    layers[0] = layers[0].module

    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    # print("position_embeddings:", position_embeddings)
    # print(cache)

    print('Ready.')
    # model.cuda()
    # layers.cuda()

    # MoE Carving
    moe_model_flag = False
    for layer in layers:
        moe_model_flag = moe_model_flag or hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts')
    if moe_model_flag:
        if hasattr(model.config, 'num_experts'):         ## olmoe，
            slice_expert_num = args.nexperts // model.config.num_experts
            assert slice_expert_num * model.config.num_experts == args.nexperts, "n_experts must be multiple of existing expert num"
            model.config.num_experts = args.nexperts
        elif hasattr(model.config, 'n_routed_experts'):  ## DeepSeek-V1-MoE-16B
            slice_expert_num = args.nexperts // model.config.n_routed_experts
            assert slice_expert_num * model.config.n_routed_experts == args.nexperts, "n_experts must be multiple of existing expert num"
            model.config.n_routed_experts = args.nexperts

        model.config.num_experts_per_tok = args.nactivated
        if hasattr(model.config, 'moe_intermediate_size'): ## DeepSeek-V1-MoE-16B
            model.config.moe_intermediate_size = model.config.moe_intermediate_size // slice_expert_num
        elif hasattr(model.config, 'intermediate_size'): ## olmoe，
            model.config.intermediate_size = model.config.intermediate_size // slice_expert_num
        print("The model is already a MoE model. Proceeding to split experts. ")
        print(f"Slice expert by : {slice_expert_num} to {args.nexperts}, with {args.nactivated} activated experts.")
    else:
        print("The model is a dense model. Proceeding to carve MoE layers. ")
        slice_expert_num = args.nexperts
    
    if model.config.model_type == 'qwen3':
        model.config.num_experts_per_tok = args.nactivated
        model.config.num_experts = slice_expert_num
        model.config.norm_topk_prob = True
        model.config.intermediate_size = model.config.intermediate_size // slice_expert_num
        model.config.moe_intermediate_size = model.config.intermediate_size

    inps = inps.squeeze(1)
    
    all_quanted_size = 0

    for layer_idx, layer in tqdm(enumerate(layers), desc = 'Carving MoE layers...'):
        moe_out, quanted_size = construct_moe(model,
            moe_model_flag,
            layer, 
            layer_idx,
            inps, 
            attention_mask, 
            position_ids,
            position_embeddings,
            n_experts = args.nexperts,
            n_activated = args.nactivated,
            slice_expert_num = slice_expert_num,
            n_shared = args.nshared,
            args = args
        )

        all_quanted_size += quanted_size
        inps = moe_out
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"Total quantized size of all quanted layers: {all_quanted_size} b, in GB: {all_quanted_size / 8 / 1024 / 1024 / 1024:.4f}")
    # From model.lm_head and vocab_embedding, unquantized with fp16 size, most of models used same weights
    # if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
    #     all_quanted_size += model.model.vocab_size * model.config.hidden_size * 16
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        all_quanted_size += model.lm_head.weight.numel() * 16 
    print(f"Total quantized size of all layers (+vol emb): {all_quanted_size} b, in GB: {all_quanted_size / 8 / 1024 / 1024 / 1024:.4f}")

    # print('Training_free_ppl:')
    pre_ppl = []
    datasets = ['wikitext2', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, tokenizer=tokenizer, seqlen=model.seqlen, bsz = args.carve_bsz
        )
        print(dataset)
        eval_set = dataset
        ppl_i = ppl_eval(model, testloader, eval_set, args)
        pre_ppl.append(f"{dataset}: {ppl_i}")
    
    return model