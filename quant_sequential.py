import json
import os
from reconstruct_sequential import construct_moe
import torch
import torch.nn as nn
import copy
import time
from tqdm import tqdm
from reconstruct_utils import *
from gptq_utils import save_gptq_export_bundle
from datautils import *
from eval_reconstruct import load_model, ppl_eval

DEV = torch.device('cuda:0')

def get_depth_sensitivity(layer_idx, total_layers):
    norm_pos = layer_idx / max(total_layers - 1, 1)
    sensitivity = 1.0 + 2.0 * norm_pos
    return sensitivity

def assign_quant_scheme_from_gptq_loss(gptq_losses_all, gptq_loss_lm_head, 
                                       weight_sizes, weight_size_lm_head, 
                                       vram_quota, FIXED_OP_ORDER, OP_SENSITIVITY_WEIGHTS, FIXED_BITS):
    layer_ids = sorted(gptq_losses_all.keys())
    layer_ids = [int(i) for i in layer_ids]
    weight_sizes = {int(k): v for k, v in weight_sizes.items()}
    gptq_losses_all = {int(k): v for k, v in gptq_losses_all.items()}
    gptq_loss_lm_head = {int(k): v for k, v in gptq_loss_lm_head.items()}
    
    ops = []
    
    if 'lm_head' in FIXED_OP_ORDER:
        lm_head_weight_count = weight_size_lm_head.get('lm_head', 0)
        options = []
        if lm_head_weight_count > 0:
            for bit in gptq_loss_lm_head.keys():
                if bit not in FIXED_BITS:
                    continue
                bpw = bit + 16.0 * 2 / 128.0
                mem_usage = lm_head_weight_count * bpw / 8.0
                revised_loss = gptq_loss_lm_head[bit]['lm_head'] * OP_SENSITIVITY_WEIGHTS.get('lm_head', 1.0)
                options.append( (bit, revised_loss, mem_usage) )

        if options:
            ops.append( ('lm_head', 'lm_head', options) )
    
    for layer_id in layer_ids:
        weight_size = weight_sizes[layer_id]
        gptq_losses = gptq_losses_all[layer_id]
        gptq_losses = {int(k): v for k, v in gptq_losses.items()}
        
        for op_name in FIXED_OP_ORDER:
            if op_name == 'lm_head':
                continue

            available_bits = sorted([int(b) for b in gptq_losses.keys()])
            options = []
            
            for bit in available_bits:
                if bit not in FIXED_BITS:
                    continue
                if bit not in gptq_losses:
                    continue
                
                weight_count = weight_size[op_name]
                bpw = bit + 16.0 * 2 / 128.0
                mem_usage = weight_count * bpw / 8.0
                
                revised_loss = gptq_losses[bit][op_name]
                revised_loss = revised_loss * OP_SENSITIVITY_WEIGHTS.get(op_name, 1.0)
                # revised_loss = revised_loss * get_depth_sensitivity(layer_id, len(layer_ids))
                
                options.append( (bit, revised_loss, mem_usage) )
            
            if not options:
                raise ValueError(f"[配置错误] 算子 {layer_id}/{op_name} 无可用的bit配置")
            
            ops.append( (layer_id, op_name, options) )

    # -------------------------- 2. 正向动态规划 --------------------------
    # dp_steps 保存每一步的状态：dp_steps[i] 是处理完前i个算子后的状态字典
    # 状态字典结构: {当前总显存: 最小总loss}
    dp_steps = []
    dp_steps.append( {0.0: 0.0} )
    
    eps = 1e-6
    vram_quota = vram_quota * 1024 * 1024 * 1024  

    for i in range(len(ops)):
        _, _, options = ops[i]
        prev_dp = dp_steps[i]
        curr_dp = {}
        
        for curr_mem, curr_loss in prev_dp.items():
            for (bit, opt_loss, opt_mem) in options:
                new_mem = curr_mem + opt_mem
                
                if new_mem > vram_quota + eps:
                    continue
                
                new_loss = curr_loss + opt_loss
                
                # 更新当前状态：
                # 1. 如果该显存未记录，直接记录
                # 2. 如果该显存已记录，但新loss更小，更新
                # 3. 如果loss相同，选择更大的显存（更接近配额，潜在更优）
                if new_mem not in curr_dp:
                    curr_dp[new_mem] = new_loss
                else:
                    if new_loss < curr_dp[new_mem] - eps:
                        curr_dp[new_mem] = new_loss
                    elif abs(new_loss - curr_dp[new_mem]) <= eps:
                        if new_mem > curr_dp[new_mem]:
                            curr_dp[new_mem] = new_loss
        
        if not curr_dp:
            raise RuntimeError(f"[无解] 处理完第 {i+1} 个算子后无可行配置，请检查显存配额")
        
        dp_steps.append(curr_dp)

    # -------------------------- 3. 寻找最优终态 --------------------------
    final_dp = dp_steps[-1]
    best_loss = float('inf')
    best_final_mem = 0.0
    
    for mem, loss in final_dp.items():
        if mem > vram_quota + eps:
            continue
        # 优先选loss最小的，loss相同选mem最大的
        if loss < best_loss - eps:
            best_loss = loss
            best_final_mem = mem
        elif abs(loss - best_loss) <= eps:
            if mem > best_final_mem:
                best_final_mem = mem
    
    if best_final_mem < eps and len(ops) > 0:
        raise RuntimeError("[无解] 显存配额过小，无法找到可行配置")

    # -------------------------- 4. 回溯路径获取方案 --------------------------
    # 
    result = {layer_id: {} for layer_id in layer_ids}
    result_loss = {layer_id: {} for layer_id in layer_ids}
    if 'lm_head' in FIXED_OP_ORDER:
        result_loss['lm_head'] = {}
        result['lm_head'] = {}

    current_mem = best_final_mem
    
    for i in range(len(ops)-1, -1, -1):
        layer_id, op_name, options = ops[i]
        prev_dp = dp_steps[i]
        found = False
        
        for (bit, opt_loss, opt_mem) in options:
            target_prev_mem = current_mem - opt_mem
            if target_prev_mem < -eps:
                continue
            
            for prev_mem_candidate, prev_loss_candidate in prev_dp.items():
                if abs(prev_mem_candidate - target_prev_mem) > eps:
                    continue
                # 2. loss match：prev_loss + opt_loss ≈ current_loss
                expected_loss = prev_loss_candidate + opt_loss
                actual_loss = dp_steps[i+1][current_mem]
                if abs(expected_loss - actual_loss) > eps:
                    continue

                result[layer_id][op_name] = [bit]
                result_loss[layer_id][op_name] = opt_loss
                current_mem = prev_mem_candidate
                found = True
                break
            if found:
                break
        
        if not found:
            raise RuntimeError(f"[回溯失败] 无法在第 {i+1} 个算子 ({layer_id}/{op_name}) 找到匹配配置")

    # 最终校验：确保所有算子都有结果
    for layer_id in layer_ids:
        for op_name in FIXED_OP_ORDER:
            if op_name == 'lm_head':
                continue 
            if op_name not in result[layer_id]:
                raise RuntimeError(f"[结果缺失] 算子 {layer_id}/{op_name} 未分配到bit配置")
    
    if 'lm_head' in result and 'lm_head' in result['lm_head']:
        print(f"lm_head: {result['lm_head']['lm_head']} ({result_loss['lm_head']['lm_head']:.3f})")
    
    for layer_id in layer_ids:
        print(f"Layer {layer_id}: ", end="")
        for op_name in FIXED_OP_ORDER:
            if op_name == 'lm_head':
                continue
            print(f"{op_name}: {result[layer_id][op_name]} ({result_loss[layer_id][op_name]:.3f}) ", end="")
        print()

    print(f"Best total loss: {best_loss:.6f}, Total quantized memory: {best_final_mem} B within quota {vram_quota} B")
    return result, result_loss

@torch.no_grad()
def get_ffn_pre_quant_loss(model, layer, layer_idx, d_wbit, layer_inps, mlp_inps, attention_mask, position_ids, position_embeddings, args):

    if args.rank_mode == "quant_outlier":
        # save_path=f"plot/_{model.config.model_type}_quant_outlier_{layer_idx}.png"
        save_path = None
        rates = {}
        if layer_idx == -1:
            if hasattr(model, 'lm_head'):
                rates['lm_head'] = analyze_quant_outlier(layer, layer_idx, layer_inps,
                                                       n_experts = 1, wbits=d_wbit, if_dense=True, 
                                                       filters = ['lm_head'], save_path=save_path, args=args)
        else:
            for i in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                rates[i] = analyze_quant_outlier(layer, layer_idx, layer_inps, 
                                                attention_mask, position_ids, position_embeddings, 
                                                n_experts = 1, wbits=d_wbit, if_dense=True, 
                                                filters = [i], save_path=save_path, args=args)
            for i in ['up_proj', 'gate_proj', 'down_proj']:
                rates[i] = analyze_quant_outlier(layer, layer_idx, mlp_inps, 
                                                n_experts = 1, wbits=d_wbit, if_dense=True, 
                                                filters = [i], save_path=save_path, args=args)
    else:
        assert False, f"Unsupported rank mode in mixqdense: {args.rank_mode}"

    losses = {}
    if layer_idx == -1 and 'lm_head' in rates:
        losses['lm_head'] = rates['lm_head'][0].mean().detach().cpu().numpy().item()
    else:
        for i in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj']:
            losses[i] = rates[i][0].mean().detach().cpu().numpy().item()

    print(f"Layer {layer_idx}, d_wbit: {d_wbit}: {losses}")

    return losses

@torch.no_grad()
def sequential_layer(model, pre_quant_flag, op_bits, ops, layer, layer_idx, inp, attention_mask, position_ids, position_embeddings, 
                    dyn_qschemes, args):
    
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
    
    gptq_losses = {}
    quanted_sizes = {}
    if pre_quant_flag:
        for base_bit in op_bits:
            gptq_losses[base_bit] = get_ffn_pre_quant_loss(model, layer, layer_idx, base_bit, 
                                hidden_states_inorm, hidden_states, 
                                attention_mask, position_ids, position_embeddings, 
                                args)
        qmodule_all = find_layers(layer, filters=ops)
        qmodule = {k: qmodule_all[k] for k in list(qmodule_all.keys())}
        for name in qmodule.keys():
            sname = name.split('.')[-1]
            quanted_sizes[sname] = qmodule[name].weight.numel()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Pre-quantization GPTQ losses for layer {layer_idx}: ", gptq_losses)
    
    if not pre_quant_flag:
        quanted_sizes['all'] = quant_layer_mix_precision(layer, layer_idx, True, 1, 
                    hidden_states_inorm, hidden_states, attention_mask, position_ids, position_embeddings, 
                    dyn_qschemes, args)
        print(f"Quantized size of layer {layer_idx}: {quanted_sizes['all']}")
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
    return moe_out, quanted_sizes, gptq_losses


def _module_device(module):
    first_param = next(iter(module.parameters()), None)
    if first_param is not None:
        return first_param.device

    first_buffer = next(iter(module.buffers()), None)
    if first_buffer is not None:
        return first_buffer.device

    return None


@torch.no_grad()
def _forward_to_lm_head_hidden_states(model, layers, inp, attention_mask, position_ids, position_embeddings):
    hidden_states = inp
    modeltype = model.config.model_type

    for layer in tqdm(layers, desc='Prepare lm_head inputs...'):
        batchsize = hidden_states.shape[0]
        device = next(layer.parameters()).device

        hidden_states = hidden_states.to(device)
        layer_attention_mask = attention_mask.to(device) if attention_mask is not None else None
        layer_position_ids = position_ids.to(device) if position_ids is not None else None

        residual = hidden_states
        hidden_states_inorm = layer.input_layernorm(hidden_states)

        attn_out = torch.zeros_like(hidden_states_inorm)
        for b_i in range(0, batchsize):
            if modeltype == 'olmoe' or modeltype == 'llama' or modeltype == 'qwen3' or modeltype == 'deepseek_v3':
                attn_out[b_i:b_i+1] = layer.self_attn(
                    hidden_states=hidden_states_inorm[b_i:b_i+1],
                    attention_mask=layer_attention_mask,
                    position_ids=layer_position_ids,
                    position_embeddings=position_embeddings)[0]
            else:
                attn_out[b_i:b_i+1] = layer.self_attn(
                    hidden_states=hidden_states_inorm[b_i:b_i+1],
                    attention_mask=layer_attention_mask,
                    position_ids=layer_position_ids)[0]

        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)

        moe_out = torch.zeros_like(hidden_states)
        for b_i in range(0, batchsize):
            out = layer.mlp(hidden_states[b_i:b_i+1])
            if isinstance(out, tuple):
                moe_out[b_i:b_i+1] = out[0]
            else:
                moe_out[b_i:b_i+1] = out

        hidden_states = moe_out + residual

    return hidden_states


@torch.no_grad()
def _prepare_lm_head_inputs(model, hidden_states):
    lm_head_inputs = hidden_states

    final_norm = getattr(getattr(model, 'model', None), 'norm', None)
    if final_norm is not None:
        norm_device = _module_device(final_norm)
        if norm_device is not None and lm_head_inputs.device != norm_device:
            lm_head_inputs = lm_head_inputs.to(norm_device)
        lm_head_inputs = final_norm(lm_head_inputs)

    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        lm_head_device = model.lm_head.weight.device
        if lm_head_inputs.device != lm_head_device:
            lm_head_inputs = lm_head_inputs.to(lm_head_device)

    return lm_head_inputs

def quant_sequential(model, tokenizer, dataloader, testloader, args):
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

    inps = inps.squeeze(1)
    device = next(layers[0].parameters()).device
    inps = inps.to(device)
    inps_ori = inps.clone()

    gptq_losses_all = {}
    gptq_loss_lm_head = {}
    weight_sizes = {}
    weight_size_lm_head = {}
    op_bits = [3, 4, 5, 6]
    # op_bits = [4]
    ops = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
    model_id = getattr(model, 'model_id', None)
    if not model_id:
        model_id = getattr(model.config, '_name_or_path', None) or getattr(model.config, 'name_or_path', None) or args.model
    model_id = str(model_id).split('/')[-1].split('\\')[-1]
    export_enabled = bool(getattr(args, "export_gptq_data", False))
    if export_enabled:
        args._gptq_export_store = {}
    loss_file = f"loss_{model_id}.json"
    _loss_all = {}
    ran_pre_quant_profiling = False
    lm_head_original_device = None
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        lm_head_original_device = model.lm_head.weight.device

    if os.path.exists(loss_file):
        with open(loss_file, 'r') as f:
            _loss_all = json.load(f)
            gptq_losses_all = _loss_all['gptq_losses_all']
            gptq_loss_lm_head = _loss_all.get('gptq_loss_lm_head', {})
            weight_sizes = _loss_all['weight_sizes']
            weight_size_lm_head = _loss_all.get('weight_size_lm_head', {})
        print(f"Loaded quantization data from files for model: {model_id}")
    else:
        print(f"Can NOT Loaded quantization data, run pre-quantization profiling for model: {model_id}")
        ran_pre_quant_profiling = True
        for layer_idx, layer in tqdm(enumerate(layers), desc = 'Pre Quant layers...'):
            out, quanted_sizes, gptq_losses = sequential_layer(model,
                True, # pre_quant_flag, pre-quant phase
                op_bits, #
                ops,
                layer, 
                layer_idx,
                inps, 
                attention_mask, 
                position_ids,
                position_embeddings,
                None, # dyn_qschemes
                args
            )

            weight_sizes[layer_idx] = quanted_sizes
            gptq_losses_all[layer_idx] = gptq_losses
            inps = out
            gc.collect()
            torch.cuda.empty_cache()

        lm_head_cache_valid = False
        if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
            lm_head_hidden_states = inps
            lm_head_inputs = _prepare_lm_head_inputs(model, lm_head_hidden_states)
            for base_bit in op_bits:
                gptq_loss_lm_head[base_bit] = get_ffn_pre_quant_loss(model, model.lm_head, -1, base_bit,
                                lm_head_inputs, None,
                                attention_mask, position_ids, position_embeddings,
                                args)
            weight_size_lm_head = {'lm_head': model.lm_head.weight.numel()}

        _loss_all['gptq_losses_all'] = gptq_losses_all
        _loss_all['gptq_loss_lm_head'] = gptq_loss_lm_head
        _loss_all['weight_sizes'] = weight_sizes
        _loss_all['weight_size_lm_head'] = weight_size_lm_head
        _loss_all['lm_head_input_stage'] = 'post_layers_post_norm'
        _loss_all['lm_head_profile_version'] = 2
        with open(loss_file, 'w') as f:
            json.dump(_loss_all, f)

    lm_head_cache_valid = (
        _loss_all.get('lm_head_input_stage') == 'post_layers_post_norm'
        and _loss_all.get('lm_head_profile_version') == 2
    )
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight') and (not gptq_loss_lm_head or not weight_size_lm_head or not lm_head_cache_valid):
        print("Refreshing lm_head quantization data with post-layer hidden states")
        if ran_pre_quant_profiling:
            lm_head_hidden_states = inps
        else:
            lm_head_hidden_states = _forward_to_lm_head_hidden_states(
                model,
                layers,
                inps_ori,
                attention_mask,
                position_ids,
                position_embeddings,
            )
        lm_head_inputs = _prepare_lm_head_inputs(model, lm_head_hidden_states)
        gptq_loss_lm_head = {}
        for base_bit in op_bits:
            gptq_loss_lm_head[base_bit] = get_ffn_pre_quant_loss(model, model.lm_head, -1, base_bit,
                            lm_head_inputs, None,
                            attention_mask, position_ids, position_embeddings,
                            args)
        weight_size_lm_head = {'lm_head': model.lm_head.weight.numel()}
        _loss_all['gptq_loss_lm_head'] = gptq_loss_lm_head
        _loss_all['weight_size_lm_head'] = weight_size_lm_head
        _loss_all['lm_head_input_stage'] = 'post_layers_post_norm'
        _loss_all['lm_head_profile_version'] = 2
        with open(loss_file, 'w') as f:
            json.dump(_loss_all, f)

    print(f"GPTQ losses for all layers: ", gptq_losses_all)
    print(f"GPTQ losses for lm_head: ", gptq_loss_lm_head)
    print(f"Weight sizes for all layers: ", weight_sizes)
    print(f"Weight sizes for lm_head: ", weight_size_lm_head)

    profile_base_bit = 8
    profile_low_bit = 2
    # profile_low_bit = 4
    dyn_qschemes = {layer_idx: {op_name: [profile_base_bit] for op_name in ops} for layer_idx in range(len(layers))}
    dyn_qschemes['lm_head'] = {'lm_head': [profile_base_bit]}
    
    quant_lm_head = getattr(args, 'quant_lm_head', True)
    fix_lm_head_bit = getattr(args, 'fix_lm_head_bit', 8)
    # dyn_qschemes['lm_head'] = {'lm_head': [fix_lm_head_bit]}

    print(args.profile_only_quant_layers, args.profile_only_quant_op)
    if args.profile_only_quant_layers == None and args.profile_only_quant_op == None:
        sensitivity = {
            'q_proj': 0.4,
            'k_proj': 0.4,
            'v_proj': 1.4,
            'o_proj': 0.8,
            'up_proj': 2.0,
            'down_proj': 4.0,
            'gate_proj': 2.0,
            'lm_head': 100.0,
            }
        sensitivity = {
            'q_proj': 0.113069338,
            'k_proj': 0.091652651,
            'v_proj': 0.08202362,
            'o_proj': 0.089444818,
            'up_proj': 0.335445031,
            'down_proj': 0.625093361,
            'gate_proj': 0.320789516
        }
        if fix_lm_head_bit is not None and fix_lm_head_bit != -1:
            gptq_loss_lm_head_fixed = {fix_lm_head_bit: {'lm_head': 0}}
        else:
            gptq_loss_lm_head_fixed = gptq_loss_lm_head
        
        ops_with_lm_head = ops + ['lm_head'] if quant_lm_head else ops
        dyn_qschemes, dyn_losses = assign_quant_scheme_from_gptq_loss(
            gptq_losses_all, gptq_loss_lm_head_fixed,
            weight_sizes, weight_size_lm_head,                                                                      
            args.vram_quota, ops_with_lm_head, sensitivity, op_bits)
    elif args.profile_only_quant_layers != None:
        if args.profile_only_quant_layers == 'lm_head':
            dyn_qschemes['lm_head'] = {'lm_head': [profile_low_bit]}
        elif isinstance(args.profile_only_quant_layers, int) and args.profile_only_quant_layers >= len(layers):
            raise ValueError(f"Invalid layer index for profiling: {args.profile_only_quant_layers}, total layers: {len(layers)}")
        elif args.profile_only_quant_layers != -1 and args.profile_only_quant_layers != '-1':
            profile_only_quant_layers = int(args.profile_only_quant_layers)
            dyn_qschemes[profile_only_quant_layers] = {op_name: [profile_low_bit] for op_name in ops}
        elif args.profile_only_quant_layers == -1 or args.profile_only_quant_layers == '-1':
            pass
            ## profile_only_quant_layers == -1, profile all layers with high bit
            ## do nothing
    elif args.profile_only_quant_op != None:
        for layer_idx in range(len(layers)):
            dyn_qschemes[layer_idx][args.profile_only_quant_op] = [profile_low_bit]
    print(f"Dynamic quantization schemes for each layer and op: ", dyn_qschemes)

    all_quanted_size = 0
    quanted_size_lm_head = 0
    inps = inps_ori

    for layer_idx, layer in tqdm(enumerate(layers), desc = 'Actual Quant layers...'):
        out, quanted_sizes, _ = sequential_layer(model,
            False, # pre_quant_flag, pre-quant phase
            None,
            None,
            layer, 
            layer_idx,
            inps, 
            attention_mask, 
            position_ids,
            position_embeddings,
            dyn_qschemes[layer_idx],
            args
        )

        all_quanted_size += quanted_sizes['all']
        inps = out
        gc.collect()
        torch.cuda.empty_cache()

    if quant_lm_head and hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        lm_head_inputs = _prepare_lm_head_inputs(model, inps)
        quanted_size_lm_head = quant_layer_mix_precision(
            model.lm_head,
            -1,
            False,
            1,
            lm_head_inputs,
            lm_head_inputs,
            attention_mask,
            position_ids,
            position_embeddings,
            dyn_qschemes.get('lm_head', {'lm_head': [4]}),
            args,
        )

        all_quanted_size += quanted_size_lm_head
        print(f"lm_head quantized, size: {quanted_size_lm_head} b, in GB: {quanted_size_lm_head / 8 / 1024 / 1024 / 1024:.4f}")
        if lm_head_original_device is not None and model.lm_head.weight.device != lm_head_original_device:
            model.lm_head = model.lm_head.to(lm_head_original_device)

    if quanted_size_lm_head > 0:
        print(f"lm_head quantized, size: {quanted_size_lm_head} b, in GB: {quanted_size_lm_head / 8 / 1024 / 1024 / 1024:.4f}")
    print(f"Total quantized size of all quanted layers: {all_quanted_size} b, in GB: {all_quanted_size / 8 / 1024 / 1024 / 1024:.4f}")
    
    if not quant_lm_head and hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        all_quanted_size += model.lm_head.weight.numel() * 16
        print(f"Total quantized size of all layers (+lm_head FP16): {all_quanted_size} b, in GB: {all_quanted_size / 8 / 1024 / 1024 / 1024:.4f}")

    if export_enabled:
        save_gptq_export_bundle(
            export_store=getattr(args, "_gptq_export_store", {}),
            export_dir=getattr(args, "gptq_export_dir", "gptq_export"),
            model_id=model_id,
            export_file=getattr(args, "gptq_export_file", "gptq_export.pt"),
        )

    # print('Training_free_ppl:')
    pre_ppl = []
    datasets = ['wikitext2', 'c4-new']
    # datasets = [args.dataset]
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, tokenizer=tokenizer, seqlen=model.seqlen, bsz = args.carve_bsz
        )
        print(dataset)
        eval_set = dataset
        ppl_i = ppl_eval(model, testloader, eval_set, args)
        pre_ppl.append(f"{dataset}: {ppl_i}")
    
    return model