from reconstruct_sequential import construct_moe
import torch
import torch.nn as nn
import copy
import time
from tqdm import tqdm
from reconstruct_utils import *
from datautils import *
from eval_reconstruct import ppl_eval

DEV = torch.device('cuda:0')

@torch.no_grad()
def dynamic_quant_scheme(model, layer, layer_idx, d_wbit, inps, slice_expert_num, device, args):
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
        up_proj_rates = analyze_quant_outlier(layer, layer_idx, inps, 1, wbits=d_wbit, if_dense=True, filters = ['up_proj'], save_path=save_path, args=args)
        gate_proj_rates = analyze_quant_outlier(layer, layer_idx, inps, 1, wbits=d_wbit, if_dense=True, filters = ['gate_proj'], save_path=save_path, args=args)
        down_proj_rates = analyze_quant_outlier(layer, layer_idx, inps, 1, wbits=d_wbit, if_dense=True, filters = ['down_proj'], save_path=save_path, args=args)
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
def sequential_layer(model, if_quant_layer, if_quant_attn, layer, layer_idx, inp, attention_mask, position_ids, position_embeddings, 
                                args):
    
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
    if args.mixqdense:
        # moe, dyn_qscheme = reconstruct_dense_dynamic_split_dense(model, layer, layer_idx, hidden_states, n_experts, n_activated, slice_expert_num, device, args)
        _, dyn_qscheme = dynamic_quant_scheme(model, layer, layer_idx, base_bit, hidden_states, 1, device, args)
    else:
        assert False, "Currently only support mixqdense model."
    
    gc.collect()
    torch.cuda.empty_cache()

    # print(layer)
    
    if if_quant_layer:
        quanted_size = quant_layer_mix_precision(layer, layer_idx, if_quant_attn, 1, 
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

def quant_sequential(model, tokenizer, dataloader, args):
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
    
    all_quanted_size = 0

    for layer_idx, layer in tqdm(enumerate(layers), desc = 'Pre Quant layers...'):
        moe_out, quanted_size = sequential_layer(model,
            True, # if_quant_layer
            True, # if_quant_attn
            layer, 
            layer_idx,
            inps, 
            attention_mask, 
            position_ids,
            position_embeddings,
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