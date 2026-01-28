import torch
import torch.nn as nn
import copy
import time
from tqdm import tqdm
from CMoE_utils import *
from datautils import *
from sft_utils import simple_sft
from eval_cmoe import cmoe_ppl_eval

DEV = torch.device('cuda:0')

def cmoe_sequential(model, tokenizer, dataloader, args):
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

    inps = inps.squeeze(1)

    for layer_idx, layer in tqdm(enumerate(layers), desc = 'Carving MoE layers...'):
        moe_out = construct_moe(model, 
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

        inps = moe_out
        gc.collect()
        torch.cuda.empty_cache()
    
    tick_1 = time.time()

    # print('Training_free_ppl:')
    pre_ppl = []
    datasets = ['wikitext2', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, tokenizer=tokenizer, seqlen=model.seqlen, bsz = args.carve_bsz
        )
        print(dataset)
        eval_set = dataset
        ppl_i = cmoe_ppl_eval(model, testloader, eval_set, args)
        pre_ppl.append(f"{dataset}: {ppl_i}")
    
    tick_2 = time.time()

    sft_flag = args.epoch > 0
    if sft_flag:
        print('Starting SFT...')    
        # LoRa-based Supervised Fine-tuning
        for layer in layers:
                layer.mlp.cus_training = True

        model.cuda()
        model = simple_sft(model, tokenizer, args, epoch = args.epoch)

        for layer in layers:
            layer.mlp.cus_training = False

        model.eval()

        model.config.use_cache = use_cache
        print('SFT_ppl:')
        ppl = []
        datasets = ['wikitext2', 'c4-new']
        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, tokenizer=tokenizer, seqlen=model.seqlen, bsz = args.carve_bsz
            )
            print(dataset)
            eval_set = dataset
            ppl_i = cmoe_ppl_eval(model, testloader, eval_set, args)
            ppl.append(f"{dataset}: {ppl_i}")
        
        print("SFT_ppl: ", ppl)

    return model, tick_1, tick_2, pre_ppl, ppl if sft_flag else None