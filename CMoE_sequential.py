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
    layers[0] = Catcher(layers[0])
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch[0].to(DEV))
            except ValueError:
                pass

    layers[0] = layers[0].module

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    moe_outs = torch.zeros_like(inps)
    
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    # print("position_embeddings:", position_embeddings)
    # print(cache)

    print('Ready.')
    # model.cuda()
    # layers.cuda()

    inp = copy.deepcopy(inps[0])

    # Check if the layer is already a MoE layer
  
    # MoE Carving
    carve_inp = copy.deepcopy(inp)
    for layer_idx, layer in tqdm(enumerate(layers), desc = 'Carving MoE layers...'):
        moe_model_flag = hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts')
        # if moe_model_flag:
        #     print("The model is already a MoE model. Proceeding to split experts. ")
        # else:
        #     print("The model is a dense model. Proceeding to carve MoE layers. ")
        if moe_model_flag:
            moe_out = construct_moe_from_existing(model, layer, 
                layer_idx,
                carve_inp, 
                attention_mask, 
                position_ids,
                position_embeddings,
                n_experts = args.nexperts,
                n_activated = args.nactivated,
                n_shared = args.nshared,
                args = args
            )
        else:
            moe_out = construct_moe(model, layer, 
                layer_idx,
                carve_inp, 
                attention_mask, 
                position_ids,
                position_embeddings,
                n_experts = args.nexperts,
                n_activated = args.nactivated,
                n_shared = args.nshared,
                args = args
            )
        carve_inp = moe_out
    
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