import time

import torch
import torch.nn as nn

from tqdm import *

import os 

import copy

from CMoE_utils import *
from CMoE_model import *
from zero_eval import *
from sft_utils import simple_sft

DEV = torch.device('cuda:0')

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    return model

def get_llava(model):
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip

    from llava.model import LlavaLlamaForCausalLM

    model = LlavaLlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    return model


def cmoe_sequential(model, dataloader, dev, args):
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
                model(batch[0].to(dev))
            except ValueError:
                pass

    layers[0] = layers[0].module

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    moe_outs = torch.zeros_like(inps)
    
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    print('Ready.')
    # model.cuda()
    # layers.cuda()

    inp = copy.deepcopy(inps[0])

    # MoE Carving
    carve_inp = copy.deepcopy(inp)
    for layer in tqdm(layers, desc = 'Carving MoE layers...'):
        moe_out = construct_moe(layer, 
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

    print('Training_free_ppl:')
    pre_ppl = []
    datasets = ['wikitext2', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, bsz = args.carve_bsz
        )
        print(dataset)
        eval_set = dataset
        ppl_i = cmoe_ppl_eval(model, testloader, DEV, eval_set, args)
        pre_ppl.append(f"{dataset}: {ppl_i}")
    
    tick_2 = time.time()
    

    # LoRa-based Supervised Fine-tuning
    for layer in layers:
            layer.mlp.cus_training = True

    model.cuda()
    model = simple_sft(model, args, epoch = args.epoch)

    for layer in layers:
        layer.mlp.cus_training = False

    model.eval()

    model.config.use_cache = use_cache
    
    return model, tick_1, tick_2, pre_ppl

@torch.no_grad()
def cmoe_ppl_eval(model, testloader, dev, eval_set, args):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # Don't move model parts to single device, let them stay on their distributed devices
    dtype = next(iter(model.parameters())).dtype
    inps = []
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # Store input from the first layer (which is already on the correct device)
            inps.append(inp.clone())
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
    
    # Get the first layer's device
    first_layer_device = next(layers[0].parameters()).device
    layers[0] = Catcher(layers[0].to(first_layer_device))
    
    testenc = testloader.input_ids
    nsamples = testenc.shape[1] // model.seqlen
    nsamples = 64
    print('ppl evaluation samples:', nsamples)


    # Get input samples from all layers
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(first_layer_device)
        try:
            model(batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    
    # Convert list to tensor on the first layer's device
    print(len(inps), nsamples, testenc.shape[1], model.seqlen)
    inps = torch.stack(inps)
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    # Process each layer, keeping it on its current device
    for i in tqdm(range(len(layers)), desc='Processing...'):
        layer = layers[i]
        layer_device = next(layer.parameters()).device
        
        # Move input to layer's device
        layer_inps = inps.to(layer_device)
        layer_outs = torch.zeros_like(layer_inps)
        
        for j in range(nsamples):
            layer_outs[j] = layer(layer_inps[j], 
                                attention_mask=attention_mask, 
                                position_ids=position_ids,
                                position_embeddings=position_embeddings)[0]
        
        # Move output back to first layer's device
        outs = layer_outs.to(first_layer_device)
        
        # Swap inps and outs for next iteration
        inps, outs = outs, torch.zeros_like(outs)

    final_layer_device = next(layers[-1].parameters()).device
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(final_layer_device)
    model.lm_head = model.lm_head.to(final_layer_device)

    testenc = testenc.to(final_layer_device)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states).contiguous()
        
        hidden_states = hidden_states.to(final_layer_device)

        lm_head_weight = model.lm_head.weight.to(final_layer_device)
        with torch.no_grad():
            lm_logits = torch.nn.functional.linear(hidden_states, lm_head_weight, None)
        lm_logits = lm_logits.squeeze(1)
 #       lm_logits = model.lm_head(hidden_states)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print("ppl: ", ppl.item())
    model.config.use_cache = use_cache

    return ppl.item()

def save_results(file_name, results):
    if results is not str:
        results = str(results)
    results = results + '\n'
    if not os.path.exists(file_name):
        with open(file_name, "w") as file:
            file.write(results)
    else:
        with open(file_name, "a") as file:
            file.write(results)


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(        'model', type=str,
        help='Model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(        '--nsamples', type=int, default=128,
        help='Number of Fine-tuning data for CMoE.'
    )
    parser.add_argument(        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(        '--extra-lr',
        type=float, default=0.001, 
        help='Initial learning rate for extra scale for router.'
    )
    parser.add_argument(        '--k-act', type=int, default=10,
        help='TopK number for the ATopK. K_a in paper.'
    )
    parser.add_argument(        '--bias-speed',
        type=float, default=0.001, 
        help='Bias update speed for load balancing. Gamma in paper.'
    )
    parser.add_argument(        '--nexperts', type=int, default=16,
        help='Total number of experts. N in paper.'
    )
    parser.add_argument(        '--nactivated', type=int, default=2,
        help='Number of activated routed experts.'
    )
    parser.add_argument(        '--nshared', type=int, default=2,
        help='Number of shared experts.'
    )
    parser.add_argument(        '--epoch', type=int, default=1,
        help='SFT epoch for CMoE.'
    )
    parser.add_argument(        '--sft-bsz', type=int, default=2,
        help='SFT batch size for CMoE.'
    )
    parser.add_argument(        '--carve-bsz', type=int, default=2,
        help='Carve batch size for CMoE.'
    )
    parser.add_argument(        '--eval-zero', action='store_true',
        help='Whether to run downstream tasks evaluation.'
    )
    parser.add_argument(        '--prefix', type=str, default=None,
        help='Prefix the results folder if needed.'
    )

    args = parser.parse_args()
    
    print(args.model.lower())
    if 'llava' in args.model.lower():
        model = get_llava(args.model)
    else:
        model = get_llama(args.model)
    model.eval()
    
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, bsz = args.carve_bsz
    )

    print("number of data: ", args.nsamples)
    print("model: ", args.model)
    print("cali_data: ", args.dataset)

    tick = time.time()
    carved_model, tick_1, tick_2, pre_ppl = cmoe_sequential(model, dataloader, DEV, args)
    rt_construct = tick_1 - tick
    extra_time = tick_2 - tick_1
    rt = time.time() - tick - extra_time
    print("Runtime of training-free construction: ", rt_construct)
    print("Runtime of fine-tuning construction: ", rt)
    

    # model_name = model.split("/")[-1]
    # save_dir = f"{model_name}_carved_{args.dataset}_{args.nsamples}_epoch_{args.epoch}_S{args.nshared}_A{args.nactivated}_E{args.nexperts}_K{args.k_act}_B{args.bias_speed}"
    # print(f"Saving carved model to {save_dir}...")
    # os.makedirs(save_dir, exist_ok=True)
    # model.save_pretrained(save_dir)

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    if 'llama-3' in args.model.lower():
        name = "meta-llama/Meta-Llama-3-8B"
    else:
        name = "meta-llama/Llama-2-7b-hf"

    # datasets = ['c4-new', 'wikitext2']
    datasets = ['wikitext2']
    ppl = []
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, bsz = args.carve_bsz
        )
        print(dataset)
        eval_set = dataset
        ppl_i = cmoe_ppl_eval(carved_model, testloader, DEV, eval_set, args)
        ppl.append(f"{dataset}: {ppl_i}")

    model_name = args.model.split("/")[-1]
    file_name = f"{model_name}_{args.dataset}_{args.nsamples}_epoch_{args.epoch}_S{args.nshared}_A{args.nactivated}_E{args.nexperts}.txt"
    dir_path = os.path.join('./result_logs', args.prefix) if args.prefix is not None else './result_logs'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    file_name = os.path.join(dir_path, file_name)

    save_results(file_name, f"pre_ppl: {str(pre_ppl)}")
    save_results(file_name, f"ft_ppl: {str(ppl)}")
    save_results(file_name, f"runtime_construct: {rt_construct}")
    save_results(file_name, f"runtime_all: {rt}")

    if args.eval_zero:
        task_list = ["winogrande"]
        results_1 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=5)
        save_results(file_name, results_1)

        task_list = ["arc_challenge"]
        results_2 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=25)
        save_results(file_name, results_2)

        task_list = ["hellaswag"]
        results_3 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=10)
        save_results(file_name, results_3)

        task_list = ["sciq","piqa"]
        results_4 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=0)
        save_results(file_name, results_4)

        task_list = ["boolq"]
        results_5 = eval_zero_shot(name, carved_model, tokenizer, task_list=task_list, num_fewshot=32)
        save_results(file_name, results_5)


    print("number of data: ", args.nsamples)
    print("model: ", args.model)
    print("cali_data: ", args.dataset)