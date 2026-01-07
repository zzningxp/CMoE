import time

import torch
import torch.nn as nn

from tqdm import *

import os 

import copy

from CMoE_utils import *
from CMoE_model import *
from CMoE_sequential import *
from zero_eval import *
from sft_utils import simple_sft
from transformers import AutoTokenizer 

def get_llama(model):
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    # model.seqlen = 4096
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

def get_olmoe(model):
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    from transformers import OlmoeForCausalLM

    model = OlmoeForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    model.seqlen = 2048
    return model

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
    parser.add_argument(        '--reconstruct_start_layer', type=int, default=0,
        help='Start layer for reconstruction.'
    )
    parser.add_argument(        '--reconstruct_end_layer', type=int, default=15,
        help='End layer for reconstruction.'
    )

    args = parser.parse_args()
    
    print("Loading model: ", args.model.lower())
    if 'llava' in args.model.lower():
        model = get_llava(args.model)
    elif 'olmoe' in args.model.lower():
        model = get_olmoe(args.model)
    else:
        model = get_llama(args.model)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataloader, testloader = get_loaders(
        args.dataset, 
        nsamples=args.nsamples, 
        seed=args.seed, 
        tokenizer=tokenizer, 
        seqlen=model.seqlen, 
        bsz = args.carve_bsz
    )

    print("number of data: ", args.nsamples)
    print("model: ", args.model)
    print("cali_data: ", args.dataset)

    tick = time.time()
    # ori_ppl = cmoe_ppl_eval(model, testloader, args.dataset, args)
    # print(f"Original model ppl on {args.dataset}: {ori_ppl}")

    carved_model, tick_1, tick_2, pre_ppl, ppl = cmoe_sequential(model, tokenizer, dataloader, args)
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

    # if 'llama-3' in args.model.lower():
    #     name = "meta-llama/Meta-Llama-3-8B"
    # else:
    #     name = "meta-llama/Llama-2-7b-hf"

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