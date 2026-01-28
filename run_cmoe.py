import time

import torch
import torch.nn as nn

from tqdm import *

import os 

import copy

from CMoE_utils import *
from CMoE_model import *
from CMoE_sequential import *
from sft_utils import simple_sft
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_cmoe import cmoe_ppl_eval, load_model

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
    parser.add_argument(        '--epoch', type=int, default=0,
        help='SFT epoch for CMoE.'
    )
    parser.add_argument(        '--sft-bsz', type=int, default=1,
        help='SFT batch size for CMoE.'
    )
    parser.add_argument(        '--carve-bsz', type=int, default=1,
        help='Carve batch size for CMoE.'
    )
    parser.add_argument(        '--eval-zero', action='store_true',
        help='Whether to run downstream tasks evaluation.'
    )
    parser.add_argument(        '--prefix', type=str, default=None,
        help='Prefix the results folder if needed.'
    )
    parser.add_argument(        '--quant-scheme', type=str, default=None,
        help='Quantization scheme like a8s4m3221.'
    )
    parser.add_argument(        '--reconstruct_start_layer', type=int, default=0,
        help='Start layer for reconstruction.'
    )
    parser.add_argument(        '--reconstruct_end_layer', type=int, default=15,
        help='End layer for reconstruction.'
    )

    args = parser.parse_args()
    
    print("-" * 50)
    print("Loading model: ", args.model)
    model, tokenizer = load_model(args.model)

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
    save_carved_model = False
    if save_carved_model:
        carved_save_dir = "model/carved_olmoe_cmoe_e" + str(args.nexperts) + "_a" + str(args.nactivated)
        print(carved_model)
        carved_model.save_pretrained(carved_save_dir)
        tokenizer.save_pretrained(carved_save_dir)

    rt_construct = tick_1 - tick
    extra_time = tick_2 - tick_1
    rt = time.time() - tick - extra_time
    print(f"Runtime of training-free construction: {rt_construct:.2f}")
    print(f"Runtime of fine-tuning construction: {rt:.2f}")
