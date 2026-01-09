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
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompress import apply_quantization
from run_cmoe import *

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(        'model', type=str,
        help='Model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(        '--nsamples', type=int, default=128,
        help='Number of Fine-tuning data for CMoE.'
    )
    parser.add_argument(        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    args = parser.parse_args()
    
    print("Loading model: ", args.model.lower())
    if 'llava' in args.model.lower():
        model = get_llava(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    elif 'olmoe' in args.model.lower():
        model = get_olmoe(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    elif 'deepseek-v2-lite-gptq' in args.model.lower():
        model, tokenizer = get_deepseek_v2_lite_gptq(args.model)
    else:
        model = get_llama(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    print("model: ", args.model)

    ppl = []
    datasets = ['wikitext2', 'c4-new']
    # datasets = ['wikitext2', ]
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, tokenizer=tokenizer, seqlen=model.seqlen, bsz = 1
        )
        print(dataset)
        eval_set = dataset
        ppl_i = cmoe_ppl_eval(model, testloader, eval_set, args)
        ppl.append(f"{dataset}: {ppl_i}")
        print("PPL on {}: {:.4f}".format(dataset, ppl_i))
