# export CUDA_VISIBLE_DEVICES=0
# MODEL_PATH=""
# python run_cmoe.py $MODEL_PATH wikitext2 --new-eval --nshared 2 --nactivated 2 --nexperts 16 --nsamples 64 --extra-lr 0.001 --bias-speed 0.001
#       https://huggingface.co/allenai/OLMoE-1B-7B-0924

export HF_DATASETS_OFFLINE=1 
# python run_cmoe.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --new-eval --nshared 0 --nactivated 64 --nexperts 512 --nsamples 16 --extra-lr 0.001 --bias-speed 0.001 --sft-bsz 1 --carve-bsz 1 --epoch 0
# python run_cmoe.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --new-eval --nshared 0 --nactivated 32 --nexperts 256 --nsamples 16 --extra-lr 0.001 --bias-speed 0.001 --sft-bsz 1 --carve-bsz 1 --epoch 0
python run_cmoe.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --new-eval --nshared 0 --nactivated 8  --nexperts 64  --nsamples 16 --extra-lr 0.001 --bias-speed 0.001 --sft-bsz 1 --carve-bsz 1 --epoch 0