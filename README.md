# MoE

## Dependencies

```bash
conda create -n cmoe python=3.11
conda activate cmoe
conda install pytorch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 pytorch-cuda=12.8 -c pytorch -c nvidia
pip install datasets==4.4.1
pip install transformers==4.57.5
pip install accelerate==1.12.0
pip install sentencepiece==0.2.0
pip install protobuf==6.33.3
pip install matplotlib==3.10.0
pip install lap==0.5.12
pip install peft==0.14.0
```
Note: please modify the version of some packages for your own environment.

## Supported Models

OLMoE / Deepseek(v1)-MoE-16B0-base / Deepseek-v2-lite(16B-A3B)

## Quick Start

You can run the pre-defined testing script 'run.sh' by:
```bash
bash run.sh
```

```python
python run_cmoe.py $MODEL_PATH wikitext2 \ 
--nshared 2 \
--nactivated 2 \
--nexperts 16 \
--nsamples 2048 \
--extra-lr 0.001 --bias-speed 0.001 --new-eval
```

## Evaluation

bash run.sh
```

```python
python eval_cmoe.py $MODEL_PATH 
```

## Code sources

Framework code is referenced from: https://github.com/JarvisPei/CMoE
GPTQ code is referenced from: https://github.com/cat538/MxMoE
Important code is inspired from: https://github.com/xuyuzhuang11/CAMERA

## Cite

If you found this work useful, please consider citing:

```

```
