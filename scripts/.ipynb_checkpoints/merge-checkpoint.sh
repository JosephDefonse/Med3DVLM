#!/bin/bash

export PYTHONPATH="$(pwd)"

export TRANSFORMERS_CACHE=/home/sdef0001/iq38_scratch/nmdid/cache/transformers
export HF_HOME=/home/sdef0001/iq38_scratch/nmdid/cache/huggingface
export XDG_CACHE_HOME=/home/sdef0001/iq38_scratch/nmdid/cache/xdg

python src/utils/merge_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_dir ./output/Med3DVLM-Qwen-2.5-7B-lora-adapter \
  --out_dir   ./models/Med3DVLM-Qwen-2.5-7B
