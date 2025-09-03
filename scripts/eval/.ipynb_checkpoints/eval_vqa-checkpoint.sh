#!/bin/bash

export PYTHONPATH="$(pwd)"

export TRANSFORMERS_CACHE=/home/sdef0001/iq38_scratch/nmdid/cache/transformers
export HF_HOME=/home/sdef0001/iq38_scratch/nmdid/cache/huggingface
export XDG_CACHE_HOME=/home/sdef0001/iq38_scratch/nmdid/cache/xdg

python src/eval/eval_vqa.py \
    --model_name_or_path ./models/Med3DVLM-Qwen-2.5-7B-vqa-20 \
    --data_root /home/sdef0001/iq38_scratch/nmdid \
    --vqa_data_test_path /home/sdef0001/iq38_scratch/nmdid/nmdid_cause_vqa_8_test.csv \
    --max_length 1024 \
    --proj_out_num 256 \
    --do_sample \
    --output_dir ./output/eval