#!/bin/bash

export PYTHONPATH="$(pwd)"

export TRANSFORMERS_CACHE=/home/sdef0001/iq38_scratch/nmdid/cache/transformers
export HF_HOME=/home/sdef0001/iq38_scratch/nmdid/cache/huggingface
export XDG_CACHE_HOME=/home/sdef0001/iq38_scratch/nmdid/cache/xdg

python src/demo/demo.py \
    --model_name_or_path ./models/Med3DVLM-Qwen-2.5-7B-vqa-8 \
    --data_root /home/sdef0001/iq38_scratch/nmdid \
    --vqa_data_test_path /home/sdef0001/iq38_scratch/nmdid/niigz/case-104228/THIN_BONE_HEAD.nii.gz \
    --max_length 512 \
    --proj_out_num 256 \