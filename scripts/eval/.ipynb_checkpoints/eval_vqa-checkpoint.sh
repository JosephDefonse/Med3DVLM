#!/bin/bash

python src/eval/eval_vqa.py \
    --model_name_or_path ./models/Med3DVLM-Qwen-2.5-7B \
    --data_root /home/sdef0001/iq38_scratch/nmdid \
    --vqa_data_test_path /home/sdef0001/iq38_scratch/nmdid/nmdid_vqa_test.csv \
    --max_length 512 \
    --proj_out_num 256 \
    --do_sample \
    --output_dir ./output/eval