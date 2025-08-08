#!/bin/bash

export PYTHONPATH="$(pwd)"

export TRANSFORMERS_CACHE=/home/sdef0001/iq38_scratch/nmdid/cache/transformers
export HF_HOME=/home/sdef0001/iq38_scratch/nmdid/cache/huggingface
export XDG_CACHE_HOME=/home/sdef0001/iq38_scratch/nmdid/cache/xdg

python src/utils/merge_lora_weights_and_save_hf_model2.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --model_type vlm_qwen \
    --vision_tower dcformer \
    --mm_projector_type mixer \
    --pretrain_vision_model ./output/DCFormer_SigLIP/pretrained_ViT.bin \
    --adapter_dir ./output/Med3DVLM-Qwen-2.5-7B-finetune_vqa_18/ADAPTER_ONLY \
    --non_lora_path ./output/Med3DVLM-Qwen-2.5-7B-finetune_vqa_18/non_lora_weights.bin \
    --output_dir ./models/Med3DVLM-Qwen-2.5-7B-vqa-18