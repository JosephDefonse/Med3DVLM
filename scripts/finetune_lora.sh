#!/bin/bash
export PYTHONPATH="$(pwd)"

# 1) HuggingFace “hub” & Transformers cache
export HF_HOME=/home/sdef0001/iq38_scratch/nmdid/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# 2) Triton autotune cache
export TRITON_CACHE_DIR=/home/sdef0001/iq38_scratch/nmdid/cache/triton

# make sure those dirs exist
mkdir -p $TRANSFORMERS_CACHE $TRITON_CACHE_DIR

deepspeed src/train/train_vlm.py \
    --deepspeed ./scripts/zero2.json \
    --wb_name Med3DVLM-Qwen-2.5-7B-finetune \
    --vision_tower "dcformer" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --model_type vlm_qwen \
    --mm_projector_type "mixer" \
    --lora_enable True \
    --vision_select_layer -2 \
    --pretrain_vision_model ./output/DCFormer_SigLIP/pretrained_ViT.bin \
    --pretrain_mm_mlp_adapter ./output/Med3DVLM-Qwen-2.5-7B/mm_projector.bin \
    --data_root /home/sdef0001/iq38_scratch/nmdid \
    --vqa_data_train_path /home/sdef0001/iq38_scratch/nmdid/nmdid_cause_vqa_13_train.csv \
    --vqa_data_val_path   /home/sdef0001/iq38_scratch/nmdid/nmdid_cause_vqa_13_val.csv \
    --vqa_data_test_path  /home/sdef0001/iq38_scratch/nmdid/nmdid_cause_vqa_13_test.csv \
    --bf16 True \
    --output_dir ./output/Med3DVLM-Qwen-2.5-7B-finetune_vqa_31 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --metric_for_best_model "eval_choice_acc" \
    --greater_is_better True \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 3 \
    --learning_rate 3e-5 \
    --max_grad_norm 0.5 \
    --label_smoothing_factor 0.05 \
    --weight_decay 0.1 \
    --warmup_ratio 0.10 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing True \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4