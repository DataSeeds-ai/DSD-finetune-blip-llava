#!/bin/bash

export OMP_NUM_THREADS=1
LLM_VERSION="lmms-lab/llava-onevision-qwen2-0.5b-ov" 
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-minimal-adpt_lora16" 
echo "MID_RUN_NAME: ${RUN_NAME}"

# Create output directory
mkdir -p "/home/sajjad/AI_proj/outputs/${RUN_NAME}"

# Set checkpoint path
CKPT_PATH=$LLM_VERSION

EPOCHS=10

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Mixed precision fix - using bf16 instead of fp16
CUDA_VISIBLE_DEVICES=0 python llava/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --version $PROMPT_VERSION \
    --data_path /home/sajjad/AI_proj/LLaVA-NeXT/scripts/train/onevision.yaml \
    --eval_data_path /home/sajjad/AI_proj/example_data/val_data_guru.json \
    --image_folder /home/sajjad/AI_proj/example_data \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir "/home/sajjad/AI_proj/outputs/${RUN_NAME}" \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 25 \
    --save_steps 25 \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --dataloader_drop_last True \
    --attn_implementation "sdpa" \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 16 \
    --report_to wandb \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --logging_nan_inf_filter False \
    --logging_first_step True \
    --lora_dropout 0.1 \
    --max_grad_norm 0.5