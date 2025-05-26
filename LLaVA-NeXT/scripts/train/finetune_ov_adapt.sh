#!/bin/bash

export OMP_NUM_THREADS=1
LLM_VERSION="lmms-lab/llava-onevision-qwen2-0.5b-ov" 
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-adapter-only" 
echo "MID_RUN_NAME: ${RUN_NAME}"

# Create output directory
mkdir -p "/home/sajjad/AI_proj/outputs/${RUN_NAME}"

# Set checkpoint path
CKPT_PATH=$LLM_VERSION

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Mixed precision fix - using bf16 instead of fp16
CUDA_VISIBLE_DEVICES=0 python llava/train/train_mem.py \
    --model_name_or_path ${CKPT_PATH} \
    --version $PROMPT_VERSION \
    --data_path /home/sajjad/AI_proj/LLaVA-NeXT/scripts/train/onevision.yaml \
    --image_folder /home/sajjad/AI_proj/example_data/images \
    --mm_tunable_parts="mm_mlp_adapter" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type linear \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --image_grid_pinpoints "(1x1)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --fp16 False \
    --run_name $RUN_NAME \
    --output_dir "/home/sajjad/AI_proj/outputs/${RUN_NAME}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --model_max_length 128 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --dataloader_drop_last True \
    --attn_implementation "eager" \
    --max_grad_norm 0.3