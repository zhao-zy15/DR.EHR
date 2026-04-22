#!/bin/bash
# ------------------------------------------------------------------
# Stage 2 — fine-tune the Stage-1 checkpoint on LLM-generated positives.
# Input = data/stage_2/train_entities.json
#
#   MODEL=small (default) → BGE / Bio_ClinicalBERT → DR.EHR-small
#   MODEL=large            → NV-Embed-v2 (LoRA)     → DR.EHR-large
#
# Paper hyper-parameters (Table III):
#   small : num_pos=16, batch=32, epoch=1
#   large : num_pos=16, batch=16, epoch=1
# ------------------------------------------------------------------
set -e
cd "$(dirname "$0")/.."

MODEL=${MODEL:-small}

case "$MODEL" in
    small)
        ENCODER=bge
        BSZ=32; POS=16; EPOCH=1; LR=1e-4
        EXTRA=""
        ;;
    large)
        ENCODER=nv
        BSZ=16; POS=16; EPOCH=1; LR=1e-4
        EXTRA="--deepspeed ds_config.json"
        ;;
    *)
        echo "Unknown MODEL=$MODEL (expected: small | large)"; exit 1
        ;;
esac

# Resume from the Stage-1 checkpoint produced by scripts/stage_1.sh
CKPT=${CKPT:-../output/stage_1_$MODEL}

torchrun --nproc_per_node 8 main.py --do_train \
    --encoder_type "$ENCODER" \
    --model_name_or_path "$CKPT" \
    --output_dir ../output/stage_2_"$MODEL" \
    --data_dir ../data/stage_2/train_entities.json \
    --learning_rate "$LR" \
    --num_train_epochs "$EPOCH" \
    --per_device_train_batch_size "$BSZ" \
    --num_pos "$POS" \
    --max_length 160 \
    --max_entity_length 16 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --save_steps 1000000 \
    --dataloader_num_workers 16 \
    --dataloader_drop_last False \
    --remove_unused_columns False \
    --optim adamw_apex_fused \
    $EXTRA
