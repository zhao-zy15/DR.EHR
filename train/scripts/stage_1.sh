#!/bin/bash
# ------------------------------------------------------------------
# Stage 1 — entity-level pretraining on BIOS-matched + abbr + KG
# augmented positives.
# Input = data/stage_1/train_entities_abbr_syn.json
#
#   MODEL=small (default) → BGE / Bio_ClinicalBERT → DR.EHR-small
#   MODEL=large            → NV-Embed-v2 (LoRA)     → DR.EHR-large
#
# Paper hyper-parameters (Table III):
#   small : num_pos=128, batch=32, epoch=3
#   large : num_pos= 32, batch=16, epoch=1
# ------------------------------------------------------------------
set -e
cd "$(dirname "$0")/.."

MODEL=${MODEL:-small}

case "$MODEL" in
    small)
        ENCODER=bge
        MODEL_PATH=${MODEL_PATH:-BAAI/bge-base-en-v1.5}
        BSZ=32; POS=128; EPOCH=3; LR=1e-4
        EXTRA=""
        ;;
    large)
        ENCODER=nv
        MODEL_PATH=${MODEL_PATH:-nvidia/NV-Embed-v2}
        BSZ=16; POS=32;  EPOCH=1; LR=1e-4
        EXTRA="--deepspeed ds_config.json"
        ;;
    *)
        echo "Unknown MODEL=$MODEL (expected: small | large)"; exit 1
        ;;
esac

torchrun --nproc_per_node 8 main.py --do_train \
    --encoder_type "$ENCODER" \
    --model_name_or_path "$MODEL_PATH" \
    --output_dir ../output/stage_1_"$MODEL" \
    --data_dir ../data/stage_1/train_entities_abbr_syn.json \
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
