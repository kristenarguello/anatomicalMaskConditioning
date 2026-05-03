#!/usr/bin/env bash
# Train SegGuidedDiff baseline — SDEdit translation without mask conditioning.
# Usage: ./run_baseline.sh [ntfy_topic]

TOPIC="${1:-}"
GPUS="${GPUS:-0,1}"
IMG_DIR="${IMG_DIR:-dataset/png_dataset}"

CMD=(python main.py
    --mode              train
    --model_type        DDPM
    --img_size          512
    --num_img_channels  1
    --img_dir           "${IMG_DIR}"
    --train_batch_size  8
    --learning_rate     5e-5
    --num_epochs        200
)

export CUDA_VISIBLE_DEVICES="${GPUS}"
exec "${CMD[@]}"
