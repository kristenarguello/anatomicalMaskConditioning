#!/usr/bin/env bash
# Train SegGuidedDiff with mask conditioning (single-channel label map appended at each denoising step).
# Usage: ./run_masked.sh [ntfy_topic]

TOPIC="${1:-}"
GPUS="${GPUS:-0,1}"
IMG_DIR="${IMG_DIR:-dataset/png_dataset}"
SEG_DIR="${SEG_DIR:-dataset/multilabel}"

CMD=(python main.py
    --mode                        train
    --model_type                  DDPM
    --img_size                    512
    --num_img_channels            1
    --img_dir                     "${IMG_DIR}"
    --seg_dir                     "${SEG_DIR}"
    --segmentation_guided
    --segmentation_channel_mode   single
    --num_segmentation_classes    5
    --train_batch_size            8
    --learning_rate               5e-5
    --num_epochs                  200
)

export CUDA_VISIBLE_DEVICES="${GPUS}"
exec "${CMD[@]}"
