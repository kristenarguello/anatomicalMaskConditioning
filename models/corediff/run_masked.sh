#!/usr/bin/env bash
# Train CoreDiff with mask conditioning and inter-slice context (4-channel input: prev/curr/next/mask).
# Usage: ./run_masked.sh [ntfy_topic]

TOPIC="${1:-}"
GPUS="${GPUS:-0,1}"
IMG_ROOT="${IMG_ROOT:-./dataset/png_dataset}"
MASK_DIR="${MASK_DIR:-./dataset/multilabel}"

CMD=(python main.py
    --model_name          corediff
    --run_name            masked_ctx
    --img_root            "${IMG_ROOT}"
    --mask_dir            "${MASK_DIR}"
    --in_channels         3
    --batch_size          8
    --max_iter            150000
    --context
    --only_adjust_two_step
    --dose                25
    --save_freq           2500
)

export CUDA_VISIBLE_DEVICES="${GPUS}"
if [[ -n "${TOPIC}" ]]; then
    exec "$(dirname "$0")/notify_run.sh" "${TOPIC}" "${CMD[@]}"
else
    exec "${CMD[@]}"
fi
