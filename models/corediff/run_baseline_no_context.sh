#!/usr/bin/env bash
# Train CoreDiff baseline without inter-slice context (single-channel input).
# Usage: ./run_baseline_no_context.sh [ntfy_topic]

TOPIC="${1:-}"
GPUS="${GPUS:-0,1}"
IMG_ROOT="${IMG_ROOT:-./dataset/png_dataset}"

CMD=(python main.py
    --model_name          corediff
    --run_name            baseline_noctx
    --img_root            "${IMG_ROOT}"
    --in_channels         1
    --batch_size          8
    --max_iter            150000
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
