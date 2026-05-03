#!/usr/bin/env bash
# Train RED-CNN baseline (single-channel input, no mask conditioning).
# Usage: ./run_baseline.sh [ntfy_topic]

TOPIC="${1:-}"
GPU="${GPU:-0}"

DATA_PATH="${DATA_PATH:-./dataset/png_dataset}"
MASK_DIR="${MASK_DIR:-./dataset/multilabel}"

CMD=(python main.py
    --mode train
    --data_path "${DATA_PATH}"
    --mask_dir  "${MASK_DIR}"
    --save_path ./save/baseline
    --lr              1e-4
    --num_epochs      100
    --batch_size      128
    --patch_n         10
    --patch_size      64
    --save_iters      1000
    --save_freq       10
    --print_iters     20
    --decay_iters     3000
    --num_workers     6
)

export CUDA_VISIBLE_DEVICES="${GPU}"

if [[ -n "${TOPIC}" ]]; then
    exec "$(dirname "$0")/notify_run.sh" "${TOPIC}" "${CMD[@]}"
else
    exec "${CMD[@]}"
fi
