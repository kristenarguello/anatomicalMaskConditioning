#!/usr/bin/env bash
# Test both RED-CNN variants (baseline and masked) and print per-slice metrics.
# Usage: ./run_test.sh

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

DATA_PATH="${DATA_PATH:-./dataset/png_dataset}"
MASK_DIR="${MASK_DIR:-./dataset/multilabel}"

echo "=== Testing baseline ==="
python main.py \
    --mode test \
    --load_best \
    --data_path "${DATA_PATH}" \
    --mask_dir  "${MASK_DIR}" \
    --save_path ./save/baseline \
    --num_workers 6

echo ""
echo "=== Testing masked ==="
python main.py \
    --mode test \
    --load_best \
    --use_mask \
    --data_path "${DATA_PATH}" \
    --mask_dir  "${MASK_DIR}" \
    --save_path ./save/masked \
    --num_workers 6
