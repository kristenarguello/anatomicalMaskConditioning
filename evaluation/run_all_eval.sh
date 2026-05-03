# ============================================================
# run_all_eval.sh — Full unified evaluation pipeline
#
# Runs inference for every model/variant, then computes metrics,
# permutation tests, and summary table. Sends ntfy notifications.
#
# Usage:
#   ./run_all_eval.sh <ntfy_topic> <gpu1> [gpu2]
#
# Examples:
#   ./run_all_eval.sh kristen-eval 3        # single GPU
#   ./run_all_eval.sh kristen-eval 3 0      # two GPUs (iDDPM+SegGuided split across both)
# ============================================================

TOPIC="${1:?Usage: ./run_all_eval.sh <ntfy_topic> <gpu1> [gpu2]}"
GPU1="${2:-3}"
GPU2="${3:-}"
GPUS="${GPU1}${GPU2:+,${GPU2}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOG_DIR}"

START_TIME=$(date +%s)

notify() {
    local title="$1" msg="$2" priority="${3:-default}" tags="${4:-computer}"
    curl -s \
        -H "Title: ${title}" \
        -H "Priority: ${priority}" \
        -H "Tags: ${tags}" \
        -d "${msg}" \
        "https://ntfy.sh/${TOPIC}" > /dev/null
}

die() {
    notify "Eval FAILED" "$1" urgent "x,rotating_light"
    echo "[ERROR] $1" >&2
    exit 1
}

step() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

PY="${PYTHON:-python3}"

# ============================================================
# 0. Notify start
# ============================================================
notify "Eval pipeline started" \
    "GPUs=${GPUS}  $(date '+%H:%M')" low "arrow_forward,computer"

# ============================================================
# 1. Inference — RED-CNN
# ============================================================
step "RED-CNN inference"
for V in no_mask mask; do
    LOG="${LOG_DIR}/infer_redcnn_${V}.log"
    echo "  variant=${V} → ${LOG}"
    CUDA_VISIBLE_DEVICES="${GPU1}" $PY "${SCRIPT_DIR}/infer_redcnn.py" \
        --variant "${V}" 2>&1 | tee "${LOG}" \
        || die "infer_redcnn.py --variant ${V} failed"
done

# ============================================================
# 2. Inference — CoreDiff
# ============================================================
step "CoreDiff inference"
for V in corediff_no_mask_noctx corediff_mask_noctx \
         corediff_no_mask_ctx corediff_mask_ctx; do
    LOG="${LOG_DIR}/infer_corediff_${V}.log"
    echo "  variant=${V} → ${LOG}"
    CUDA_VISIBLE_DEVICES="${GPU1}" $PY "${SCRIPT_DIR}/infer_corediff.py" \
        --variant "${V}" 2>&1 | tee "${LOG}" \
        || die "infer_corediff.py --variant ${V} failed"
done

# ============================================================
# 3. Inference — iDDPM  (splits slices across GPUs when GPU2 is set)
# ============================================================
step "iDDPM inference"
for V in no_mask mask; do
    LOG="${LOG_DIR}/infer_iddpm_${V}.log"
    echo "  variant=${V}  gpus=${GPUS} → ${LOG}"
    $PY "${SCRIPT_DIR}/infer_iddpm.py" --variant "${V}" \
        --gpus "${GPUS}" 2>&1 | tee "${LOG}" \
        || die "infer_iddpm.py --variant ${V} failed"
done

# ============================================================
# 4. Inference — SegGuidedDiff  (splits slices across GPUs when GPU2 is set)
# ============================================================
step "SegGuidedDiff inference"
for V in no_mask mask; do
    LOG="${LOG_DIR}/infer_segguideddiff_${V}.log"
    echo "  variant=${V}  gpus=${GPUS} → ${LOG}"
    $PY "${SCRIPT_DIR}/infer_segguideddiff.py" --variant "${V}" \
        --gpus "${GPUS}" 2>&1 | tee "${LOG}" \
        || die "infer_segguideddiff.py --variant ${V} failed"
done

notify "All inference done" \
    "Starting metric computation…  $(date '+%H:%M')" default "white_check_mark"

# ============================================================
# 5. Unified evaluation — per-slice metrics + qualitative images
# ============================================================
step "Unified evaluation"

# Maps: (result_dir_name, model_name, variant)
declare -a EVAL_TARGETS=(
    "redcnn_no_mask|redcnn|no_mask"
    "redcnn_mask|redcnn|mask"
    "corediff_no_mask_noctx|corediff|no_mask_noctx"
    "corediff_mask_noctx|corediff|mask_noctx"
    "corediff_no_mask_ctx|corediff|no_mask_ctx"
    "corediff_mask_ctx|corediff|mask_ctx"
    "iddpm_no_mask|iddpm|no_mask"
    "iddpm_mask|iddpm|mask"
    "segguideddiff_no_mask|segguideddiff|no_mask"  # pending
    "segguideddiff_mask|segguideddiff|mask"        # pending
)

for ENTRY in "${EVAL_TARGETS[@]}"; do
    IFS="|" read -r DIR_NAME MODEL_NAME VARIANT <<< "${ENTRY}"
    PRED_DIR="${RESULTS_DIR}/${DIR_NAME}/predictions"
    OUT_DIR="${RESULTS_DIR}/${DIR_NAME}"
    LOG="${LOG_DIR}/eval_${DIR_NAME}.log"

    if [[ ! -d "${PRED_DIR}" ]]; then
        echo "  [SKIP] no predictions at ${PRED_DIR}"
        continue
    fi

    echo "  ${DIR_NAME} → ${LOG}"
    $PY "${SCRIPT_DIR}/unified_eval.py" \
        --pred_dir   "${PRED_DIR}" \
        --out_dir    "${OUT_DIR}" \
        --model_name "${MODEL_NAME}" \
        --variant    "${VARIANT}" \
        2>&1 | tee "${LOG}" \
        || die "unified_eval.py failed for ${DIR_NAME}"
done

# ============================================================
# 6. Permutation tests — one per model family
# ============================================================
step "Permutation tests"

declare -a PERMTEST_PAIRS=(
    "redcnn_no_mask|redcnn_mask|redcnn"
    "corediff_no_mask_noctx|corediff_mask_noctx|corediff_noctx"
    "corediff_no_mask_ctx|corediff_mask_ctx|corediff_ctx"
    "iddpm_no_mask|iddpm_mask|iddpm"
    "segguideddiff_no_mask|segguideddiff_mask|segguideddiff"
)

for ENTRY in "${PERMTEST_PAIRS[@]}"; do
    IFS="|" read -r BASE_DIR MASK_DIR_NAME MODEL_LABEL <<< "${ENTRY}"
    B_CSV="${RESULTS_DIR}/${BASE_DIR}/per_slice_metrics.csv"
    M_CSV="${RESULTS_DIR}/${MASK_DIR_NAME}/per_slice_metrics.csv"
    P_OUT="${RESULTS_DIR}/permtest_${MODEL_LABEL}.csv"
    LOG="${LOG_DIR}/permtest_${MODEL_LABEL}.log"

    if [[ ! -f "${B_CSV}" ]] || [[ ! -f "${M_CSV}" ]]; then
        echo "  [SKIP] missing CSV for ${MODEL_LABEL}"
        continue
    fi

    echo "  ${MODEL_LABEL} → ${LOG}"
    $PY "${SCRIPT_DIR}/run_permutation_test.py" \
        --baseline "${B_CSV}" \
        --masked   "${M_CSV}" \
        --model    "${MODEL_LABEL}" \
        --out_csv  "${P_OUT}" \
        2>&1 | tee "${LOG}"
done

# ============================================================
# 7. Summary table
# ============================================================
step "Summary table"
$PY "${SCRIPT_DIR}/build_summary_table.py" \
    --results_root "${RESULTS_DIR}" \
    2>&1 | tee "${LOG_DIR}/summary_table.log"

# ============================================================
# Done
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
H=$(( ELAPSED / 3600 ))
M=$(( (ELAPSED % 3600) / 60 ))
S=$(( ELAPSED % 60 ))
DURATION="${H}h ${M}m ${S}s"

echo ""
echo "All done in ${DURATION}."
echo "Results in: ${RESULTS_DIR}"

notify "Eval pipeline DONE ✓" \
    "Finished in ${DURATION}. Results: ${RESULTS_DIR}" \
    default "white_check_mark,tada"
