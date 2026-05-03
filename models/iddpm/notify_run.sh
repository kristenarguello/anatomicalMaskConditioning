#!/usr/bin/env bash
# Wrapper: runs a command and sends an ntfy notification on finish or crash.
# Also monitors GPU idle state and notifies if <1 GB usage for >1 minute.
# Usage: ./notify_run.sh <ntfy_topic> <command...>

TOPIC="${1:?Provide the ntfy topic as the first argument}"
shift
CMD=("$@")

NTFY_URL="https://ntfy.sh/${TOPIC}"
START_TIME=$(date +%s)

notify() {
    local title="$1" msg="$2" priority="$3" tags="$4"
    curl -s \
        -H "Title: ${title}" \
        -H "Priority: ${priority}" \
        -H "Tags: ${tags}" \
        -d "${msg}" \
        "${NTFY_URL}" > /dev/null
}

echo "[notify_run] Starting: ${CMD[*]}"
notify "Training started" "Command: ${CMD[*]}" "low" "arrow_forward,computer"

# Background GPU idle monitor: notifies if all selected GPUs use <1 GB for >1 minute
GPU_MONITOR_PID=""
if command -v nvidia-smi &>/dev/null && [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    _monitor_gpu_idle() {
        local gpus="${CUDA_VISIBLE_DEVICES}"
        local idle_since=0 notified=0
        while true; do
            sleep 15
            local all_idle=1
            for gpu_id in $(echo "$gpus" | tr ',' ' '); do
                local mem_used
                mem_used=$(nvidia-smi --id="${gpu_id}" --query-gpu=memory.used \
                    --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
                if [[ -z "$mem_used" ]] || (( mem_used >= 1024 )); then
                    all_idle=0; break
                fi
            done
            if (( all_idle )); then
                (( idle_since == 0 )) && idle_since=$(date +%s)
                local elapsed=$(( $(date +%s) - idle_since ))
                if (( elapsed >= 60 && notified == 0 )); then
                    notify "GPUs idle" "GPUs ${gpus} using <1 GB for over 1 minute." "high" "warning,computer"
                    notified=1
                fi
            else
                idle_since=0; notified=0
            fi
        done
    }
    _monitor_gpu_idle &
    GPU_MONITOR_PID=$!
fi

"${CMD[@]}"
EXIT_CODE=$?

[[ -n "${GPU_MONITOR_PID}" ]] && kill "${GPU_MONITOR_PID}" 2>/dev/null

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
DURATION="$(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m $(( ELAPSED%60 ))s"

if [ "${EXIT_CODE}" -eq 0 ]; then
    notify "Training done" "Finished in ${DURATION}." "default" "white_check_mark,tada"
    echo "[notify_run] Finished successfully in ${DURATION}."
else
    notify "Training FAILED" "Exit code ${EXIT_CODE} after ${DURATION}." "urgent" "x,rotating_light"
    echo "[notify_run] FAILED with exit code ${EXIT_CODE} after ${DURATION}."
fi

exit "${EXIT_CODE}"
