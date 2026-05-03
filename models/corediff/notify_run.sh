#!/usr/bin/env bash
# Wrapper: runs a command and sends an ntfy notification on finish or crash.
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

"${CMD[@]}"
EXIT_CODE=$?

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
