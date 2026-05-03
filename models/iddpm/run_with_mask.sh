#!/usr/bin/env bash
# Train iDDPM with mask conditioning — label map appended as a third input channel.
# Usage: ./run_with_mask.sh [ntfy_topic]

TOPIC="${1:-}"
GPUS="${GPUS:-0,1}"

DATA_DIR="${DATA_DIR:-./dataset/png_dataset/train,./dataset/png_dataset/val}"
VAL_DIR="${VAL_DIR:-./dataset/png_dataset/test}"
MASK_ROOT="${MASK_ROOT:-./dataset/multilabel}"
LOG_DIR="${LOG_DIR:-./runs/with_mask}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export OPENAI_LOGDIR="${LOG_DIR}"
mkdir -p "${LOG_DIR}"

N_GPUS=$(echo "${GPUS}" | tr ',' '\n' | wc -l | tr -d ' ')
export GPUS_PER_NODE="${N_GPUS}"

# Resume from latest checkpoint if one exists
RESUME_ARG=""
LATEST=$(ls "${LOG_DIR}"/model*.pt 2>/dev/null | sort -V | tail -1)
[[ -n "${LATEST}" ]] && RESUME_ARG="--resume_checkpoint ${LATEST}"

CMD=(mpiexec -n "${N_GPUS}" python scripts/image_train.py
    --data_dir                  "${DATA_DIR}"
    --use_mask                  True
    --mask_root                 "${MASK_ROOT}"
    --image_size                512
    --image_channels            1
    --num_channels              64
    --num_res_blocks            2
    --attention_resolutions     32,16,8
    --learn_sigma               True
    --diffusion_steps           1000
    --noise_schedule            cosine
    --rescale_learned_sigmas    True
    --batch_size                6
    --microbatch                -1
    --lr                        1.5e-4
    --ema_rate                  0.9999
    --lr_anneal_steps           100000
    --save_interval             2500
    --log_interval              100
    --val_data_dir              "${VAL_DIR}"
    --val_interval              2500
    --val_num_samples           16
    --trans_noise_level         0.4
    ${RESUME_ARG}
)

if [[ -n "${TOPIC}" ]]; then
    exec "$(dirname "$0")/notify_run.sh" "${TOPIC}" "${CMD[@]}"
else
    exec "${CMD[@]}"
fi
