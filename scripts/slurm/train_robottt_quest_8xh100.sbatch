#!/usr/bin/env bash
#SBATCH --account=p33200
#SBATCH --partition=gengpu
#SBATCH --job-name=robottt-s1-8xh100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=/gpfs/home/qge0476/project/yiyun/robottt-quest/logs/robottt-8xh100-%j.out
#SBATCH --error=/gpfs/home/qge0476/project/yiyun/robottt-quest/logs/robottt-8xh100-%j.err
#SBATCH --open-mode=append

set -euo pipefail

quest_root="${ROBOTTT_QUEST_ROOT:-/gpfs/home/qge0476/project/yiyun/robottt-quest}"
repo_dir="${quest_root}/Isaac-GR00T"
output_dir="${quest_root}/runs/stage1"
torchrun_bin="${repo_dir}/.venv/bin/torchrun"

test -x "${torchrun_bin}"
test -f "${quest_root}/data/subset/manifest.json"
test -d "${quest_root}/models/GR00T-N1.7-3B"
test -d "${quest_root}/hf-cache/hub/models--nvidia--Cosmos-Reason2-2B"
mkdir -p "${output_dir}"

master_addr="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
master_port="$((20000 + SLURM_JOB_ID % 20000))"

export ROBOTTT_REPO="${repo_dir}"
export ROBOTTT_TORCHRUN="${torchrun_bin}"
export ROBOTTT_OUTPUT_DIR="${output_dir}"
export ROBOTTT_MASTER_ADDR="${master_addr}"
export ROBOTTT_MASTER_PORT="${master_port}"
export ROBOTTT_RESUME_FROM_CHECKPOINT=""
if compgen -G "${output_dir}/checkpoint-*" >/dev/null; then
  export ROBOTTT_RESUME_FROM_CHECKPOINT=1
fi

export HF_HOME="${quest_root}/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-${USER}"

requeue_at_timeout() {
  echo "Requeueing ${SLURM_JOB_ID} from the latest periodic checkpoint" >&2
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
trap requeue_at_timeout USR1

srun --kill-on-bad-exit=1 --label bash -lc '
  set -euo pipefail
  cd "${ROBOTTT_REPO}"
  resume_args=()
  if [[ -n "${ROBOTTT_RESUME_FROM_CHECKPOINT}" ]]; then
    resume_args+=(--resume-from-checkpoint)
  fi
  exec "${ROBOTTT_TORCHRUN}" \
    --nnodes="${SLURM_NNODES}" \
    --nproc-per-node=4 \
    --node-rank="${SLURM_NODEID}" \
    --master-addr="${ROBOTTT_MASTER_ADDR}" \
    --master-port="${ROBOTTT_MASTER_PORT}" \
    gr00t/experiment/launch_robottt.py \
    --stage stage1 \
    --base-model-path /gpfs/home/qge0476/project/yiyun/robottt-quest/models/GR00T-N1.7-3B \
    --dataset-path /gpfs/home/qge0476/project/yiyun/robottt-quest/data/subset/pretrain:/gpfs/home/qge0476/project/yiyun/robottt-quest/data/subset/composite_seen:/gpfs/home/qge0476/project/yiyun/robottt-quest/data/subset/composite_unseen \
    --manifest-path /gpfs/home/qge0476/project/yiyun/robottt-quest/data/subset/manifest.json \
    --output-dir "${ROBOTTT_OUTPUT_DIR}" \
    --num-gpus 8 \
    --gradient-accumulation-steps 1 \
    --save-steps 1000 \
    --save-total-limit 1 \
    --dataloader-num-workers 0 \
    --tbptt-steps 1 \
    --backbone-micro-batch-size 8 \
    "${resume_args[@]}"
' &
launcher_pid=$!
wait "${launcher_pid}"
