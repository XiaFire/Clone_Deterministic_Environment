#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"

if [[ -n "${CONDA_SH:-}" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "${CONDA_ENV:-grwm}"
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_DIR="${TRAIN_DIR:?Set TRAIN_DIR to the Maze9 training directory, e.g. /path/to/9x9_n0.8}"
VAL_DIR="${VAL_DIR:?Set VAL_DIR to the Maze9 validation directory, e.g. /path/to/9x9_val}"

mkdir -p logs/maze9 outputs/maze9_metrics ckpts

AE_EPOCHS="${AE_EPOCHS:-200}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-64}"
AE_MICRO_BATCH_SIZE="${AE_MICRO_BATCH_SIZE:-512}"
AE_GPU_VAE="${AE_GPU_VAE:-0}"
AE_GPU_GRWM="${AE_GPU_GRWM:-1}"

PRECOMPUTE_GPUS="${PRECOMPUTE_GPUS:-0,1,2,3,4,5,6,7}"
DYNAMICS_GPUS="${DYNAMICS_GPUS:-0,1,2,3,4,5,6,7}"
DYNAMICS_STEPS="${DYNAMICS_STEPS:-20000}"
DYNAMICS_BATCH_SIZE="${DYNAMICS_BATCH_SIZE:-64}"
PART_GPUS="${PART_GPUS:-0,1,2,3,4,5,6,7}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
PRE_K="${PRE_K:-15}"
CONTEXT_LEN="${CONTEXT_LEN:-15}"
SAMPLING_TIMESTEPS="${SAMPLING_TIMESTEPS:-10}"
DYN_CONFIG="${DYN_CONFIG:-configs/mc_unet_maze9.yaml}"

LATENT_ROOT="${LATENT_ROOT:-$(dirname "$TRAIN_DIR")}"

VAE_RUN="${VAE_RUN:-maze9_vae32}"
GRWM_RUN="${GRWM_RUN:-maze9_grwm32}"

VAE_AE_CKPT="checkpoints/mcvae32_no_transformer/${VAE_RUN}/last.ckpt"
GRWM_AE_CKPT="checkpoints/mcgr32/${GRWM_RUN}/last.ckpt"

VAE_LATENT_DIR="${VAE_LATENT_DIR:-${LATENT_ROOT}/maze9_tmp_vae32}"
GRWM_LATENT_DIR="${GRWM_LATENT_DIR:-${LATENT_ROOT}/maze9_tmp_grwm32}"

VAE_DYN_NAME="${VAE_DYN_NAME:-maze9_vae32_len64_latent_pred_steps${DYNAMICS_STEPS}}"
GRWM_DYN_NAME="${GRWM_DYN_NAME:-maze9_grwm32_len64_latent_pred_steps${DYNAMICS_STEPS}}"

VAE_EVAL_NAME="${VAE_EVAL_NAME:-maze9_vae32_dyna${DYNAMICS_STEPS}_sample${SAMPLING_TIMESTEPS}_p${PRE_K}c${CONTEXT_LEN}}"
GRWM_EVAL_NAME="${GRWM_EVAL_NAME:-maze9_grwm32_dyna${DYNAMICS_STEPS}_sample${SAMPLING_TIMESTEPS}_p${PRE_K}c${CONTEXT_LEN}}"

count_csv_items() {
  local value="$1"
  awk -F',' '{print NF}' <<< "$value"
}

wait_for_jobs() {
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  return "$status"
}

train_ae() {
  local exp="$1"
  local run="$2"
  local slow_weight="$3"
  local uniform_weight="$4"
  local gpu="$5"
  local master_port="$6"
  local log_file="$7"

  echo "[$(date '+%F %T')] AE train ${run}: gpu=${gpu}, epochs=${AE_EPOCHS}, batch=${AE_BATCH_SIZE}" | tee "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" MASTER_ADDR=127.0.0.1 MASTER_PORT="$master_port" \
    stdbuf -oL -eL python pipeline.py \
      --exp "$exp" \
      --run "$run" \
      dataset=maze \
      'gpus=[0]' \
      data.train_dir="$TRAIN_DIR" \
      data.val_dir="$VAL_DIR" \
      data.scale=4.5 \
      data.pose_center=1 \
      data.num_workers=2 \
      data.batch_size="$AE_BATCH_SIZE" \
      data.max_length=64 \
      data.train_subset_size=0 \
      data.val_subset_size=0 \
      model.model_name=vae-32-small \
      'model.map_size=(4,4,4)' \
      model.ae_micro_batch_size="$AE_MICRO_BATCH_SIZE" \
      model.uniform_max_points=4096 \
      model.uniform_chunk_size=512 \
      training.epochs="$AE_EPOCHS" \
      loss.cls=MaskedLPIPSLoss \
      loss.kwargs.perceptual_weight=0.0 \
      loss.kwargs.disc_weight=0.0 \
      loss.kwargs.disc_factor=0.0 \
      loss.kwargs.slow_weight="$slow_weight" \
      loss.kwargs.uniform_weight="$uniform_weight" \
      2>&1 | tee -a "$log_file"
}

run_precompute() {
  local label="$1"
  local ckpt="$2"
  local output_dir="$3"
  local log_file="$4"
  local num_gpus
  num_gpus="$(count_csv_items "$PRECOMPUTE_GPUS")"

  [[ -f "$ckpt" ]] || { echo "Missing AE checkpoint: $ckpt" >&2; exit 1; }

  echo "[$(date '+%F %T')] Precompute ${label} latents -> ${output_dir}" | tee "$log_file"
  stdbuf -oL -eL python precompute_latent_new.py \
    --ckpt "$ckpt" \
    --cls grvae \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --output_dir "$output_dir" \
    --type maze \
    --scale 4.5 \
    --size 32 \
    --chunk_size 64 \
    --chunks_per_batch 4 \
    --frames_per_sample 64 \
    --window_stride 64 \
    --num_gpus "$num_gpus" \
    --gpu_ids "$PRECOMPUTE_GPUS" \
    2>&1 | tee -a "$log_file"
}

run_dynamics() {
  local label="$1"
  local name="$2"
  local latent_dir="$3"
  local log_file="$4"

  echo "[$(date '+%F %T')] Dynamics train ${label}: steps=${DYNAMICS_STEPS}" | tee "$log_file"
  stdbuf -oL -eL python dyna_efficient.py \
    --name "$name" \
    --config "$DYN_CONFIG" \
    --input_key latents \
    --precomputed True \
    --train_path "${latent_dir}/train" \
    --val_path "${latent_dir}/val" \
    --type maze \
    --gpu "$DYNAMICS_GPUS" \
    --batch_size "$DYNAMICS_BATCH_SIZE" \
    --learning_rate 1e-4 \
    --steps "$DYNAMICS_STEPS" \
    --save_steps "$DYNAMICS_STEPS" \
    --val_freq -1 \
    --max_length 64 \
    --pre_k "$PRE_K" \
    --context_len "$CONTEXT_LEN" \
    --load_images False \
    --size 32 \
    'x_shape=[4,4,4]' \
    noise_level=random_uniform \
    lr_scheduler.num_warmup_steps=500 \
    "lr_scheduler.num_training_steps=${DYNAMICS_STEPS}" \
    2>&1 | tee -a "$log_file"
}

run_mse_part() {
  local name="$1"
  local part_idx="$2"
  local gpu="$3"
  local num_parts="$4"
  local dyna_ckpt="$5"
  local ae_ckpt="$6"
  local metadata_dir="$7"
  local log_prefix="$8"
  local part_name="${name}_part${part_idx}"
  local log_file="logs/maze9/${log_prefix}_part${part_idx}.log"

  echo "[$(date '+%F %T')] Eval ${part_name}: gpu=${gpu}" | tee "$log_file"
  stdbuf -oL -eL python evaluate_maze9_rollout_mse.py \
    --mode latent \
    --name "$part_name" \
    --dyna_config "$DYN_CONFIG" \
    --dyna_ckpt "$dyna_ckpt" \
    --vae_ckpt "$ae_ckpt" \
    --metadata_dir "${metadata_dir}/val" \
    --maze_dir "$VAL_DIR" \
    --output_dir outputs/maze9_metrics \
    --num_samples "$NUM_SAMPLES" \
    --sample_start "$part_idx" \
    --sample_stride "$num_parts" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --pre_k "$PRE_K" \
    --context_len "$CONTEXT_LEN" \
    --sample_method 3d \
    --size 32 \
    --gpu "$gpu" \
    'x_shape=[4,4,4]' \
    noise_level=random_uniform \
    "diffusion.sampling_timesteps=${SAMPLING_TIMESTEPS}" \
    2>&1 | tee -a "$log_file"
}

run_parallel_mse() {
  local name="$1"
  local dyna_ckpt="$2"
  local ae_ckpt="$3"
  local metadata_dir="$4"
  local log_prefix="$5"
  local pids=()
  local gpu_ids=()
  IFS=',' read -r -a gpu_ids <<< "$PART_GPUS"
  local num_parts="${#gpu_ids[@]}"

  echo "[$(date '+%F %T')] Eval ${name}: parts=${num_parts}, pre_k=${PRE_K}, context_len=${CONTEXT_LEN}, sampling=${SAMPLING_TIMESTEPS}"
  for idx in "${!gpu_ids[@]}"; do
    run_mse_part "$name" "$idx" "${gpu_ids[$idx]}" "$num_parts" "$dyna_ckpt" "$ae_ckpt" "$metadata_dir" "$log_prefix" &
    pids+=("$!")
  done
  wait_for_jobs "${pids[@]}"

  python scripts/aggregate_rollout_mse_parts.py \
    --output_dir outputs/maze9_metrics \
    --name "$name" \
    --num_parts "$num_parts" \
    2>&1 | tee "logs/maze9/${log_prefix}_aggregate.log"
}

plot_norm_mse() {
  VAE_EVAL_NAME="$VAE_EVAL_NAME" GRWM_EVAL_NAME="$GRWM_EVAL_NAME" python - <<'PY'
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = Path("outputs/maze9_metrics")
experiments = [
    (os.environ["VAE_EVAL_NAME"], "VAE", "#4C78A8"),
    (os.environ["GRWM_EVAL_NAME"], "GRWM u=w=0.1", "#54A24B"),
]

fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
series = {}
max_step = 0
for name, label, color in experiments:
    data = json.loads((out_dir / f"{name}_rollout_pixel_mse_summary.json").read_text(encoding="utf-8"))
    values = data["mse_norm_per_step_mean"]
    pre_k = int(data.get("pre_k", 15))
    valid = [(idx - pre_k + 1, val) for idx, val in enumerate(values) if val is not None]
    x = [step for step, _ in valid]
    y = [val for _, val in valid]
    series[label] = dict(valid)
    max_step = max(max_step, max(x))
    ax.plot(x, y, label=label, linewidth=2.6, color=color)

ax.set_title("Maze9 Rollout Normalized MSE Per Step", pad=12)
ax.set_xlabel("Rollout step")
ax.set_ylabel("Normalized MSE")
ax.grid(True, which="major", color="#D9D9D9", linewidth=0.8, alpha=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

stem = "maze9_vae_vs_grwm_norm_mse_per_step"
png_path = out_dir / f"{stem}.png"
pdf_path = out_dir / f"{stem}.pdf"
csv_path = out_dir / f"{stem}.csv"
fig.savefig(png_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

with csv_path.open("w", encoding="utf-8") as f:
    f.write("rollout_step,VAE,GRWM_u_w_0p1\n")
    for step in range(1, max_step + 1):
        f.write(",".join([
            str(step),
            str(series["VAE"].get(step, "")),
            str(series["GRWM u=w=0.1"].get(step, "")),
        ]) + "\n")

print(png_path)
print(pdf_path)
print(csv_path)
PY
}

append_table_rows() {
  VAE_EVAL_NAME="$VAE_EVAL_NAME" GRWM_EVAL_NAME="$GRWM_EVAL_NAME" SAMPLING_TIMESTEPS="$SAMPLING_TIMESTEPS" python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path("outputs/maze9_metrics")
combined_path = out_dir / "maze9_rollout_pixel_mse_table.json"
existing = []
if combined_path.exists():
    loaded = json.loads(combined_path.read_text(encoding="utf-8"))
    if isinstance(loaded, list):
        existing = loaded

by_name = {row.get("name"): row for row in existing if isinstance(row, dict)}
rows = []
for name in [os.environ["VAE_EVAL_NAME"], os.environ["GRWM_EVAL_NAME"]]:
    path = out_dir / f"{name}_rollout_pixel_mse_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "name": name,
        "mode": data["mode"],
        "num_samples": data["num_samples"],
        "pre_k": data["pre_k"],
        "context_len": data["context_len"],
        "sample_method": data["sample_method"],
        "noise_level": "random_uniform",
        "diffusion_sampling_timesteps": int(os.environ["SAMPLING_TIMESTEPS"]),
        "mse_255_global_mean": data["mse_255_global_mean"],
        "mse_255_global_std": data["mse_255_global_std"],
        "mse_norm_global_mean": data["mse_norm_global_mean"],
        "mse_norm_global_std": data["mse_norm_global_std"],
        "mse_255_per_step_mean": data["mse_255_per_step_mean"],
        "mse_norm_per_step_mean": data["mse_norm_per_step_mean"],
    }
    by_name[name] = row
    rows.append(row)

combined_path.write_text(json.dumps(list(by_name.values()), indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2), flush=True)
print(f"Updated {combined_path}", flush=True)
PY
}

run_stage="${1:-all}"

case "$run_stage" in
  all|ae)
    echo "[$(date '+%F %T')] Launching VAE and GRWM AE training"
    ae_pids=()
    train_ae "mcvae32_no_transformer" "$VAE_RUN" "0.0" "0.0" "$AE_GPU_VAE" "29641" "logs/maze9/${VAE_RUN}_ae.log" &
    ae_pids+=("$!")
    train_ae "mcgr32" "$GRWM_RUN" "0.1" "0.1" "$AE_GPU_GRWM" "29642" "logs/maze9/${GRWM_RUN}_ae.log" &
    ae_pids+=("$!")
    wait_for_jobs "${ae_pids[@]}"
    [[ "$run_stage" == "ae" ]] && exit 0
    ;;
esac

case "$run_stage" in
  all|precompute)
    run_precompute "VAE" "$VAE_AE_CKPT" "$VAE_LATENT_DIR" "logs/maze9/${VAE_RUN}_precompute.log"
    run_precompute "GRWM u=w=0.1" "$GRWM_AE_CKPT" "$GRWM_LATENT_DIR" "logs/maze9/${GRWM_RUN}_precompute.log"
    [[ "$run_stage" == "precompute" ]] && exit 0
    ;;
esac

case "$run_stage" in
  all|dynamics)
    run_dynamics "VAE" "$VAE_DYN_NAME" "$VAE_LATENT_DIR" "logs/maze9/${VAE_DYN_NAME}.log"
    run_dynamics "GRWM u=w=0.1" "$GRWM_DYN_NAME" "$GRWM_LATENT_DIR" "logs/maze9/${GRWM_DYN_NAME}.log"
    [[ "$run_stage" == "dynamics" ]] && exit 0
    ;;
esac

case "$run_stage" in
  all|eval)
    run_parallel_mse "$VAE_EVAL_NAME" "ckpts/${VAE_DYN_NAME}_last.pth" "$VAE_AE_CKPT" "$VAE_LATENT_DIR" "${VAE_EVAL_NAME}_mse_parallel"
    run_parallel_mse "$GRWM_EVAL_NAME" "ckpts/${GRWM_DYN_NAME}_last.pth" "$GRWM_AE_CKPT" "$GRWM_LATENT_DIR" "${GRWM_EVAL_NAME}_mse_parallel"
    append_table_rows
    plot_norm_mse
    ;;
  *)
    echo "Usage: $0 [all|ae|precompute|dynamics|eval]" >&2
    exit 2
    ;;
esac

echo "[$(date '+%F %T')] Done stage=${run_stage}"
