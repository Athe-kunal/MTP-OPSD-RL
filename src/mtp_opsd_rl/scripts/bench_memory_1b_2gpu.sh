#!/usr/bin/env bash
# End-to-end memory benchmark: explicit forward KL (repo) vs soft-KL, Qwen3-1.7B, 2 GPUs.
# Varies --max_completion_length; vLLM pads every completion to that length, so sequence
# length (and memory) is deterministic per setting. Everything else = run_opsd_1b.sh.
#
# Usage (from repo root):  bash src/mtp_opsd_rl/scripts/bench_memory_1b_2gpu.sh [lengths...]
#   env: METHODS="soft_kl explicit"  MAX_STEPS=3  CUDA_VISIBLE_DEVICES=2,3
# Summary:                 .venv/bin/python src/mtp_opsd_rl/scripts/summarize_memprof.py outputs/bench_memory
set -uo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(cd ../.. && pwd)"
OUT="$REPO_ROOT/outputs/bench_memory"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export WANDB_MODE=disabled
export TRL_EXPERIMENTAL_SILENCE=1
export OPSD_MEM_PROFILE=1

LENGTHS=("${@:-1024 2048 4096}")
read -ra LENGTHS <<< "${LENGTHS[*]}"
read -ra METHODS <<< "${METHODS:-soft_kl explicit}"
MAX_STEPS="${MAX_STEPS:-3}"

for L in "${LENGTHS[@]}"; do
  for M in "${METHODS[@]}"; do
    EXTRA=()
    [[ "$M" == "soft_kl" ]] && EXTRA=(--soft_kl)
    LOG="$OUT/${M}_${L}.log"
    echo "=== $M  max_completion_length=$L  -> $LOG"
    "$REPO_ROOT/.venv/bin/accelerate" launch \
        --config_file accelerate.yaml \
        --num_processes 2 \
        --gradient_accumulation_steps 2 \
        --main_process_port 12951 \
        opsd_train.py \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --learning_rate 5e-6 \
        --max_grad_norm 0.1 \
        --per_device_train_batch_size 4 \
        --gradient_checkpointing \
        --gradient_accumulation_steps 2 \
        --output_dir "$OUT/ckpt" \
        --run_config "${M}_${L}" \
        --max_steps "$MAX_STEPS" \
        --max_completion_length "$L" \
        --save_strategy no \
        --logging_steps 1 \
        --attn_implementation flash_attention_2 \
        --torch_dtype bfloat16 \
        --max_length 20000 \
        --beta 0 \
        --use_vllm \
        --vllm_mode colocate \
        --vllm_gpu_memory_utilization 0.6 \
        --vllm_tensor_parallel_size 1 \
        --use_peft \
        --lora_r 64 \
        --lora_alpha 128 \
        --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --temperature 1.1 \
        --top_p 0.95 \
        --top_k 20 \
        --lmbda 1 \
        --fixed_teacher \
        --jsd_token_clip 0.05 \
        --report_to tensorboard \
        --logging_dir "$OUT/tb/${M}_${L}" \
        "${EXTRA[@]}" > "$LOG" 2>&1
    echo "exit_code=$?" >> "$LOG"
    tail -1 "$LOG"
  done
done
