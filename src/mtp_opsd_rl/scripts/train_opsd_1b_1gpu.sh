#!/usr/bin/env bash
# Paper reproduction (Self-Distilled Reasoner, Qwen3-1.7B recipe from run_opsd_1b.sh) on ONE GPU.
# Paper: 4 GPUs x 4 per device x 2 accumulation = 32. Here: 1 GPU x 4 x 8 = 32 (same effective batch).
# Paper LR schedule = linear decay over ~28k steps (30 epochs) -> still 99.6% of peak at step 100,
# so for a 100-step run we use a constant LR.
#
# Usage (from repo root):  METHOD=explicit|soft_kl GPU=2 PORT=12960 bash src/mtp_opsd_rl/scripts/train_opsd_1b_1gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(cd ../.. && pwd)"
METHOD="${METHOD:?set METHOD=explicit or soft_kl}"
GPU="${GPU:?set GPU}"
PORT="${PORT:-12960}"
MAX_STEPS="${MAX_STEPS:-100}"
EXP="${EXP:-$REPO_ROOT/outputs/opsd_1.7b}"
RUN="${RUN:-qwen3_1.7b_${METHOD}}"

export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_MODE=disabled
export TRL_EXPERIMENTAL_SILENCE=1

if [[ "$METHOD" == "explicit" ]]; then
  LOSS_ARGS=(--jsd_token_clip 0.05)                       # repo loss: explicit forward KL, per-vocab-entry clip
elif [[ "$METHOD" == "soft_kl" ]]; then
  LOSS_ARGS=(--jsd_token_clip 0 --soft_kl --soft_kl_fp32 --soft_kl_log_kl)   # exact forward-KL grad, no clip
else
  echo "unknown METHOD=$METHOD"; exit 1
fi

"$REPO_ROOT/.venv/bin/accelerate" launch \
    --config_file accelerate.yaml \
    --num_processes 1 \
    --gradient_accumulation_steps 8 \
    --main_process_port "$PORT" \
    opsd_train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --lr_scheduler_type constant \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 8 \
    --output_dir "$EXP/ckpt" \
    --run_config "$RUN" \
    --max_steps "$MAX_STEPS" \
    --max_completion_length 1024 \
    --save_steps 25 \
    --save_only_model \
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
    --report_to tensorboard \
    --logging_dir "$EXP/tb/$RUN" \
    "${LOSS_ARGS[@]}"
