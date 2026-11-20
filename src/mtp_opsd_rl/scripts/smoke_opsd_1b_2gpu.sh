#!/usr/bin/env bash
# Smoke test: original OPSD (explicit forward KL) on Qwen3-1.7B, 2 GPUs, 3 optimizer steps.
# Same hyperparameters as run_opsd_1b.sh except: GPUs, model path, max_steps, output dir, no wandb.
# Run from src/mtp_opsd_rl/:   bash scripts/smoke_opsd_1b_2gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT="$(cd ../.. && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export WANDB_MODE=disabled
export TRL_EXPERIMENTAL_SILENCE=1

"$REPO_ROOT/.venv/bin/accelerate" launch \
    --config_file accelerate.yaml \
    --num_processes 2 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12950 \
    opsd_train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir "$REPO_ROOT/outputs/smoke" \
    --run_config smoke_qwen31b_explicit \
    --max_steps 3 \
    --max_completion_length 1024 \
    --save_steps 1000 \
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
    --logging_dir "$REPO_ROOT/outputs/smoke/tb"
