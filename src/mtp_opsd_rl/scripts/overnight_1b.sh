#!/usr/bin/env bash
# Overnight: train OPSD baseline (GPU 2) and soft-KL OPSD (GPU 3) in parallel, then evaluate
# base + checkpoints {100,50,75,25} on aime24/aime25/hmmt25 across both GPUs.
# Usage (from repo root):  setsid nohup bash src/mtp_opsd_rl/scripts/overnight_1b.sh > outputs/opsd_1.7b/overnight.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../.."
EXP=outputs/opsd_1.7b
mkdir -p "$EXP/logs"
rm -f "$EXP/TRAIN_DONE" "$EXP/ALL_DONE"

echo "[$(date)] training start"
METHOD=explicit GPU=2 PORT=12960 bash src/mtp_opsd_rl/scripts/train_opsd_1b_1gpu.sh > "$EXP/logs/train_explicit.log" 2>&1 &
P1=$!
METHOD=soft_kl  GPU=3 PORT=12961 bash src/mtp_opsd_rl/scripts/train_opsd_1b_1gpu.sh > "$EXP/logs/train_soft_kl.log" 2>&1 &
P2=$!
wait $P1; echo "[$(date)] explicit training exit=$?"
wait $P2; echo "[$(date)] soft_kl training exit=$?"
touch "$EXP/TRAIN_DONE"

echo "[$(date)] eval start"
.venv/bin/python src/mtp_opsd_rl/scripts/eval_queue.py --exp "$EXP" \
    --runs qwen3_1.7b_explicit qwen3_1.7b_soft_kl --steps 100 50 75 25 --gpus 2 3
echo "[$(date)] eval exit=$?"
touch "$EXP/ALL_DONE"
