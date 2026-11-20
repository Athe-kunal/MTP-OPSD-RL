"""Loss-only memory benchmark: explicit forward KL (repo) vs soft-KL, on ONE fixed batch of rollouts.

Loads the model + LoRA once (same LoRA config, bf16, flash-attn 2, gradient checkpointing as
run_opsd_1b.sh), truncates the saved completions to each L, and runs one forward+backward of
OPSDTrainer.compute_loss for each method -- the exact training code path, via a stub `self`.
No vLLM and no optimizer: the reported delta is what the loss forward+backward adds.

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python src/mtp_opsd_rl/scripts/bench_loss_memory.py \
      --rollouts outputs/bench_memory_once/rollouts_Qwen3-1.7B_B4_L4096.pt --lengths 1024 2048 4096
"""
import argparse
import json
import pathlib
import statistics
import sys
import time
from collections import defaultdict
from types import MethodType, SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from opsd_trainer import OPSDTrainer  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--rollouts", required=True)
p.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096])
p.add_argument("--methods", nargs="+", default=["explicit", "soft_kl"])
p.add_argument("--reps", type=int, default=3, help="timed repetitions per config (after 1 warm-up)")
p.add_argument("--jsd_token_clip", type=float, default=0.05)
p.add_argument("--out", default="outputs/bench_memory_once/results.json")
args = p.parse_args()

data = torch.load(args.rollouts)
batch, completions, model_name = data["batch"], data["completions"], data["model"]
assert max(args.lengths) <= completions.shape[1], "generate longer rollouts first"
dev = torch.device("cuda")
GB = 1024**3

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", use_cache=False
).to(dev)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
model.train()


def make_stub(method):
    """Just enough of OPSDTrainer for compute_loss / _compute_soft_kl_loss to run unchanged."""
    s = SimpleNamespace(
        soft_kl=method.startswith("soft_kl"), soft_kl_token_clip=None, soft_kl_log_kl=False,
        soft_kl_fp32=(method == "soft_kl_fp32"),
        beta=0.0, temperature=1.1, top_k_loss=None, use_thinking_machines_loss=False,
        jsd_token_clip=args.jsd_token_clip, use_ema_teacher=False, fixed_teacher=True,
        accelerator=SimpleNamespace(unwrap_model=lambda m: m),
        _metrics={"train": defaultdict(list)},
        generalized_jsd_loss=OPSDTrainer.generalized_jsd_loss,
    )
    s._compute_soft_kl_loss = MethodType(OPSDTrainer._compute_soft_kl_loss, s)
    return s


def build_inputs(L):
    """Same construction as OPSDTrainer.training_step, with completions truncated to L."""
    gen = completions[:, :L]
    pad = tok.pad_token_id
    s_ids = torch.cat([batch["student_prompts"], gen], 1)
    t_ids = torch.cat([batch["teacher_prompts"], gen], 1)
    s_mask, t_mask = (s_ids != pad).long(), (t_ids != pad).long()
    labels = s_ids.clone()
    for i, n in enumerate(batch["student_prompt_lengths_per_example"].tolist()):
        labels[i, :n] = -100
    labels[labels == pad] = -100
    return {
        "student_input_ids": s_ids.to(dev), "student_attention_mask": s_mask.to(dev),
        "teacher_input_ids": t_ids.to(dev), "teacher_attention_mask": t_mask.to(dev),
        "labels": labels.to(dev),
        "student_prompt_length": batch["student_prompt_length"],
        "teacher_prompt_length": batch["teacher_prompt_length"],
    }


def run_once(stub, inputs):
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.time()
    loss = OPSDTrainer.compute_loss(stub, model, inputs)
    loss.backward()
    torch.cuda.synchronize()
    return {
        "base_gb": base / GB,
        "peak_gb": torch.cuda.max_memory_allocated() / GB,
        "delta_gb": (torch.cuda.max_memory_allocated() - base) / GB,
        "reserved_gb": torch.cuda.max_memory_reserved() / GB,
        "time_s": time.time() - t0,
        "loss": loss.item(),
    }


results = []
for L in args.lengths:
    inputs = build_inputs(L)
    for method in args.methods:
        stub = make_stub(method)
        rec = {"L": L, "method": method, "B": completions.shape[0],
               "seq_student": inputs["student_input_ids"].shape[1],
               "seq_teacher": inputs["teacher_input_ids"].shape[1],
               "valid_tokens": int((inputs["labels"] != -100).sum())}
        try:
            run_once(stub, inputs)  # warm-up (allocator, kernels)
            reps = [run_once(stub, inputs) for _ in range(args.reps)]
            rec.update(status="ok",
                       base_gb=reps[0]["base_gb"],
                       peak_gb=max(r["peak_gb"] for r in reps),
                       delta_gb=max(r["delta_gb"] for r in reps),
                       reserved_gb=max(r["reserved_gb"] for r in reps),
                       time_s=statistics.median(r["time_s"] for r in reps),
                       loss=reps[0]["loss"])
        except torch.cuda.OutOfMemoryError:
            rec.update(status="OOM")
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        results.append(rec)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)

pathlib.Path(args.out).write_text(json.dumps(results, indent=2))

print("\n| L | method | status | loss fwd+bwd delta GB | peak alloc GB | peak reserved GB | time s |")
print("|---|---|---|---|---|---|---|")
for r in results:
    if r["status"] == "ok":
        print(f"| {r['L']} | {r['method']} | ok | {r['delta_gb']:.2f} | {r['peak_gb']:.2f} | "
              f"{r['reserved_gb']:.2f} | {r['time_s']:.2f} |")
    else:
        print(f"| {r['L']} | {r['method']} | {r['status']} | - | - | - | - |")
