"""GPU gradient-equivalence test: soft-KL vs explicit forward KL on a real model.

Runs the trainer's own compute_loss code (via a stub `self`) on one fixed batch of real OPSD
rollouts and compares LoRA gradients against an EXACT reference: forward KL computed from fp32
logits built from the same post-norm hidden states (HF's model(...).logits are bf16, which
rounds logits of magnitude ~20 by ~0.06 and is itself a noticeable error source).

  noise   reference vs itself (flash-attn backward is non-deterministic) -> the floor
  T2      no clip: repo explicit (bf16 logits), soft-KL bf16, soft-KL fp32  vs reference
  T3      per-token clip at q90 (realistic, ~10% clipped) and q50 (stress): soft-KL vs clamped reference
  T5      lm_head gets no gradient in soft-KL mode

Pass (T2/T3 with soft-KL fp32): cosine > 0.999 and rel. norm diff <= max(3 x noise floor, 1e-3).

Usage (from repo root; needs rollouts from src/mtp_opsd_rl/scripts/gen_once.py):
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python tests/gpu_soft_kl_equivalence.py \
      --rollouts outputs/bench_memory_once/rollouts_Qwen3-1.7B_B4_L4096.pt --L 1024
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict
from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "mtp_opsd_rl"))
from opsd_trainer import OPSDTrainer, capture_final_hidden  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--rollouts", required=True)
p.add_argument("--L", type=int, default=1024)
p.add_argument("--temperature", type=float, default=1.1)
p.add_argument("--lora_b_std", type=float, default=1e-3, help="perturb LoRA B so A-grads are non-zero")
p.add_argument("--out", default="outputs/gpu_equivalence.json")
args = p.parse_args()

torch.manual_seed(0)
dev = torch.device("cuda")
data = torch.load(args.rollouts)
batch, completions, model_name = data["batch"], data["completions"][:, : args.L], data["model"]
TAU = args.temperature

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, attn_implementation="flash_attention_2", use_cache=False
).to(dev)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.0, task_type="CAUSAL_LM",  # training disables dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
with torch.no_grad():
    for n, prm in model.named_parameters():
        if "lora_B" in n:
            prm.normal_(0, args.lora_b_std)
model.train()
lora_params = [prm for prm in model.parameters() if prm.requires_grad]
causal_lm = model.get_base_model()
lm_head = causal_lm.lm_head.weight
W32 = lm_head.detach().float()

# ---------- inputs: identical construction to OPSDTrainer.training_step ----------
pad = tok.pad_token_id
s_ids = torch.cat([batch["student_prompts"], completions], 1)
t_ids = torch.cat([batch["teacher_prompts"], completions], 1)
labels = s_ids.clone()
for i, n in enumerate(batch["student_prompt_lengths_per_example"].tolist()):
    labels[i, :n] = -100
labels[labels == pad] = -100
inputs = {
    "student_input_ids": s_ids.to(dev), "student_attention_mask": (s_ids != pad).long().to(dev),
    "teacher_input_ids": t_ids.to(dev), "teacher_attention_mask": (t_ids != pad).long().to(dev),
    "labels": labels.to(dev),
    "student_prompt_length": batch["student_prompt_length"],
    "teacher_prompt_length": batch["teacher_prompt_length"],
}
sp, tp = inputs["student_prompt_length"], inputs["teacher_prompt_length"]
mask = inputs["labels"][:, sp:] != -100


def make_stub(soft_kl, token_clip=None, fp32=False, log_kl=False):
    s = SimpleNamespace(
        soft_kl=soft_kl, soft_kl_token_clip=token_clip, soft_kl_log_kl=log_kl, soft_kl_fp32=fp32,
        beta=0.0, temperature=TAU, top_k_loss=None, use_thinking_machines_loss=False,
        jsd_token_clip=None, use_ema_teacher=False, fixed_teacher=True,
        accelerator=SimpleNamespace(unwrap_model=lambda m: m),
        _metrics={"train": defaultdict(list)},
        generalized_jsd_loss=OPSDTrainer.generalized_jsd_loss,
    )
    s._compute_soft_kl_loss = MethodType(OPSDTrainer._compute_soft_kl_loss, s)
    return s


def exact_per_token_kl():
    """Reference per-token KL(q||p) [B, C]: fp32 logits from the post-norm hidden states. Student with grad."""
    with capture_final_hidden(causal_lm) as hs:
        model(input_ids=inputs["student_input_ids"], attention_mask=inputs["student_attention_mask"],
              logits_to_keep=1)
    with torch.no_grad(), model.disable_adapter(), capture_final_hidden(causal_lm) as ht:
        model(input_ids=inputs["teacher_input_ids"], attention_mask=inputs["teacher_attention_mask"],
              logits_to_keep=1)
    h_s = hs[-1][:, sp - 1 : -1].float()
    h_t = ht[-1][:, tp - 1 : -1].float()
    ls = F.log_softmax((h_s @ W32.T) / TAU, -1)
    with torch.no_grad():
        lt = F.log_softmax((h_t @ W32.T) / TAU, -1)
    return (lt.exp() * (lt - ls)).sum(-1)


def grads_of(loss_fn):
    model.zero_grad(set_to_none=True)
    loss_fn().backward()
    g = torch.cat([prm.grad.float().flatten() for prm in lora_params])
    head_grad = lm_head.grad
    model.zero_grad(set_to_none=True)
    return g, head_grad


report = {}


def compare(name, g_ref, g_new, floor=None):
    cos = F.cosine_similarity(g_ref, g_new, dim=0).item()
    rel = ((g_new - g_ref).norm() / g_ref.norm()).item()
    tol = None if floor is None else max(3 * floor, 1e-3)
    ok = None if tol is None else (cos > 0.999 and rel <= tol)
    verdict = "" if ok is None else ("PASS" if ok else "FAIL")
    print(f"{name:<52} cos={cos:.6f}  rel={rel:.2e}  |ref|={g_ref.norm():.3e}  {verdict}")
    report[name.strip()] = {"cosine": cos, "rel_norm_diff": rel, "ref_norm": g_ref.norm().item(), "pass": ok}
    return ok


print(f"model={model_name} B={s_ids.shape[0]} L={args.L} valid_tokens={int(mask.sum())} "
      f"seq S/T={s_ids.shape[1]}/{t_ids.shape[1]}")
results = []

# ---- noise floor ----
ref_loss = lambda: exact_per_token_kl()[mask].sum() / mask.sum()  # noqa: E731
g_ref, _ = grads_of(ref_loss)
g_ref2, _ = grads_of(ref_loss)
compare("noise: reference vs reference", g_ref, g_ref2)
floor = report["noise: reference vs reference"]["rel_norm_diff"]

# ---- T2: no clip ----
g_repo, _ = grads_of(lambda: OPSDTrainer.compute_loss(make_stub(False), model, inputs))
compare("T2 repo explicit (bf16 logits) vs ref", g_ref, g_repo)
g_soft16, _ = grads_of(lambda: OPSDTrainer.compute_loss(make_stub(True), model, inputs))
compare("T2 soft-KL bf16 vs ref", g_ref, g_soft16)
g_soft32, head_grad = grads_of(lambda: OPSDTrainer.compute_loss(make_stub(True, fp32=True), model, inputs))
results.append(compare("T2 soft-KL fp32 vs ref", g_ref, g_soft32, floor))

# ---- T5: head untouched ----
t5 = head_grad is None or head_grad.abs().max().item() == 0
print(f"{'T5 lm_head grad in soft-KL mode is None':<52} {'PASS' if t5 else 'FAIL'}")
report["T5 lm_head untouched"] = {"pass": t5}
results.append(t5)

# ---- T3: per-token clip ----
with torch.no_grad():
    kv = exact_per_token_kl()[mask]
print(f"per-token KL: mean={kv.mean():.3e} median={kv.median():.3e} q90={kv.quantile(0.9):.3e} max={kv.max():.3e}")
report["per_token_kl"] = {"mean": kv.mean().item(), "median": kv.median().item(),
                          "q90": kv.quantile(0.9).item(), "max": kv.max().item()}
for qname, q in (("q90", 0.9), ("q50", 0.5)):
    c = kv.quantile(q).item()
    g_c, _ = grads_of(lambda: exact_per_token_kl().clamp(max=c)[mask].sum() / mask.sum())
    g_c2, _ = grads_of(lambda: exact_per_token_kl().clamp(max=c)[mask].sum() / mask.sum())
    compare(f"noise: clipped {qname} ref vs ref", g_c, g_c2)
    floor_c = report[f"noise: clipped {qname} ref vs ref"]["rel_norm_diff"]
    for fp32 in (False, True):
        stub = make_stub(True, token_clip=c, fp32=fp32, log_kl=True)
        g_s, _ = grads_of(lambda: OPSDTrainer.compute_loss(stub, model, inputs))
        st = stub._metrics["train"]
        tag = f"T3 {qname} c={c:.2e} soft-KL {'fp32' if fp32 else 'bf16'} vs ref"
        print(f"   clip_frac ref={(kv > c).float().mean():.4f} soft={st['soft_kl/clip_frac'][0]:.4f}   "
              f"KL mean ref={kv.mean():.5e} soft={st['soft_kl/kl_mean'][0]:.5e}")
        ok = compare(tag, g_c, g_s, floor_c if fp32 else None)
        if fp32:
            results.append(ok)

pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(args.out).write_text(json.dumps(report, indent=2))
print("\nALL PASS" if all(results) else "\nSOME FAILED")
sys.exit(0 if all(results) else 1)
