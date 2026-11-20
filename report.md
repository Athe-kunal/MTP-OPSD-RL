# Soft-KL OPSD — Experiment Report

Applying the **KL gradient-recovery trick** (Teoh 2026, *Multi-Token Prediction Done Right*) to
**OPSD** (Zhao et al. 2026, *Self-Distilled Reasoner*). Living document. Newest results at the top of each section.

Last updated: 2026-11-20

---

## 1. Summary so far

| Question | Answer | Evidence |
|---|---|---|
| Does soft-KL give the same gradient as OPSD's forward KL? | **Yes** (fp32 variant matches the exact KL gradient to the run-to-run noise floor) | §4 |
| How much memory does it save? | **6× at 1024 tokens, 9× at 4096** for the loss forward+backward (Qwen3-1.7B, batch 4) | §3 |
| Is it slower? | bf16: no. fp32 (exact): loss ~1.6× slower, about 10% per training step | §3 |
| Can it reproduce the repo's `--jsd_token_clip`? | **No.** The repo clips per *vocabulary entry*, not per token; the CE trick can only do per-token clipping | §2 |
| Does soft-KL OPSD train as well as OPSD? | Running (overnight, §5) | §5 |

---

## 2. Findings about the OPSD codebase

- **Loss used by all released scripts:** `--beta 0` → forward KL $D_{KL}(q_{teacher}\|p_{student})$ over the full vocabulary, temperature 1.1, fixed teacher (LoRA disabled), LoRA does not touch `lm_head`. These are exactly the preconditions of the trick.
- **The clip is per vocabulary entry, not per token.** `F.kl_div(..., reduction="none")` returns `[B, C, V]`, and `jsd.clamp(max=c)` (`opsd_trainer.py`, `generalized_jsd_loss`) caps each term $q_v\log(q_v/p_v)$ independently. The docstring says "per-token". The resulting gradient is $W^\top(Q_A p - q\odot\mathbb 1_A)$ — a truncated teacher target — which two CEs cannot express. At `1e-6`/`1e-7` (non-thinking scripts) nearly every positive entry is clipped.
- **Per-token KL is extremely heavy-tailed** (Qwen3-1.7B, 4093 completion tokens): mean 0.15, **median 1.1e-4**, q90 0.21, max 42. Most tokens have near-zero teacher/student disagreement; a few tokens carry almost all of the signal. Relevant to whether clipping is needed.
- **The 1024 completion cap** comes from `--max_completion_length 1024` (README gives no reason). vLLM pads every completion to that length. §3 suggests memory is the reason: the explicit loss would not fit at 2048–4096 alongside vLLM.
- Bugs/quirks fixed along the way: `opsd_train.py` always initialised wandb (now only if requested); `torch_dtype` passed as a `torch.dtype` broke tensorboard logging (now a string).

---

## 3. Memory and speed

One fixed batch of real rollouts (4 × 4096 tokens, generated once with `ignore_eos`, truncated to L),
Qwen3-1.7B + LoRA r=64, bf16, flash-attn 2, gradient checkpointing. One forward+backward of the
trainer's actual `compute_loss`. "Delta" = memory added by the loss on top of model weights.
Scripts: `src/mtp_opsd_rl/scripts/gen_once.py`, `bench_loss_memory.py`.

| L | explicit (repo) delta GB | soft-KL bf16 delta GB | soft-KL fp32 delta GB | explicit s | soft-KL bf16 s | soft-KL fp32 s |
|---|---|---|---|---|---|---|
| 1024 | 14.27 | **2.34** | 3.55 | 1.27 | 1.26 | 2.04 |
| 2048 | 26.33 | **2.91** | 4.11 | 2.14 | 2.14 | 3.71 |
| 4096 | 50.45 | **5.60** | 5.60 | 4.59 | 4.00 | 7.15 |

- Explicit grows ~12 GB per 1024 completion tokens (full-vocab logits, including over the long teacher prompt). Soft-KL grows only with ordinary activations.
- Cross-check: end-to-end training runs at 1024 measured 13.6–14.1 GB (explicit) and 2.35 GB (soft-KL), matching.
- **What fits in a real run** (vLLM colocate + model ≈ 50.5 GB before the loss, 80 GB GPU): explicit ≈ 65 / 77 / 101 GB at 1024 / 2048 / 4096 (OOM at 4096); soft-KL ≈ 53–56 GB at all three.

---

## 4. Gradient equivalence (Qwen3-1.7B, bf16, real rollouts)

`tests/gpu_soft_kl_equivalence.py`. Compares LoRA gradients (392 tensors) against an **exact
reference**: forward KL from fp32 logits built from the same post-norm hidden states. LoRA B is
randomly perturbed so A-gradients are non-zero. Results in `outputs/gpu_equivalence.json`.

| Comparison | cosine | rel. norm diff |
|---|---|---|
| **Noise floor**: reference vs itself (flash-attn backward is non-deterministic) | 0.99997 | 0.75% |
| Repo explicit loss (bf16 logits) vs reference | 0.99986 | 1.65% |
| Soft-KL bf16 vs reference | 0.99992 | 1.30% |
| **Soft-KL fp32 vs reference** | **0.99997** | **0.73%** (= noise floor) |
| Per-token clip q90 (10% clipped), soft-KL fp32 | 0.99996 | 0.90% (floor 0.63%) |
| Per-token clip q50 (stress: only KL<1e-4 tokens left), soft-KL fp32 | 0.99982 | 1.97% (floor 0.66%) |
| `lm_head` gradient in soft-KL mode | — | None ✓ |

Plus 6 CPU float64 tests (`tests/test_soft_kl.py`): gradient = forward KL to 1e-10, label choice irrelevant, chunking invariant, per-token clip exact.

**Numerical lessons**
1. The repo's own loss is not exact: HF returns bf16 logits (magnitude ~20, rounding ~0.06), which perturbs p and q by several percent. bf16 soft-KL is already *closer* to the exact gradient than the baseline.
2. Dividing the bf16 hidden state by τ before upcasting rounds every logit; the fix (upcast, then scale) is in `soft_kl_loss(tau=...)`.
3. In bf16, clipped/low-KL regimes degrade (cosine 0.92 at q50) — use `--soft_kl_fp32` (+1.2 GB for an fp32 copy of `lm_head`).

---

## 5. Overnight experiment: paper reproduction, Qwen3-1.7B (running)

**Setup** (`src/mtp_opsd_rl/scripts/overnight_1b.sh`, `train_opsd_1b_1gpu.sh`, `eval_queue.py`)

| | A: OPSD (baseline) | B: Soft-KL OPSD |
|---|---|---|
| Loss | explicit forward KL, `--jsd_token_clip 0.05` (repo, per-entry clip) | soft-KL fp32, **no clip** (`--soft_kl --soft_kl_fp32`) |
| GPU | 2 | 3 |
| Everything else | paper `run_opsd_1b.sh`: lr 5e-6, max_grad_norm 0.1, LoRA r64/α128, τ=1.1, top-p 0.95, top-k 20, fixed teacher, 1024 completion tokens, effective batch 32 | same |

Deviations from the paper, all forced by 2 GPUs: 1 GPU × 4 × 8 accumulation instead of 4 × 4 × 2 (same effective batch 32); constant LR instead of linear decay over ~28k steps (identical to within 0.4% over 100 steps); 100 steps total (paper reports 25/50/75/100). LoRA-only checkpoints at 25/50/75/100.

**Confound to keep in mind (measured, steps 1–2):** A clips per vocab entry, B does not clip. The per-entry clip shrinks A's gradient norm to **≈0.04**, *below* `max_grad_norm 0.1`, so A's updates are never norm-clipped. B's gradient norm is **≈0.26**, clipped to 0.1. So **B takes ~2.5× larger steps than A** as well as a different direction. A clean "same objective" comparison needs a third run (explicit, no clip; its grad norm should match B's) — see §6.

**Evaluation** (paper protocol, `eval/run_eval.sh`): AIME24, AIME25, HMMT25; Avg@12; temperature 1.0; thinking mode; max 38,912 new tokens; top-p 0.95 (the script's auto value; README says "none"). Base model + checkpoints 100, 50, 75, 25 for each run.

**Paper reference (README, Qwen3-1.7B thinking, Avg@12):**

| Step | AIME24 | AIME25 | HMMT25 |
|---|---|---|---|
| Base | 51.5 | 36.7 | 23.1 |
| 25 | 51.4 | 42.5 | 24.7 |
| 50 | 52.8 | 43.9 | 27.8 |
| 75 | 54.4 | 40.6 | 26.9 |
| 100 | 57.2 | 41.1 | 29.2 |

**Results:** _pending_

**Where to look:** `tensorboard --logdir outputs/opsd_1.7b/tb` (training curves + `eval/*` scalars per run); `outputs/opsd_1.7b/eval/summary.jsonl`; logs in `outputs/opsd_1.7b/logs/`.

---

## 6. Open questions / next steps

- Third run: explicit forward KL **without** clip, to isolate "same objective, less memory" from "different clipping".
- Spend the memory: soft-KL at 2048 / 4096 completion tokens (E3), which the explicit loss cannot fit.
- Per-token clip ablation (`--soft_kl_token_clip`), given the heavy-tailed KL distribution.
- Avg@12 on 30 problems has a standard error of several points; multi-seed eval before drawing conclusions.
- GRPO + λ·soft-KL distillation (dense privileged-teacher signal), where memory at 16k tokens actually binds.
