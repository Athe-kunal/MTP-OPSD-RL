"""Generate ONE batch of OPSD rollouts at a fixed length and save the token ids.

Uses the same prompts (SelfDistillationDataCollator) and sampling settings as run_opsd_1b.sh.
ignore_eos=True makes every completion exactly --max_tokens long, so later truncation to any
L <= max_tokens yields L real (unmasked) tokens per sequence.

Usage (from repo root):
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python src/mtp_opsd_rl/scripts/gen_once.py --max_tokens 4096
"""
import argparse
import pathlib
import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data_collator import SelfDistillationDataCollator  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-1.7B")
p.add_argument("--batch_size", type=int, default=4)       # = per_device_train_batch_size
p.add_argument("--max_tokens", type=int, default=4096)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="outputs/bench_memory_once")
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
collator = SelfDistillationDataCollator(tok, max_length=20000, reason_first=False,
                                        student_thinking=False, teacher_thinking=True)
ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd", split="train")
batch = collator([ds[i] for i in range(args.batch_size)])

from vllm import LLM, SamplingParams  # noqa: E402  (import after tokenizer work)

llm = LLM(model=args.model, gpu_memory_utilization=0.5, max_model_len=20000, seed=args.seed)
sp = SamplingParams(n=1, temperature=1.1, top_p=0.95, top_k=20, max_tokens=args.max_tokens,
                    ignore_eos=True, seed=args.seed)
prompts = [{"prompt_token_ids": ids[:n].tolist()}
           for ids, n in zip(batch["student_prompts"], batch["student_prompt_lengths_per_example"])]
outs = llm.generate(prompts, sp)
completions = torch.tensor([list(o.outputs[0].token_ids) for o in outs])  # [B, max_tokens]
assert completions.shape == (args.batch_size, args.max_tokens), completions.shape

out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
path = out / f"rollouts_{args.model.split('/')[-1]}_B{args.batch_size}_L{args.max_tokens}.pt"
torch.save({"batch": dict(batch), "completions": completions, "model": args.model}, path)
print(f"saved {path}  completions={tuple(completions.shape)}")
print("sample:", tok.decode(completions[0, :200]))
