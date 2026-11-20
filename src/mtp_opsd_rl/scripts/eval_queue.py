"""Run the paper's evaluation (eval/evaluate_math.py) over base + checkpoints on several GPUs.

Protocol (eval/run_eval.sh): Avg@12, temperature 1.0, thinking mode, max_new_tokens 38912,
top_p auto (0.95 for thinking, as in the authors' script), datasets aime24/aime25/hmmt25.

Each job = (model, dataset) on one GPU (tensor_parallel_size=1). Jobs already done (result JSON
exists) are skipped, so the queue is resumable. After each job, metrics are written to
tensorboard under <exp>/tb/<run>/ at global_step = checkpoint step (base model = step 0, written
to every run) and appended to <exp>/eval/summary.jsonl.

Usage (from repo root):
  .venv/bin/python src/mtp_opsd_rl/scripts/eval_queue.py --exp outputs/opsd_1.7b \
      --runs qwen3_1.7b_explicit qwen3_1.7b_soft_kl --steps 100 50 75 25 --gpus 2 3
"""
import argparse
import json
import pathlib
import queue
import subprocess
import threading
import time

from torch.utils.tensorboard import SummaryWriter

p = argparse.ArgumentParser()
p.add_argument("--exp", default="outputs/opsd_1.7b")
p.add_argument("--runs", nargs="+", required=True)
p.add_argument("--steps", nargs="+", type=int, default=[100, 50, 75, 25], help="priority order")
p.add_argument("--datasets", nargs="+", default=["aime24", "aime25", "hmmt25"])
p.add_argument("--gpus", nargs="+", default=["2", "3"])
p.add_argument("--base_model", default="Qwen/Qwen3-1.7B")
p.add_argument("--val_n", type=int, default=12)
p.add_argument("--max_new_tokens", type=int, default=38912)
p.add_argument("--no_base", action="store_true")
args = p.parse_args()

repo = pathlib.Path(__file__).resolve().parents[3]
exp = (repo / args.exp).resolve()
eval_dir = exp / "eval"
eval_dir.mkdir(parents=True, exist_ok=True)
evaluate_py = repo / "src/mtp_opsd_rl/eval/evaluate_math.py"
python = repo / ".venv/bin/python"

# job priority: base first, then steps in the given order, alternating runs
jobs = []
if not args.no_base:
    jobs += [("base", 0, None, ds) for ds in args.datasets]
for step in args.steps:
    for ds in args.datasets:
        for run in args.runs:
            jobs.append((run, step, exp / "ckpt" / run / f"checkpoint-{step}", ds))

q = queue.Queue()
for j in jobs:
    q.put(j)
lock = threading.Lock()


def log_result(run, step, ds, res_path, minutes):
    d = json.loads(res_path.read_text())
    lengths = [len(r.get("response", "") or "") for r in d.get("results", []) if isinstance(r, dict)]
    rec = {
        "run": run, "step": step, "dataset": ds,
        "avg_at_n_pct": d["average_at_n_pct"], "pass_at_n_pct": d["pass_at_n_pct"],
        "maj_at_n_pct": d["majority_vote_at_n_pct"], "format_rate": d["format_rate"],
        "val_n": d["val_n"], "num_problems": d["num_problems"], "minutes": round(minutes, 1),
    }
    with lock:
        with open(eval_dir / "summary.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        targets = args.runs if run == "base" else [run]
        for r in targets:
            w = SummaryWriter(log_dir=str(exp / "tb" / r))
            w.add_scalar(f"eval/{ds}_avg@{d['val_n']}", d["average_at_n_pct"], step)
            w.add_scalar(f"eval/{ds}_pass@{d['val_n']}", d["pass_at_n_pct"], step)
            w.add_scalar(f"eval/{ds}_maj@{d['val_n']}", d["majority_vote_at_n_pct"], step)
            w.add_scalar(f"eval/{ds}_format_rate", d["format_rate"], step)
            w.flush()
            w.close()
    print(f"[done] {rec}", flush=True)


def worker(gpu):
    while True:
        try:
            run, step, ckpt, ds = q.get_nowait()
        except queue.Empty:
            return
        name = f"{run}_step{step}_{ds}"
        res = eval_dir / f"{name}.json"
        if res.exists():
            print(f"[skip] {name} (exists)", flush=True)
            continue
        if ckpt is not None and not (ckpt / "adapter_model.safetensors").exists():
            print(f"[missing] {name}: no adapter at {ckpt}", flush=True)
            continue
        cmd = [str(python), str(evaluate_py), "--base_model", args.base_model, "--dataset", ds,
               "--val_n", str(args.val_n), "--temperature", "1.0", "--max_new_tokens", str(args.max_new_tokens),
               "--tensor_parallel_size", "1", "--gpu_memory_utilization", "0.85",
               "--output_file", str(res.with_suffix(".tmp.json"))]
        if ckpt is not None:
            cmd += ["--checkpoint_dir", str(ckpt)]
        print(f"[start] {name} on GPU {gpu}", flush=True)
        t0 = time.time()
        with open(eval_dir / f"{name}.log", "w") as logf:
            rc = subprocess.run(cmd, cwd=evaluate_py.parent, stdout=logf, stderr=subprocess.STDOUT,
                                env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": gpu}).returncode
        tmp = res.with_suffix(".tmp.json")
        if rc == 0 and tmp.exists():
            tmp.rename(res)
            log_result(run, step, ds, res, (time.time() - t0) / 60)
        else:
            print(f"[fail] {name} rc={rc} (see {name}.log)", flush=True)


threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("EVAL QUEUE FINISHED", flush=True)
