"""Markdown table of eval results from <exp>/eval/summary.jsonl (Avg@12, with pass@12 / maj@12).

Usage: .venv/bin/python src/mtp_opsd_rl/scripts/eval_table.py outputs/opsd_1.7b
"""
import json
import pathlib
import sys

exp = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/opsd_1.7b")
recs = {}
for line in (exp / "eval" / "summary.jsonl").read_text().splitlines():
    r = json.loads(line)
    recs[(r["run"], r["step"], r["dataset"])] = r

datasets = ["aime24", "aime25", "hmmt25"]
runs = sorted({k[0] for k in recs if k[0] != "base"}) or ["qwen3_1.7b_explicit", "qwen3_1.7b_soft_kl"]
paper = {0: (51.5, 36.7, 23.1), 25: (51.4, 42.5, 24.7), 50: (52.8, 43.9, 27.8),
         75: (54.4, 40.6, 26.9), 100: (57.2, 41.1, 29.2)}


def cell(key):
    r = recs.get(key)
    return "…" if r is None else f"{r['avg_at_n_pct']:.1f} ({r['pass_at_n_pct']:.0f}/{r['maj_at_n_pct']:.0f})"


print("Avg@12 % (pass@12 / maj@12). Paper = README Qwen3-1.7B, Avg@12.\n")
print("| Step | Model | " + " | ".join(datasets) + " | mean |")
print("|---|---|" + "---|" * (len(datasets) + 1))
for step in (0, 25, 50, 75, 100):
    p = paper[step]
    print(f"| {step} | paper | " + " | ".join(f"{x:.1f}" for x in p) + f" | {sum(p)/3:.1f} |")
    models = ["base"] if step == 0 else runs
    for m in models:
        vals = [recs.get((m, step, d)) for d in datasets]
        mean = (f"{sum(v['avg_at_n_pct'] for v in vals)/3:.1f}" if all(vals) else "…")
        name = {"base": "base (ours)", "qwen3_1.7b_explicit": "OPSD explicit (ours)",
                "qwen3_1.7b_soft_kl": "soft-KL OPSD (ours)"}.get(m, m)
        print(f"| {step} | {name} | " + " | ".join(cell((m, step, d)) for d in datasets) + f" | {mean} |")
