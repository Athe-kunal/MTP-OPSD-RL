"""Summarize MEMPROF lines from bench_memory logs into a side-by-side table.

Usage: python summarize_memprof.py outputs/bench_memory
"""
import json
import pathlib
import re
import sys
from collections import defaultdict

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/bench_memory")
rows = defaultdict(list)
status = {}
for log in sorted(root.glob("*.log")):
    m = re.match(r"(soft_kl|explicit)_(\d+)\.log", log.name)
    if not m:
        continue
    key = (int(m.group(2)), m.group(1))
    text = log.read_text(errors="ignore")
    for line in text.splitlines():
        if line.startswith("MEMPROF "):
            rows[key].append(json.loads(line[len("MEMPROF "):]))
    code = re.findall(r"exit_code=(\d+)", text)
    oom = "OutOfMemoryError" in text or "CUDA out of memory" in text
    status[key] = "OOM" if oom else ("ok" if code and code[-1] == "0" else f"exit {code[-1] if code else '?'}")


def agg(recs, field, fn=max):
    return fn(r[field] for r in recs) if recs else float("nan")


lengths = sorted({k[0] for k in status})
print("| max_completion | method | status | peak alloc GB | loss fwd+bwd delta GB | peak reserved GB | fwd+bwd s (median) | seq len S/T |")
print("|---|---|---|---|---|---|---|---|")
for L in lengths:
    for M in ("explicit", "soft_kl"):
        if (L, M) not in status:
            continue
        r = rows[(L, M)]
        times = sorted(x["loss_fwd_bwd_s"] for x in r[2:] or r)  # skip warm-up micro-steps
        med = times[len(times) // 2] if times else float("nan")
        seq = f'{agg(r, "seq_len_student")}/{agg(r, "seq_len_teacher")}' if r else "-"
        print(f"| {L} | {M} | {status[(L, M)]} | {agg(r, 'peak_gb'):.2f} | {agg(r, 'loss_delta_gb'):.2f} "
              f"| {agg(r, 'reserved_peak_gb'):.2f} | {med:.2f} | {seq} |")
