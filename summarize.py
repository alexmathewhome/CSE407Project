"""Print a table of all runs in runs/<experiment>/ and the mean and standard deviation over seeds.
Usage: python summarize.py demo
"""
import json, statistics, sys
from pathlib import Path

exp = sys.argv[1]
rows = []
for meta_file in sorted((Path(__file__).resolve().parent / "runs" / exp).glob("seed_*/meta.json")):
    rows.append(json.loads(meta_file.read_text()))

print(f"\n{'seed':>5} {'status':<14} {'epochs':>6} {'best epoch':>10} {'best val loss':>14}")
for m in rows:
    best = "-" if m["best_val_loss"] is None else f"{m['best_val_loss']:.4f}"
    print(f"{m['seed']:>5} {m['status']:<14} {m['epochs_run']:>6} {str(m['best_epoch']):>10} {best:>14}")

vals = [m["best_val_loss"] for m in rows if m["status"] != "running" and m["best_val_loss"] is not None]
if len(vals) > 1:
    print(f"\nbest val loss over {len(vals)} seeds: mean {statistics.mean(vals):.4f}, "
          f"std {statistics.stdev(vals):.4f}")
