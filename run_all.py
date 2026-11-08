"""Run the lookup_grid prototype and the MLP baselines on MNIST and Fashion-MNIST, single CPU thread.

Writes results/results.json and results/results.md, and refreshes the results section of
README.md between the RESULTS markers.  Usage:  .venv/bin/python run_all.py [--quick]
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import json
import platform
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)

import importlib

from baseline import mlp
from common import data

SUITES = ["lookup_grid"]
ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["mnist", "fashion"]


def fmt_bytes(b):
    return f"{b / 1e6:.2f} MB" if b >= 1e6 else f"{b / 1e3:.0f} kB"


def table(rows):
    out = ["| Model | Test acc | Train (s) | Infer 10k (s) | Model memory |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"| {r['model']} | {r['acc'] * 100:.2f}% | {r['train_s']:.2f} | {r['infer_s']:.3f} | "
                   f"{fmt_bytes(r['mem_bytes'])} |")
    return "\n".join(out)


def main():
    quick = "--quick" in sys.argv
    results = {"machine": f"{platform.machine()} {platform.processor()} / {platform.platform()}",
               "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
               "threads": 1, "date": time.strftime("%Y-%m-%d %H:%M"), "datasets": {}}
    for name in DATASETS[:1] if quick else DATASETS:
        print(f"== {name}")
        xtr, ytr, xte, yte = data.load(name)
        suites = {"baseline": lambda: [mlp.run(xtr, ytr, xte, yte, hidden=256),
                                       mlp.run(xtr, ytr, xte, yte, hidden=32)]}
        for s in SUITES:
            suites[s] = (lambda m: lambda: m.run(xtr, ytr, xte, yte))(importlib.import_module(f"{s}.run"))
        results["datasets"][name] = {}
        for suite, fn in suites.items():
            rows = fn()
            for r in rows:
                print(f"  {r['model']:<48} acc {r['acc']:.4f}  train {r['train_s']:7.2f}s  "
                      f"infer {r['infer_s']:.3f}s  mem {fmt_bytes(r['mem_bytes'])}")
            results["datasets"][name][suite] = rows

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    md = [f"Measured {results['date']} on `{results['machine']}`, Python {results['python']}, "
          f"numpy {results['numpy']}, torch {results['torch']}, **1 CPU thread** for everything. "
          "Train time is wall-clock for the whole fit; inference is the best of repeated runs over "
          "the 10,000 test images."]
    for name, suites in results["datasets"].items():
        md.append(f"\n### {'MNIST' if name == 'mnist' else 'Fashion-MNIST'}\n")
        md.append(table([r for rows in suites.values() for r in rows]))
        if "rns_arith" not in suites:
            continue
        rns = suites["rns_arith"][0]["extra"]
        md.append(f"\nRNS check: float32 accuracy of the same trained net {rns['float_acc'] * 100:.2f}%; "
                  f"residue result identical to int64 arithmetic: **{rns['exact']}** "
                  f"(max |accumulator| {rns['max_abs_acc']:,} vs dynamic range M = {rns['dynamic_range']:,}). "
                  f"Modelled energy per image ({rns['macs']:,} MACs): fp32 {rns['fp32_nJ']:.0f} nJ, "
                  f"RNS 3x9-bit {rns['rns_nJ']:.0f} nJ, plain int8/int32 {rns['int8_nJ']:.0f} nJ.")
    md = "\n".join(md) + "\n"
    with open(os.path.join(ROOT, "results", "results.md"), "w") as f:
        f.write(md)

    readme = os.path.join(ROOT, "README.md")
    if os.path.exists(readme) and not quick:
        s = open(readme).read()
        a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if a in s and b in s:
            s = s[: s.index(a) + len(a)] + "\n" + md + s[s.index(b):]
            open(readme, "w").write(s)
    print("wrote results/results.json, results/results.md")


if __name__ == "__main__":
    main()
