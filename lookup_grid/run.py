"""Benchmark the lookup-grid network: train by counting, infer with bit-sliced ops."""
import numpy as np

from common.bench import Timer, best_of
from lookup_grid.lgn import LookupGrid

CONFIGS = [
    ("LUT grid, 0 hidden, vote", dict(layers=(), n_readout=4000, readout="vote")),
    ("LUT grid, 2 hidden, vote", dict(layers=((28, 28, 4, 2), (14, 14, 8, 2)), n_readout=4000, readout="vote")),
    ("LUT grid, 0 hidden, log-count", dict(layers=(), n_readout=4000, readout="logcount")),
    ("LUT grid, 1 hidden, log-count", dict(layers=((28, 28, 4, 2),), n_readout=4000, readout="logcount")),
]


def run(xtr, ytr, xte, yte):
    rows = []
    for name, cfg in CONFIGS:
        with Timer() as t:
            m = LookupGrid(**cfg).fit(xtr, ytr)
        pred, t_inf = best_of(lambda: m.predict_bitwise(xte), repeats=2)
        ref = m.predict_gather(xte)
        assert (pred == ref).all(), "bit-sliced inference disagrees with table-gather reference"
        tables, wires = m.mem_bytes()
        rows.append({
            "model": name,
            "acc": float((pred == yte).mean()),
            "train_s": t.s,
            "infer_s": t_inf,
            "mem_bytes": tables + wires,
            "note": f"tables {tables/1e3:.0f} kB + wiring {wires/1e3:.0f} kB; bitwise == gather reference",
        })
    return rows
