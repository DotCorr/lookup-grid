"""Timing helpers shared by the prototypes. Everything is pinned to one CPU thread
(see run_all.py) so the numbers compare like-for-like on an edge-style budget."""
import time


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self.t0


def best_of(fn, repeats=3):
    """Run fn() `repeats` times, return (last result, best wall-clock seconds)."""
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best
