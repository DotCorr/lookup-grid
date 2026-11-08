# lookup-grid

A binary lookup-table network for image classification, trained **without gradients**: every
parameter is set by counting, in one pass over the data per layer. Inference uses only bitwise ops.
It is benchmarked on MNIST and Fashion-MNIST against a small PyTorch MLP, on one CPU thread.

## How it works ([`lookup_grid/lgn.py`](lookup_grid/lgn.py))

- **Input:** pixels are thermometer-coded into bits (thresholds 64 and 160).
- **Cells:** cells sit on a 2D grid. Each reads K = 6 bits from a random local neighbourhood of the
  previous map and holds a 2^6 = 64-entry truth table.
- **Hidden-layer training (counting):** each cell is assigned a random half S of the classes. For each
  6-bit pattern p, `table[p] = 1` iff P(y ∈ S | p) > P(y ∈ S), using Laplace-smoothed counts from the
  training set. Layers are filled one after another in a single pass each.
- **Readout:** two variants.
  - `vote`: one 1-bit table per class per readout cell; the class score is a popcount.
  - `log-count`: a uint8-quantised log P(p | class) table (an n-tuple / naive-Bayes readout); the
    class score is a sum of looked-up bytes.
- **Inference:** 64 samples are packed into each `uint64` word. Each 6-input table is evaluated as a
  63-mux Shannon tree of AND/OR/NOT, so no multiplies are involved. The benchmark asserts that this
  bit-sliced path gives exactly the same predictions as a plain table-gather reference.

## Results

Written by `run_all.py` from an actual run. Everything runs on **one CPU thread**. Model memory counts
truth tables plus wiring indices.

<!-- RESULTS:START -->
Measured 2026-11-08 14:28 on `arm64 arm / macOS-27.0-arm64-arm-64bit-Mach-O`, Python 3.13.15, numpy 2.5.4, torch 2.15.0, **1 CPU thread** for everything. Train time is wall-clock for the whole fit; inference is the best of repeated runs over the 10,000 test images.

### MNIST

| Model | Test acc | Train (s) | Infer 10k (s) | Model memory |
|---|---:|---:|---:|---:|
| MLP 784-256-10 fp32 (Adam, 5 ep) | 97.47% | 1.41 | 0.007 | 814 kB |
| MLP 784-32-10 fp32 (Adam, 5 ep) | 95.29% | 0.63 | 0.002 | 102 kB |
| LUT grid, 0 hidden, vote | 72.28% | 3.75 | 1.557 | 368 kB |
| LUT grid, 2 hidden, vote | 86.82% | 19.20 | 1.249 | 462 kB |
| LUT grid, 0 hidden, log-count | 92.75% | 5.07 | 1.967 | 2.61 MB |
| LUT grid, 1 hidden, log-count | 91.91% | 19.12 | 2.326 | 2.67 MB |

### Fashion-MNIST

| Model | Test acc | Train (s) | Infer 10k (s) | Model memory |
|---|---:|---:|---:|---:|
| MLP 784-256-10 fp32 (Adam, 5 ep) | 87.22% | 2.34 | 0.007 | 814 kB |
| MLP 784-32-10 fp32 (Adam, 5 ep) | 85.32% | 0.92 | 0.002 | 102 kB |
| LUT grid, 0 hidden, vote | 71.00% | 5.30 | 2.157 | 368 kB |
| LUT grid, 2 hidden, vote | 74.86% | 26.84 | 1.738 | 462 kB |
| LUT grid, 0 hidden, log-count | 74.66% | 5.28 | 2.432 | 2.61 MB |
| LUT grid, 1 hidden, log-count | 76.37% | 22.37 | 2.886 | 2.67 MB |
<!-- RESULTS:END -->

## What the numbers say

- **It works, but accuracy trails a small MLP.** The fully bitwise version (popcount vote) reaches
  only about 72–87% on MNIST. Hidden layers help it (about +15 points on MNIST).
- **The log-count readout is better, but larger.** It reaches about 93% on MNIST and 75–76% on
  Fashion-MNIST, but its uint8 tables take 2.6 MB. Hidden layers give it no gain on MNIST.
- **Why it plateaus:** counting statistics are independent per table, so correlated tables are
  double-counted. That ceiling is a known property of n-tuple classifiers, not a bug.
- **Inference is slow in numpy.** It takes over a second per 10k images: the operations are cheap,
  but numpy pays heavy per-array overhead to run them. The design targets an FPGA/ASIC or a C kernel,
  which this repo does not demonstrate.
- **No energy was measured** (`powermetrics` needs root). Wall-clock time on one thread is the only
  cost measured.
- **Timing is noisy:** runs were on a shared laptop, so read differences under ~30% as noise.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run_all.py          # downloads MNIST + Fashion-MNIST into data/ on first run
.venv/bin/python run_all.py --quick  # MNIST only, does not touch README
```

## Next steps

- A C or SIMD kernel for the bit-sliced table evaluation, to measure the real speed of
  multiply-free inference.

Part of a set of three edge-ML prototypes, alongside
[rns-arith](https://github.com/DotCorr/rns-arith) and
[closed-form-nets](https://github.com/DotCorr/closed-form-nets).
