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
