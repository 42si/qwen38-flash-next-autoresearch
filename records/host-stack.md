# Host software stack changes

Measurements are only comparable within one stack. When the stack changes, the previous ledger is archived and a new trial 0 baseline is measured first.

## DGX Spark (GB10)

| Stack | Period (JST) | Kernel | NVIDIA driver | DGX OTA meta | CUDA toolkit | Ledger |
|---|---|---|---|---|---|---|
| stack-20261118 | until 2026-11-19 06:50 | 6.17.0-1029-nvidia | 580.173.02 | 26.09.2 | 13.0 | `spark/results-stack-20261118.tsv`, `spark/progress-stack-20261118.png` |
| stack-20261119 | from 2026-11-19 08:38 | 7.0.0-1025-nvidia | 580.178.04 | 26.09.3 | 13.0 (V13.0.88) | `spark/results.tsv`, `spark/progress.png` |

Other packages updated at the same time: nvidia-container-toolkit 1.19.1 -> 1.20.2, containerd.io 2.2.1 -> 2.2.5, Ubuntu 24.04 security/updates (incl. mesa, webkit). A hardware watchdog (60 s) and auto-reboot on kernel panic / soft lockup were enabled after two hard hangs (2026-11-19 04:31 and 06:50 JST; no OOM record).

Notes on stack-20261118 results (not comparable with the new stack):

- trial 0 baseline 74.87 tok/s (keep), trial 1 92.59 tok/s (keep, commit `a7c98672`), trial 2 94.74 tok/s (discard, tag `trial/spark-20261118/2`).
- trial 3 (commit `839354f`, MXFP8 packing of excluded BF16 trunk + `lm_head`) was never measured: the host hung during the run. Preserved as tag `trial/spark-20261118/3-unmeasured`.
