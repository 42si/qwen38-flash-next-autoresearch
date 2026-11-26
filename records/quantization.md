# Quantization Policy & Selection Log (`records/quantization.md`)

## 1. Published GPQA Diamond Accuracy Reference (No Fabricated Local Claims)

Published evaluation figures from official model cards (distinct checkpoints separated explicitly):

| Checkpoint | Quantization Method | Published GPQA Diamond | Reference Baseline on Same Card | Source URL |
|---|---|---:|---|---|
| `nvidia/Qwen3.8-Flash-Next-NVFP4` (`fc694b54`) | NVIDIA ModelOpt NVFP4 | **91.5%** | FP8: **92.0%** (same model card) | <https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4> |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` (`b9ce18a6`) | NVIDIA QAD FP4-Local (~96 GiB on disk) | **91.5%** | BF16: **92.2%** (same model card) | <https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/blob/b9ce18a6887e18c089fdea8509a33e63429a3300/README.md> |
| Hibana calibrated 3-bit (`sharar-project/hibana`) | 3-bit expert weights + NVFP4 KV | Table 4: **90.4%** (3-bit only), **89.1%** (fast profile); Table 6 (248K budget): **89.6%** (3-bit), **87.0%** (fast) | Exact profile: **90.7%** | <https://arxiv.org/abs/2611.10469v1> |

*Note*: The shorthand comparison "NVFP4 91.5% vs BF16 92.2%" combines `nvidia/Qwen3.8-Flash-Next-NVFP4` (91.5% vs FP8 92.0% on its card) with the BF16 figure (92.2%) reported on `nvidia/Qwen3.8-Flash-Next-FP4-Local`.

## 2. Initial Selection Rule (Q0)

1. **Default starting quantization**: `NVFP4` (`nvidia/Qwen3.8-Flash-Next-NVFP4@fc694b54`).
2. **Short pre-loop check (Q0)**: Compare baseline `NVFP4` measurements against published data and at most 2 candidate checkpoints that show clear promise of exceeding NVFP4.
3. **"Clearly better" threshold**: On the same engine and frozen `bench/`, `decode_c1_tps` must exceed NVFP4 by **>= 10%** AND GPQA Diamond correct count on the identical split must be **>= NVFP4**. Otherwise, quantization is locked to **NVFP4**, and autoresearch trials focus exclusively on engines, kernels, schedulers, speculative decoding (MTP), CUDA graphs, and KV cache.

## 3. Plateau & Quantization Switch Rule

Switching away from NVFP4 is permitted **only** when all three plateau criteria in `program.md` hold simultaneously:
1. Zero `keep` trials over the last 24 trials, OR total `decode_c1_tps` gain < 3% across the last 3 rounds.
2. Best guard-passing `decode_c1_tps` remains below 85% of the verified target published record.
3. No high-leverage unattempted NVFP4 hypotheses remain in `records/research-log.md`.

When switching, candidate formats are evaluated in accuracy-preserving order (`FP4-Local`, `MXFP4`, calibrated `INT4` AWQ/GPTQ/AutoRound, `FP8`, calibrated 3-bit), and the new starting point must pass the full `G198` GPQA Diamond guard before becoming the parent commit.

## 4. Decision Log (Q0 Completed — Locked to NVFP4)

### Q0 Same-Engine / Same-Harness Comparison on DGX Spark (GB10 × 1)

Measured on `TensorFold v1.2.9` (`6b9de695`) with the identical frozen 16-prompt 400-token code benchmark (`temperature=0`, `ignore_eos=True`, thinking disabled, 3 repeats median) and identical GPQA Diamond `G13` split (`temperature=1.0`, `top_p=0.95`, `top_k=20`, thinking enabled, `seed=42`, choice order `seed=0`):

| Candidate Checkpoint | `decode_c1_tps` | Gain vs NVFP4 | `total_c4_tps` | `total_c8_tps` | `total_c16_tps` | `ttft_c1_s` | `prefill_tps` | GPQA `G13` | Meets Q0 Rule (`>= +10%` speed AND `>=` GPQA)? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`nvidia/Qwen3.8-Flash-Next-NVFP4` (`fc694b54`)** | **74.87** | baseline | 137.76 | 187.93 | 228.41 | 0.185 | 1,743.19 | **13 / 13** | **Selected (Locked)** |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` (`b9ce18a6`) | 80.19 | +7.11% | 159.10 | 206.31 | 242.83 | 0.174 | 1,638.51 | 12 / 13 | No (`+7.11% < +10%`, and `12/13 < 13/13`) |

*(Reference check on `Hibana v0.1.0` `exact` profile with `RadixArk/Qwen3.8-Flash-Next-NVFP4`: `decode_c1_tps = 78.67`, `total_c4_tps = 162.11`, `total_c16_tps = 269.84`, GPQA `G13 = 13 / 13`.)*

| Date/Time (JST) | Site | Selected Quantization | Status | Evidence & Notes |
|---|---|---|---|---|
| 2026-11-19 04:20 | spark | **`NVFP4` (`fc694b54`)** | **Locked after Q0** | `FP4-Local` achieved `+7.11%` single-stream decode (below `+10%` threshold) and `12/13` vs `13/13` on `G13` (`-6.0%` slower prefill). Quantization is locked to **NVFP4**; all subsequent trials focus on engines, kernels, schedulers, speculative decoding (MTP), CUDA graphs, and KV cache until plateau criteria (§3) are met. |
