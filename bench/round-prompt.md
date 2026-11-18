# Autoresearch Trial Agent Round (8 Trials)

You are the autonomous implementation trial agent for the `Qwen/Qwen3.8-Flash-Next` inference speed autoresearch loop.

## Mandatory Reading Before Trial 1
1. Read `program.md` in full and obey every constraint without exception.
2. Read `../repo/records/world-records.tsv`, `../repo/records/quantization.md`, `../repo/records/research-log.md`, and `../repo/<site>/results.tsv` (where `<site>` is `spark` for `loop-spark` or `colab` for `loop-colab`).
3. Inspect the current `<site>/candidate.sh` (your parent `keep` configuration).

## Strict Rules
- **Single-file rule**: In this worktree, edit and commit **ONLY `<site>/candidate.sh`**. Even kernel patches, scheduler patches, speculative decoding tweaks, or custom build steps must be embedded inside `<site>/candidate.sh` (e.g., via heredocs applied in `prepare`). Do **NOT** edit `program.md` or anything under `bench/`.
- **Do not fabricate numbers**: All measurements are executed exclusively by `bash bench/run_trial.sh <site>`.
- **No secrets or private identifiers**: Never write hostnames, usernames, absolute personal home-directory paths, internal SSH aliases, or GPQA question/choice text into any file or commit message. Use `$HOME`, `$QFN_WORKDIR`, and `$QFN_PORT`.
- **No new cron/monitoring jobs** and **no new authentication/OAuth/paid resources**. If a stop condition in `program.md` is hit, stop immediately and report the exact condition.

## Per-Trial Workflow (Repeat for Up to 8 Trials This Round)
For each trial in this round:
1. **Check upstream releases & PRs**: Use `gh api` (read-only) to check recent releases, commits, and PRs on `vllm-project/vllm`, `sgl-project/sglang`, `NVIDIA/TensorRT-LLM`, `ggml-org/llama.cpp`, `ashhart/TensorFold`, and `sharar-project/hibana` relevant to `Qwen3.8-Flash-Next`, GB10 / SM121 (or RTX PRO 6000), NVFP4, MTP / speculative decoding, KV cache, or CUDA graphs. Append a concise 1-line entry (JST timestamp, URL, summary, whether adopted) to `../repo/records/research-log.md`.
2. **Formulate one focused hypothesis**: Combine upstream advances or your own kernel/scheduler/MTP/CUDA-graph/KV-cache/offload optimization. Unless NVFP4 has hit the formal plateau criteria in `program.md`, keep quantization fixed to NVFP4.
3. **Edit and commit `<site>/candidate.sh` ONLY**:
   - Ensure `bash <site>/candidate.sh meta` prints valid JSON (`engine`, `engine_rev`, `quant`, `weights_rev`, `context`).
   - Stage only `<site>/candidate.sh`: `git add <site>/candidate.sh && git commit -m "trial <N>: <concise English summary>"`
4. **Run the fixed benchmark & guard**:
   - Execute `bash bench/run_trial.sh <site>`
   - Wait for completion. `run_trial.sh` automatically runs the pre-flight gate, speed/prefill benchmark, GPQA Diamond guard (`hash_match` or `G40`/`G198`), cold-start reproduction on `keep`, updates `../repo/<site>/results.tsv` and `../repo/<site>/progress.png`, scans for secrets, pushes `main`, and either advances this branch (`keep`) or tags `trial/<site>/<N>` and resets to the parent `keep` (`discard`/`crash`).
   - If `run_trial.sh` exits with code `3` (host busy with another process), sleep 60 seconds and retry the gate without creating duplicate commits; if it exits with code `2` or `4` (frozen hash mismatch, auth required, or CU floor), stop immediately.
5. Inspect the new row in `../repo/<site>/results.tsv`, refine your next hypothesis, and proceed directly to the next trial until 8 trials are completed in this round. Do not ask a human whether to continue.
