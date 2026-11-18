#!/usr/bin/env bash
# Single editable file for the Colab RTX PRO 6000 (G4, 96 GB VRAM, x86_64) autoresearch loop.
# Exposes three entrypoints called by bench/run_trial.sh:
#   meta     — print JSON metadata (engine, engine_rev, quant, weights_rev, context)
#   prepare  — fetch/patch/build engine and verify cached weights
#   start    — launch OpenAI-compatible server in the foreground on 127.0.0.1:$QFN_PORT
set -euo pipefail

ACTION="${1:?Usage: colab/candidate.sh <meta|prepare|start> [speed|guard|nvidia]}"
PROFILE="${2:-speed}"

WORKDIR="${QFN_WORKDIR:-$HOME/qfn-autoresearch}"
PORT="${QFN_PORT:-29771}"

ENGINE_NAME="SGLang"
ENGINE_REV="v0.5.23"
QUANT_NAME="NVFP4"
MODEL_REPO="nvidia/Qwen3.8-Flash-Next-NVFP4"
WEIGHTS_REV="fc694b54fb0174e0913e6adf86691ef85a4ead47"
SPEED_CONTEXT="16384"

MODEL_DIR="$WORKDIR/models/Qwen3.8-Flash-Next-NVFP4-${WEIGHTS_REV:0:8}"

case "$ACTION" in
  meta)
    cat <<EOF
{"engine":"${ENGINE_NAME}","engine_rev":"${ENGINE_REV}","quant":"${QUANT_NAME}","weights_rev":"${WEIGHTS_REV:0:8}","context":${SPEED_CONTEXT}}
EOF
    ;;

  prepare)
    mkdir -p "$WORKDIR/models"
    if [[ ! -f "$MODEL_DIR/config.json" ]]; then
      python3 - "$MODEL_REPO" "$WEIGHTS_REV" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo_id, rev, local_dir = sys.argv[1:4]
snapshot_download(repo_id=repo_id, revision=rev, local_dir=local_dir)
PY
    fi
    ;;

  start)
    CTX="$SPEED_CONTEXT"
    if [[ "$PROFILE" == "guard" ]]; then
      CTX="139264"
    fi
    exec python3 -m sglang.launch_server \
      --model-path "$MODEL_DIR" \
      --served-model-name Qwen3.8-Flash-Next \
      --host 127.0.0.1 \
      --port "$PORT" \
      --context-length "$CTX" \
      --quantization modelopt_fp4 \
      --disable-radix-cache
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
