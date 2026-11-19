#!/usr/bin/env bash
# Single editable file for the DGX Spark (GB10) autoresearch loop.
# Exposes three entrypoints called by bench/run_trial.sh:
#   meta     — print JSON metadata (engine, engine_rev, quant, weights_rev, context)
#   prepare  — fetch/patch/build engine and verify cached weights
#   start    — launch OpenAI-compatible server in the foreground on 127.0.0.1:$QFN_PORT
set -euo pipefail

ACTION="${1:?Usage: spark/candidate.sh <meta|prepare|start> [speed|guard|nvidia]}"
PROFILE="${2:-speed}"

WORKDIR="${QFN_WORKDIR:-$HOME/qfn-autoresearch}"
PORT="${QFN_PORT:-29771}"

ENGINE_NAME="TensorFold"
ENGINE_VERSION="1.2.9"
ENGINE_COMMIT="6b9de695104b6b8b30c20458da87ac06c9480289"
QUANT_NAME="NVFP4"
MODEL_REPO="nvidia/Qwen3.8-Flash-Next-NVFP4"
WEIGHTS_REV="fc694b54fb0174e0913e6adf86691ef85a4ead47"
SPEED_CONTEXT="16384"
SPEED_PARALLEL="16"
GUARD_CONTEXT="139264"
GUARD_PARALLEL="2"

ENGINE_DIR="$WORKDIR/engines/tensorfold-${ENGINE_VERSION}-aarch64"
RELEASE_BIN="$ENGINE_DIR/tensorfold-${ENGINE_VERSION}-linux-aarch64/bin/tensorfold-native"
BUILD_FLAVOR="mtp-vocab-sm121"
BUILD_DIR="$WORKDIR/engines/tf-${ENGINE_COMMIT:0:8}-$BUILD_FLAVOR"
ENGINE_BIN="$BUILD_DIR/native/bin/tensorfold-native"
MODEL_DIR="$WORKDIR/models/Qwen3.8-Flash-Next-NVFP4-${WEIGHTS_REV:0:8}"

case "$ACTION" in
  meta)
    cat <<EOF
{"engine":"${ENGINE_NAME} v${ENGINE_VERSION}+${BUILD_FLAVOR}","engine_rev":"${ENGINE_COMMIT:0:8}","quant":"${QUANT_NAME}","weights_rev":"${WEIGHTS_REV:0:8}","context":${SPEED_CONTEXT}}
EOF
    ;;

  prepare)
    mkdir -p "$WORKDIR/engines" "$WORKDIR/models"
    if [[ ! -x "$RELEASE_BIN" ]]; then
      mkdir -p "$ENGINE_DIR"
      ARCHIVE="$ENGINE_DIR/tensorfold-${ENGINE_VERSION}-linux-aarch64.tar.gz"
      curl -fsSL "https://github.com/ashhart/TensorFold/releases/download/v${ENGINE_VERSION}/tensorfold-${ENGINE_VERSION}-linux-aarch64.tar.gz" -o "$ARCHIVE"
      tar -xzf "$ARCHIVE" -C "$ENGINE_DIR"
      chmod +x "$RELEASE_BIN"
    fi


    if [[ ! -x "$ENGINE_BIN" ]]; then
      mkdir -p "$WORKDIR/engines"
      stage="$(mktemp -d "$WORKDIR/engines/.tf-build.XXXXXXXX")"
      trap 'rm -rf -- "$stage"' EXIT
      zig_dir="$WORKDIR/engines/zig-0.17.0-aarch64"
      if [[ ! -x "$zig_dir/zig" ]]; then
        curl -fsSL "https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz" -o "$stage/zig.tar.xz"
        echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $stage/zig.tar.xz" | sha256sum -c -
        mkdir "$stage/zig"
        tar -xJf "$stage/zig.tar.xz" --strip-components=1 -C "$stage/zig"
        mv "$stage/zig" "$zig_dir"
      fi
      git init -q "$stage/src"
      git -C "$stage/src" fetch -q --depth 1 https://github.com/ashhart/TensorFold.git "$ENGINE_COMMIT"
      git -C "$stage/src" checkout -q --detach "$ENGINE_COMMIT"
      python3 - "$stage/src/zig/src/families/flashnext/cuda_mtp.zig" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text()
old = 'return proposal.legacy.ids(gpa, @import("proposal_table.zig").text, .{ .vocab = vocab,'
new = '''const cutoff: u32 = if (std.c.getenv("QFN_DRAFT_VOCAB_LIMIT")) |raw|
        try std.fmt.parseInt(u32, std.mem.span(raw), 10)
    else
        vocab;
    if (cutoff == 0 or cutoff > vocab) return error.BadDraftVocab;
    return proposal.legacy.ids(gpa, @import("proposal_table.zig").text, .{ .vocab = cutoff,'''
if source.count(old) != 1:
    raise ValueError("unexpected draft vocabulary loader")
path.write_text(source.replace(old, new))
PY
      (
        cd "$stage/src"
        "$zig_dir/zig" build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc \
          -Doptimize=fast -j6 --prefix "$stage/install"
      )
      test -x "$stage/install/native/bin/tensorfold-native"
      mv "$stage/install" "$BUILD_DIR"
    fi

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
    export LD_LIBRARY_PATH="${RELEASE_BIN%/bin/*}/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    unset QFN_DRAFT_VOCAB_LIMIT
    export TF_FLASHNEXT_DRAFT_HEAD="affine4"
    export TF_FLASHNEXT_DRAFTS_GREEDY="14"
    export TF_FLASHNEXT_CONFIDENCE="-0.10"
    export TF_FLASHNEXT_GREEDY_WIDE="1"
    if [[ "$PROFILE" == "guard" || "$PROFILE" == "nvidia" ]]; then
      export TENSORFOLD_MEMORY_RESERVE_GIB=6
      CTX="$GUARD_CONTEXT"
      PAR="$GUARD_PARALLEL"
      THINK="--thinking"
    else
      CTX="$SPEED_CONTEXT"
      PAR="$SPEED_PARALLEL"
      THINK="--no-thinking"
    fi
    exec "$ENGINE_BIN" serve "$MODEL_DIR" \
      --host 127.0.0.1 \
      --port "$PORT" \
      --name Qwen3.8-Flash-Next \
      --backend cuda \
      --context "$CTX" \
      --parallel "$PAR" \
      --prompt-cache-gib 0 \
      --max-tokens "$CTX" \
      "$THINK" \
      --no-update-check
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
