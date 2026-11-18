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
ENGINE_BUILD_TAG="spark2"
ENGINE_COMMIT="6b9de695104b6b8b30c20458da87ac06c9480289"
QUANT_NAME="NVFP4"
MODEL_REPO="nvidia/Qwen3.8-Flash-Next-NVFP4"
WEIGHTS_REV="fc694b54fb0174e0913e6adf86691ef85a4ead47"
SPEED_CONTEXT="16384"
SPEED_PARALLEL="16"
GUARD_CONTEXT="139264"
GUARD_PARALLEL="2"

ZIG_VERSION="0.17.0"
ZIG_SHA256="9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8"
ZIG_DIR="$WORKDIR/engines/zig-${ZIG_VERSION}-aarch64"
ZIG_BIN="$ZIG_DIR/zig"

REL_DIR="$WORKDIR/engines/tensorfold-${ENGINE_VERSION}-aarch64"
REL_GRAMMAR="$REL_DIR/tensorfold-${ENGINE_VERSION}-linux-aarch64/lib/libtfgrammar.so"
SRC_DIR="$WORKDIR/engines/tf-src-${ENGINE_COMMIT:0:8}"
BUILD_DIR="$WORKDIR/engines/tf-build-${ENGINE_COMMIT:0:8}-${ENGINE_BUILD_TAG}"
ENGINE_BIN="$BUILD_DIR/native/bin/tensorfold-native"
MODEL_DIR="$WORKDIR/models/Qwen3.8-Flash-Next-NVFP4-${WEIGHTS_REV:0:8}"

apply_patches() {
  python3 - "$SRC_DIR" <<'PY'
import pathlib, sys

root = pathlib.Path(sys.argv[1])

def patch_one(rel: str, target: str, replacement: str) -> None:
    p = root / rel
    txt = p.read_text(encoding="utf-8")
    if target not in txt:
        raise RuntimeError(f"Patch target missing in {rel}")
    out = txt.replace(target, replacement, 1)
    lines = out.splitlines()
    if len(lines) >= 600:
        raise RuntimeError(f"Line limit exceeded in {rel}: {len(lines)}")
    p.write_text(out, encoding="utf-8")

proj_add = """    p.native = cuda.mxfp8.weight(p.w.ptr, p.s.ptr, n, k);
    return p;
}

fn f32ToE4m3(x: f32) u8 {
    const ub: u32 = @bitCast(x);
    const sign: u8 = if (ub & 0x80000000 != 0) 0x80 else 0;
    const ab: u32 = ub & 0x7fffffff;
    const ax: f32 = @bitCast(ab);
    if (ax >= 448.0) return sign | 0x7e;
    if (ax < (1.0 / 64.0)) {
        const magic: f32 = @bitCast(@as(u32, 0x4b400000));
        const s: f32 = ax * 512.0 + magic;
        const m_int: u32 = @as(u32, @bitCast(s)) - 0x4b400000;
        return sign | @as(u8, @intCast(m_int));
    }
    const rem = ab & 0xfffff;
    const inc: u32 = if (rem > 0x80000 or (rem == 0x80000 and ((ab >> 20) & 1) != 0)) 0x100000 else 0;
    const rnd = ab + inc;
    const re: i32 = @intCast((rnd >> 23) & 0xff);
    const rm: u8 = @intCast((rnd >> 20) & 7);
    const e8: i32 = re - 120;
    if (e8 > 15 or (e8 == 15 and rm == 7)) return sign | 0x7e;
    return sign | (@as(u8, @intCast(e8)) << 3) | rm;
}

pub fn useMxfp8(prefix: []const u8) bool {
    const raw = std.c.getenv("TF_FLASHNEXT_BF16_MXFP8") orelse return false;
    const m = std.mem.span(raw);
    if (std.mem.eql(u8, m, "all")) return true;
    const is_mtp = std.mem.indexOf(u8, prefix, ".mtp.") != null or std.mem.startsWith(u8, prefix, "mtp.");
    if (std.mem.eql(u8, m, "mtp")) return is_mtp;
    if (std.mem.eql(u8, m, "trunk")) return !std.mem.endsWith(u8, prefix, "lm_head");
    return false;
}

pub fn bf16AsMxfp8(gpa: std.mem.Allocator, driver: *const cuda.Driver, t: core.safetensors.Tensor) !Projection {
    if (t.rank != 2 or t.dtype != .bf16 or t.dim(0) == 0 or t.dim(1) == 0 or t.dim(1) % 64 != 0)
        return error.UnexpectedTensor;
    const n = t.dim(0);
    const k = t.dim(1);
    const count = try std.math.mul(usize, n, k);
    if (t.bytes.len != try std.math.mul(usize, count, 2)) return error.UnexpectedTensor;
    const raw_codes = try gpa.alloc(u8, count);
    defer gpa.free(raw_codes);
    const groups = count / 32;
    const raw_scales = try gpa.alloc(u8, groups);
    defer gpa.free(raw_scales);
    for (0..groups) |g| {
        const base = g * 32;
        var max_ab: u16 = 0;
        for (0..32) |i| {
            const b16 = std.mem.readInt(u16, t.bytes[(base + i) * 2 ..][0..2], .little);
            const ab16 = b16 & 0x7fff;
            if (ab16 > max_ab) max_ab = ab16;
        }
        if (max_ab == 0) {
            raw_scales[g] = 127;
            @memset(raw_codes[base..][0..32], 0);
            continue;
        }
        const eb: i32 = @intCast((max_ab >> 7) & 0xff);
        const mb: u16 = max_ab & 0x7f;
        const shift: i32 = if (eb == 0) -134 else (eb - 127) - (if (mb <= 96) @as(i32, 8) else @as(i32, 7));
        const ue: u8 = @intCast(std.math.clamp(shift + 127, 1, 253));
        raw_scales[g] = ue;
        const inv_scale: f32 = @bitCast(@as(u32, @intCast(254 - @as(i32, ue))) << 23);
        for (0..32) |i| {
            const b16 = std.mem.readInt(u16, t.bytes[(base + i) * 2 ..][0..2], .little);
            const val: f32 = @bitCast(@as(u32, b16) << 16);
            raw_codes[base + i] = f32ToE4m3(val * inv_scale);
        }
    }
    const cw: core.safetensors.Tensor = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = raw_codes };
    const cs: core.safetensors.Tensor = .{ .dtype = .u8, .rank = 2, .shape = .{ n, k / 32, 1, 1, 1 }, .bytes = raw_scales };
    return mxfp8(gpa, driver, cw, cs);
}

pub fn rootCopiesOn() bool {
    const raw = std.c.getenv("TF_FLASHNEXT_ROOT_COPIES") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "1");
}

pub fn copyMatch() i64 {
    const raw = std.c.getenv("TF_FLASHNEXT_COPY_MATCH") orelse return 8;
    return std.fmt.parseInt(i64, std.mem.span(raw), 10) catch 8;
}

pub fn firstCopyRows(def: u32) u32 {
    const raw = std.c.getenv("TF_FLASHNEXT_FIRST_COPY") orelse return def;
    return std.fmt.parseInt(u32, std.mem.span(raw), 10) catch def;
}"""

patch_one(
    "zig/src/families/flashnext/cuda_projection.zig",
    "    p.native = cuda.mxfp8.weight(p.w.ptr, p.s.ptr, n, k);\n    return p;\n}",
    proj_add,
)

patch_one(
    "zig/src/families/flashnext/cuda_projection_load.zig",
    "        .bf16 => {\n            if (!t.is(.bf16, &.{ n, k })) return error.UnexpectedTensor;\n            return projection.dense(a, d, t);\n        },",
    "        .bf16 => {\n            if (!t.is(.bf16, &.{ n, k })) return error.UnexpectedTensor;\n            return if (projection.useMxfp8(prefix)) projection.bf16AsMxfp8(a, d, t) else projection.dense(a, d, t);\n        },",
)

patch_one(
    "zig/src/families/flashnext/cuda_projection_load.zig",
    "        return projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = bytes });",
    "        const st_t: core.safetensors.Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = bytes };\n        return if (projection.useMxfp8(prefix)) projection.bf16AsMxfp8(a, d, st_t) else projection.dense(a, d, st_t);",
)

patch_one(
    "zig/src/families/flashnext/cuda_prompt.zig",
    '            e.head_q = try @import("cuda_projection.zig").dense(gpa, driver, if (try @import("cuda_head.zig").enabled()) try @import("cuda_head.zig").shard(head_tensor, world, rank) else head_tensor);',
    '            const ht = if (try @import("cuda_head.zig").enabled()) try @import("cuda_head.zig").shard(head_tensor, world, rank) else head_tensor;\n            e.head_q = try (if (@import("cuda_projection.zig").useMxfp8("lm_head")) @import("cuda_projection.zig").bf16AsMxfp8(gpa, driver, ht) else @import("cuda_projection.zig").dense(gpa, driver, ht));',
)

patch_one(
    "zig/src/families/flashnext/cuda_lanes.zig",
    "                    .first_copy_rows = if (drafting) mtp.max_chain + 1 else 0,",
    '                    .root_copies = @import("cuda_projection.zig").rootCopiesOn(), .copy_match = @import("cuda_projection.zig").copyMatch(), .first_copy_rows = if (drafting) @import("cuda_projection.zig").firstCopyRows(mtp.max_chain + 1) else 0,',
)

patch_one(
    "zig/src/core/lanes/proposer.zig",
    "    ngram: usize = 3,\n    min_match: i64 = 6,",
    "    ngram: usize = 3,\n    min_match: i64 = 3,",
)
PY
}

case "$ACTION" in
  meta)
    cat <<EOF
{"engine":"${ENGINE_NAME} v${ENGINE_VERSION}+${ENGINE_BUILD_TAG}","engine_rev":"${ENGINE_COMMIT:0:8}","quant":"${QUANT_NAME}","weights_rev":"${WEIGHTS_REV:0:8}","context":${SPEED_CONTEXT}}
EOF
    ;;

  prepare)
    mkdir -p "$WORKDIR/engines" "$WORKDIR/models"
    if [[ ! -f "$REL_GRAMMAR" ]]; then
      mkdir -p "$REL_DIR"
      ARCHIVE="$REL_DIR/tensorfold-${ENGINE_VERSION}-linux-aarch64.tar.gz"
      curl -fsSL "https://github.com/ashhart/TensorFold/releases/download/v${ENGINE_VERSION}/tensorfold-${ENGINE_VERSION}-linux-aarch64.tar.gz" -o "$ARCHIVE"
      tar -xzf "$ARCHIVE" -C "$REL_DIR"
    fi

    if [[ ! -x "$ZIG_BIN" ]]; then
      mkdir -p "$ZIG_DIR"
      ZIG_TAR="$ZIG_DIR/zig.tar.xz"
      curl -fsSL "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" -o "$ZIG_TAR"
      echo "${ZIG_SHA256}  ${ZIG_TAR}" | sha256sum -c -
      tar -xJf "$ZIG_TAR" -C "$ZIG_DIR" --strip-components=1
      rm -f "$ZIG_TAR"
    fi

    if [[ ! -x "$ENGINE_BIN" ]]; then
      if [[ ! -d "$SRC_DIR/.git" ]]; then
        rm -rf "$SRC_DIR"
        git init -q "$SRC_DIR"
        git -C "$SRC_DIR" remote add origin "https://github.com/ashhart/TensorFold.git"
        git -C "$SRC_DIR" fetch -q --depth 1 origin "$ENGINE_COMMIT"
      fi
      git -C "$SRC_DIR" checkout -q -f "$ENGINE_COMMIT"
      git -C "$SRC_DIR" clean -fdq -e .zig-cache
      apply_patches
      rm -rf "$BUILD_DIR"
      mkdir -p "$BUILD_DIR"
      (
        cd "$SRC_DIR"
        "$ZIG_BIN" build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix "$BUILD_DIR" -j"$(nproc)"
      )
      mkdir -p "$BUILD_DIR/native/lib" "$BUILD_DIR/lib"
      cp -f "$REL_GRAMMAR" "$BUILD_DIR/native/lib/libtfgrammar.so"
      cp -f "$REL_GRAMMAR" "$BUILD_DIR/lib/libtfgrammar.so"
      chmod +x "$ENGINE_BIN"
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
    export LD_LIBRARY_PATH="$BUILD_DIR/native/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export TF_FLASHNEXT_DRAFT_HEAD="affine4"
    export TF_FLASHNEXT_BF16_MXFP8="all"
    export TF_FLASHNEXT_ATTN_LANES="1"
    export TF_FLASHNEXT_WARM_AHEAD="1"
    export TF_FLASHNEXT_PROMPT_EXPERT_TILE="2"
    export TF_FLASHNEXT_CHUNK="3072"
    if [[ "$PROFILE" == "guard" || "$PROFILE" == "nvidia" ]]; then
      export TENSORFOLD_MEMORY_RESERVE_GIB=6
      CTX="$GUARD_CONTEXT"
      PAR="$GUARD_PARALLEL"
      THINK="--thinking"
    else
      export TF_FLASHNEXT_DRAFTS_GREEDY="14"
      export TF_FLASHNEXT_CONFIDENCE="-0.10"
      export TF_FLASHNEXT_GREEDY_WIDE="1"
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
