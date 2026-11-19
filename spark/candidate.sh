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
BUILD_FLAVOR="mtp-w2-nvfp4-g4-head4-fp8g"
QUANT_NAME="NVFP4"
MODEL_REPO="nvidia/Qwen3.8-Flash-Next-NVFP4"
WEIGHTS_REV="fc694b54fb0174e0913e6adf86691ef85a4ead47"
SPEED_CONTEXT="16384"
SPEED_PARALLEL="16"
GUARD_CONTEXT="139264"
GUARD_PARALLEL="2"

ENGINE_DIR="$WORKDIR/engines/tensorfold-${ENGINE_VERSION}-aarch64"
RELEASE_BIN="$ENGINE_DIR/tensorfold-${ENGINE_VERSION}-linux-aarch64/bin/tensorfold-native"
RELEASE_LIB="$ENGINE_DIR/tensorfold-${ENGINE_VERSION}-linux-aarch64/lib"
SRC_DIR="$WORKDIR/engines/tf-src-${ENGINE_COMMIT:0:8}"
BUILD_DIR="$WORKDIR/engines/tf-${ENGINE_COMMIT:0:8}-${BUILD_FLAVOR}"
ENGINE_BIN="$BUILD_DIR/native/bin/tensorfold-native"
MODEL_DIR="$WORKDIR/models/Qwen3.8-Flash-Next-NVFP4-${WEIGHTS_REV:0:8}"

apply_patches() {
  local fn_dir="$SRC_DIR/zig/src/families/flashnext"

  # 1a. Append W2 group-32 proposal GEMV + shared-BF16 exact top-64 rescore kernels to flashnext_mtp.cu
  cat >> "$SRC_DIR/zig/kernels/cuda/flashnext_mtp.cu" <<'CU'

extern "C" __global__ void __launch_bounds__(128) fn_w2_gemv(
    const __nv_bfloat16* __restrict__ X,
    const uint64_t* __restrict__ W,
    const unsigned short* __restrict__ S,
    __nv_bfloat16* __restrict__ out,
    int V) {
    __shared__ float sx[2560];
    __shared__ float sg_sum[80];
    const int tid = threadIdx.x;
    for (int i = tid; i < 2560; i += 128) {
        sx[i] = __bfloat162float(X[i]);
    }
    __syncthreads();
    if (tid < 80) {
        float s = 0.0f;
        const float* xg = sx + tid * 32;
        #pragma unroll
        for (int j = 0; j < 32; ++j) s += xg[j];
        sg_sum[tid] = 1.5f * s;
    }
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31;
    const int blk = blockIdx.x * 4 + warp;
    if (blk * 32 >= V) return;
    const uint64_t* wp = W + static_cast<size_t>(blk) * 80 * 32 + lane;
    const unsigned short* sp = S + static_cast<size_t>(blk) * 80 * 32 + lane;
    float acc = 0.0f;
    for (int g = 0; g < 80; ++g) {
        const uint64_t bits = __ldg(wp + g * 32);
        const unsigned short sb = __ldg(sp + g * 32);
        const float sc = __uint_as_float(static_cast<uint32_t>(sb) << 16);
        const float* xg = sx + g * 32;
        float dot = 0.0f;
        const uint32_t lo = static_cast<uint32_t>(bits);
        const uint32_t hi = static_cast<uint32_t>(bits >> 32);
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            dot = fmaf(xg[j], static_cast<float>((lo >> (2 * j)) & 3u), dot);
        }
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            dot = fmaf(xg[16 + j], static_cast<float>((hi >> (2 * j)) & 3u), dot);
        }
        acc = fmaf(sc, dot - sg_sum[g], acc);
    }
    const int row = blk * 32 + lane;
    if (row < V) out[row] = __float2bfloat16_rn(acc);
}

extern "C" __global__ void __launch_bounds__(256) fn_w2_rescore(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W_full_bf16,
    const uint32_t* __restrict__ vocab_ids,
    const int* __restrict__ cols,
    __nv_bfloat16* __restrict__ dlogits,
    int V) {
    const int c = cols[blockIdx.x];
    if (c < 0 || c >= V) return;
    const uint32_t gid = vocab_ids[c];
    const uint32_t* xp = reinterpret_cast<const uint32_t*>(X);
    const uint32_t* wp = reinterpret_cast<const uint32_t*>(W_full_bf16 + static_cast<size_t>(gid) * 2560);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < 1280; i += 256) {
        const uint32_t xv = __ldg(xp + i);
        const uint32_t wv = __ldg(wp + i);
        const float x0 = __uint_as_float((xv & 0xFFFFu) << 16);
        const float x1 = __uint_as_float(xv & 0xFFFF0000u);
        const float w0 = __uint_as_float((wv & 0xFFFFu) << 16);
        const float w1 = __uint_as_float(wv & 0xFFFF0000u);
        sum = fmaf(x0, w0, fmaf(x1, w1, sum));
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    __shared__ float ws[8];
    if (lane == 0) ws[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < 8 ? ws[lane] : 0.0f;
        for (int o = 4; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) dlogits[c] = __float2bfloat16_rn(sum);
    }
}
CU

  # 1b. Append NVFP4 head quantizer + fused top-128 & floor + exact BF16 rescore kernels to flashnext_batch.cu
  cat >> "$SRC_DIR/zig/kernels/cuda/flashnext_batch.cu" <<'CU'

#include <cuda_fp8.h>

extern "C" __global__ void __launch_bounds__(256) fn_bf16_to_nvfp4(
    const uint16_t* __restrict__ bf16,
    float global_scale,
    uint8_t* __restrict__ codes,
    uint8_t* __restrict__ scales,
    long long total_groups) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_groups) return;
    float w[16];
    float mmax = 0.0f;
    const uint16_t* gp = bf16 + idx * 16;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        const uint32_t bits = gp[j];
        const float f = __uint_as_float(bits << 16);
        w[j] = f;
        mmax = fmaxf(mmax, fabsf(f));
    }
    const float target = fminf(fmaxf(mmax / (global_scale * 6.0f), 0.0f), 448.0f);
    const __nv_fp8_storage_t sc_code = __nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3);
    scales[idx] = static_cast<uint8_t>(sc_code);
    const float sc_val = __half2float(__half(__nv_cvt_fp8_to_halfraw(sc_code, __NV_E4M3))) * global_scale;
    uint64_t word = 0;
    if (sc_val > 0.0f) {
        const float inv = 1.0f / sc_val;
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            auto q4 = [](float v) -> uint64_t {
                const uint64_t s = signbit(v) ? 8ull : 0ull;
                const float a = fabsf(v);
                const uint64_t m = a < 0.25f ? 0ull : a <= 0.75f ? 1ull : a < 1.25f ? 2ull : a <= 1.75f ? 3ull : a < 2.5f ? 4ull : a <= 3.5f ? 5ull : a < 5.0f ? 6ull : 7ull;
                return s | m;
            };
            const uint64_t b = q4(w[2 * p] * inv) | (q4(w[2 * p + 1] * inv) << 4);
            word |= (b << (8 * p));
        }
    }
    reinterpret_cast<uint64_t*>(codes)[idx] = word;
}

extern "C" __global__ void __launch_bounds__(1024) fn_head_topk_floor(
    __nv_bfloat16* __restrict__ L,
    int V,
    long long ld,
    int K,
    int* __restrict__ out_c) {
    const int r = blockIdx.x, t = threadIdx.x, nt = blockDim.x;
    unsigned short* row = reinterpret_cast<unsigned short*>(L + static_cast<long long>(r) * ld);
    int* oc = out_c + static_cast<long long>(r) * K;
    __shared__ unsigned int hist[256];
    __shared__ unsigned int scan[1024], scan_eq[1024];
    __shared__ unsigned int pick_hi, pick_lo, above;
    for (int i = t; i < 256; i += nt) hist[i] = 0;
    __syncthreads();
    int bad = 0;
    for (int i = t; i < V; i += nt) {
        const unsigned short u = row[i];
        bad |= ((u & 0x7fff) > 0x7f80) | (u == 0x7f80);
        const unsigned short uk = (u == 0x8000u) ? 0u : u;
        const unsigned int ok = (uk & 0x8000u) ? (~uk & 0xFFFFu) : (uk | 0x8000u);
        atomicAdd(&hist[ok >> 8], 1u);
    }
    const int row_bad = __syncthreads_or(bad);
    if (row_bad) {
        if (t == 0) oc[0] = -1;
        return;
    }
    if (t == 0) {
        unsigned int cum = 0;
        int b = 255;
        for (; b > 0; --b) {
            if (cum + hist[b] >= static_cast<unsigned int>(K)) break;
            cum += hist[b];
        }
        pick_hi = b;
        above = cum;
    }
    __syncthreads();
    const unsigned int hi = pick_hi;
    for (int i = t; i < 256; i += nt) hist[i] = 0;
    __syncthreads();
    for (int i = t; i < V; i += nt) {
        unsigned short uk = row[i];
        if (uk == 0x8000u) uk = 0u;
        const unsigned int ok = (uk & 0x8000u) ? (~uk & 0xFFFFu) : (uk | 0x8000u);
        if ((ok >> 8) == hi) atomicAdd(&hist[ok & 255u], 1u);
    }
    __syncthreads();
    if (t == 0) {
        unsigned int cum = above;
        int b = 255;
        for (; b > 0; --b) {
            if (cum + hist[b] >= static_cast<unsigned int>(K)) break;
            cum += hist[b];
        }
        pick_lo = b;
        above = cum;
    }
    __syncthreads();
    const unsigned int kth = (hi << 8) | pick_lo, high = above, need = K - high;
    const int per = (V + nt - 1) / nt, lo = t * per, end = min(V, lo + per);
    unsigned int gt = 0, eq = 0;
    for (int i = lo; i < end; ++i) {
        unsigned short uk = row[i];
        if (uk == 0x8000u) uk = 0u;
        const unsigned int ok = (uk & 0x8000u) ? (~uk & 0xFFFFu) : (uk | 0x8000u);
        gt += ok > kth;
        eq += ok == kth;
    }
    scan[t] = gt;
    scan_eq[t] = eq;
    __syncthreads();
    if (t == 0) {
        unsigned int s = 0, q = 0;
        for (int i = 0; i < nt; ++i) {
            const unsigned int x = scan[i], y = scan_eq[i];
            scan[i] = s;
            scan_eq[i] = q;
            s += x;
            q += y;
        }
    }
    __syncthreads();
    unsigned int at = scan[t], rank = scan_eq[t];
    for (int i = lo; i < end; ++i) {
        unsigned short uk = row[i];
        if (uk == 0x8000u) uk = 0u;
        const unsigned int ok = (uk & 0x8000u) ? (~uk & 0xFFFFu) : (uk | 0x8000u);
        if (ok > kth) {
            oc[at++] = i;
        } else if (ok == kth && rank < need) {
            oc[high + rank] = i;
            ++rank;
        }
    }
    __syncthreads();
    if ((reinterpret_cast<uintptr_t>(row) & 15u) == 0u) {
        const uint4 fill4 = make_uint4(0xC680C680u, 0xC680C680u, 0xC680C680u, 0xC680C680u);
        uint4* row4 = reinterpret_cast<uint4*>(row);
        const int v8 = V / 8;
        for (int i = t; i < v8; i += nt) row4[i] = fill4;
        for (int i = v8 * 8 + t; i < V; i += nt) row[i] = 0xC680u;
    } else {
        for (int i = t; i < V; i += nt) row[i] = 0xC680u;
    }
}

extern "C" __global__ void __launch_bounds__(128) fn_head_rescore(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W_bf16,
    const int* __restrict__ cols,
    __nv_bfloat16* __restrict__ out,
    int V,
    int K) {
    const int r = blockIdx.y;
    if (cols[static_cast<long long>(r) * K] < 0) return;
    __shared__ float sx[320 * 9];
    const __nv_bfloat16* xr = X + static_cast<size_t>(r) * 2560;
    for (int idx = threadIdx.x; idx < 2560; idx += 128) {
        sx[(idx >> 3) * 9 + (idx & 7)] = __bfloat162float(xr[idx]);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * 4 + warp;
    if (j >= K) return;
    const int c = cols[static_cast<long long>(r) * K + j];
    if (c < 0 || c >= V) return;
    const uint4* wp4 = reinterpret_cast<const uint4*>(W_bf16 + static_cast<size_t>(c) * 2560);
    float sum = 0.0f;
    #pragma unroll
    for (int i = lane; i < 320; i += 32) {
        const uint4 wv = __ldg(wp4 + i);
        const float* xp = sx + i * 9;
        const uint32_t u0 = wv.x, u1 = wv.y, u2 = wv.z, u3 = wv.w;
        sum = fmaf(xp[0], __uint_as_float((u0 & 0xFFFFu) << 16), sum);
        sum = fmaf(xp[1], __uint_as_float(u0 & 0xFFFF0000u), sum);
        sum = fmaf(xp[2], __uint_as_float((u1 & 0xFFFFu) << 16), sum);
        sum = fmaf(xp[3], __uint_as_float(u1 & 0xFFFF0000u), sum);
        sum = fmaf(xp[4], __uint_as_float((u2 & 0xFFFFu) << 16), sum);
        sum = fmaf(xp[5], __uint_as_float(u2 & 0xFFFF0000u), sum);
        sum = fmaf(xp[6], __uint_as_float((u3 & 0xFFFFu) << 16), sum);
        sum = fmaf(xp[7], __uint_as_float(u3 & 0xFFFF0000u), sum);
    }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) out[static_cast<size_t>(r) * V + c] = __float2bfloat16_rn(sum);
}
CU

  # 1c. Append strided FP8G shared_down kernels to qmmf.cu so shared_expert.down_proj skips sharedScatter
  cat >> "$SRC_DIR/zig/kernels/cuda/qmmf.cu" <<'CU'

namespace tf_qmmf {
#define TF_SHARED_DOWN_FP8G(NAME, BM, BN) \
extern "C" __global__ void __launch_bounds__(128) NAME( \
        const __nv_bfloat16* x, const unsigned char* w, const uint8_t* bs, float scale, float* out, \
        int M, int N, int K, int npad, int ldx, int group, int ldy) { \
    qmmf_body<FP8G, BM, BN, 1, 4, 4, true, false, false>( \
        x, w, bs, scale, out, nullptr, M, N, K, 1, npad, ldx, group, ldy); \
}
TF_SHARED_DOWN_FP8G(tf_qmmf_shared_fp8g16_half, 16, 32)
TF_SHARED_DOWN_FP8G(tf_qmmf_shared_fp8g16, 16, 64)
TF_SHARED_DOWN_FP8G(tf_qmmf_shared_fp8g32, 32, 64)
TF_SHARED_DOWN_FP8G(tf_qmmf_shared_fp8g64, 64, 64)
#undef TF_SHARED_DOWN_FP8G
}
CU

  # 2a. Least-squares Group-64 exact-FP32-scale FP8G projection encoder for MTP & non-recurrent BF16 trunk
  cat > "$fn_dir/cuda_mtp_mxfp8.zig" <<'ZIG'
//! Encodes MTP drafter and non-recurrent BF16 projections to group-64 exact-FP32-scale FP8G; recurrent GDN stays BF16.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const projection = @import("cuda_projection.zig");

const group: usize = 64;

fn nonrecEnabled() bool {
    const raw = std.c.getenv("TF_FLASHNEXT_NONREC_FP8G") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "1");
}

pub fn useForLoad(prefix: []const u8) bool {
    if (std.mem.startsWith(u8, prefix, "language_model.mtp.")) return true;
    return nonrecEnabled() and std.mem.startsWith(u8, prefix, "language_model.model.");
}

pub fn useForStitched(prefix: []const u8) bool {
    if (std.mem.startsWith(u8, prefix, "language_model.mtp.")) return true;
    if (!nonrecEnabled() or !std.mem.startsWith(u8, prefix, "language_model.model.")) return false;
    return !std.mem.endsWith(u8, prefix, ".linear_attn");
}

pub fn useForHc(prefix: []const u8) bool {
    if (std.mem.startsWith(u8, prefix, "language_model.mtp.")) return true;
    return nonrecEnabled() and std.mem.startsWith(u8, prefix, "language_model.model.");
}

const EncodeCtx = struct {
    bf16: []const u8,
    lut: *const [256]f32,
    raw_codes: []u8,
    packed_scales: []f32,
    n: usize,
    k: usize,
    kg: usize,
    per: usize,

    fn worker(ctx: *const EncodeCtx, tid: usize) void {
        const lo = @min(ctx.n, tid * ctx.per);
        const hi = @min(ctx.n, lo + ctx.per);
        for (lo..hi) |r| {
            const t = r / 64;
            const tr = r % 64;
            const rb = ctx.bf16[r * ctx.k * 2 ..][0 .. ctx.k * 2];
            for (0..ctx.kg) |g| {
                var w: [group]f32 = undefined;
                var max_abs: f32 = 0;
                for (0..group) |j| {
                    const bits = std.mem.readInt(u16, rb[(g * group + j) * 2 ..][0..2], .little);
                    const f: f32 = @bitCast(@as(u32, bits) << 16);
                    w[j] = f;
                    const a = @abs(f);
                    if (a > max_abs) max_abs = a;
                }
                const dst_c = ctx.raw_codes[r * ctx.k + g * group ..][0..group];
                if (!(max_abs > 0) or !std.math.isFinite(max_abs)) {
                    @memset(dst_c, 0);
                    ctx.packed_scales[(t * ctx.kg + g) * 64 + tr] = 1.0;
                    continue;
                }
                var sc: f32 = max_abs / 448.0;
                {
                    var num1: f32 = 0;
                    var den1: f32 = 0;
                    const inv1: f32 = 1.0 / sc;
                    for (0..group) |j| {
                        const c = core.exlr8.codec.e4m3FromF32(w[j] * inv1);
                        const q = ctx.lut[c];
                        num1 += w[j] * q;
                        den1 += q * q;
                    }
                    if (den1 > 0) sc = num1 / den1;
                }
                var num2: f32 = 0;
                var den2: f32 = 0;
                const inv2: f32 = if (sc > 0) 1.0 / sc else 0.0;
                for (0..group) |j| {
                    const c = core.exlr8.codec.e4m3FromF32(w[j] * inv2);
                    dst_c[j] = c;
                    const q = ctx.lut[c];
                    num2 += w[j] * q;
                    den2 += q * q;
                }
                ctx.packed_scales[(t * ctx.kg + g) * 64 + tr] = if (den2 > 0) num2 / den2 else sc;
            }
        }
    }
};

const PackCodesCtx = struct {
    out: []u8,
    codes: []const u8,
    n: usize,
    k: usize,
    tiles: usize,
    per: usize,

    fn kin(lane: usize, byte: usize) usize {
        const p = 16 * (byte / 4) + 4 * (lane % 4) + byte % 4;
        return 16 * (p / 16) + 2 * ((p % 16) / 4) + p % 2 + 8 * ((p % 4) / 2);
    }

    fn worker(ctx: *const PackCodesCtx, tid: usize) void {
        const lo = @min(ctx.tiles, tid * ctx.per);
        const hi = @min(ctx.tiles, lo + ctx.per);
        const kg = ctx.k / group;
        for (lo..hi) |t| {
            var i: usize = t * kg * 4096;
            for (0..kg) |g| for (0..8) |a| for (0..32) |lane| for (0..2) |h| for (0..8) |b| {
                const row = t * 64 + a * 8 + lane / 4;
                const col = g * group + h * 32 + kin(lane, b);
                ctx.out[i] = if (row < ctx.n) ctx.codes[row * ctx.k + col] else 0;
                i += 1;
            };
        }
    }
};

pub fn project(gpa: std.mem.Allocator, driver: *const cuda.Driver, bf16: []const u8, n: usize, k: usize) !projection.Projection {
    if (n == 0 or k == 0 or k % group != 0 or bf16.len != n * k * 2) return error.UnexpectedTensor;
    const npad = cuda.fp8.padded(n);
    const kg = k / group;
    const raw_codes = try gpa.alloc(u8, n * k);
    defer gpa.free(raw_codes);
    const packed_codes = try gpa.alloc(u8, npad * k);
    defer gpa.free(packed_codes);
    const packed_scales = try gpa.alloc(f32, npad * kg);
    defer gpa.free(packed_scales);
    @memset(packed_scales, 1.0);

    var lut: [256]f32 = undefined;
    for (0..256) |b| lut[b] = if ((b & 0x7f) == 0x7f) 0.0 else core.exlr8.codec.e4m3Value(@intCast(b));

    const n_th: usize = if (n * kg >= 2048) 8 else 1;
    const ectx: EncodeCtx = .{
        .bf16 = bf16,
        .lut = &lut,
        .raw_codes = raw_codes,
        .packed_scales = packed_scales,
        .n = n,
        .k = k,
        .kg = kg,
        .per = (n + n_th - 1) / n_th,
    };
    if (n_th == 1) {
        EncodeCtx.worker(&ectx, 0);
    } else {
        var ths: [8]?std.Thread = @splat(null);
        for (0..n_th) |ti| ths[ti] = std.Thread.spawn(.{}, EncodeCtx.worker, .{ &ectx, ti }) catch null;
        for (0..n_th) |ti| {
            if (ths[ti]) |t| t.join() else EncodeCtx.worker(&ectx, ti);
        }
    }

    const tiles = npad / 64;
    const p_th: usize = @min(8, @max(1, tiles));
    const pctx: PackCodesCtx = .{
        .out = packed_codes,
        .codes = raw_codes,
        .n = n,
        .k = k,
        .tiles = tiles,
        .per = (tiles + p_th - 1) / p_th,
    };
    if (p_th == 1) {
        PackCodesCtx.worker(&pctx, 0);
    } else {
        var ths2: [8]?std.Thread = @splat(null);
        for (0..p_th) |ti| ths2[ti] = std.Thread.spawn(.{}, PackCodesCtx.worker, .{ &pctx, ti }) catch null;
        for (0..p_th) |ti| {
            if (ths2[ti]) |t| t.join() else PackCodesCtx.worker(&pctx, ti);
        }
    }

    var p = projection.Projection.empty(driver);
    p.w = try cuda.DeviceBuffer.fromHost(driver, packed_codes);
    errdefer p.w.free();
    p.s = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(packed_scales));
    p.n = n;
    p.k = k;
    p.native = cuda.fp8.weight(p.w.ptr, p.s.ptr, n, k);
    return p;
}
ZIG

  # 2c. Thread-safe 32K-entry in-RAM PLE 3-gram row cache (pre-faulted at init, 64 striped spinlocks)
  cat > "$fn_dir/cuda_ple_cache.zig" <<'ZIG'
//! Exact in-RAM 4-way set-associative cache for immutable Layer-2 PLE 3-gram embeddings and group-32 sums.
const std = @import("std");

pub const row_w: usize = 2560;
pub const xs_w: usize = row_w / 32;
const n_sets: usize = 8192;
const ways: usize = 4;
const n_locks: usize = 64;

const Lock = struct {
    state: std.atomic.Value(u32) align(64) = .init(0),

    fn acquire(self: *Lock) void {
        while (self.state.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }

    fn release(self: *Lock) void {
        self.state.store(0, .release);
    }
};

const Entry = struct {
    h0: i64,
    h1: i64,
    tok: u32,
    valid: u32,
    emb: [row_w]u16,
    xs: [xs_w]f32,
};

const Set = struct {
    hand: u32,
    items: [ways]Entry,
};

pub const Cache = struct {
    sets: []Set,
    locks: [n_locks]Lock = @splat(.{}),

    pub fn create(gpa: std.mem.Allocator) !*Cache {
        const sets = try gpa.alloc(Set, n_sets);
        errdefer gpa.free(sets);
        @memset(std.mem.sliceAsBytes(sets), 0);
        const c = try gpa.create(Cache);
        c.* = .{ .sets = sets };
        return c;
    }

    pub fn destroy(self: *Cache, gpa: std.mem.Allocator) void {
        gpa.free(self.sets);
        gpa.destroy(self);
    }

    fn setIndex(h0: i64, h1: i64, tok: u32) usize {
        var x: u64 = @bitCast(h0);
        x = (x ^ (@as(u64, @bitCast(h1)) *% 0x9E3779B97F4A7C15)) *% 0xBF58476D1CE4E5B9;
        x = (x ^ (@as(u64, tok) *% 0x94D049BB133111EB)) *% 0x9E3779B97F4A7C15;
        return @intCast((x >> 51) & (n_sets - 1));
    }

    pub fn has(self: *Cache, h0: i64, h1: i64, tok: u32) bool {
        const idx = setIndex(h0, h1, tok);
        const lk = &self.locks[idx & (n_locks - 1)];
        lk.acquire();
        defer lk.release();
        const set = &self.sets[idx];
        for (&set.items) |*e| {
            if (e.valid != 0 and e.tok == tok and e.h0 == h0 and e.h1 == h1) return true;
        }
        return false;
    }

    pub fn get(self: *Cache, h0: i64, h1: i64, tok: u32, emb_out: *[row_w]u16, xs_out: *[xs_w]f32) bool {
        const idx = setIndex(h0, h1, tok);
        const lk = &self.locks[idx & (n_locks - 1)];
        lk.acquire();
        defer lk.release();
        const set = &self.sets[idx];
        for (&set.items) |*e| {
            if (e.valid != 0 and e.tok == tok and e.h0 == h0 and e.h1 == h1) {
                @memcpy(emb_out, &e.emb);
                @memcpy(xs_out, &e.xs);
                return true;
            }
        }
        return false;
    }

    pub fn put(self: *Cache, h0: i64, h1: i64, tok: u32, emb_in: *const [row_w]u16, xs_in: *const [xs_w]f32) void {
        const idx = setIndex(h0, h1, tok);
        const lk = &self.locks[idx & (n_locks - 1)];
        lk.acquire();
        defer lk.release();
        const set = &self.sets[idx];
        for (&set.items) |*e| {
            if (e.valid != 0 and e.tok == tok and e.h0 == h0 and e.h1 == h1) return;
        }
        const way: usize = @intCast(set.hand & (ways - 1));
        set.hand +%= 1;
        const dst = &set.items[way];
        dst.h0 = h0;
        dst.h1 = h1;
        dst.tok = tok;
        dst.valid = 1;
        @memcpy(&dst.emb, emb_in);
        @memcpy(&dst.xs, xs_in);
    }
};
ZIG

  # 2b. Two-stage main lm_head loader: GPU NVFP4 coarse packed weight + shared resident BF16 rows in p.b
  cat > "$fn_dir/cuda_head_nvfp4.zig" <<'ZIG'
//! Load-time NVFP4 coarse projection + resident BF16 rows for exact top-128 main lm_head rescoring.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const projection = @import("cuda_projection.zig");

const MaxCtx = struct {
    bytes: []const u8,
    total: usize,
    per: usize,
    out: []u16,

    fn worker(ctx: *const MaxCtx, tid: usize) void {
        const lo = @min(ctx.total, tid * ctx.per);
        const hi = @min(ctx.total, lo + ctx.per);
        var m: u16 = 0;
        for (lo..hi) |i| {
            const u = std.mem.readInt(u16, ctx.bytes[i * 2 ..][0..2], .little) & 0x7fff;
            if (u <= 0x7f80 and u > m) {
                m = u;
            }
        }
        ctx.out[tid] = m;
    }
};

pub fn enabled() bool {
    const raw = std.c.getenv("TF_FLASHNEXT_HEAD_NVFP4") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "1");
}

pub fn load(driver: *cuda.Driver, stream: cuda.Stream, store: anytype, t: core.safetensors.Tensor) !projection.Projection {
    if (t.rank != 2 or t.dtype != .bf16 or t.dim(0) == 0 or t.dim(1) == 0 or t.dim(1) % 64 != 0)
        return error.UnexpectedTensor;
    const n = t.dim(0);
    const k = t.dim(1);
    const count = try std.math.mul(usize, n, k);
    if (t.bytes.len != count * 2) return error.UnexpectedTensor;

    const n_th: usize = 8;
    var max_u16: [n_th]u16 = @splat(0);
    const ctx: MaxCtx = .{ .bytes = t.bytes, .total = count, .per = (count + n_th - 1) / n_th, .out = &max_u16 };
    var ths: [n_th]?std.Thread = @splat(null);
    for (0..n_th) |ti| ths[ti] = try std.Thread.spawn(.{}, MaxCtx.worker, .{ &ctx, ti });
    for (&ths) |*th| if (th.*) |tr| tr.join();
    var top_bits: u16 = 0;
    for (max_u16) |u| {
        if (u > top_bits) {
            top_bits = u;
        }
    }
    const max_abs: f32 = @bitCast(@as(u32, top_bits) << 16);
    const global_scale: f32 = if (max_abs > 0) max_abs / (6.0 * 448.0) else 1.0;

    var bf16_buf = try cuda.DeviceBuffer.fromHost(driver, t.bytes);
    errdefer bf16_buf.free();
    store.release();

    const npad = cuda.nvfp4.padded(n);
    const words_bytes = (npad / 64) * (k / 64) * 512 * 4;
    const tiles_bytes = npad * (k / 16);
    var w_buf = try cuda.DeviceBuffer.alloc(driver, words_bytes);
    errdefer w_buf.free();
    var s_buf = try cuda.DeviceBuffer.alloc(driver, tiles_bytes);
    errdefer s_buf.free();

    const groups = count / 16;
    var raw_codes = try cuda.DeviceBuffer.alloc(driver, count / 2);
    defer raw_codes.free();
    var raw_scales = try cuda.DeviceBuffer.alloc(driver, groups);
    defer raw_scales.free();
    errdefer cuda.completion.drain(driver, stream, "head nvfp4 pack did not drain");

    var batch_mod = try cuda.Module.load(driver, cuda.kernels.flashnext_batch);
    defer batch_mod.unload();
    const quant_fn = try batch_mod.function("fn_bf16_to_nvfp4");
    var qa: cuda.Args = .{};
    qa.add(bf16_buf.ptr);
    qa.add(global_scale);
    qa.add(raw_codes.ptr);
    qa.add(raw_scales.ptr);
    qa.add(@as(c_longlong, @intCast(groups)));
    try cuda.launch.launch(quant_fn, .{ .grid = .{ .x = @intCast((groups + 255) / 256) }, .block = .{ .x = 256 } }, stream, &qa);

    var packer = try cuda.modelopt.Packer.load(driver);
    defer packer.unload();
    try packer.fp4Lane(stream, raw_codes.ptr, raw_scales.ptr, w_buf.ptr, s_buf.ptr, n, k);
    try stream.synchronize();

    return .{
        .w = w_buf,
        .s = s_buf,
        .b = bf16_buf,
        .n = n,
        .k = k,
        .native = cuda.nvfp4.weight(w_buf.ptr, s_buf.ptr, global_scale, n, k),
    };
}
ZIG

  # 3. 2.5-bit Lloyd-Max group-32 MTP proposal head sharing Engine's resident BF16 lm_head buffer
  cat > "$fn_dir/cuda_mtp_w2.zig" <<'ZIG'
//! 2.5-bit symmetric group-32 proposal head + GPU top-64 exact BF16 rescore sharing Engine's resident BF16 lm_head.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");
const affine = @import("cuda_draft_affine.zig");

const kg = wt.dims / 32;

const PackCtx = struct {
    full_rows: []const u8,
    ids: []const u32,
    codes: []u64,
    scales: []u16,
    v: usize,
    per: usize,

    fn worker(ctx: *const PackCtx, tid: usize) void {
        const lo = @min(ctx.v, tid * ctx.per);
        const hi = @min(ctx.v, lo + ctx.per);
        for (lo..hi) |r| {
            const blk = r / 32;
            const lane = r % 32;
            const gid: usize = ctx.ids[r];
            const rb = ctx.full_rows[gid * wt.dims * 2 ..][0 .. wt.dims * 2];
            for (0..kg) |g| {
                var w: [32]f32 = undefined;
                var sq: f32 = 0;
                for (0..32) |j| {
                    const bits = std.mem.readInt(u16, rb[(g * 32 + j) * 2 ..][0..2], .little);
                    const f: f32 = @bitCast(@as(u32, bits) << 16);
                    w[j] = f;
                    sq += f * f;
                }
                var s: f32 = @sqrt(sq / 32.0) * 0.9957;
                if (s < 1e-12) {
                    ctx.codes[(blk * kg + g) * 32 + lane] = 0;
                    ctx.scales[(blk * kg + g) * 32 + lane] = 0;
                    continue;
                }
                var q: [32]u2 = undefined;
                for (0..3) |_| {
                    var num: f32 = 0;
                    var den: f32 = 0;
                    const inv = 1.0 / @max(s, 1e-12);
                    for (0..32) |j| {
                        const raw = @round(w[j] * inv + 1.5);
                        const qi: u2 = if (raw <= 0.0) 0 else if (raw >= 3.0) 3 else @intFromFloat(raw);
                        q[j] = qi;
                        const qc: f32 = @as(f32, @floatFromInt(qi)) - 1.5;
                        num += w[j] * qc;
                        den += qc * qc;
                    }
                    s = if (den > 0) @max(num / den, 0.0) else 0.0;
                }
                var word: u64 = 0;
                for (0..32) |j| word |= @as(u64, q[j]) << @intCast(2 * j);
                ctx.codes[(blk * kg + g) * 32 + lane] = word;
                ctx.scales[(blk * kg + g) * 32 + lane] = affine.bf16(s);
            }
        }
    }
};

pub const W2 = struct {
    codes: ?cuda.DeviceBuffer = null,
    scales: ?cuda.DeviceBuffer = null,
    vocab_ids: ?cuda.DeviceBuffer = null,
    bf16_ptr: u64 = 0,
    scratch: ?cuda.DeviceBuffer = null,
    gemv_fn: ?cuda.Function = null,
    rescore_fn: ?cuda.Function = null,

    pub fn ready(self: *const W2) bool {
        return self.codes != null and self.bf16_ptr != 0;
    }

    pub fn deinit(self: *W2) void {
        if (self.scratch) |*b| b.free();
        if (self.vocab_ids) |*b| b.free();
        if (self.scales) |*b| b.free();
        if (self.codes) |*b| b.free();
        self.* = .{};
    }

    pub fn init(gpa: std.mem.Allocator, driver: *cuda.Driver, store: *wt.Store, artifact: *const core.proposal_vocab.Artifact, module: *const cuda.Module) !W2 {
        const raw = std.c.getenv("TF_FLASHNEXT_MTP_W2") orelse return .{};
        if (!std.mem.eql(u8, std.mem.span(raw), "1")) return .{};
        const tensor = try store.tensor("language_model.lm_head.weight");
        if (tensor.dtype != .bf16 or !tensor.is(.bf16, &.{ wt.vocab, wt.dims })) return .{};
        const ids = artifact.ids();
        const v = ids.len;
        const nb = (v + 31) / 32;
        const total = nb * kg * 32;
        const codes_host = try gpa.alloc(u64, total);
        defer gpa.free(codes_host);
        @memset(codes_host, 0);
        const scales_host = try gpa.alloc(u16, total);
        defer gpa.free(scales_host);
        @memset(scales_host, 0);

        const n_threads: usize = 8;
        const ctx: PackCtx = .{ .full_rows = tensor.bytes, .ids = ids, .codes = codes_host, .scales = scales_host, .v = v, .per = (v + n_threads - 1) / n_threads };
        var ths: [n_threads]?std.Thread = @splat(null);
        for (0..n_threads) |ti| ths[ti] = try std.Thread.spawn(.{}, PackCtx.worker, .{ &ctx, ti });
        for (&ths) |*th| if (th.*) |t| t.join();
        store.release();

        var ids_buf = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(ids));
        errdefer ids_buf.free();
        var codes_buf = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(codes_host));
        errdefer codes_buf.free();
        var scales_buf = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(scales_host));
        errdefer scales_buf.free();
        var scratch_buf = try cuda.DeviceBuffer.alloc(driver, 4096);
        errdefer scratch_buf.free();

        return .{
            .codes = codes_buf,
            .scales = scales_buf,
            .vocab_ids = ids_buf,
            .scratch = scratch_buf,
            .gemv_fn = try module.function("fn_w2_gemv"),
            .rescore_fn = try module.function("fn_w2_rescore"),
        };
    }

    pub fn project(self: *const W2, h: *const mtp.Head, stream: cuda.Stream, input: u64, out: u64, v: usize) !void {
        const nb = (v + 31) / 32;
        var a1: cuda.Args = .{};
        a1.add(input);
        a1.add(self.codes.?.ptr);
        a1.add(self.scales.?.ptr);
        a1.add(out);
        a1.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(self.gemv_fn.?, .{ .grid = .{ .x = @intCast((nb + 3) / 4) }, .block = .{ .x = 128 } }, stream, &a1);
        const sc = self.scratch.?.ptr;
        try h.topk(stream, out, v, v, 1, 64, sc, sc + 256);
        var a2: cuda.Args = .{};
        a2.add(input);
        a2.add(self.bf16_ptr);
        a2.add(self.vocab_ids.?.ptr);
        a2.add(sc + 256);
        a2.add(out);
        a2.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(self.rescore_fn.?, .{ .grid = .{ .x = 64 }, .block = .{ .x = 256 } }, stream, &a2);
    }
};
ZIG

  # 4. MTP routed experts: re-quantize block-128 FP8 -> native SM121 NVFP4 at load
  cat > "$fn_dir/cuda_fp8_experts_load.zig" <<'ZIG'
//! The MTP head's block-FP8 routed experts, with opt-in load-time conversion to SM121 NVFP4.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const experts = @import("cuda_modelopt_experts.zig");
const fx = cuda.experts.Fp8Experts;
const ex = cuda.experts.Nvfp4Experts;

const blk = fx.block;

pub const Expert = struct { up: []u32, down: []u32, up_scale: []f32, down_scale: []f32 };

pub const Shape = struct {
    up: usize,
    down: usize,
    up_scale: usize,
    down_scale: usize,

    pub fn of(dims: usize, width: usize) !Shape {
        if (dims == 0 or width == 0 or dims % blk != 0 or width % blk != 0) return error.UnexpectedTensor;
        const weights = std.math.mul(usize, dims, width) catch return error.UnexpectedTensor;
        const blocks = (dims / blk) * (width / blk);
        return .{ .up = weights / 2, .down = weights / 4, .up_scale = 2 * blocks, .down_scale = blocks };
    }
};

fn useNvfp4() bool {
    const raw = std.c.getenv("TF_FLASHNEXT_MTP_EXPERTS") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "nvfp4");
}

fn e2m1(v: f32) u8 {
    const sign: u8 = if (std.math.signbit(v)) 8 else 0;
    const a = @abs(v);
    const mag: u8 = if (a < 0.25) 0 else if (a <= 0.75) 1 else if (a < 1.25) 2 else if (a <= 1.75) 3 else if (a < 2.5) 4 else if (a <= 3.5) 5 else if (a < 5.0) 6 else 7;
    return sign | mag;
}

fn quantizeFp8ToNvfp4(lut: *const [256]f32, fp8_bytes: []const u8, blk_scales: []const f32, n: usize, k: usize, codes: []u8, scales: []u8) f32 {
    const kb = k / blk;
    var max_abs: f32 = 0;
    for (0..n) |r| {
        const br = (r / blk) * kb;
        const row = fp8_bytes[r * k ..][0..k];
        for (0..kb) |cb| {
            var m7: u8 = 0;
            for (row[cb * blk ..][0..blk]) |b| {
                const u = b & 0x7f;
                if (u != 0x7f and u > m7) m7 = u;
            }
            const cand = lut[m7] * @abs(blk_scales[br + cb]);
            if (cand > max_abs) max_abs = cand;
        }
    }
    const global: f32 = if (max_abs > 0) max_abs / (6.0 * 448.0) else 1.0;
    const inv_global: f32 = 1.0 / global;
    const kg16 = k / 16;
    for (0..n) |r| {
        const br = (r / blk) * kb;
        const row = fp8_bytes[r * k ..][0..k];
        for (0..kg16) |g| {
            const bs = blk_scales[br + g / 8];
            const abs_bs = @abs(bs);
            const seg = row[g * 16 ..][0..16];
            var m7: u8 = 0;
            for (seg) |b| {
                const u = b & 0x7f;
                if (u != 0x7f and u > m7) m7 = u;
            }
            const sc_code = core.exlr8.codec.e4m3FromF32(std.math.clamp(lut[m7] * abs_bs * inv_global / 6.0, 0.0, 448.0));
            scales[r * kg16 + g] = sc_code;
            const denom = lut[sc_code] * global;
            const out_c = codes[r * (k / 2) + g * 8 ..][0..8];
            if (denom == 0 or bs == 0) {
                @memset(out_c, 0);
            } else {
                const factor = bs / denom;
                for (0..8) |p| {
                    const v0 = lut[seg[2 * p]] * factor;
                    const v1 = lut[seg[2 * p + 1]] * factor;
                    out_c[p] = e2m1(v0) | (e2m1(v1) << 4);
                }
            }
        }
    }
    return global;
}

const BatchJob = struct {
    lut: *const [256]f32,
    raw_gate: []const u8,
    raw_up: []const u8,
    raw_down: []const u8,
    sc_gate: []const f32,
    sc_up: []const f32,
    sc_down: []const f32,
    packed_up: []u32,
    packed_down: []u32,
    up_globals: []f32,
    down_globals: []f32,
    dims: usize,
    width: usize,
    n_items: usize,
    up_words: usize,
    down_words: usize,
    blocks: usize,

    fn run(job: *const BatchJob, tid: usize, stride: usize) void {
        const mat = job.dims * job.width;
        var g_codes: [640 * 1280]u8 = undefined;
        var u_codes: [640 * 1280]u8 = undefined;
        var g_scales: [640 * 160]u8 = undefined;
        var u_scales: [640 * 160]u8 = undefined;
        var i = tid;
        while (i < job.n_items) : (i += stride) {
            const gg = quantizeFp8ToNvfp4(job.lut, job.raw_gate[i * mat ..][0..mat], job.sc_gate[i * job.blocks ..][0..job.blocks], job.width, job.dims, g_codes[0 .. mat / 2], g_scales[0 .. mat / 16]);
            const ug = quantizeFp8ToNvfp4(job.lut, job.raw_up[i * mat ..][0..mat], job.sc_up[i * job.blocks ..][0..job.blocks], job.width, job.dims, u_codes[0 .. mat / 2], u_scales[0 .. mat / 16]);
            ex.packGateUp(job.packed_up[i * job.up_words ..][0..job.up_words], .{ g_codes[0 .. mat / 2], g_scales[0 .. mat / 16] }, .{ u_codes[0 .. mat / 2], u_scales[0 .. mat / 16] }, job.width, job.dims);
            job.up_globals[i * 2] = gg;
            job.up_globals[i * 2 + 1] = ug;

            const dg = quantizeFp8ToNvfp4(job.lut, job.raw_down[i * mat ..][0..mat], job.sc_down[i * job.blocks ..][0..job.blocks], job.dims, job.width, g_codes[0 .. mat / 2], g_scales[0 .. mat / 16]);
            ex.packOne(job.packed_down[i * job.down_words ..][0..job.down_words], g_codes[0 .. mat / 2], g_scales[0 .. mat / 16], job.dims, job.width);
            job.down_globals[i] = dg;
        }
    }
};

fn loadNvfp4(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, world: usize, dims: usize, width: usize, count: usize) !experts.Owned {
    if (world != 1) return error.UnevenExpertSplit;
    const one = try Shape.of(dims, width);
    if (count == 0 or dims != 2560 or width != 640) return error.UnexpectedTensor;
    const up_bytes = try experts.packedBytes(width, dims, 2);
    const down_bytes = try experts.packedBytes(dims, width, 1);
    var up = try cuda.DeviceBuffer.alloc(d, count * up_bytes);
    errdefer up.free();
    var down = try cuda.DeviceBuffer.alloc(d, count * down_bytes);
    errdefer down.free();
    const up_globals = try a.alloc(f32, count * 2);
    defer a.free(up_globals);
    const down_globals = try a.alloc(f32, count);
    defer a.free(down_globals);

    var lut: [256]f32 = undefined;
    for (0..256) |b| lut[b] = if ((b & 0x7f) == 0x7f) 0.0 else core.exlr8.codec.e4m3Value(@intCast(b));

    const batch_max: usize = 16;
    const mat = dims * width;
    const blocks = one.down_scale;
    const raw_gate = try a.alloc(u8, batch_max * mat);
    defer a.free(raw_gate);
    const raw_up = try a.alloc(u8, batch_max * mat);
    defer a.free(raw_up);
    const raw_down = try a.alloc(u8, batch_max * mat);
    defer a.free(raw_down);
    const sc_gate = try a.alloc(f32, batch_max * blocks);
    defer a.free(sc_gate);
    const sc_up = try a.alloc(f32, batch_max * blocks);
    defer a.free(sc_up);
    const sc_down = try a.alloc(f32, batch_max * blocks);
    defer a.free(sc_down);
    const up_words = up_bytes / 4;
    const down_words = down_bytes / 4;
    const packed_up = try a.alloc(u32, batch_max * up_words);
    defer a.free(packed_up);
    const packed_down = try a.alloc(u32, batch_max * down_words);
    defer a.free(packed_down);

    var base: usize = 0;
    while (base < count) : (base += batch_max) {
        const bn = @min(batch_max, count - base);
        for (0..bn) |bi| {
            const g_slice = try part(store, prefix, base + bi, "gate_proj", width, dims, sc_gate[bi * blocks ..][0..blocks]);
            @memcpy(raw_gate[bi * mat ..][0..mat], g_slice);
            const u_slice = try part(store, prefix, base + bi, "up_proj", width, dims, sc_up[bi * blocks ..][0..blocks]);
            @memcpy(raw_up[bi * mat ..][0..mat], u_slice);
            const d_slice = try part(store, prefix, base + bi, "down_proj", dims, width, sc_down[bi * blocks ..][0..blocks]);
            @memcpy(raw_down[bi * mat ..][0..mat], d_slice);
            store.release();
        }
        const n_th: usize = @min(8, bn);
        const job: BatchJob = .{
            .lut = &lut,
            .raw_gate = raw_gate,
            .raw_up = raw_up,
            .raw_down = raw_down,
            .sc_gate = sc_gate,
            .sc_up = sc_up,
            .sc_down = sc_down,
            .packed_up = packed_up,
            .packed_down = packed_down,
            .up_globals = up_globals[base * 2 ..][0 .. bn * 2],
            .down_globals = down_globals[base..][0..bn],
            .dims = dims,
            .width = width,
            .n_items = bn,
            .up_words = up_words,
            .down_words = down_words,
            .blocks = blocks,
        };
        var ths: [8]?std.Thread = @splat(null);
        for (0..n_th) |ti| ths[ti] = try std.Thread.spawn(.{}, BatchJob.run, .{ &job, ti, n_th });
        for (&ths) |*th| if (th.*) |t| t.join();

        up.upload(base * up_bytes, std.mem.sliceAsBytes(packed_up[0 .. bn * up_words])) catch |e| return drained(d, e);
        down.upload(base * down_bytes, std.mem.sliceAsBytes(packed_down[0 .. bn * down_words])) catch |e| return drained(d, e);
    }

    var us = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(up_globals));
    errdefer us.free();
    const ds = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(down_globals));
    return .{ .up = up, .down = down, .up_scale = us, .down_scale = ds, .width = width, .dims = dims, .count = count, .format = .nvfp4 };
}

pub fn load(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, world: usize, dims: usize, width: usize, count: usize) !experts.Owned {
    defer store.release();
    if (useNvfp4() and dims == 2560 and width == 640) return loadNvfp4(a, d, store, prefix, world, dims, width, count);
    if (world != 1) return error.UnevenExpertSplit;
    const one = try Shape.of(dims, width);
    if (count == 0) return error.UnexpectedTensor;
    const up_total = std.math.mul(usize, count, one.up * 4) catch return error.UnexpectedTensor;
    const down_total = std.math.mul(usize, count, one.down * 4) catch return error.UnexpectedTensor;
    var up = try cuda.DeviceBuffer.alloc(d, up_total);
    errdefer up.free();
    var down = try cuda.DeviceBuffer.alloc(d, down_total);
    errdefer down.free();
    const up_scale = try a.alloc(f32, count * one.up_scale);
    defer a.free(up_scale);
    const down_scale = try a.alloc(f32, count * one.down_scale);
    defer a.free(down_scale);
    const up_words = try a.alloc(u32, one.up);
    defer a.free(up_words);
    const down_words = try a.alloc(u32, one.down);
    defer a.free(down_words);
    for (0..count) |i| {
        const at: Expert = .{ .up = up_words, .down = down_words, .up_scale = up_scale[i * one.up_scale ..][0..one.up_scale], .down_scale = down_scale[i * one.down_scale ..][0..one.down_scale] };
        try packExpert(a, store, prefix, i, dims, width, at);
        up.upload(i * one.up * 4, std.mem.sliceAsBytes(up_words)) catch |e| return drained(d, e);
        down.upload(i * one.down * 4, std.mem.sliceAsBytes(down_words)) catch |e| return drained(d, e);
    }
    var us = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(up_scale));
    errdefer us.free();
    const ds = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(down_scale));
    return .{ .up = up, .down = down, .up_scale = us, .down_scale = ds, .width = width, .dims = dims, .count = count, .format = .fp8g };
}

fn drained(d: *const cuda.Driver, e: anyerror) anyerror {
    cuda.completion.drain(d, .{ .d = d, .handle = null }, "an FP8 expert upload's default stream and context did not drain");
    return e;
}

pub fn packExpert(a: std.mem.Allocator, store: anytype, prefix: []const u8, i: usize, dims: usize, width: usize, out: Expert) !void {
    defer store.release();
    const one = try Shape.of(dims, width);
    if (out.up.len != one.up or out.down.len != one.down or out.up_scale.len != one.up_scale or out.down_scale.len != one.down_scale) return error.UnexpectedTensor;
    const half = one.up_scale / 2;
    const gate = try part(store, prefix, i, "gate_proj", width, dims, out.up_scale[0..half]);
    const up = try part(store, prefix, i, "up_proj", width, dims, out.up_scale[half..]);
    try fx.packGateUp(a, out.up, gate, up, width, dims);
    fx.packOne(out.down, try part(store, prefix, i, "down_proj", dims, width, out.down_scale), dims, width);
}

fn part(store: anytype, prefix: []const u8, i: usize, name: []const u8, n: usize, k: usize, scale: []f32) ![]const u8 {
    var buf: [512]u8 = undefined;
    const h = try core.quant.fp8.read(store, try std.fmt.bufPrint(&buf, "{s}.experts.{d}.{s}", .{ prefix, i, name }), n, k, .bf16, null);
    try core.quant.fp8.widen(h.scale_inv, scale, true);
    return h.weight.bytes;
}
ZIG

  # 5. Wire Zig call sites cleanly
  python3 - "$fn_dir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])

def patch_exact(rel: str, old: str, new: str) -> None:
    p = root / rel
    txt = p.read_text(encoding="utf-8")
    if txt.count(old) != 1:
        raise RuntimeError(f"Expected exactly 1 match in {rel}, got {txt.count(old)}")
    p.write_text(txt.replace(old, new, 1), encoding="utf-8")

# 5a. qmmf.zig + qmmf_shared.zig + cuda_projection_load.zig + cuda_prompt_wt.zig: non-recurrent FP8G + strided FP8G shared_down
cuda_dir = root.parent.parent / "cuda"
def patch_cuda(rel: str, old: str, new: str) -> None:
    p = cuda_dir / rel
    txt = p.read_text(encoding="utf-8")
    if txt.count(old) != 1:
        raise RuntimeError(f"Expected exactly 1 match in cuda/{rel}, got {txt.count(old)}")
    p.write_text(txt.replace(old, new, 1), encoding="utf-8")

patch_cuda(
    "qmmf.zig",
    "    shared_down: [4]Function = undefined,",
    "    shared_down: [4]Function = undefined,\n    shared_down_fp8g: [4]Function = undefined,",
)
patch_cuda(
    "qmmf.zig",
    """        for (@import("qmmf_shared.zig").symbols, &l.shared_down) |name, *f| {
            f.* = try mod.function(name);
            try f.allowDynamicShared(sharedBytes(.bf16, 64));
        }""",
    """        for (@import("qmmf_shared.zig").symbols, &l.shared_down) |name, *f| {
            f.* = try mod.function(name);
            try f.allowDynamicShared(sharedBytes(.bf16, 64));
        }
        for (@import("qmmf_shared.zig").fp8g_symbols, &l.shared_down_fp8g) |name, *f| {
            f.* = try mod.function(name);
            try f.allowDynamicShared(sharedBytes(.fp8g, 64));
        }""",
)
patch_cuda(
    "qmmf_shared.zig",
    'pub const symbols = [4][:0]const u8{ "tf_qmmf_shared16_half", "tf_qmmf_shared16", "tf_qmmf_shared32", "tf_qmmf_shared64" };',
    'pub const symbols = [4][:0]const u8{ "tf_qmmf_shared16_half", "tf_qmmf_shared16", "tf_qmmf_shared32", "tf_qmmf_shared64" };\npub const fp8g_symbols = [4][:0]const u8{ "tf_qmmf_shared_fp8g16_half", "tf_qmmf_shared_fp8g16", "tf_qmmf_shared_fp8g32", "tf_qmmf_shared_fp8g64" };',
)
patch_cuda(
    "qmmf_shared.zig",
    "    return w.mode == .bf16 and q.planFor(m, w, l.clusters).sk == 1;",
    "    return (w.mode == .bf16 or w.mode == .fp8g) and q.planFor(m, w, l.clusters).sk == 1;",
)
patch_cuda(
    "qmmf_shared.zig",
    """    var a: launch.Args = .{};
    a.add(x);
    a.add(w.codes);
    a.add(w.scale);
    a.add(y);
    for ([_]usize{ m, w.n, w.k, w.npad, if (m == 1) w.k else ldx, band, ldy }) |v| a.add(@as(i32, @intCast(v)));
    try launch.launch(l.shared_down[index], .{
        .grid = .{ .x = @intCast(rows * ((w.n + cols - 1) / cols)) },
        .block = .{ .x = 128 },
        .shared = if (half) q.halfShared(.bf16) else q.sharedBytes(.bf16, @intCast(bm)),
    }, s, &a);""",
    """    if (w.mode == .fp8g) {
        var a8: launch.Args = .{};
        a8.add(x);
        a8.add(w.codes);
        a8.add(w.scales);
        a8.add(w.scale);
        a8.add(y);
        for ([_]usize{ m, w.n, w.k, w.npad, if (m == 1) w.k else ldx, band, ldy }) |v| a8.add(@as(i32, @intCast(v)));
        const smem_half: u32 = 4 * ((16 * 128 + 32 * 64 + 32 * 4 + 127) / 128 * 128);
        return launch.launch(l.shared_down_fp8g[index], .{
            .grid = .{ .x = @intCast(rows * ((w.n + cols - 1) / cols)) },
            .block = .{ .x = 128 },
            .shared = if (half) smem_half else q.sharedBytes(.fp8g, @intCast(bm)),
        }, s, &a8);
    }
    var a: launch.Args = .{};
    a.add(x);
    a.add(w.codes);
    a.add(w.scale);
    a.add(y);
    for ([_]usize{ m, w.n, w.k, w.npad, if (m == 1) w.k else ldx, band, ldy }) |v| a.add(@as(i32, @intCast(v)));
    try launch.launch(l.shared_down[index], .{
        .grid = .{ .x = @intCast(rows * ((w.n + cols - 1) / cols)) },
        .block = .{ .x = 128 },
        .shared = if (half) q.halfShared(.bf16) else q.sharedBytes(.bf16, @intCast(bm)),
    }, s, &a);""",
)

patch_exact(
    "cuda_projection_load.zig",
    "            return projection.dense(a, d, t);",
    "            if (@import(\"cuda_mtp_mxfp8.zig\").useForLoad(prefix))\n                return @import(\"cuda_mtp_mxfp8.zig\").project(a, d, t.bytes, n, k);\n            return projection.dense(a, d, t);",
)
patch_exact(
    "cuda_projection_load.zig",
    "        return projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = bytes });",
    "        if (@import(\"cuda_mtp_mxfp8.zig\").useForStitched(prefix))\n            return @import(\"cuda_mtp_mxfp8.zig\").project(a, d, bytes, n, k);\n        return projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = bytes });",
)
patch_exact(
    "cuda_prompt_wt.zig",
    """    var dq = try projection.dense(gpa, driver, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, wide, 1, 1, 1 }, .bytes = data });
    errdefer dq.free();
    const up = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_up.weight", .{prefix}));
    if (!up.is(.bf16, &.{ wide, low })) return error.UnexpectedTensor;
    var uq = try projection.dense(gpa, driver, up);""",
    """    var dq = if (@import("cuda_mtp_mxfp8.zig").useForHc(prefix))
        try @import("cuda_mtp_mxfp8.zig").project(gpa, driver, data, n, wide)
    else
        try projection.dense(gpa, driver, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, wide, 1, 1, 1 }, .bytes = data });
    errdefer dq.free();
    const up = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_up.weight", .{prefix}));
    if (!up.is(.bf16, &.{ wide, low })) return error.UnexpectedTensor;
    var uq = if (@import("cuda_mtp_mxfp8.zig").useForHc(prefix))
        try @import("cuda_mtp_mxfp8.zig").project(gpa, driver, up.bytes, wide, low)
    else
        try projection.dense(gpa, driver, up);""",
)

# 5a2. cuda_mtp_pieces.zig: direct sk=1 output for native fc_e / fc_h
patch_exact(
    "cuda_mtp_pieces.zig",
    """            const esk = try pp.lane(ps.mixed, ps.xsm, hh.fc_e, ps.dummy, ps.part, m, fc_sk);
            try pp.sumSlices(ps.part, ps.branch, true, m * dims, esk);
            // the hidden branch: the four streams normed as one row, fc_hidden on each stream
            try t.rmsnorm(r.input, hh.norm_h.ptr, ps.norm, ps.gq, eps, wide, m, wide, wide);
            const hsk = try pp.lane(ps.norm, ps.gq, hh.fc_h, ps.dummy, ps.part, streams * m, fc_sk);
            try pp.sumSlices(ps.part, ps.ple_gated, true, streams * m * dims, hsk);""",
    """            if (hh.fc_e.native != null) {
                _ = try pp.lane(ps.mixed, ps.xsm, hh.fc_e, ps.branch, ps.dummy, m, 1);
            } else {
                const esk = try pp.lane(ps.mixed, ps.xsm, hh.fc_e, ps.dummy, ps.part, m, fc_sk);
                try pp.sumSlices(ps.part, ps.branch, true, m * dims, esk);
            }
            try t.rmsnorm(r.input, hh.norm_h.ptr, ps.norm, ps.gq, eps, wide, m, wide, wide);
            if (hh.fc_h.native != null) {
                _ = try pp.lane(ps.norm, ps.gq, hh.fc_h, ps.ple_gated, ps.dummy, streams * m, 1);
            } else {
                const hsk = try pp.lane(ps.norm, ps.gq, hh.fc_h, ps.dummy, ps.part, streams * m, fc_sk);
                try pp.sumSlices(ps.part, ps.ple_gated, true, streams * m * dims, hsk);
            }""",
)
patch_exact(
    "cuda_mtp_pieces.zig",
    """            const esk = try pp.lane(ps.mixed, ps.xsm, hh.fc_e, ps.dummy, ps.part, tot, fc_sk);
            try pp.sumSlices(ps.part, ps.branch, true, tot * dims, esk);
            // each lane's rows of main-model streams, side by side (the head reads them as one block of rows)
            var row0: usize = 0;
            for (r.group.rows, r.group.nexts) |rw, nx| {
                try pp.copy(ps.ple_nrow + row0 * wide * 2, rw, nx.len * wide * 2);
                row0 += nx.len;
            }
            try t.rmsnorm(ps.ple_nrow, hh.norm_h.ptr, ps.norm, ps.gq, eps, wide, tot, wide, wide);
            const hsk = try pp.lane(ps.norm, ps.gq, hh.fc_h, ps.dummy, ps.part, streams * tot, fc_sk);
            try pp.sumSlices(ps.part, ps.ple_gated, true, streams * tot * dims, hsk);""",
    """            if (hh.fc_e.native != null) {
                _ = try pp.lane(ps.mixed, ps.xsm, hh.fc_e, ps.branch, ps.dummy, tot, 1);
            } else {
                const esk = try pp.lane(ps.mixed, ps.xsm, hh.fc_e, ps.dummy, ps.part, tot, fc_sk);
                try pp.sumSlices(ps.part, ps.branch, true, tot * dims, esk);
            }
            var row0: usize = 0;
            for (r.group.rows, r.group.nexts) |rw, nx| {
                try pp.copy(ps.ple_nrow + row0 * wide * 2, rw, nx.len * wide * 2);
                row0 += nx.len;
            }
            try t.rmsnorm(ps.ple_nrow, hh.norm_h.ptr, ps.norm, ps.gq, eps, wide, tot, wide, wide);
            if (hh.fc_h.native != null) {
                _ = try pp.lane(ps.norm, ps.gq, hh.fc_h, ps.ple_gated, ps.dummy, streams * tot, 1);
            } else {
                const hsk = try pp.lane(ps.norm, ps.gq, hh.fc_h, ps.dummy, ps.part, streams * tot, fc_sk);
                try pp.sumSlices(ps.part, ps.ple_gated, true, streams * tot * dims, hsk);
            }""",
)

# 5a3. cuda_ple.zig + cuda_mtp_draft.zig + cuda_mtp_ops.zig + cuda_lanes.zig + cuda_lanes_draft.zig: 32K PLE cache + async CPU pre-warm
patch_exact(
    "cuda_ple.zig",
    "const ple_fp8 = @import(\"cuda_ple_fp8.zig\");",
    "const ple_fp8 = @import(\"cuda_ple_fp8.zig\");\nconst ple_cache = @import(\"cuda_ple_cache.zig\");",
)
patch_exact(
    "cuda_ple.zig",
    "    pool: ?*ple_pool.Pool = null, // the gather's parked workers (null: every row on the caller's thread)\n    warmer: ?std.Thread = null, // warm's read-ahead thread until joined (waitWarm, deinit)",
    "    pool: ?*ple_pool.Pool = null,\n    warmer: ?std.Thread = null,\n    cache: ?*ple_cache.Cache = null,",
)
patch_exact(
    "cuda_ple.zig",
    "        g.pool = try ple_pool.Pool.create(gpa, io);\n        return g;",
    "        g.pool = try ple_pool.Pool.create(gpa, io);\n        if (g.width() == ple_cache.row_w and g.hash.n == 3) {\n            g.cache = try ple_cache.Cache.create(gpa);\n        }\n        return g;",
)
patch_exact(
    "cuda_ple.zig",
    "    pub fn deinit(g: *Ngram) void {\n        g.waitWarm();\n        if (g.pool) |p| p.destroy();",
    "    pub fn deinit(g: *Ngram) void {\n        g.waitWarm();\n        if (g.cache) |c| c.destroy(g.gpa);\n        if (g.pool) |p| p.destroy();",
)
patch_exact(
    "cuda_ple.zig",
    """    pub fn rows(g: *const Ngram, history: []i64, tokens: []const u32, out: []u16, xs: []f32) !void {
        const heads = g.hash.heads;
        const w = g.width();
        if (history.len != g.hash.n - 1 or out.len != tokens.len * w or xs.len != out.len / 32) return error.UnexpectedTensor;""",
    """    pub fn warmToken(g: *const Ngram, hist: *[2]i64, tok: u32) void {
        const cache = g.cache orelse return;
        if (g.hash.n != 3 or g.width() != ple_cache.row_w) return;
        if (cache.has(hist[0], hist[1], tok)) {
            hist.* = .{ hist[1], tok };
            return;
        }
        var emb: [ple_cache.row_w]u16 = undefined;
        var xs_tmp: [ple_cache.xs_w]f32 = undefined;
        g.rows(hist[0..], &.{tok}, &emb, &xs_tmp) catch {};
    }

    pub fn warmBatch(g: *const Ngram, ds: anytype, nexts: []const []const u32) void {
        const cache = g.cache orelse return;
        if (g.hash.n != 3 or g.width() != ple_cache.row_w or ds.len == 0 or ds.len != nexts.len) return;
        const heads = g.hash.heads;
        var miss_tok: [16]u32 = undefined;
        var miss_h0: [16]i64 = undefined;
        var miss_h1: [16]i64 = undefined;
        var n_miss: usize = 0;
        const lim = @min(ds.len, 16);
        for (0..lim) |i| {
            const nx = nexts[i];
            if (nx.len == 0) continue;
            const tok = nx[nx.len - 1];
            const d = ds[i];
            const h0 = d.draft_hist[0];
            const h1 = d.draft_hist[1];
            d.draft_hist = .{ h1, tok };
            if (!cache.has(h0, h1, tok)) {
                miss_tok[n_miss] = tok;
                miss_h0[n_miss] = h0;
                miss_h1[n_miss] = h1;
                n_miss += 1;
            }
        }
        if (n_miss == 0) return;
        const miss_items = n_miss * heads;
        const miss_ids = g.gpa.alloc(i64, miss_items) catch return;
        defer g.gpa.free(miss_ids);
        const miss_emb = g.gpa.alloc(u16, n_miss * ple_cache.row_w) catch return;
        defer g.gpa.free(miss_emb);
        for (0..n_miss) |m| {
            const mh: [2]i64 = .{ miss_h0[m], miss_h1[m] };
            lastIds(g.hash, &mh, @intCast(miss_tok[m]), miss_ids[m * heads ..][0..heads]) catch return;
        }
        if (g.format != .mlx and miss_items > 1) {
            @import("cuda_ple_gather.zig").run(g, miss_ids, miss_emb) catch return;
        } else {
            g.fill(miss_ids, miss_emb, 0, miss_items) catch return;
        }
        var xs_tmp: [ple_cache.xs_w]f32 = undefined;
        for (0..n_miss) |m| {
            const row: *[ple_cache.row_w]u16 = miss_emb[m * ple_cache.row_w ..][0..ple_cache.row_w];
            if (g.format == .mlx and g.scale != 1) {
                for (row) |*v| {
                    const f: f32 = @bitCast(@as(u32, v.*) << 16);
                    v.* = toBf16(f * g.scale);
                }
            }
            sumGroups(row, &xs_tmp);
            cache.put(miss_h0[m], miss_h1[m], miss_tok[m], row, &xs_tmp);
        }
    }

    pub fn rows(g: *const Ngram, history: []i64, tokens: []const u32, out: []u16, xs: []f32) !void {
        const heads = g.hash.heads;
        const w = g.width();
        if (history.len != g.hash.n - 1 or out.len != tokens.len * w or xs.len != out.len / 32) return error.UnexpectedTensor;
        if (g.cache) |cache| if (history.len == 2 and w == ple_cache.row_w and tokens.len > 0) {
            var hist2: [2]i64 = .{ history[0], history[1] };
            var stack_idx: [64]usize = undefined;
            var stack_h0: [64]i64 = undefined;
            var stack_h1: [64]i64 = undefined;
            const dyn = tokens.len > stack_idx.len;
            const miss_idx = if (dyn) try g.gpa.alloc(usize, tokens.len) else stack_idx[0..tokens.len];
            defer if (dyn) g.gpa.free(miss_idx);
            const miss_h0 = if (dyn) try g.gpa.alloc(i64, tokens.len) else stack_h0[0..tokens.len];
            defer if (dyn) g.gpa.free(miss_h0);
            const miss_h1 = if (dyn) try g.gpa.alloc(i64, tokens.len) else stack_h1[0..tokens.len];
            defer if (dyn) g.gpa.free(miss_h1);
            var n_miss: usize = 0;
            for (tokens, 0..) |tok, t| {
                const out_row: *[ple_cache.row_w]u16 = out[t * ple_cache.row_w ..][0..ple_cache.row_w];
                const xs_row: *[ple_cache.xs_w]f32 = xs[t * ple_cache.xs_w ..][0..ple_cache.xs_w];
                if (!cache.get(hist2[0], hist2[1], tok, out_row, xs_row)) {
                    miss_idx[n_miss] = t;
                    miss_h0[n_miss] = hist2[0];
                    miss_h1[n_miss] = hist2[1];
                    n_miss += 1;
                }
                hist2 = .{ hist2[1], tok };
            }
            if (n_miss > 0) {
                const miss_items = n_miss * heads;
                const miss_ids = try g.gpa.alloc(i64, miss_items);
                defer g.gpa.free(miss_ids);
                const miss_emb = try g.gpa.alloc(u16, n_miss * ple_cache.row_w);
                defer g.gpa.free(miss_emb);
                for (0..n_miss) |m| {
                    const t = miss_idx[m];
                    const mh: [2]i64 = .{ miss_h0[m], miss_h1[m] };
                    try lastIds(g.hash, &mh, @intCast(tokens[t]), miss_ids[m * heads ..][0..heads]);
                }
                const nt = @min(ple_pool.max_parts, miss_items / 4);
                if (g.format != .mlx and miss_items > 1) {
                    try @import("cuda_ple_gather.zig").run(g, miss_ids, miss_emb);
                } else if (nt <= 1 or g.pool == null) {
                    try g.fill(miss_ids, miss_emb, 0, miss_items);
                } else {
                    const split_m: Split = .{ .g = g, .ids = miss_ids, .out = miss_emb, .per = (miss_items + nt - 1) / nt };
                    try g.pool.?.run(nt, .{ .ctx = &split_m, .run = Split.part });
                }
                for (0..n_miss) |m| {
                    const t = miss_idx[m];
                    const src_row = miss_emb[m * ple_cache.row_w ..][0..ple_cache.row_w];
                    if (g.format == .mlx and g.scale != 1) {
                        for (src_row) |*v| {
                            const f: f32 = @bitCast(@as(u32, v.*) << 16);
                            v.* = toBf16(f * g.scale);
                        }
                    }
                    const dst_row: *[ple_cache.row_w]u16 = out[t * ple_cache.row_w ..][0..ple_cache.row_w];
                    const dst_xs: *[ple_cache.xs_w]f32 = xs[t * ple_cache.xs_w ..][0..ple_cache.xs_w];
                    @memcpy(dst_row, src_row);
                    sumGroups(dst_row, dst_xs);
                    cache.put(miss_h0[m], miss_h1[m], tokens[t], dst_row, dst_xs);
                }
            }
            history[0] = hist2[0];
            history[1] = hist2[1];
            return;
        };""",
)
patch_exact(
    "cuda_mtp_draft.zig",
    "    history: [2]i64 = .{ 0, 0 },\n    unused_history: [2]i64 = .{ 0, 0 },",
    "    history: [2]i64 = .{ 0, 0 },\n    draft_hist: [2]i64 = .{ 0, 0 },\n    unused_history: [2]i64 = .{ 0, 0 },",
)
patch_exact(
    "cuda_lanes.zig",
    "                if (r.position != l.seq.fed + 1) return error.PositionMismatch;",
    "                if (r.position != l.seq.fed + 1) return error.PositionMismatch;\n                d.draft_hist = l.seq.history;",
)
patch_exact(
    "cuda_lanes_draft.zig",
    "        if (r.position != l.seq.fed + 1) return error.PositionMismatch;",
    "        if (r.position != l.seq.fed + 1) return error.PositionMismatch;\n        d.draft_hist = l.seq.history;",
)
patch_exact(
    "cuda_mtp_ops.zig",
    """        pub fn draft(o: Self, d: *Draft, next: []const u32, rows: u64, position: u64, count: usize, sm: ?Sampling, confidence: f64, out: *Chain) !void {
            out.n = 0;
            try o.absorb(d, next, rows, count > 0);
            var stop: mtp.Stop = .{ .c = confidence };
            var j: usize = 0;
            while (j < @min(count, max_chain)) : (j += 1) {
                const got = try o.pick(d, position + j, sm);
                const t = stop.take(j, got.prob);
                if (!t.propose) break;
                out.tokens[out.n] = got.token;
                out.probs[out.n] = got.prob;
                out.n += 1;
                if (!t.more) break;
                if (j + 1 < count) {
                    try o.forwardHead(d, &.{got.token}, d.out.ptr, true);
                    d.drafted += 1;
                }
            }
        }""",
    """        pub fn draft(o: Self, d: *Draft, next: []const u32, rows: u64, position: u64, count: usize, sm: ?Sampling, confidence: f64, out: *Chain) !void {
            out.n = 0;
            try o.absorb(d, next, rows, count > 0);
            if (next.len > 0) o.e.ngram.warmToken(&d.draft_hist, next[next.len - 1]);
            var stop: mtp.Stop = .{ .c = confidence };
            var j: usize = 0;
            while (j < @min(count, max_chain)) : (j += 1) {
                const got = try o.pick(d, position + j, sm);
                const t = stop.take(j, got.prob);
                if (!t.propose) break;
                out.tokens[out.n] = got.token;
                out.probs[out.n] = got.prob;
                out.n += 1;
                if (!t.more) break;
                if (j + 1 < count) {
                    try o.forwardHead(d, &.{got.token}, d.out.ptr, true);
                    d.drafted += 1;
                    o.e.ngram.warmToken(&d.draft_hist, got.token);
                }
            }
        }""",
)
patch_exact(
    "cuda_mtp_ops.zig",
    "    try o.forwardHeadMany(ds, nexts, rows, true);",
    "    try o.forwardHeadMany(ds, nexts, rows, true);\n    o.e.ngram.warmBatch(ds, nexts);",
)
patch_exact(
    "cuda_mtp_ops.zig",
    "            try o.forwardHeadMany(step_ds[0..n_step], step_nexts[0..n_step], step_rows[0..n_step], true);\n            for (step_ds[0..n_step]) |d| d.drafted += 1;",
    "            try o.forwardHeadMany(step_ds[0..n_step], step_nexts[0..n_step], step_rows[0..n_step], true);\n            for (step_ds[0..n_step]) |d| d.drafted += 1;\n            o.e.ngram.warmBatch(step_ds[0..n_step], step_nexts[0..n_step]);",
)

# 5b. cuda_mtp.zig + cuda_lanes_load.zig: hook W2 proposal head & share Engine's BF16 lm_head buffer
patch_exact(
    "cuda_mtp.zig",
    "    topk_fn: cuda.Function, // fn_row_topk: a row's top_k candidates as drawRow ranks them (cuda_mtp_draw.zig)",
    "    topk_fn: cuda.Function,\n    w2: @import(\"cuda_mtp_w2.zig\").W2 = .{},",
)
patch_exact(
    "cuda_mtp.zig",
    "        h.draw_batch = try @import(\"cuda_draw_batch.zig\").Batch.init(gpa, driver, batch_on, h.ids.len);\n        return h;",
    "        h.draw_batch = try @import(\"cuda_draw_batch.zig\").Batch.init(gpa, driver, batch_on, h.ids.len);\n        errdefer h.draw_batch.deinit(gpa);\n        h.w2 = try @import(\"cuda_mtp_w2.zig\").W2.init(gpa, driver, &store, &h.artifact, &h.module);\n        return h;",
)
patch_exact(
    "cuda_mtp.zig",
    "    pub fn deinit(h: *Head) void {\n        h.draw_batch.deinit(h.gpa);",
    "    pub fn deinit(h: *Head) void {\n        h.w2.deinit();\n        h.draw_batch.deinit(h.gpa);",
)
patch_exact(
    "cuda_mtp.zig",
    "    pub fn project(h: *const Head, pass: anytype, input: u64, sums: u64, out: u64, rows: usize) !void {\n        if (h.draft.native) |weight| {",
    "    pub fn project(h: *const Head, pass: anytype, input: u64, sums: u64, out: u64, rows: usize) !void {\n        if (rows == 1 and h.w2.ready()) return h.w2.project(h, pass.stream, input, out, h.ids.len);\n        if (h.draft.native) |weight| {",
)
patch_exact(
    "cuda_lanes_load.zig",
    "        own.head = try mtp.Head.init(gpa, io, driver, &own.stream, dir);",
    "        own.head = try mtp.Head.init(gpa, io, driver, &own.stream, dir);\n        own.head.?.w2.bf16_ptr = own.engine.head_q.b.ptr;",
)

# 5c. cuda_lanes_bytes.zig: reflect NVFP4 MTP expert size in budget check
patch_exact(
    "cuda_lanes_bytes.zig",
    "pub fn headFp8(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {",
    "pub fn headFp8(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {\n    if (std.c.getenv(\"TF_FLASHNEXT_MTP_EXPERTS\")) |v| if (std.mem.eql(u8, std.mem.span(v), \"nvfp4\")) return false;",
)

# 5d. cuda_prompt.zig + cuda_batch_kernels.zig + cuda_batch_head.zig: two-stage exact NVFP4 + top-128 BF16 rescore head
patch_exact(
    "cuda_prompt.zig",
    "            e.head_q = try @import(\"cuda_projection.zig\").dense(gpa, driver, if (try @import(\"cuda_head.zig\").enabled()) try @import(\"cuda_head.zig\").shard(head_tensor, world, rank) else head_tensor);",
    "            e.head_q = if (world == 1 and !(try @import(\"cuda_head.zig\").enabled()) and @import(\"cuda_head_nvfp4.zig\").enabled())\n                try @import(\"cuda_head_nvfp4.zig\").load(driver, stream.*, &e.store, head_tensor)\n            else\n                try @import(\"cuda_projection.zig\").dense(gpa, driver, if (try @import(\"cuda_head.zig\").enabled()) try @import(\"cuda_head.zig\").shard(head_tensor, world, rank) else head_tensor);",
)
patch_exact(
    "cuda_batch_kernels.zig",
    "    select: cuda.Function,\n    select_lanes: cuda.Function,",
    "    select: cuda.Function,\n    select_lanes: cuda.Function,\n    head_topk_floor: cuda.Function,\n    head_rescore: cuda.Function,",
)
patch_exact(
    "cuda_batch_kernels.zig",
    "        k.late_roll = try k.batch.function(\"fn_late_roll\");\n        return k;",
    "        k.late_roll = try k.batch.function(\"fn_late_roll\");\n        k.head_topk_floor = try k.batch.function(\"fn_head_topk_floor\");\n        k.head_rescore = try k.batch.function(\"fn_head_rescore\");\n        return k;",
)
patch_exact(
    "cuda_batch_head.zig",
    """    if (self.m.head.n == vocab) {
        _ = try self.lane(s.mixed, s.xsm, self.m.head, out, s.dummy, n, 1);
        return;
    }""",
    """    if (self.m.head.n == vocab) {
        _ = try self.lane(s.mixed, s.xsm, self.m.head, out, s.dummy, n, 1);
        if (self.m.head.b.ptr != 0) if (self.m.head.native) |nw| if (nw.mode == .fp4) {
            const k_top: usize = 128;
            if (n * k_top * 4 > s.cap * (wt.experts_n + 1) * 4) return error.UnexpectedTensor;
            var a1: cuda.Args = .{};
            a1.add(out);
            a1.add(@as(c_int, vocab));
            a1.add(@as(c_longlong, vocab));
            a1.add(@as(c_int, k_top));
            a1.add(s.router_l);
            try cuda.launch.launch(self.k.head_topk_floor, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 1024 } }, self.stream, &a1);
            var a2: cuda.Args = .{};
            a2.add(s.mixed);
            a2.add(self.m.head.b.ptr);
            a2.add(s.router_l);
            a2.add(out);
            a2.add(@as(c_int, vocab));
            a2.add(@as(c_int, k_top));
            try cuda.launch.launch(self.k.head_rescore, .{ .grid = .{ .x = @intCast((k_top + 3) / 4), .y = @intCast(n) }, .block = .{ .x = 128 } }, self.stream, &a2);
        };
        return;
    }""",
)

# 5e. cuda_window_graph.zig: 4-way LRU cache per shape slot to eliminate state-swap & streams/out re-recording
old_cache = """pub fn Cache(comptime K: type) type {
    return struct {
        const Self = @This();
        /// A held graph and the whole key it was last recorded for; the map holds it under its slot (slotOf).
        const Held = struct { key: K, exec: cuda.graph.Exec };
        map: std.AutoHashMapUnmanaged(K, Held) = .empty,
        /// Slots run once eagerly and are captured the second time (a one-off pass would never pay its capture back).
        seen: std.AutoHashMapUnmanaged(K, void) = .empty,
        captures: u64 = 0,
        replays: u64 = 0,
        /// Held graphs re-recorded at a new length, for a new sequence or on other buffers and updated in place.
        updates: u64 = 0,
        /// A capture failed: every later pass runs eagerly (logged once).
        off: bool = false,

        pub fn clear(c: *Self, gpa: std.mem.Allocator) void {
            var it = c.map.valueIterator();
            while (it.next()) |h| h.exec.deinit();
            c.map.clearAndFree(gpa);
            c.seen.clearAndFree(gpa);
        }

        /// `pass.run()` (kernels only: its uploads already queued) as `key`'s graph, captured first when new.
        pub fn run(c: *Self, gpa: std.mem.Allocator, stream: cuda.Stream, key: K, pass: anytype) !void {
            if (c.off) return pass.run();
            const slot = slotOf(key);
            if (c.map.getPtr(slot)) |h| {
                if (std.meta.eql(h.key, key)) {
                    c.replays += 1;
                    return h.exec.launchOn(stream);
                }
                // the same pass at another length, sequence or lane: recorded again, the held graph updated in place
                refresh(stream, pass, &h.exec) catch |err| return c.fail(err, pass);
                h.key = key;
                c.updates += 1;
                return h.exec.launchOn(stream);
            }
            if (c.map.count() >= max_graphs or c.seen.count() >= 4 * max_graphs) c.clear(gpa);
            if (!c.seen.contains(slot)) {
                try c.seen.put(gpa, slot, {});
                return pass.run();
            }
            var exec = capture(stream, pass) catch |err| return c.fail(err, pass);
            errdefer exec.deinit();
            try c.map.put(gpa, slot, .{ .key = key, .exec = exec });
            c.captures += 1;
            if (c.captures % 64 == 1) std.log.info("{s} graphs: {d} captured, {d} updated, {d} replayed, {d} held", .{ @typeName(K), c.captures, c.updates, c.replays, c.map.count() });
            try exec.launchOn(stream);
        }"""

new_cache = """pub fn Cache(comptime K: type) type {
    return struct {
        const Self = @This();
        const ways = 4;
        const Entry = struct { key: K, exec: cuda.graph.Exec, stamp: u64 };
        const Bucket = struct { items: [ways]?Entry = @splat(null) };
        map: std.AutoHashMapUnmanaged(K, Bucket) = .empty,
        seen: std.AutoHashMapUnmanaged(K, void) = .empty,
        captures: u64 = 0,
        replays: u64 = 0,
        updates: u64 = 0,
        tick: u64 = 0,
        off: bool = false,

        pub fn clear(c: *Self, gpa: std.mem.Allocator) void {
            var it = c.map.valueIterator();
            while (it.next()) |b| {
                for (&b.items) |*se| {
                    if (se.*) |*e| e.exec.deinit();
                    se.* = null;
                }
            }
            c.map.clearAndFree(gpa);
            c.seen.clearAndFree(gpa);
        }

        pub fn run(c: *Self, gpa: std.mem.Allocator, stream: cuda.Stream, key: K, pass: anytype) !void {
            if (c.off) return pass.run();
            c.tick +%= 1;
            const slot = slotOf(key);
            if (c.map.getPtr(slot)) |b| {
                for (&b.items) |*se| {
                    if (se.*) |*e| {
                        if (std.meta.eql(e.key, key)) {
                            e.stamp = c.tick;
                            c.replays += 1;
                            return e.exec.launchOn(stream);
                        }
                    }
                }
                for (&b.items) |*se| {
                    if (se.* == null) {
                        const exec = capture(stream, pass) catch |err| return c.fail(err, pass);
                        se.* = .{ .key = key, .exec = exec, .stamp = c.tick };
                        c.captures += 1;
                        return exec.launchOn(stream);
                    }
                }
                var lru: usize = 0;
                for (1..ways) |i| {
                    if (b.items[i].?.stamp < b.items[lru].?.stamp) lru = i;
                }
                refresh(stream, pass, &b.items[lru].?.exec) catch |err| return c.fail(err, pass);
                b.items[lru].?.key = key;
                b.items[lru].?.stamp = c.tick;
                c.updates += 1;
                return b.items[lru].?.exec.launchOn(stream);
            }
            if (c.map.count() >= max_graphs or c.seen.count() >= 4 * max_graphs) c.clear(gpa);
            if (!c.seen.contains(slot)) {
                try c.seen.put(gpa, slot, {});
                return pass.run();
            }
            var exec = capture(stream, pass) catch |err| return c.fail(err, pass);
            errdefer exec.deinit();
            var bucket: Bucket = .{};
            bucket.items[0] = .{ .key = key, .exec = exec, .stamp = c.tick };
            try c.map.put(gpa, slot, bucket);
            c.captures += 1;
            if (c.captures % 64 == 1) std.log.info("{s} graphs: {d} captured, {d} updated, {d} replayed, {d} held", .{ @typeName(K), c.captures, c.updates, c.replays, c.map.count() });
            try exec.launchOn(stream);
        }"""

patch_exact("cuda_window_graph.zig", old_cache, new_cache)
PY
}

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
      stage="$(mktemp -d "$WORKDIR/engines/.tf-stage.XXXXXXXX")"
      trap 'rm -rf -- "$stage"' EXIT
      zig_dir="$WORKDIR/engines/zig-0.17.0-aarch64"
      if [[ ! -x "$zig_dir/zig" ]]; then
        curl -fsSL "https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz" -o "$stage/zig.tar.xz"
        echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  $stage/zig.tar.xz" | sha256sum -c -
        mkdir "$stage/zig"
        tar -xJf "$stage/zig.tar.xz" --strip-components=1 -C "$stage/zig"
        mv "$stage/zig" "$zig_dir"
      fi

      if [[ ! -d "$SRC_DIR/.git" ]]; then
        rm -rf "$SRC_DIR"
        git init -q "$SRC_DIR"
        git -C "$SRC_DIR" remote add origin https://github.com/ashhart/TensorFold.git
        git -C "$SRC_DIR" fetch -q --depth 1 origin "$ENGINE_COMMIT"
      fi
      git -C "$SRC_DIR" checkout -q -f "$ENGINE_COMMIT"
      git -C "$SRC_DIR" clean -fdq -e .zig-cache

      apply_patches

      (
        cd "$SRC_DIR"
        "$zig_dir/zig" build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc \
          -Doptimize=fast -j6 --prefix "$stage/install"
      )
      test -x "$stage/install/native/bin/tensorfold-native"
      rm -rf "$BUILD_DIR"
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
    export LD_LIBRARY_PATH="${RELEASE_LIB}:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export TF_FLASHNEXT_DRAFT_HEAD="affine4"
    export TF_FLASHNEXT_MTP_W2="1"
    export TF_FLASHNEXT_MTP_EXPERTS="nvfp4"
    export TF_FLASHNEXT_HEAD_NVFP4="1"
    export TF_FLASHNEXT_NONREC_FP8G="1"
    export TF_FLASHNEXT_GREEDY_WIDE="1"
    if [[ "$PROFILE" == "guard" ]]; then
      export TENSORFOLD_MEMORY_RESERVE_GIB=6
      export TF_FLASHNEXT_DRAFTS="8"
      export TF_FLASHNEXT_DRAFTS_GREEDY="14"
      export TF_FLASHNEXT_CONFIDENCE="-0.25"
      CTX="65536"
      PAR="$GUARD_PARALLEL"
      THINK="--thinking"
    elif [[ "$PROFILE" == "nvidia" ]]; then
      export TENSORFOLD_MEMORY_RESERVE_GIB=6
      export TF_FLASHNEXT_DRAFTS="6"
      export TF_FLASHNEXT_DRAFTS_GREEDY="14"
      export TF_FLASHNEXT_CONFIDENCE="-0.35"
      CTX="$GUARD_CONTEXT"
      PAR="$GUARD_PARALLEL"
      THINK="--thinking"
    else
      export TF_FLASHNEXT_DRAFTS_GREEDY="14"
      export TF_FLASHNEXT_CONFIDENCE="-0.18"
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
