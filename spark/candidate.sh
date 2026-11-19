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
BUILD_FLAVOR="mtp-w2-nvfp4-g4-head4-bf12"
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
  local cuda_dir="$SRC_DIR/zig/src/cuda"
  local kern_dir="$SRC_DIR/zig/kernels/cuda"

  # 1a. Append W2 group-32 proposal GEMV + shared-BF16 exact top-64 rescore kernels to flashnext_mtp.cu
  cat >> "$kern_dir/flashnext_mtp.cu" <<'CU'

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
  cat >> "$kern_dir/flashnext_batch.cu" <<'CU'

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

  # 1c. Patch qmmf.cu + qmmf_body.cuh: lossless Mode::BF12 (12.125-bit + 0.01% escape table, half/prompt/strided) + PR #3475 fused split-K
  python3 - "$kern_dir" <<'PYCU'
from pathlib import Path
import sys

kdir = Path(sys.argv[1])

def patch_k(rel: str, old: str, new: str) -> None:
    p = kdir / rel
    txt = p.read_text(encoding="utf-8")
    if txt.count(old) != 1:
        raise RuntimeError(f"Expected 1 match in {rel}, got {txt.count(old)}")
    p.write_text(txt.replace(old, new, 1), encoding="utf-8")

patch_k(
    "qmmf.cu",
    "enum Mode : int { FP4 = 0, FP8 = 1, MXFP8 = 2, FP8G = 3, BF16 = 4 };",
    """enum Mode : int { FP4 = 0, FP8 = 1, MXFP8 = 2, FP8G = 3, BF16 = 4, BF12 = 5 };

__device__ __forceinline__ uint16_t bf12_esc_lookup(
        const uint8_t* __restrict__ bs, size_t total_tiles, size_t tile_idx, uint32_t key12) {
    const size_t dir_off = (total_tiles * 64 + 15) & ~static_cast<size_t>(15);
    const uint32_t* dir = reinterpret_cast<const uint32_t*>(bs + dir_off);
    const uint32_t* ents = dir + (total_tiles + 1);
    const uint32_t lo = __ldg(dir + tile_idx);
    const uint32_t hi = __ldg(dir + tile_idx + 1);
    for (uint32_t idx = lo; idx < hi; ++idx) {
        const uint32_t e = __ldg(ents + idx);
        if ((e >> 16) == key12) return static_cast<uint16_t>(e & 0xFFFFu);
    }
    return 0u;
}

__device__ __forceinline__ uint32_t dec12_pair(
        uint32_t ms, uint32_t perm, uint32_t d_pair, uint32_t em_pair,
        bool lane_has_esc, const uint8_t* __restrict__ bs, size_t total_tiles, size_t tile_idx, uint32_t key_base) {
    const uint32_t x = __byte_perm(ms, 0u, perm);
    if (__builtin_expect(!lane_has_esc, 1)) {
        return ((x & 0x00800080u) << 8) | ((em_pair - d_pair) << 7) | (x & 0x007F007Fu);
    }
    const uint32_t d0 = d_pair & 0xFu, d1 = (d_pair >> 16) & 0xFu;
    const uint32_t em = em_pair & 0xFFu;
    const uint16_t lo = (d0 == 0xFu)
        ? bf12_esc_lookup(bs, total_tiles, tile_idx, key_base)
        : static_cast<uint16_t>(((x & 0x80u) << 8) | ((em - d0) << 7) | (x & 0x7Fu));
    const uint16_t hi = (d1 == 0xFu)
        ? bf12_esc_lookup(bs, total_tiles, tile_idx, key_base + 1u)
        : static_cast<uint16_t>((((x >> 16) & 0x80u) << 8) | ((em - d1) << 7) | ((x >> 16) & 0x7Fu));
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}""",
)

patch_k(
    "qmmf.cu",
    """    static constexpr int W = MODE == FP4 ? BN * GS / 2 : MODE == BF16 ? BN * GS * 2 : BN * GS;
    static constexpr int S = MODE == FP4 || MODE == FP8G ? BN * 4 : MODE == MXFP8 ? BN * 2 : 0;""",
    """    static constexpr int W = MODE == FP4 ? BN * GS / 2 : MODE == BF16 ? BN * GS * 2 : MODE == BF12 ? (BN * GS * 3) / 2 : BN * GS;
    static constexpr int S = MODE == FP4 || MODE == FP8G ? BN * 4 : MODE == MXFP8 ? BN * 2 : MODE == BF12 ? BN : 0;""",
)

patch_k(
    "qmmf.cu",
    """TF_SHARED_DOWN(tf_qmmf_shared64, 64, 64)
#undef TF_SHARED_DOWN""",
    """TF_SHARED_DOWN(tf_qmmf_shared64, 64, 64)
#undef TF_SHARED_DOWN

#define TF_SHARED_DOWN_BF12(NAME, BM, BN) \\
extern "C" __global__ void __launch_bounds__(128) NAME( \\
        const __nv_bfloat16* x, const unsigned char* w, const uint8_t* bs, float scale, float* out, \\
        int M, int N, int K, int npad, int ldx, int group, int ldy) { \\
    qmmf_body<BF12, BM, BN, 1, 4, 4, true, false, false>( \\
        x, w, bs, scale, out, nullptr, M, N, K, 1, npad, ldx, group, ldy); \\
}
TF_SHARED_DOWN_BF12(tf_qmmf_shared_bf12_16_half, 16, 32)
TF_SHARED_DOWN_BF12(tf_qmmf_shared_bf12_16, 16, 64)
TF_SHARED_DOWN_BF12(tf_qmmf_shared_bf12_32, 32, 64)
TF_SHARED_DOWN_BF12(tf_qmmf_shared_bf12_64, 64, 64)
#undef TF_SHARED_DOWN_BF12""",
)

patch_k(
    "qmmf.cu",
    "BF16_PROMPT(true, true)\n#undef BF16_PROMPT",
    """BF16_PROMPT(true, true)
#undef BF16_PROMPT

// Lossless BF12 decode, half, fused split-K (PR #3475), and 128x128 prompt instances.
#define BF12_INST(BM, BN, WM, WN, ST, F32, CL, FU) \\
    template __global__ void qmmf_kernel<BF12, BM, BN, WM, WN, ST, F32, CL, FU>( \\
        const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, \\
        void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
BF12_INST(16, 64, 1, 4, 4, false, false, false)
BF12_INST(16, 64, 1, 4, 4, false, true, false)
BF12_INST(16, 64, 1, 4, 4, false, false, true)
BF12_INST(32, 64, 1, 4, 4, false, false, false)
BF12_INST(32, 64, 1, 4, 4, false, true, false)
BF12_INST(32, 64, 1, 4, 4, false, false, true)
BF12_INST(64, 64, 1, 4, 4, false, false, false)
BF12_INST(64, 64, 1, 4, 4, false, true, false)
BF12_INST(64, 64, 1, 4, 4, false, false, true)
BF12_INST(16, 64, 1, 4, 4, true, false, false)
BF12_INST(16, 64, 1, 4, 4, true, true, false)
BF12_INST(16, 64, 1, 4, 4, true, false, true)
BF12_INST(32, 64, 1, 4, 4, true, false, false)
BF12_INST(32, 64, 1, 4, 4, true, true, false)
BF12_INST(32, 64, 1, 4, 4, true, false, true)
BF12_INST(64, 64, 1, 4, 4, true, false, false)
BF12_INST(64, 64, 1, 4, 4, true, true, false)
BF12_INST(64, 64, 1, 4, 4, true, false, true)
BF12_INST(16, 32, 1, 4, 4, false, false, false)
BF12_INST(16, 32, 1, 4, 4, false, true, false)
BF12_INST(16, 32, 1, 4, 4, true, false, false)
BF12_INST(16, 32, 1, 4, 4, true, true, false)
BF12_INST(128, 128, 2, 4, 3, false, false, false)
BF12_INST(128, 128, 2, 4, 3, false, false, true)
BF12_INST(128, 128, 2, 4, 3, true, false, false)
BF12_INST(128, 128, 2, 4, 3, true, false, true)
#undef BF12_INST

// PR #3475 fused split-K at row tiles 16 and 32 for MXFP8 and BF16.
#define FUSE_SPLIT(MD, BM, F32) \\
    template __global__ void qmmf_kernel<MD, BM, 64, 1, 4, 4, F32, false, true>( \\
        const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, \\
        void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
FUSE_SPLIT(MXFP8, 16, false)
FUSE_SPLIT(MXFP8, 32, false)
FUSE_SPLIT(MXFP8, 16, true)
FUSE_SPLIT(MXFP8, 32, true)
FUSE_SPLIT(BF16, 16, false)
FUSE_SPLIT(BF16, 32, false)
FUSE_SPLIT(BF16, 16, true)
FUSE_SPLIT(BF16, 32, true)
#undef FUSE_SPLIT""",
)

patch_k(
    "qmmf_body.cuh",
    'static_assert(BN == 64 || BN == 32 || (MODE == BF16 && BN == 128), "packed column tiles");',
    'static_assert(BN == 64 || BN == 32 || ((MODE == BF16 || MODE == BF12) && BN == 128), "packed column tiles");',
)

patch_k(
    "qmmf_body.cuh",
    "        constexpr int TILE_BYTES = MODE == FP4 ? 64 * GS / 2 : MODE == BF16 ? 64 * GS * 2 : 64 * GS;",
    "        constexpr int TILE_BYTES = MODE == FP4 ? 64 * GS / 2 : MODE == BF16 ? 64 * GS * 2 : MODE == BF12 ? 6144 : 64 * GS;",
)

patch_k(
    "qmmf_body.cuh",
    """        if constexpr (T::S > 0) {
            unsigned char* ps = pw + T::W;
            for (int c = tid; c < T::S / 16; c += T::THREADS)
                cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64) * KG + g) * T::S + c * 16);
        }""",
    """        if constexpr (T::S > 0) {
            unsigned char* ps = pw + T::W;
            constexpr int S_TILE = MODE == BF12 ? 64 : T::S;
            for (int c = tid; c < T::S / 16; c += T::THREADS) {
                const int st = c / (S_TILE / 16), soff = c % (S_TILE / 16);
                cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64 + st) * KG + g) * S_TILE + soff * 16);
            }
        }""",
)

patch_k(
    "qmmf_body.cuh",
    """            if constexpr (MODE == FP4) {
                const uint2 u = reinterpret_cast<const uint2*>(pw)[jj * 32 + lane];
                wq[j] = make_uint4(u.x, u.y, 0u, 0u);
            } else if constexpr (MODE != BF16) {
                wq[j] = reinterpret_cast<const uint4*>(pw)[jj * 32 + lane];
            }""",
    """            if constexpr (MODE == FP4) {
                const uint2 u = reinterpret_cast<const uint2*>(pw)[jj * 32 + lane];
                wq[j] = make_uint4(u.x, u.y, 0u, 0u);
            } else if constexpr (MODE == BF12) {
                const unsigned char* pjj = pw + jj * 768;
                const uint32_t d01 = reinterpret_cast<const uint32_t*>(pjj + 512)[lane];
                const uint32_t d23 = reinterpret_cast<const uint32_t*>(pjj + 640)[lane];
                const uint32_t f01 = d01 & (d01 >> 1) & (d01 >> 2) & (d01 >> 3) & 0x11111111u;
                const uint32_t f23 = d23 & (d23 >> 1) & (d23 >> 2) & (d23 >> 3) & 0x11111111u;
                const uint32_t em = static_cast<uint32_t>(ps[jj * 8 + (lane >> 2)]) * 0x00010001u;
                wq[j] = make_uint4(d01, d23, em, f01 | f23);
            } else if constexpr (MODE != BF16) {
                wq[j] = reinterpret_cast<const uint4*>(pw)[jj * 32 + lane];
            }""",
)

patch_k(
    "qmmf_body.cuh",
    """                } else {
                    const uint32_t word = comp(wq[j], kt);
                    b0 = fp8pair(word & 0xFFFFu);
                    b1 = fp8pair(word >> 16);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    if constexpr (MODE == FP8 || MODE == BF16) mma(acc[i][j], a[i], b0, b1);""",
    """                } else if constexpr (MODE == BF12) {
                    const int jj_in_pw = wn * T::NT + j;
                    const unsigned char* pjj = pw + jj_in_pw * 768;
                    const uint32_t ms = reinterpret_cast<const uint32_t*>(pjj)[kt * 32 + lane];
                    const uint32_t dlo = kt < 2 ? wq[j].x : wq[j].y;
                    const int sh = (kt & 1) * 8;
                    const bool has_esc = wq[j].w != 0u;
                    const int local_jj = (BN == 32) ? (((n0 & 32) ? 4 : 0) + jj_in_pw) : (jj_in_pw & 7);
                    const size_t total_tiles = static_cast<size_t>(npad / 64) * KG;
                    const size_t tile_idx = static_cast<size_t>(n0 / 64 + (BN == 128 ? (jj_in_pw >> 3) : 0)) * KG + (g0 + it);
                    const uint32_t kbase = (static_cast<uint32_t>(local_jj) << 9) | (static_cast<uint32_t>(kt) << 7) | (static_cast<uint32_t>(lane) << 2);
                    b0 = dec12_pair(ms, 0x4140u, (dlo >> sh) & 0x000F000Fu, wq[j].z, has_esc, bs, total_tiles, tile_idx, kbase);
                    b1 = dec12_pair(ms, 0x4342u, (dlo >> (sh + 4)) & 0x000F000Fu, wq[j].z, has_esc, bs, total_tiles, tile_idx, kbase + 2u);
                } else {
                    const uint32_t word = comp(wq[j], kt);
                    b0 = fp8pair(word & 0xFFFFu);
                    b1 = fp8pair(word >> 16);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    if constexpr (MODE == FP8 || MODE == BF16 || MODE == BF12) mma(acc[i][j], a[i], b0, b1);""",
)
PYCU

  # 2a. MTP BF16 -> MXFP8 projection encoder (PR #3218, Apache-2.0, 8-thread load)
  cat > "$fn_dir/cuda_mtp_mxfp8.zig" <<'ZIG'
//! Re-codes the MTP drafter's BF16 linears and HC mixers to group-32 UE8M0 MXFP8 at load; target verify stays untouched.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const projection = @import("cuda_projection.zig");

const top_code: f32 = 448.0;
const group = 32;

fn value(bf16: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, bf16[i * 2 ..][0..2], .little)) << 16);
}

fn exponentFor(top: f32) i32 {
    if (!(top > 0) or !std.math.isFinite(top)) return 0;
    var e: i32 = @intFromFloat(@ceil(@log2(top / top_code)));
    while (std.math.ldexp(top, -e) > top_code) e += 1;
    while (e > -126 and std.math.ldexp(top, -(e - 1)) <= top_code) e -= 1;
    return std.math.clamp(e, -126, 127);
}

const Job = struct {
    bf16: []const u8,
    codes: []u8,
    scales: []u8,
    groups: usize,
    per: usize,

    fn worker(ctx: *const Job, tid: usize) void {
        const lo = @min(ctx.groups, tid * ctx.per);
        const hi = @min(ctx.groups, lo + ctx.per);
        for (lo..hi) |g| {
            const base = g * group;
            var top: f32 = 0;
            for (0..group) |i| top = @max(top, @abs(value(ctx.bf16, base + i)));
            const e = exponentFor(top);
            ctx.scales[g] = @intCast(127 + e);
            for (0..group) |i| ctx.codes[base + i] = core.exlr8.codec.e4m3FromF32(std.math.ldexp(value(ctx.bf16, base + i), -e));
        }
    }
};

pub fn encode(bf16: []const u8, n: usize, k: usize, codes: []u8, scales: []u8) void {
    std.debug.assert(k % group == 0 and bf16.len == n * k * 2 and codes.len == n * k and scales.len == n * k / group);
    const groups = n * k / group;
    const n_th: usize = if (groups >= 4096) 8 else 1;
    const job: Job = .{ .bf16 = bf16, .codes = codes, .scales = scales, .groups = groups, .per = (groups + n_th - 1) / n_th };
    if (n_th == 1) {
        Job.worker(&job, 0);
        return;
    }
    var ths: [8]?std.Thread = @splat(null);
    for (0..n_th) |ti| ths[ti] = std.Thread.spawn(.{}, Job.worker, .{ &job, ti }) catch null;
    for (0..n_th) |ti| {
        if (ths[ti]) |t| t.join() else Job.worker(&job, ti);
    }
}

pub fn project(gpa: std.mem.Allocator, driver: *const cuda.Driver, bf16: []const u8, n: usize, k: usize) !projection.Projection {
    if (k % 64 != 0 or bf16.len != n * k * 2) return error.UnexpectedTensor;
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    const scales = try gpa.alloc(u8, n * k / group);
    defer gpa.free(scales);
    encode(bf16, n, k, codes, scales);
    return projection.mxfp8(gpa, driver, .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = codes }, .{ .dtype = .u8, .rank = 2, .shape = .{ n, k / group, 1, 1, 1 }, .bytes = scales });
}
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

  # 2d. Lossless 12.125-bit BF12 8-thread packer (100% bit-exact for every BF16 bit pattern)
  cat > "$fn_dir/cuda_bf12_pack.zig" <<'ZIG'
//! 100% lossless 12.125-bit group-64 delta-exponent BF16 weight packer with sparse 0.01% escape directory.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const projection = @import("cuda_projection.zig");

pub fn enabled() bool {
    const raw = std.c.getenv("TF_FLASHNEXT_TRUNK_BF12") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "1");
}

const Pass1Ctx = struct {
    src: []const u8,
    n: usize,
    k: usize,
    kg: usize,
    n_tiles: usize,
    per: usize,
    w_out: []u8,
    emax_out: []u8,
    esc_counts: []u32,

    inline fn readU16(ctx: *const Pass1Ctx, row: usize, col: usize) u16 {
        const idx = (row * ctx.k + col) * 2;
        return std.mem.readInt(u16, ctx.src[idx..][0..2], .little);
    }

    fn worker(ctx: *const Pass1Ctx, tid: usize) void {
        const lo = @min(ctx.n_tiles, tid * ctx.per);
        const hi = @min(ctx.n_tiles, lo + ctx.per);
        for (lo..hi) |n64| {
            for (0..ctx.kg) |g| {
                const tg = n64 * ctx.kg + g;
                const em_slice = ctx.emax_out[tg * 64 ..][0..64];
                for (0..64) |r| {
                    const row = n64 * 64 + r;
                    if (row >= ctx.n) {
                        em_slice[r] = 0;
                        continue;
                    }
                    var em: u8 = 0;
                    const base = (row * ctx.k + g * 64) * 2;
                    for (0..64) |c| {
                        const u = std.mem.readInt(u16, ctx.src[base + c * 2 ..][0..2], .little);
                        if ((u & 0x7FFF) != 0) {
                            const exp: u8 = @intCast((u >> 7) & 0xFF);
                            if (exp > em) {
                                em = exp;
                            }
                        }
                    }
                    em_slice[r] = em;
                }

                const tile = ctx.w_out[tg * 6144 ..][0..6144];
                var ec: u32 = 0;
                for (0..8) |jj| {
                    const pjj = jj * 768;
                    for (0..32) |lane| {
                        const r = jj * 8 + (lane >> 2);
                        const row = n64 * 64 + r;
                        const em: u16 = em_slice[r];
                        var d01: u32 = 0;
                        var d23: u32 = 0;
                        for (0..4) |kt| {
                            var ms: u32 = 0;
                            for (0..4) |e| {
                                const col = g * 64 + kt * 16 + 2 * (lane & 3) + (e & 1) + 8 * (e >> 1);
                                const u: u16 = if (row < ctx.n) ctx.readU16(row, col) else 0;
                                var msb: u8 = 0;
                                var dn: u32 = 0;
                                if (!(em == 0 and u == 0)) {
                                    const exp: u16 = (u >> 7) & 0xFF;
                                    if ((u & 0x7FFF) == 0 or exp == 0 or (em - exp) >= 15) {
                                        msb = 0;
                                        dn = 15;
                                        ec += 1;
                                    } else {
                                        msb = @truncate(((u >> 8) & 0x80) | (u & 0x7F));
                                        dn = @intCast(em - exp);
                                    }
                                }
                                ms |= @as(u32, msb) << @intCast(8 * e);
                                const sh: u5 = @intCast((kt & 1) * 8 + (e & 1) * 16 + (e >> 1) * 4);
                                if (kt < 2) {
                                    d01 |= dn << sh;
                                } else {
                                    d23 |= dn << sh;
                                }
                            }
                            std.mem.writeInt(u32, tile[pjj + kt * 128 + lane * 4 ..][0..4], ms, .little);
                        }
                        std.mem.writeInt(u32, tile[pjj + 512 + lane * 4 ..][0..4], d01, .little);
                        std.mem.writeInt(u32, tile[pjj + 640 + lane * 4 ..][0..4], d23, .little);
                    }
                }
                ctx.esc_counts[tg] = ec;
            }
        }
    }
};

const Pass2Ctx = struct {
    src: []const u8,
    n: usize,
    k: usize,
    kg: usize,
    n_tiles: usize,
    per: usize,
    emax_out: []const u8,
    dir: []const u32,
    ents: []u32,

    inline fn readU16(ctx: *const Pass2Ctx, row: usize, col: usize) u16 {
        const idx = (row * ctx.k + col) * 2;
        return std.mem.readInt(u16, ctx.src[idx..][0..2], .little);
    }

    fn worker(ctx: *const Pass2Ctx, tid: usize) void {
        const lo = @min(ctx.n_tiles, tid * ctx.per);
        const hi = @min(ctx.n_tiles, lo + ctx.per);
        for (lo..hi) |n64| {
            for (0..ctx.kg) |g| {
                const tg = n64 * ctx.kg + g;
                const d_lo = ctx.dir[tg];
                const d_hi = ctx.dir[tg + 1];
                if (d_lo == d_hi) continue;
                const em_slice = ctx.emax_out[tg * 64 ..][0..64];
                var at: usize = d_lo;
                for (0..8) |jj| {
                    for (0..32) |lane| {
                        const r = jj * 8 + (lane >> 2);
                        const row = n64 * 64 + r;
                        const em: u16 = em_slice[r];
                        for (0..4) |kt| {
                            for (0..4) |e| {
                                const col = g * 64 + kt * 16 + 2 * (lane & 3) + (e & 1) + 8 * (e >> 1);
                                const u: u16 = if (row < ctx.n) ctx.readU16(row, col) else 0;
                                if (!(em == 0 and u == 0)) {
                                    const exp: u16 = (u >> 7) & 0xFF;
                                    if ((u & 0x7FFF) == 0 or exp == 0 or (em - exp) >= 15) {
                                        const key12: u32 = (@as(u32, @intCast(jj)) << 9) | (@as(u32, @intCast(kt)) << 7) | (@as(u32, @intCast(lane)) << 2) | @as(u32, @intCast(e));
                                        ctx.ents[at] = (key12 << 16) | @as(u32, u);
                                        at += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};

pub fn pack(gpa: std.mem.Allocator, driver: *const cuda.Driver, t: core.safetensors.Tensor) !projection.Projection {
    if (t.rank != 2 or t.dtype != .bf16 or t.dim(0) == 0 or t.dim(1) == 0 or t.dim(1) % 64 != 0)
        return error.UnexpectedTensor;
    const n = t.dim(0);
    const k = t.dim(1);
    const count = try std.math.mul(usize, n, k);
    if (t.bytes.len != try std.math.mul(usize, count, 2)) return error.UnexpectedTensor;

    const npad = cuda.nvfp4.padded(n);
    const n_tiles = npad / 64;
    const kg = k / 64;
    const total_tiles = n_tiles * kg;

    const w_host = try gpa.alloc(u8, total_tiles * 6144);
    defer gpa.free(w_host);
    const emax_host = try gpa.alloc(u8, total_tiles * 64);
    defer gpa.free(emax_host);
    const esc_counts = try gpa.alloc(u32, total_tiles);
    defer gpa.free(esc_counts);

    const n_th: usize = if (n_tiles >= 8) 8 else n_tiles;
    const p1: Pass1Ctx = .{
        .src = t.bytes,
        .n = n,
        .k = k,
        .kg = kg,
        .n_tiles = n_tiles,
        .per = (n_tiles + n_th - 1) / n_th,
        .w_out = w_host,
        .emax_out = emax_host,
        .esc_counts = esc_counts,
    };
    if (n_th <= 1) {
        Pass1Ctx.worker(&p1, 0);
    } else {
        var ths: [8]?std.Thread = @splat(null);
        for (0..n_th) |ti| ths[ti] = std.Thread.spawn(.{}, Pass1Ctx.worker, .{ &p1, ti }) catch null;
        for (0..n_th) |ti| {
            if (ths[ti]) |tr| tr.join() else Pass1Ctx.worker(&p1, ti);
        }
    }

    const dir_off = (total_tiles * 64 + 15) & ~@as(usize, 15);
    var total_esc: usize = 0;
    for (esc_counts) |c| total_esc += c;
    const s_total_bytes = dir_off + (total_tiles + 1 + total_esc) * 4;
    const s_words = try gpa.alloc(u32, s_total_bytes / 4);
    defer gpa.free(s_words);
    const s_host = std.mem.sliceAsBytes(s_words);
    @memcpy(s_host[0 .. total_tiles * 64], emax_host);
    if (dir_off > total_tiles * 64) {
        @memset(s_host[total_tiles * 64 .. dir_off], 0);
    }

    const dir_idx = dir_off / 4;
    const dir = s_words[dir_idx..][0 .. total_tiles + 1];
    var acc: u32 = 0;
    for (0..total_tiles) |tg| {
        dir[tg] = acc;
        acc += esc_counts[tg];
    }
    dir[total_tiles] = acc;

    if (total_esc > 0) {
        const ents = s_words[dir_idx + total_tiles + 1 ..][0..total_esc];
        const p2: Pass2Ctx = .{
            .src = t.bytes,
            .n = n,
            .k = k,
            .kg = kg,
            .n_tiles = n_tiles,
            .per = (n_tiles + n_th - 1) / n_th,
            .emax_out = emax_host,
            .dir = dir,
            .ents = ents,
        };
        if (n_th <= 1) {
            Pass2Ctx.worker(&p2, 0);
        } else {
            var ths: [8]?std.Thread = @splat(null);
            for (0..n_th) |ti| ths[ti] = std.Thread.spawn(.{}, Pass2Ctx.worker, .{ &p2, ti }) catch null;
            for (0..n_th) |ti| {
                if (ths[ti]) |tr| tr.join() else Pass2Ctx.worker(&p2, ti);
            }
        }
    }

    var p = projection.Projection.empty(driver);
    p.w = try cuda.DeviceBuffer.fromHost(driver, w_host);
    errdefer p.w.free();
    p.s = try cuda.DeviceBuffer.fromHost(driver, s_host);
    p.n = n;
    p.k = k;
    p.native = .{
        .mode = .bf12,
        .codes = p.w.ptr,
        .scales = p.s.ptr,
        .scale = 1.0,
        .n = @intCast(n),
        .k = @intCast(k),
        .npad = @intCast(npad),
    };
    return p;
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

  # 5. Wire Zig call sites cleanly across cuda/ and families/flashnext/
  python3 - "$fn_dir" "$cuda_dir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
cdir = Path(sys.argv[2])

def patch_exact(base: Path, rel: str, old: str, new: str) -> None:
    p = base / rel
    txt = p.read_text(encoding="utf-8")
    if txt.count(old) != 1:
        raise RuntimeError(f"Expected exactly 1 match in {rel}, got {txt.count(old)}")
    p.write_text(txt.replace(old, new, 1), encoding="utf-8")

# 5a. zig/src/cuda/qmmf.zig + qmmf_shared.zig: Mode::bf12 (decode, half, prompt, strided) + PR #3475 fused split-K
patch_exact(
    cdir, "qmmf.zig",
    "pub const Mode = enum(u3) { fp4 = 0, fp8 = 1, mxfp8 = 2, fp8g = 3, bf16 = 4 };\n\nconst modes = [_]Mode{ .fp4, .fp8, .mxfp8, .fp8g, .bf16 };",
    "pub const Mode = enum(u3) { fp4 = 0, fp8 = 1, mxfp8 = 2, fp8g = 3, bf16 = 4, bf12 = 5 };\n\nconst modes = [_]Mode{ .fp4, .fp8, .mxfp8, .fp8g, .bf16, .bf12 };",
)
patch_exact(
    cdir, "qmmf.zig",
    """        .bf16 => bn * group * 2,
    };
    const s: u32 = if (mode == .fp4 or mode == .fp8g) bn * 4 else if (mode == .mxfp8) bn * 2 else 0;""",
    """        .bf16 => bn * group * 2,
        .bf12 => (bn * group * 3) / 2,
    };
    const s: u32 = if (mode == .fp4 or mode == .fp8g) bn * 4 else if (mode == .mxfp8) bn * 2 else if (mode == .bf12) bn else 0;""",
)
patch_exact(
    cdir, "qmmf.zig",
    """pub fn halfShared(mode: Mode) u32 {
    const w: u32 = if (mode == .bf16) 32 * group * 2 else 32 * group;
    const s: u32 = if (mode == .mxfp8) 32 * 2 else 0;
    const stage = (16 * group * 2 + w + s + 127) / 128 * 128;
    const partials = 1 * 1 * 4 * threads * 4;
    return @max(4 * stage, partials);
}""",
    """pub fn halfShared(mode: Mode) u32 {
    const w: u32 = if (mode == .bf16) 32 * group * 2 else if (mode == .bf12) (32 * group * 3) / 2 else 32 * group;
    const s: u32 = if (mode == .mxfp8) 32 * 2 else if (mode == .bf12) 32 else 0;
    const stage = (16 * group * 2 + w + s + 127) / 128 * 128;
    const partials = 1 * 1 * 4 * threads * 4;
    return @max(4 * stage, partials);
}""",
)
patch_exact(
    cdir, "qmmf.zig",
    """pub fn halfModeIndex(mode: Mode) ?usize {
    return switch (mode) {
        .mxfp8 => 0,
        .bf16 => 1,
        else => null,
    };
}""",
    """pub fn halfModeIndex(mode: Mode) ?usize {
    return switch (mode) {
        .mxfp8 => 0,
        .bf16 => 1,
        .bf12 => 2,
        else => null,
    };
}""",
)
patch_exact(
    cdir, "qmmf.zig",
    """pub const prompt_shared: u32 = 3 * (128 + 128) * group * 2 + 128;
pub const prompt_rows: usize = 512;
pub const prompt_k32_shared: u32 = 3 * (128 + 128) * 32 * 2;""",
    """pub const prompt_shared: u32 = 3 * (128 + 128) * group * 2 + 128;
pub const prompt_bf12_shared: u32 = 3 * ((128 * group * 2 + (128 * group * 3) / 2 + 128 + 127) / 128 * 128);
pub const prompt_rows: usize = 512;
pub const prompt_k32_shared: u32 = 3 * (128 + 128) * 32 * 2;""",
)
patch_exact(
    cdir, "qmmf.zig",
    """pub fn promptSymbol(comptime out: Out, comptime fused: bool) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN7tf_qmmf11qmmf_kernelILi4ELi128ELi128ELi2ELi4ELi3ELb{d}ELb0ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{ @intFromBool(out == .f32), @intFromBool(fused) });
}

/// Only large BF16 grids use the prompt tile; the packed allocation contains complete 128-column tiles.
pub fn promptEligible(m: usize, w: Weight) bool {
    return m >= prompt_rows and w.mode == .bf16 and w.n >= 128 and
        (w.k >= 512 or (w.n == 10240 and w.k == 320)) and w.npad % 128 == 0;
}""",
    """pub fn promptSymbol(comptime out: Out, comptime fused: bool) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN7tf_qmmf11qmmf_kernelILi4ELi128ELi128ELi2ELi4ELi3ELb{d}ELb0ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{ @intFromBool(out == .f32), @intFromBool(fused) });
}

pub fn promptBf12Symbol(comptime out: Out, comptime fused: bool) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN7tf_qmmf11qmmf_kernelILi5ELi128ELi128ELi2ELi4ELi3ELb{d}ELb0ELb{d}EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{ @intFromBool(out == .f32), @intFromBool(fused) });
}

/// Only large BF16/BF12 grids use the prompt tile; the packed allocation contains complete 128-column tiles.
pub fn promptEligible(m: usize, w: Weight) bool {
    return m >= prompt_rows and (w.mode == .bf16 or w.mode == .bf12) and w.n >= 128 and
        (w.k >= 512 or (w.n == 10240 and w.k == 320)) and w.npad % 128 == 0;
}""",
)
patch_exact(
    cdir, "qmmf.zig",
    """    half: [2][2][2]Function, // by halfModeIndex, Out, then plain / cluster
    half_below: usize = half_below_default,
    shared_down: [4]Function = undefined,
    prompt: ?[2][2]Function = null, // Out, then plain / fused; loadPrompt() fills it on sm_12x
    prompt_k32: ?[2]Function = null, // Out; unsplit BF16 prompt calls, loadPrompt() fills it on sm_12x""",
    """    half: [3][2][2]Function, // by halfModeIndex (mxfp8, bf16, bf12), Out, then plain / cluster
    half_below: usize = half_below_default,
    shared_down: [4]Function = undefined,
    shared_down_bf12: [4]Function = undefined,
    fuse_s: [3][2][2]Function = undefined, // by halfModeIndex, Out, bm=16/32 (PR #3475)
    fuse_split_n: usize = 0,
    prompt: ?[2][2]Function = null, // Out, then plain / fused; loadPrompt() fills it on sm_12x
    prompt_bf12: ?[2][2]Function = null, // Out, then plain / fused for BF12
    prompt_k32: ?[2]Function = null, // Out; unsplit BF16 prompt calls, loadPrompt() fills it on sm_12x""",
)
patch_exact(
    cdir, "qmmf.zig",
    """        inline for (.{ Mode.mxfp8, Mode.bf16 }, 0..) |mode, mi| inline for (.{ false, true }, 0..) |wide, oi| inline for (.{ false, true }, 0..) |cl, ci| {
            const f = try mod.function(halfSymbol(mode, cl, wide));
            try f.allowDynamicShared(halfShared(mode));
            l.half[mi][oi][ci] = f;
        };
        for (@import("qmmf_shared.zig").symbols, &l.shared_down) |name, *f| {
            f.* = try mod.function(name);
            try f.allowDynamicShared(sharedBytes(.bf16, 64));
        }
        l.half_below = try halfBelow();
        return l;""",
    """        inline for (.{ Mode.mxfp8, Mode.bf16, Mode.bf12 }, 0..) |mode, mi| inline for (.{ false, true }, 0..) |wide, oi| inline for (.{ false, true }, 0..) |cl, ci| {
            const f = try mod.function(halfSymbol(mode, cl, wide));
            try f.allowDynamicShared(halfShared(mode));
            l.half[mi][oi][ci] = f;
        };
        for (@import("qmmf_shared.zig").symbols, &l.shared_down) |name, *f| {
            f.* = try mod.function(name);
            try f.allowDynamicShared(sharedBytes(.bf16, 64));
        }
        for (@import("qmmf_shared.zig").bf12_symbols, &l.shared_down_bf12) |name, *f| {
            f.* = try mod.function(name);
            try f.allowDynamicShared(sharedBytes(.bf12, 64));
        }
        inline for (.{ Mode.mxfp8, Mode.bf16, Mode.bf12 }, 0..) |mode, mi| inline for (.{ false, true }, 0..) |wide, oi| inline for (.{ 16, 32 }, 0..) |bm, bi| {
            const f = try mod.function(symbol(mode, bm, false, true, wide));
            try f.allowDynamicShared(sharedBytes(mode, bm));
            l.fuse_s[mi][oi][bi] = f;
        };
        l.half_below = try halfBelow();
        if (std.c.getenv("TF_FLASHNEXT_FUSED_SPLIT")) |v| {
            if (std.mem.eql(u8, std.mem.span(v), "1")) l.fuse_split_n = 2048;
        }
        return l;""",
)
patch_exact(
    cdir, "qmmf.zig",
    """        l.prompt = prompt;
        l.prompt_k32 = k32;""",
    """        var prompt_bf12: [2][2]Function = undefined;
        inline for (.{ Out.bf16, Out.f32 }, 0..) |out, oi| inline for (.{ false, true }, 0..) |fused, fi| {
            const f = try l.mod.function(promptBf12Symbol(out, fused));
            try f.allowDynamicShared(prompt_bf12_shared);
            prompt_bf12[oi][fi] = f;
        };
        l.prompt = prompt;
        l.prompt_bf12 = prompt_bf12;
        l.prompt_k32 = k32;""",
)
patch_exact(
    cdir, "qmmf.zig",
    """    pub fn usesPrompt(l: Lane, m: usize, w: Weight) bool {
        const p = planFor(m, w, l.clusters);
        return promptEligible(m, w) and !p.cluster and !p.reduce and l.prompt != null and
            2 * promptBlocks(m, w.n) > l.prompt_sms;
    }""",
    """    pub fn usesPrompt(l: Lane, m: usize, w: Weight) bool {
        const p = planFor(m, w, l.clusters);
        const has_p = if (w.mode == .bf12) l.prompt_bf12 != null else l.prompt != null;
        return promptEligible(m, w) and !p.cluster and !p.reduce and has_p and
            2 * promptBlocks(m, w.n) > l.prompt_sms;
    }""",
)
patch_exact(
    cdir, "qmmf.zig",
    """        const prompt = l.usesPrompt(m, w);
        const k32 = prompt and p.sk == 1 and l.prompt_k32 != null;
        const bm: usize = if (prompt) 128 else p.bm;
        const rows_t = (m + bm - 1) / bm;
        const hm = halfModeIndex(w.mode);
        const half = hm != null and p.bm == 16 and !p.fused and rows_t * ((w.n + bn - 1) / bn) * p.sk < l.half_below;
        const f = if (k32) l.prompt_k32.?[@backingInt(out)] else if (prompt) l.prompt.?[@backingInt(out)][@intFromBool(p.fused)] else if (half) l.half[hm.?][@backingInt(out)][@intFromBool(p.cluster)] else if (p.fused) set[6] else set[ti * 2 + @intFromBool(p.cluster)];
        const cols: usize = if (prompt) 128 else if (half) 32 else bn;
        const band = @max(1, @min(rows_t, (12 << 20) / (bm * w.k * 2)));
        var args: launch_.Args = .{};
        args.add(x);
        args.add(w.codes);
        args.add(w.scales);
        args.add(w.scale);
        args.add(y);
        args.add(if (p.reduce) part else @as(u64, 0));
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(w.n)));
        args.add(@as(i32, @intCast(w.k)));
        args.add(@as(i32, @intCast(p.sk)));
        args.add(@as(i32, @intCast(w.npad)));
        args.add(@as(i32, @intCast(if (m == 1) w.k else ldx)));
        args.add(@as(i32, @intCast(band)));
        try launch_.launch(f, .{
            .grid = .{ .x = @intCast(rows_t * ((w.n + cols - 1) / cols)), .y = 1, .z = if (p.fused) 1 else p.sk },
            .block = .{ .x = if (prompt) 256 else threads, .y = 1, .z = 1 },
            .shared = if (k32) prompt_k32_shared else if (prompt) prompt_shared else if (half) halfShared(w.mode) else sharedBytes(w.mode, p.bm),
            .cluster = if (p.cluster) .{ .x = 1, .y = 1, .z = p.sk } else null,
        }, s, &args);""",
    """        const prompt = l.usesPrompt(m, w);
        const k32 = prompt and w.mode == .bf16 and p.sk == 1 and l.prompt_k32 != null;
        const bm: usize = if (prompt) 128 else p.bm;
        const rows_t = (m + bm - 1) / bm;
        const hm = halfModeIndex(w.mode);
        const fs = l.fuse_split_n != 0 and p.cluster and w.n >= l.fuse_split_n and hm != null;
        const half = hm != null and p.bm == 16 and !p.fused and !fs and rows_t * ((w.n + bn - 1) / bn) * p.sk < l.half_below;
        const f = if (k32)
            l.prompt_k32.?[@backingInt(out)]
        else if (prompt)
            (if (w.mode == .bf12) l.prompt_bf12.?[@backingInt(out)][@intFromBool(p.fused)] else l.prompt.?[@backingInt(out)][@intFromBool(p.fused)])
        else if (half)
            l.half[hm.?][@backingInt(out)][@intFromBool(p.cluster)]
        else if (fs)
            (if (p.bm == 64) set[6] else l.fuse_s[hm.?][@backingInt(out)][@intFromBool(p.bm == 32)])
        else if (p.fused)
            set[6]
        else
            set[ti * 2 + @intFromBool(p.cluster)];
        const cols: usize = if (prompt) 128 else if (half) 32 else bn;
        const band = @max(1, @min(rows_t, (12 << 20) / (bm * w.k * 2)));
        var args: launch_.Args = .{};
        args.add(x);
        args.add(w.codes);
        args.add(w.scales);
        args.add(w.scale);
        args.add(y);
        args.add(if (p.reduce) part else @as(u64, 0));
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(w.n)));
        args.add(@as(i32, @intCast(w.k)));
        args.add(@as(i32, @intCast(p.sk)));
        args.add(@as(i32, @intCast(w.npad)));
        args.add(@as(i32, @intCast(if (m == 1) w.k else ldx)));
        args.add(@as(i32, @intCast(band)));
        try launch_.launch(f, .{
            .grid = .{ .x = @intCast(rows_t * ((w.n + cols - 1) / cols)), .y = 1, .z = if (p.fused or fs) 1 else p.sk },
            .block = .{ .x = if (prompt) 256 else threads, .y = 1, .z = 1 },
            .shared = if (k32) prompt_k32_shared else if (prompt) (if (w.mode == .bf12) prompt_bf12_shared else prompt_shared) else if (half) halfShared(w.mode) else sharedBytes(w.mode, p.bm),
            .cluster = if (p.cluster and !fs) .{ .x = 1, .y = 1, .z = p.sk } else null,
        }, s, &args);""",
)

patch_exact(
    cdir, "qmmf_shared.zig",
    """pub const symbols = [4][:0]const u8{ "tf_qmmf_shared16_half", "tf_qmmf_shared16", "tf_qmmf_shared32", "tf_qmmf_shared64" };

/// The launch takes it: BF16, one K slice, a shape and strides (`ldx` in, `ldy` out) it addresses; else fall back.
pub fn eligible(l: q.Lane, m: usize, w: q.Weight, ldx: usize, ldy: usize) bool {
    if (m == 0 or w.n == 0 or w.k == 0 or w.k % q.group != 0 or w.npad < w.n or w.npad % 128 != 0) return false;
    if (ldy < w.n or ldx < w.k or (m > 1 and ldx % 8 != 0)) return false; // rows of x load as 16-byte copies
    return w.mode == .bf16 and q.planFor(m, w, l.clusters).sk == 1;
}

pub fn matmulF32Strided(l: q.Lane, s: Stream, x: u64, ldx: usize, m: usize, w: q.Weight, y: u64, ldy: usize) !void {
    if (!eligible(l, m, w, ldx, ldy)) return error.StridedLaneNotSupported;
    const bm: usize = q.bucket(m);
    const rows = (m + bm - 1) / bm;
    const half = bm == 16 and rows * ((w.n + 63) / 64) < l.half_below;
    const index: usize = if (half) 0 else if (bm == 16) 1 else if (bm == 32) 2 else 3;
    const cols: usize = if (half) 32 else 64;
    const band = @max(1, @min(rows, (12 << 20) / (bm * w.k * 2)));
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
    }, s, &a);
}""",
    """pub const symbols = [4][:0]const u8{ "tf_qmmf_shared16_half", "tf_qmmf_shared16", "tf_qmmf_shared32", "tf_qmmf_shared64" };
pub const bf12_symbols = [4][:0]const u8{ "tf_qmmf_shared_bf12_16_half", "tf_qmmf_shared_bf12_16", "tf_qmmf_shared_bf12_32", "tf_qmmf_shared_bf12_64" };

/// The launch takes it: BF16 or BF12, one K slice, a shape and strides (`ldx` in, `ldy` out) it addresses; else fall back.
pub fn eligible(l: q.Lane, m: usize, w: q.Weight, ldx: usize, ldy: usize) bool {
    if (m == 0 or w.n == 0 or w.k == 0 or w.k % q.group != 0 or w.npad < w.n or w.npad % 128 != 0) return false;
    if (ldy < w.n or ldx < w.k or (m > 1 and ldx % 8 != 0)) return false; // rows of x load as 16-byte copies
    return (w.mode == .bf16 or w.mode == .bf12) and q.planFor(m, w, l.clusters).sk == 1;
}

pub fn matmulF32Strided(l: q.Lane, s: Stream, x: u64, ldx: usize, m: usize, w: q.Weight, y: u64, ldy: usize) !void {
    if (!eligible(l, m, w, ldx, ldy)) return error.StridedLaneNotSupported;
    const bm: usize = q.bucket(m);
    const rows = (m + bm - 1) / bm;
    const half = bm == 16 and rows * ((w.n + 63) / 64) < l.half_below;
    const index: usize = if (half) 0 else if (bm == 16) 1 else if (bm == 32) 2 else 3;
    const cols: usize = if (half) 32 else 64;
    const band = @max(1, @min(rows, (12 << 20) / (bm * w.k * 2)));
    var a: launch.Args = .{};
    a.add(x);
    a.add(w.codes);
    if (w.mode == .bf12) a.add(w.scales);
    a.add(w.scale);
    a.add(y);
    for ([_]usize{ m, w.n, w.k, w.npad, if (m == 1) w.k else ldx, band, ldy }) |v| a.add(@as(i32, @intCast(v)));
    const f = if (w.mode == .bf12) l.shared_down_bf12[index] else l.shared_down[index];
    try launch.launch(f, .{
        .grid = .{ .x = @intCast(rows * ((w.n + cols - 1) / cols)) },
        .block = .{ .x = 128 },
        .shared = if (half) q.halfShared(w.mode) else q.sharedBytes(w.mode, @intCast(bm)),
    }, s, &a);
}""",
)

# 5b. cuda_projection.zig: route target dense() -> lossless BF12 when TF_FLASHNEXT_TRUNK_BF12=1
patch_exact(
    root, "cuda_projection.zig",
    """pub fn dense(gpa: std.mem.Allocator, driver: *const cuda.Driver, t: core.safetensors.Tensor) !Projection {
    if (t.rank != 2 or t.dtype != .bf16 or t.dim(0) == 0 or t.dim(1) == 0 or t.dim(1) % 64 != 0)""",
    """pub fn dense(gpa: std.mem.Allocator, driver: *const cuda.Driver, t: core.safetensors.Tensor) !Projection {
    if (@import("cuda_bf12_pack.zig").enabled()) return @import("cuda_bf12_pack.zig").pack(gpa, driver, t);
    if (t.rank != 2 or t.dtype != .bf16 or t.dim(0) == 0 or t.dim(1) == 0 or t.dim(1) % 64 != 0)""",
)

# 5c. cuda_projection_load.zig + cuda_prompt_wt.zig: MTP BF16 projections and MTP HC mixers -> MXFP8
for storage, original in (
    ("t.bytes", "            return projection.dense(a, d, t);"),
    ("bytes", "        return projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = bytes });"),
):
    indent = original[:len(original) - len(original.lstrip())]
    check = 'if (std.mem.startsWith(u8, prefix, "language_model.mtp."))'
    project = f'return @import("cuda_mtp_mxfp8.zig").project(a, d, {storage}, n, k);'
    patch_exact(root, "cuda_projection_load.zig", original, f"{indent}{check}\n{indent}    {project}\n{original}")

patch_exact(
    root, "cuda_prompt_wt.zig",
    """    var dq = try projection.dense(gpa, driver, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, wide, 1, 1, 1 }, .bytes = data });
    errdefer dq.free();
    const up = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_up.weight", .{prefix}));
    if (!up.is(.bf16, &.{ wide, low })) return error.UnexpectedTensor;
    var uq = try projection.dense(gpa, driver, up);""",
    """    const mtp_mx = std.mem.startsWith(u8, prefix, "language_model.mtp.");
    var dq = if (mtp_mx)
        try @import("cuda_mtp_mxfp8.zig").project(gpa, driver, data, n, wide)
    else
        try projection.dense(gpa, driver, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, wide, 1, 1, 1 }, .bytes = data });
    errdefer dq.free();
    const up = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_up.weight", .{prefix}));
    if (!up.is(.bf16, &.{ wide, low })) return error.UnexpectedTensor;
    var uq = if (mtp_mx)
        try @import("cuda_mtp_mxfp8.zig").project(gpa, driver, up.bytes, wide, low)
    else
        try projection.dense(gpa, driver, up);""",
)

# 5d. cuda_mtp_pieces.zig: direct sk=1 for native fc_e and fc_h (avoid zero-slice memset + sumSlices)
patch_exact(
    root, "cuda_mtp_pieces.zig",
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
            // the hidden branch: the four streams normed as one row, fc_hidden on each stream
            try t.rmsnorm(r.input, hh.norm_h.ptr, ps.norm, ps.gq, eps, wide, m, wide, wide);
            if (hh.fc_h.native != null) {
                _ = try pp.lane(ps.norm, ps.gq, hh.fc_h, ps.ple_gated, ps.dummy, streams * m, 1);
            } else {
                const hsk = try pp.lane(ps.norm, ps.gq, hh.fc_h, ps.dummy, ps.part, streams * m, fc_sk);
                try pp.sumSlices(ps.part, ps.ple_gated, true, streams * m * dims, hsk);
            }""",
)
patch_exact(
    root, "cuda_mtp_pieces.zig",
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
            // each lane's rows of main-model streams, side by side (the head reads them as one block of rows)
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

# 5e. cuda_ple.zig + cuda_mtp_draft.zig + cuda_lanes.zig + cuda_lanes_draft.zig + cuda_mtp_ops.zig: 32K PLE cache + pre-gather
patch_exact(
    root, "cuda_ple.zig",
    "const ple_fp8 = @import(\"cuda_ple_fp8.zig\");",
    "const ple_fp8 = @import(\"cuda_ple_fp8.zig\");\nconst ple_cache = @import(\"cuda_ple_cache.zig\");",
)
patch_exact(
    root, "cuda_ple.zig",
    "    pool: ?*ple_pool.Pool = null, // the gather's parked workers (null: every row on the caller's thread)\n    warmer: ?std.Thread = null, // warm's read-ahead thread until joined (waitWarm, deinit)",
    "    pool: ?*ple_pool.Pool = null, // the gather's parked workers (null: every row on the caller's thread)\n    warmer: ?std.Thread = null, // warm's read-ahead thread until joined (waitWarm, deinit)\n    cache: ?*ple_cache.Cache = null,",
)
patch_exact(
    root, "cuda_ple.zig",
    "        g.pool = try ple_pool.Pool.create(gpa, io);\n        return g;",
    "        g.pool = try ple_pool.Pool.create(gpa, io);\n        g.cache = ple_cache.Cache.create(gpa) catch null;\n        return g;",
)
patch_exact(
    root, "cuda_ple.zig",
    "        if (g.pool) |p| p.destroy();\n        for (g.files.items) |*o| o.close(g.gpa);",
    "        if (g.cache) |c| c.destroy(g.gpa);\n        if (g.pool) |p| p.destroy();\n        for (g.files.items) |*o| o.close(g.gpa);",
)
patch_exact(
    root, "cuda_ple.zig",
    """    /// `embedding` for each token after `history` (moved past the chunk) into out and xs, read by up to 128 threads.
    pub fn rows(g: *const Ngram, history: []i64, tokens: []const u32, out: []u16, xs: []f32) !void {
        const heads = g.hash.heads;
        const w = g.width();
        if (history.len != g.hash.n - 1 or out.len != tokens.len * w or xs.len != out.len / 32) return error.UnexpectedTensor;""",
    """    /// Synchronously warm one 3-gram row into the in-RAM cache while GPU runs MTP forwardHead.
    pub fn warmToken(g: *const Ngram, hist: *[2]i64, tok: u32) void {
        const cache = g.cache orelse return;
        const h0 = hist[0];
        const h1 = hist[1];
        hist.* = .{ h1, @intCast(tok) };
        if (g.width() != ple_cache.row_w or g.hash.n != 3) return;
        if (cache.has(h0, h1, tok)) return;
        var ids_buf: [max_heads]i64 = undefined;
        const heads = g.hash.heads;
        const mh: [2]i64 = .{ h0, h1 };
        lastIds(g.hash, &mh, @intCast(tok), ids_buf[0..heads]) catch return;
        var emb: [ple_cache.row_w]u16 = undefined;
        if (g.format != .mlx and heads > 1) {
            @import("cuda_ple_gather.zig").run(g, ids_buf[0..heads], &emb) catch return;
        } else {
            g.fill(ids_buf[0..heads], &emb, 0, heads) catch return;
        }
        if (g.format == .mlx and g.scale != 1) {
            for (&emb) |*v| {
                const f: f32 = @bitCast(@as(u32, v.*) << 16);
                v.* = toBf16(f * g.scale);
            }
        }
        var xs_buf: [ple_cache.xs_w]f32 = undefined;
        sumGroups(&emb, &xs_buf);
        cache.put(h0, h1, tok, &emb, &xs_buf);
    }

    /// Warm a batch of lanes' latest tokens into the PLE cache behind forwardHeadMany.
    pub fn warmBatch(g: *const Ngram, ds: anytype, nexts: anytype) void {
        const cache = g.cache orelse return;
        if (g.width() != ple_cache.row_w or g.hash.n != 3) return;
        var miss_h0: [32]i64 = undefined;
        var miss_h1: [32]i64 = undefined;
        var miss_tok: [32]u32 = undefined;
        var n_miss: usize = 0;
        for (ds, nexts) |d, nx| {
            if (nx.len == 0) continue;
            const tok = nx[nx.len - 1];
            const h0 = d.draft_hist[0];
            const h1 = d.draft_hist[1];
            d.draft_hist = .{ h1, @intCast(tok) };
            if (n_miss < miss_tok.len and !cache.has(h0, h1, tok)) {
                miss_h0[n_miss] = h0;
                miss_h1[n_miss] = h1;
                miss_tok[n_miss] = tok;
                n_miss += 1;
            }
        }
        if (n_miss == 0) return;
        const heads = g.hash.heads;
        const items = n_miss * heads;
        const ids = g.gpa.alloc(i64, items) catch return;
        defer g.gpa.free(ids);
        const emb = g.gpa.alloc(u16, n_miss * ple_cache.row_w) catch return;
        defer g.gpa.free(emb);
        for (0..n_miss) |m| {
            const mh: [2]i64 = .{ miss_h0[m], miss_h1[m] };
            lastIds(g.hash, &mh, @intCast(miss_tok[m]), ids[m * heads ..][0..heads]) catch return;
        }
        if (g.format != .mlx and items > 1) {
            @import("cuda_ple_gather.zig").run(g, ids, emb) catch return;
        } else {
            g.fill(ids, emb, 0, items) catch return;
        }
        var xs_tmp: [ple_cache.xs_w]f32 = undefined;
        for (0..n_miss) |m| {
            const row: *[ple_cache.row_w]u16 = emb[m * ple_cache.row_w ..][0..ple_cache.row_w];
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

    /// `embedding` for each token after `history` (moved past the chunk) into out and xs, read by up to 128 threads.
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
    root, "cuda_mtp_draft.zig",
    "    history: [2]i64 = .{ 0, 0 },\n    unused_history: [2]i64 = .{ 0, 0 },",
    "    history: [2]i64 = .{ 0, 0 },\n    draft_hist: [2]i64 = .{ 0, 0 },\n    unused_history: [2]i64 = .{ 0, 0 },",
)
patch_exact(
    root, "cuda_lanes.zig",
    "                if (r.position != l.seq.fed + 1) return error.PositionMismatch;",
    "                if (r.position != l.seq.fed + 1) return error.PositionMismatch;\n                d.draft_hist = l.seq.history;",
)
patch_exact(
    root, "cuda_lanes_draft.zig",
    "        if (r.position != l.seq.fed + 1) return error.PositionMismatch;",
    "        if (r.position != l.seq.fed + 1) return error.PositionMismatch;\n        d.draft_hist = l.seq.history;",
)
patch_exact(
    root, "cuda_mtp_ops.zig",
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
    root, "cuda_mtp_ops.zig",
    "    try o.forwardHeadMany(ds, nexts, rows, true);",
    "    try o.forwardHeadMany(ds, nexts, rows, true);\n    if (@hasField(@TypeOf(o), \"e\")) o.e.ngram.warmBatch(ds, nexts);",
)
patch_exact(
    root, "cuda_mtp_ops.zig",
    "            try o.forwardHeadMany(step_ds[0..n_step], step_nexts[0..n_step], step_rows[0..n_step], true);\n            for (step_ds[0..n_step]) |d| d.drafted += 1;",
    "            try o.forwardHeadMany(step_ds[0..n_step], step_nexts[0..n_step], step_rows[0..n_step], true);\n            for (step_ds[0..n_step]) |d| d.drafted += 1;\n            if (@hasField(@TypeOf(o), \"e\")) o.e.ngram.warmBatch(step_ds[0..n_step], step_nexts[0..n_step]);",
)

# 5f. cuda_mtp.zig + cuda_lanes_load.zig: hook W2 proposal head & share Engine's BF16 lm_head buffer
patch_exact(
    root, "cuda_mtp.zig",
    "    topk_fn: cuda.Function, // fn_row_topk: a row's top_k candidates as drawRow ranks them (cuda_mtp_draw.zig)",
    "    topk_fn: cuda.Function,\n    w2: @import(\"cuda_mtp_w2.zig\").W2 = .{},",
)
patch_exact(
    root, "cuda_mtp.zig",
    "        h.draw_batch = try @import(\"cuda_draw_batch.zig\").Batch.init(gpa, driver, batch_on, h.ids.len);\n        return h;",
    "        h.draw_batch = try @import(\"cuda_draw_batch.zig\").Batch.init(gpa, driver, batch_on, h.ids.len);\n        errdefer h.draw_batch.deinit(gpa);\n        h.w2 = try @import(\"cuda_mtp_w2.zig\").W2.init(gpa, driver, &store, &h.artifact, &h.module);\n        return h;",
)
patch_exact(
    root, "cuda_mtp.zig",
    "    pub fn deinit(h: *Head) void {\n        h.draw_batch.deinit(h.gpa);",
    "    pub fn deinit(h: *Head) void {\n        h.w2.deinit();\n        h.draw_batch.deinit(h.gpa);",
)
patch_exact(
    root, "cuda_mtp.zig",
    "    pub fn project(h: *const Head, pass: anytype, input: u64, sums: u64, out: u64, rows: usize) !void {\n        if (h.draft.native) |weight| {",
    "    pub fn project(h: *const Head, pass: anytype, input: u64, sums: u64, out: u64, rows: usize) !void {\n        if (rows == 1 and h.w2.ready()) return h.w2.project(h, pass.stream, input, out, h.ids.len);\n        if (h.draft.native) |weight| {",
)
patch_exact(
    root, "cuda_lanes_load.zig",
    "        own.head = try mtp.Head.init(gpa, io, driver, &own.stream, dir);",
    "        own.head = try mtp.Head.init(gpa, io, driver, &own.stream, dir);\n        own.head.?.w2.bf16_ptr = own.engine.head_q.b.ptr;",
)

# 5g. cuda_lanes_bytes.zig: reflect NVFP4 MTP expert size in budget check
patch_exact(
    root, "cuda_lanes_bytes.zig",
    "pub fn headFp8(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {",
    "pub fn headFp8(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {\n    if (std.c.getenv(\"TF_FLASHNEXT_MTP_EXPERTS\")) |v| if (std.mem.eql(u8, std.mem.span(v), \"nvfp4\")) return false;",
)

# 5h. cuda_prompt.zig + cuda_batch_kernels.zig + cuda_batch_head.zig: two-stage exact NVFP4 + top-128 BF16 rescore head
patch_exact(
    root, "cuda_prompt.zig",
    "            e.head_q = try @import(\"cuda_projection.zig\").dense(gpa, driver, if (try @import(\"cuda_head.zig\").enabled()) try @import(\"cuda_head.zig\").shard(head_tensor, world, rank) else head_tensor);",
    "            e.head_q = if (world == 1 and !(try @import(\"cuda_head.zig\").enabled()) and @import(\"cuda_head_nvfp4.zig\").enabled())\n                try @import(\"cuda_head_nvfp4.zig\").load(driver, stream.*, &e.store, head_tensor)\n            else\n                try @import(\"cuda_projection.zig\").dense(gpa, driver, if (try @import(\"cuda_head.zig\").enabled()) try @import(\"cuda_head.zig\").shard(head_tensor, world, rank) else head_tensor);",
)
patch_exact(
    root, "cuda_batch_kernels.zig",
    "    select: cuda.Function,\n    select_lanes: cuda.Function,",
    "    select: cuda.Function,\n    select_lanes: cuda.Function,\n    head_topk_floor: cuda.Function,\n    head_rescore: cuda.Function,",
)
patch_exact(
    root, "cuda_batch_kernels.zig",
    "        k.late_roll = try k.batch.function(\"fn_late_roll\");\n        return k;",
    "        k.late_roll = try k.batch.function(\"fn_late_roll\");\n        k.head_topk_floor = try k.batch.function(\"fn_head_topk_floor\");\n        k.head_rescore = try k.batch.function(\"fn_head_rescore\");\n        return k;",
)
patch_exact(
    root, "cuda_batch_head.zig",
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

# 5i. cuda_window_graph.zig: 4-way LRU cache per shape slot to eliminate state-swap & streams/out re-recording
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

patch_exact(root, "cuda_window_graph.zig", old_cache, new_cache)
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
    export TF_FLASHNEXT_TRUNK_BF12="1"
    export TF_FLASHNEXT_FUSED_SPLIT="1"
    export TF_FLASHNEXT_GREEDY_WIDE="1"
    if [[ "$PROFILE" == "guard" || "$PROFILE" == "nvidia" ]]; then
      export TENSORFOLD_MEMORY_RESERVE_GIB=6
      export TF_FLASHNEXT_DRAFTS="6"
      export TF_FLASHNEXT_DRAFTS_GREEDY="14"
      export TF_FLASHNEXT_CONFIDENCE="-0.35"
      CTX="$GUARD_CONTEXT"
      PAR="$GUARD_PARALLEL"
      THINK="--thinking"
    else
      export TF_FLASHNEXT_DRAFTS_GREEDY="14"
      export TF_FLASHNEXT_CONFIDENCE="-0.12"
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
