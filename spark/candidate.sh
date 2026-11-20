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
BUILD_FLAVOR="mtp-w2f-nvfp4-g4-head4-bf12s3-dynexp-pdl3-sk8-wbn4-hcfast-kvp16-while-fusehc-shgu-hbest-s2-gdn4t-h2-w2h-int6s-w1-tkplan3-c31-wbnv8e-cu4-fa1-hc64-r33-w1h-qtr-c256-mint6c-upmix32-hcd32-agate-outsk4-hcs1sc-s5dn-r256-t60"
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

  # 1a. Append fused 2.25-bit E4M3 W2 proposal GEMV + block-max pruned top-32 exact BF16 rescore-to-best + zero-expf argmax to flashnext_mtp.cu
  cat >> "$kern_dir/flashnext_mtp.cu" <<'CU'

extern "C" __global__ void __launch_bounds__(128) fn_w2_gemv_fused(
    const __nv_bfloat16* X,
    const uint4* __restrict__ W,
    const uint16_t* __restrict__ S,
    const float* __restrict__ G,
    __nv_bfloat16* __restrict__ out,
    float4* __restrict__ stats,
    int V) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
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
        sg_sum[tid] = 2.75f * s;
    }
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31;
    const int blk = blockIdx.x * 4 + warp;
    const int row = blk * 32 + lane;
    float fv = -CUDART_INF_F;
    if (blk * 32 < V) {
        const uint4* wp = W + static_cast<size_t>(blk) * 40 * 32 + lane;
        const uint16_t* sp = S + static_cast<size_t>(blk) * 40 * 32 + lane;
        float acc = 0.0f;
        auto ld_w256 = [](const uint4* p) -> uint4 {
            uint4 v;
#if __CUDA_ARCH__ >= 900
            asm volatile("ld.global.nc.L1::evict_first.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#else
            v = __ldg(p);
#endif
            return v;
        };
        uint4 w4 = ld_w256(wp);
        uint32_t s2 = __ldg(sp);
        auto q2f = [](uint32_t word, int j) -> float {
            const uint32_t bits = (j <= 10) ? ((word << (21 - 2 * j)) & 0x00600000u) : ((word >> (2 * j - 21)) & 0x00600000u);
            return __uint_as_float(0x40000000u | bits);
        };
        for (int g2 = 0; g2 < 40; ++g2) {
            uint4 nw4;
            uint32_t ns2;
            if (g2 + 1 < 40) {
                nw4 = ld_w256(wp + (g2 + 1) * 32);
                ns2 = __ldg(sp + (g2 + 1) * 32);
            }
            const uint32_t sx2 = (s2 & 0xFFu) | ((s2 & 0xFF00u) << 8);
            const uint32_t ts = ((sx2 & 0x007F007Fu) << 4) | ((sx2 & 0x00800080u) << 8);
            uint32_t bf2;
            asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(bf2) : "r"(ts), "r"(0x7B807B80u), "r"(0x80008000u));
            const float sc0 = 2.0f * __uint_as_float((bf2 & 0xFFFFu) << 16);
            const float sc1 = 2.0f * __uint_as_float(bf2 & 0xFFFF0000u);
            const int g0 = g2 * 2;
            const float* xg0 = sx + g0 * 32;
            float dot0 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) dot0 = fmaf(xg0[j], q2f(w4.x, j), dot0);
            #pragma unroll
            for (int j = 0; j < 16; ++j) dot0 = fmaf(xg0[16 + j], q2f(w4.y, j), dot0);
            acc = fmaf(sc0, dot0 - sg_sum[g0], acc);
            const float* xg1 = sx + (g0 + 1) * 32;
            float dot1 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) dot1 = fmaf(xg1[j], q2f(w4.z, j), dot1);
            #pragma unroll
            for (int j = 0; j < 16; ++j) dot1 = fmaf(xg1[16 + j], q2f(w4.w, j), dot1);
            acc = fmaf(sc1, dot1 - sg_sum[g0 + 1], acc);
            if (g2 + 1 < 40) {
                w4 = nw4;
                s2 = ns2;
            }
        }
        acc *= __ldg(G + row);
        if (row < V) {
            const __nv_bfloat16 bv = __float2bfloat16_rn(acc);
            out[row] = bv;
            fv = __bfloat162float(bv);
        }
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    float wm = fv;
    int wc = (row < V) ? row : 0x7fffffff;
    for (int o = 16; o; o >>= 1) {
        fn_mtp::better(wm, wc, __shfl_xor_sync(0xffffffffu, wm, o), __shfl_xor_sync(0xffffffffu, wc, o));
    }
    __shared__ float smax[4], ssum[4], sbmax;
    __shared__ int scol[4], sbcol;
    if (lane == 0) {
        smax[warp] = wm;
        scol[warp] = wc;
    }
    __syncthreads();
    if (tid == 0) {
        float bm = smax[0];
        int bc = scol[0];
        for (int w = 1; w < 4; ++w) fn_mtp::better(bm, bc, smax[w], scol[w]);
        sbmax = bm;
        sbcol = bc;
    }
    __syncthreads();
    const float bmax = sbmax;
    float ex = (row < V) ? expf(fv - bmax) : 0.0f;
    for (int o = 16; o; o >>= 1) ex += __shfl_xor_sync(0xffffffffu, ex, o);
    if (lane == 0) ssum[warp] = ex;
    __syncthreads();
    if (tid == 0) {
        stats[blockIdx.x] = make_float4(bmax, ssum[0] + ssum[1] + ssum[2] + ssum[3], __int_as_float(sbcol), 0.0f);
    }
}

extern "C" __global__ void __launch_bounds__(128) fn_w1g32_gemv_fused(
    const __nv_bfloat16* X,
    const uint4* __restrict__ WS,
    const float* __restrict__ G,
    __nv_bfloat16* __restrict__ out,
    float4* __restrict__ stats,
    int V) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
    __shared__ uint32_t sxu[2560];
    const int tid = threadIdx.x;
    for (int i = tid; i < 2560; i += 128) {
        sxu[i] = __float_as_uint(__bfloat162float(X[i]));
    }
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31;
    const int blk = blockIdx.x * 4 + warp;
    const int row = blk * 32 + lane;
    float fv = -CUDART_INF_F;
    if (blk * 32 < V) {
        const uint4* bp = WS + static_cast<size_t>(blk) * 800;
        auto ld4_256 = [](const uint4* p) -> uint4 {
            uint4 v;
#if __CUDA_ARCH__ >= 900
            asm volatile("ld.global.nc.L1::evict_first.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#else
            v = __ldg(p);
#endif
            return v;
        };
        auto ld1_256 = [](const uint32_t* p) -> uint32_t {
            uint32_t v;
#if __CUDA_ARCH__ >= 900
            asm volatile("ld.global.nc.L1::evict_first.L2::256B.u32 %0, [%1];"
                         : "=r"(v) : "l"(p));
#else
            v = __ldg(p);
#endif
            return v;
        };
        float acc0 = 0.0f, acc1 = 0.0f;
        uint4 w4 = ld4_256(bp + lane);
        uint32_t s4 = ld1_256(reinterpret_cast<const uint32_t*>(bp + 32) + lane);
        for (int g4 = 0; g4 < 20; ++g4) {
            uint4 nw4;
            uint32_t ns4;
            if (g4 + 1 < 20) {
                const uint4* np = bp + (g4 + 1) * 40;
                nw4 = ld4_256(np + lane);
                ns4 = ld1_256(reinterpret_cast<const uint32_t*>(np + 32) + lane);
            }
            const uint32_t s01 = s4 & 0xFFFFu, s23 = s4 >> 16;
            const uint32_t sx01 = (s01 & 0xFFu) | ((s01 & 0xFF00u) << 8);
            const uint32_t sx23 = (s23 & 0xFFu) | ((s23 & 0xFF00u) << 8);
            const uint32_t ts01 = ((sx01 & 0x007F007Fu) << 4) | ((sx01 & 0x00800080u) << 8);
            const uint32_t ts23 = ((sx23 & 0x007F007Fu) << 4) | ((sx23 & 0x00800080u) << 8);
            uint32_t bf01, bf23;
            asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(bf01) : "r"(ts01), "r"(0x7B807B80u), "r"(0x80008000u));
            asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(bf23) : "r"(ts23), "r"(0x7B807B80u), "r"(0x80008000u));
            const float sc[4] = {
                __uint_as_float((bf01 & 0xFFFFu) << 16), __uint_as_float(bf01 & 0xFFFF0000u),
                __uint_as_float((bf23 & 0xFFFFu) << 16), __uint_as_float(bf23 & 0xFFFF0000u)
            };
            const uint32_t ws[4] = {w4.x, w4.y, w4.z, w4.w};
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t* xg = sxu + (g4 * 4 + q) * 32;
                const uint32_t word = ws[q];
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; j += 4) {
                    d0 += __uint_as_float(xg[j + 0] ^ ((word << (31 - (j + 0))) & 0x80000000u));
                    d1 += __uint_as_float(xg[j + 1] ^ ((word << (31 - (j + 1))) & 0x80000000u));
                    d2 += __uint_as_float(xg[j + 2] ^ ((word << (31 - (j + 2))) & 0x80000000u));
                    d3 += __uint_as_float(xg[j + 3] ^ ((word << (31 - (j + 3))) & 0x80000000u));
                }
                if (q & 1) acc1 = fmaf(sc[q], (d0 + d1) + (d2 + d3), acc1);
                else acc0 = fmaf(sc[q], (d0 + d1) + (d2 + d3), acc0);
            }
            if (g4 + 1 < 20) {
                w4 = nw4;
                s4 = ns4;
            }
        }
        const float acc = (acc0 + acc1) * __ldg(G + row);
        if (row < V) {
            const __nv_bfloat16 bv = __float2bfloat16_rn(acc);
            out[row] = bv;
            fv = __bfloat162float(bv);
        }
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    float wm = fv;
    int wc = (row < V) ? row : 0x7fffffff;
    for (int o = 16; o; o >>= 1) {
        fn_mtp::better(wm, wc, __shfl_xor_sync(0xffffffffu, wm, o), __shfl_xor_sync(0xffffffffu, wc, o));
    }
    __shared__ float smax[4], ssum[4], sbmax;
    __shared__ int scol[4], sbcol;
    if (lane == 0) {
        smax[warp] = wm;
        scol[warp] = wc;
    }
    __syncthreads();
    if (tid == 0) {
        float bm = smax[0];
        int bc = scol[0];
        for (int w = 1; w < 4; ++w) fn_mtp::better(bm, bc, smax[w], scol[w]);
        sbmax = bm;
        sbcol = bc;
    }
    __syncthreads();
    const float bmax = sbmax;
    float ex = (row < V) ? expf(fv - bmax) : 0.0f;
    for (int o = 16; o; o >>= 1) ex += __shfl_xor_sync(0xffffffffu, ex, o);
    if (lane == 0) ssum[warp] = ex;
    __syncthreads();
    if (tid == 0) {
        stats[blockIdx.x] = make_float4(bmax, ssum[0] + ssum[1] + ssum[2] + ssum[3], __int_as_float(sbcol), 0.0f);
    }
}

extern "C" __global__ void __launch_bounds__(128) fn_w1g64_gemv_fused(
    const __nv_bfloat16* X,
    const uint4* __restrict__ WS,
    const float* __restrict__ G,
    __nv_bfloat16* __restrict__ out,
    float4* __restrict__ stats,
    int V) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
    __shared__ uint32_t sxu[2560];
    const int tid = threadIdx.x;
    for (int i = tid; i < 2560; i += 128) {
        sxu[i] = __float_as_uint(__bfloat162float(X[i]));
    }
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31;
    const int blk = blockIdx.x * 4 + warp;
    const int row = blk * 32 + lane;
    float fv = -CUDART_INF_F;
    if (blk * 32 < V) {
        const uint4* bp = WS + static_cast<size_t>(blk) * 720;
        auto ld4_256 = [](const uint4* p) -> uint4 {
            uint4 v;
#if __CUDA_ARCH__ >= 900
            asm volatile("ld.global.nc.L1::evict_first.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#else
            v = __ldg(p);
#endif
            return v;
        };
        auto ld1_256 = [](const uint32_t* p) -> uint32_t {
            uint32_t v;
#if __CUDA_ARCH__ >= 900
            asm volatile("ld.global.nc.L1::evict_first.L2::256B.u32 %0, [%1];"
                         : "=r"(v) : "l"(p));
#else
            v = __ldg(p);
#endif
            return v;
        };
        float acc0 = 0.0f, acc1 = 0.0f;
        uint4 wa = ld4_256(bp + lane);
        uint4 wb = ld4_256(bp + 32 + lane);
        uint32_t s4 = ld1_256(reinterpret_cast<const uint32_t*>(bp + 64) + lane);
        for (int g4 = 0; g4 < 10; ++g4) {
            uint4 nwa, nwb;
            uint32_t ns4;
            if (g4 + 1 < 10) {
                const uint4* np = bp + (g4 + 1) * 72;
                nwa = ld4_256(np + lane);
                nwb = ld4_256(np + 32 + lane);
                ns4 = ld1_256(reinterpret_cast<const uint32_t*>(np + 64) + lane);
            }
            const uint32_t s01 = s4 & 0xFFFFu, s23 = s4 >> 16;
            const uint32_t sx01 = (s01 & 0xFFu) | ((s01 & 0xFF00u) << 8);
            const uint32_t sx23 = (s23 & 0xFFu) | ((s23 & 0xFF00u) << 8);
            const uint32_t ts01 = ((sx01 & 0x007F007Fu) << 4) | ((sx01 & 0x00800080u) << 8);
            const uint32_t ts23 = ((sx23 & 0x007F007Fu) << 4) | ((sx23 & 0x00800080u) << 8);
            uint32_t bf01, bf23;
            asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(bf01) : "r"(ts01), "r"(0x7B807B80u), "r"(0x80008000u));
            asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(bf23) : "r"(ts23), "r"(0x7B807B80u), "r"(0x80008000u));
            const float sc[4] = {
                __uint_as_float((bf01 & 0xFFFFu) << 16), __uint_as_float(bf01 & 0xFFFF0000u),
                __uint_as_float((bf23 & 0xFFFFu) << 16), __uint_as_float(bf23 & 0xFFFF0000u)
            };
            const uint32_t wlo[4] = {wa.x, wa.z, wb.x, wb.z};
            const uint32_t whi[4] = {wa.y, wa.w, wb.y, wb.w};
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t* xg = sxu + (g4 * 4 + q) * 64;
                const uint32_t w0 = wlo[q], w1 = whi[q];
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                #pragma unroll
                for (int j = 0; j < 32; j += 4) {
                    d0 += __uint_as_float(xg[j + 0] ^ ((w0 << (31 - (j + 0))) & 0x80000000u));
                    d1 += __uint_as_float(xg[j + 1] ^ ((w0 << (31 - (j + 1))) & 0x80000000u));
                    d2 += __uint_as_float(xg[j + 2] ^ ((w0 << (31 - (j + 2))) & 0x80000000u));
                    d3 += __uint_as_float(xg[j + 3] ^ ((w0 << (31 - (j + 3))) & 0x80000000u));
                }
                #pragma unroll
                for (int j = 0; j < 32; j += 4) {
                    d0 += __uint_as_float(xg[32 + j + 0] ^ ((w1 << (31 - (j + 0))) & 0x80000000u));
                    d1 += __uint_as_float(xg[32 + j + 1] ^ ((w1 << (31 - (j + 1))) & 0x80000000u));
                    d2 += __uint_as_float(xg[32 + j + 2] ^ ((w1 << (31 - (j + 2))) & 0x80000000u));
                    d3 += __uint_as_float(xg[32 + j + 3] ^ ((w1 << (31 - (j + 3))) & 0x80000000u));
                }
                if (q & 1) acc1 = fmaf(sc[q], (d0 + d1) + (d2 + d3), acc1);
                else acc0 = fmaf(sc[q], (d0 + d1) + (d2 + d3), acc0);
            }
            if (g4 + 1 < 10) {
                wa = nwa;
                wb = nwb;
                s4 = ns4;
            }
        }
        const float acc = (acc0 + acc1) * __ldg(G + row);
        if (row < V) {
            const __nv_bfloat16 bv = __float2bfloat16_rn(acc);
            out[row] = bv;
            fv = __bfloat162float(bv);
        }
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    float wm = fv;
    int wc = (row < V) ? row : 0x7fffffff;
    for (int o = 16; o; o >>= 1) {
        fn_mtp::better(wm, wc, __shfl_xor_sync(0xffffffffu, wm, o), __shfl_xor_sync(0xffffffffu, wc, o));
    }
    __shared__ float smax[4], ssum[4], sbmax;
    __shared__ int scol[4], sbcol;
    if (lane == 0) {
        smax[warp] = wm;
        scol[warp] = wc;
    }
    __syncthreads();
    if (tid == 0) {
        float bm = smax[0];
        int bc = scol[0];
        for (int w = 1; w < 4; ++w) fn_mtp::better(bm, bc, smax[w], scol[w]);
        sbmax = bm;
        sbcol = bc;
    }
    __syncthreads();
    const float bmax = sbmax;
    float ex = (row < V) ? expf(fv - bmax) : 0.0f;
    for (int o = 16; o; o >>= 1) ex += __shfl_xor_sync(0xffffffffu, ex, o);
    if (lane == 0) ssum[warp] = ex;
    __syncthreads();
    if (tid == 0) {
        stats[blockIdx.x] = make_float4(bmax, ssum[0] + ssum[1] + ssum[2] + ssum[3], __int_as_float(sbcol), 0.0f);
    }
}

extern "C" __global__ void __launch_bounds__(1024) fn_w2_rescore_best(
    const __nv_bfloat16* X,
    const __nv_bfloat16* __restrict__ W_full_bf16,
    const uint32_t* __restrict__ vocab_ids,
    __nv_bfloat16* dlogits,
    const float4* stats,
    int n_blocks,
    int V,
    int k_top,
    float* __restrict__ best_out) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    __shared__ float sx[320 * 9];
    __shared__ float sv[32], ss[32];
    __shared__ int si[32];
    __shared__ unsigned int hist[128];
    __shared__ int act_blk[1024];
    __shared__ int cand_col[64];
    __shared__ float cand_old[64], cand_new[64];
    __shared__ float sgmax, sgsum, srmax;
    __shared__ int srcol, s_blk_cut, s_strict_cut, s_b_cut;
    __shared__ unsigned int s_eq_need, n_cand, n_eq_taken, n_act_blk;

    for (int i = t; i < 2560; i += 1024) {
        sx[(i >> 3) * 9 + (i & 7)] = __bfloat162float(X[i]);
    }
    if (t < 128) hist[t] = 0;

    float bm = -CUDART_INF_F;
    int bad = 0;
    float4 my_st = make_float4(-CUDART_INF_F, 0.0f, 0.0f, 0.0f);
    if (n_blocks <= 1024) {
        if (t < n_blocks) {
            my_st = stats[t];
            bad |= (!isfinite(my_st.x)) | (!isfinite(my_st.y));
            bm = my_st.x;
        }
    } else {
        for (int b = t; b < n_blocks; b += 1024) {
            const float4 st = stats[b];
            bad |= (!isfinite(st.x)) | (!isfinite(st.y));
            bm = fmaxf(bm, st.x);
        }
    }
    for (int o = 16; o; o >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, o));
    if (lane == 0) sv[warp] = bm;
    __syncthreads();
    if (warp == 0) {
        bm = sv[lane];
        for (int o = 16; o; o >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, o));
        if (lane == 0) sgmax = bm;
    }
    const int bad_stats = __syncthreads_or(bad);
    if (bad_stats || !isfinite(sgmax)) {
        if (t == 0) {
            best_out[0] = CUDART_NAN_F;
            best_out[1] = CUDART_NAN_F;
            best_out[2] = 0.0f;
            best_out[3] = 0.0f;
        }
        return;
    }

    const float gmax = sgmax;
    float ws = 0.0f;
    if (n_blocks <= 1024) {
        if (t < n_blocks) ws = my_st.y * expf(my_st.x - gmax);
    } else {
        for (int b = t; b < n_blocks; b += 1024) {
            const float4 st = stats[b];
            ws += st.y * expf(st.x - gmax);
        }
    }
    for (int o = 16; o; o >>= 1) ws += __shfl_xor_sync(0xffffffffu, ws, o);
    if (lane == 0) ss[warp] = ws;
    __syncthreads();
    if (warp == 0) {
        ws = ss[lane];
        for (int o = 16; o; o >>= 1) ws += __shfl_xor_sync(0xffffffffu, ws, o);
        if (lane == 0) sgsum = ws;
    }

    const unsigned short* drow = reinterpret_cast<const unsigned short*>(dlogits);
    float rmax = -CUDART_INF_F;
    int rcol = 0x7fffffff;
    if (n_blocks <= 1024) {
        int my_bb = 127;
        if (t < n_blocks) {
            const float diff = gmax - my_st.x;
            if (diff < 15.75f) {
                my_bb = (my_st.x >= gmax) ? 0 : (1 + min(125, static_cast<int>(diff * 8.0f)));
                atomicAdd(&hist[my_bb], 1u);
            }
        }
        __syncthreads();
        if (t == 0) {
            const unsigned int k_need = static_cast<unsigned int>(k_top);
            unsigned int cum = 0;
            int bc = 127;
            for (int b = 0; b <= 126; ++b) {
                if (cum + hist[b] >= k_need) {
                    bc = b;
                    break;
                }
                cum += hist[b];
            }
            s_blk_cut = bc;
            n_act_blk = 0;
        }
        __syncthreads();
        if (t < 128) hist[t] = 0;
        if (t < n_blocks) {
            if (my_bb <= s_blk_cut) {
                act_blk[atomicAdd(&n_act_blk, 1u)] = t;
            } else {
                fn_mtp::better(rmax, rcol, my_st.x, __float_as_int(my_st.z));
            }
        }
        __syncthreads();
        const int total_act = static_cast<int>(n_act_blk) * 128;
        for (int idx = t; idx < total_act; idx += 1024) {
            const int i = (act_blk[idx >> 7] << 7) + (idx & 127);
            if (i < V) {
                const float v = __uint_as_float(static_cast<uint32_t>(drow[i]) << 16);
                const float diff = gmax - v;
                if (diff < 15.75f) {
                    int b = (v >= gmax) ? 0 : (1 + min(125, static_cast<int>(diff * 8.0f)));
                    atomicAdd(&hist[b], 1u);
                }
            }
        }
    } else {
        for (int i = t; i < V; i += 1024) {
            const float v = __uint_as_float(static_cast<uint32_t>(drow[i]) << 16);
            const float diff = gmax - v;
            if (diff < 15.75f) {
                int b = (v >= gmax) ? 0 : (1 + min(125, static_cast<int>(diff * 8.0f)));
                atomicAdd(&hist[b], 1u);
            }
        }
    }
    __syncthreads();
    if (t == 0) {
        const unsigned int k_need = static_cast<unsigned int>(k_top);
        unsigned int cum = 0;
        int bc = 127;
        for (int b = 0; b <= 126; ++b) {
            if (cum + hist[b] >= k_need) {
                bc = b;
                break;
            }
            cum += hist[b];
        }
        if (bc == 127) {
            s_strict_cut = 127;
            s_b_cut = 127;
            s_eq_need = 0;
        } else {
            s_strict_cut = bc;
            s_b_cut = bc;
            s_eq_need = k_need - cum;
        }
        n_cand = 0;
        n_eq_taken = 0;
    }
    __syncthreads();

    const int sc = s_strict_cut, bc = s_b_cut;
    const unsigned int eq_lim = s_eq_need;
    const unsigned int k_lim = static_cast<unsigned int>(k_top);
    if (n_blocks <= 1024) {
        const int total_act = static_cast<int>(n_act_blk) * 128;
        for (int idx = t; idx < total_act; idx += 1024) {
            const int i = (act_blk[idx >> 7] << 7) + (idx & 127);
            if (i < V) {
                const float v = __uint_as_float(static_cast<uint32_t>(drow[i]) << 16);
                const float diff = gmax - v;
                const int b = (diff < 15.75f) ? ((v >= gmax) ? 0 : (1 + min(125, static_cast<int>(diff * 8.0f)))) : 127;
                bool take = false;
                if (b < sc) {
                    take = true;
                } else if (b == bc && eq_lim > 0u) {
                    if (atomicAdd(&n_eq_taken, 1u) < eq_lim) take = true;
                }
                if (take) {
                    const unsigned int slot = atomicAdd(&n_cand, 1u);
                    if (slot < k_lim) {
                        cand_col[slot] = i;
                        cand_old[slot] = v;
                    } else {
                        fn_mtp::better(rmax, rcol, v, i);
                    }
                } else {
                    fn_mtp::better(rmax, rcol, v, i);
                }
            }
        }
    } else {
        for (int i = t; i < V; i += 1024) {
            const float v = __uint_as_float(static_cast<uint32_t>(drow[i]) << 16);
            const float diff = gmax - v;
            const int b = (diff < 15.75f) ? ((v >= gmax) ? 0 : (1 + min(125, static_cast<int>(diff * 8.0f)))) : 127;
            bool take = false;
            if (b < sc) {
                take = true;
            } else if (b == bc && eq_lim > 0u) {
                if (atomicAdd(&n_eq_taken, 1u) < eq_lim) take = true;
            }
            if (take) {
                const unsigned int slot = atomicAdd(&n_cand, 1u);
                if (slot < k_lim) {
                    cand_col[slot] = i;
                    cand_old[slot] = v;
                } else {
                    fn_mtp::better(rmax, rcol, v, i);
                }
            } else {
                fn_mtp::better(rmax, rcol, v, i);
            }
        }
    }
    for (int o = 16; o; o >>= 1) {
        fn_mtp::better(rmax, rcol, __shfl_xor_sync(0xffffffffu, rmax, o), __shfl_xor_sync(0xffffffffu, rcol, o));
    }
    if (lane == 0) {
        sv[warp] = rmax;
        si[warp] = rcol;
    }
    __syncthreads();
    if (warp == 0) {
        rmax = sv[lane];
        rcol = si[lane];
        for (int o = 16; o; o >>= 1) {
            fn_mtp::better(rmax, rcol, __shfl_xor_sync(0xffffffffu, rmax, o), __shfl_xor_sync(0xffffffffu, rcol, o));
        }
        if (lane == 0) {
            srmax = rmax;
            srcol = rcol;
        }
    }
    __syncthreads();

    const int K = static_cast<int>(min(n_cand, k_lim));
    for (int j = warp; j < K; j += 32) {
        const int c = cand_col[j];
        const uint32_t gid = vocab_ids[c];
        const uint4* wp4 = reinterpret_cast<const uint4*>(W_full_bf16 + static_cast<size_t>(gid) * 2560);
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
        if (lane == 0) {
            const __nv_bfloat16 nb = __float2bfloat16_rn(sum);
            dlogits[c] = nb;
            cand_new[j] = __bfloat162float(nb);
        }
    }
    __syncthreads();

    if (warp == 0) {
        float best_v = (lane == 0) ? srmax : -CUDART_INF_F;
        int best_c = (lane == 0) ? srcol : 0x7fffffff;
        float sub_old = 0.0f;
        int bad_new = 0;
        for (int j = lane; j < K; j += 32) {
            const float nv = cand_new[j];
            bad_new |= !isfinite(nv);
            fn_mtp::better(best_v, best_c, nv, cand_col[j]);
            sub_old += expf(cand_old[j] - gmax);
        }
        const unsigned int any_bad = __ballot_sync(0xffffffffu, bad_new != 0);
        for (int o = 16; o; o >>= 1) {
            fn_mtp::better(best_v, best_c, __shfl_xor_sync(0xffffffffu, best_v, o), __shfl_xor_sync(0xffffffffu, best_c, o));
            sub_old += __shfl_xor_sync(0xffffffffu, sub_old, o);
        }
        const float new_max = __shfl_sync(0xffffffffu, best_v, 0);
        const float rest_sum = fmaxf(sgsum - sub_old, 0.0f) * expf(gmax - new_max);
        float add_new = 0.0f;
        for (int j = lane; j < K; j += 32) {
            add_new += expf(cand_new[j] - new_max);
        }
        for (int o = 16; o; o >>= 1) add_new += __shfl_xor_sync(0xffffffffu, add_new, o);
        if (lane == 0) {
            if (any_bad != 0u || !isfinite(new_max)) {
                best_out[0] = CUDART_NAN_F;
                best_out[1] = CUDART_NAN_F;
                best_out[2] = 0.0f;
                best_out[3] = 0.0f;
            } else {
                const float tot_sum = fmaxf(rest_sum + add_new, 1.0f);
                best_out[0] = new_max;
                best_out[1] = new_max + logf(tot_sum);
                best_out[2] = __int_as_float(best_c);
                best_out[3] = 0.0f;
            }
        }
    }
}

struct W2PickOut {
    int cand_col[64];
    float cand_old[64];
    float cand_new[64];
    float gmax;
    float gsum;
    float rmax;
    int rcol;
    int K;
    unsigned int done_cnt;
};

__device__ __forceinline__ uint16_t fn_bf16_ord(uint16_t u) {
    return (u & 0x8000u) ? static_cast<uint16_t>(~u) : static_cast<uint16_t>(u | 0x8000u);
}

extern "C" __global__ void __launch_bounds__(1024) fn_w2_pick_cands(
    const __nv_bfloat16* dlogits,
    const float4* stats,
    int n_blocks,
    int V,
    int k_top,
    W2PickOut* __restrict__ po,
    float* __restrict__ best_out) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    __shared__ float sv[32], ss[32];
    __shared__ int si[32];
    __shared__ uint32_t wcnt[32];
    __shared__ int act_blk[1024];
    __shared__ float sgmax, sgsum;
    __shared__ uint32_t s_hi, s_tot;
    __shared__ unsigned int n_act_blk, n_cand, n_eq_taken;

    float bm = -CUDART_INF_F;
    int bad = 0;
    float4 my_st = make_float4(-CUDART_INF_F, 0.0f, 0.0f, 0.0f);
    uint16_t my_ord = 0;
    if (t < n_blocks) {
        my_st = stats[t];
        bad |= (!isfinite(my_st.x)) | (!isfinite(my_st.y));
        bm = my_st.x;
        my_ord = fn_bf16_ord(static_cast<uint16_t>(__float_as_uint(bm) >> 16));
    }
    for (int o = 16; o; o >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, o));
    if (lane == 0) sv[warp] = bm;
    __syncthreads();
    if (warp == 0) {
        bm = sv[lane];
        for (int o = 16; o; o >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, o));
        if (lane == 0) {
            sgmax = bm;
            s_hi = fn_bf16_ord(static_cast<uint16_t>(__float_as_uint(bm) >> 16));
            n_act_blk = 0;
        }
    }
    const int bad_stats = __syncthreads_or(bad);
    if (bad_stats || !isfinite(sgmax)) {
        if (t == 0) {
            po->K = 0;
            best_out[0] = CUDART_NAN_F;
            best_out[1] = CUDART_NAN_F;
            best_out[2] = 0.0f;
            best_out[3] = 0.0f;
        }
#if __CUDA_ARCH__ >= 900
        asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
        return;
    }

    const float gmax = sgmax;
    float ws = (t < n_blocks) ? my_st.y * expf(my_st.x - gmax) : 0.0f;
    for (int o = 16; o; o >>= 1) ws += __shfl_xor_sync(0xffffffffu, ws, o);
    if (lane == 0) ss[warp] = ws;
    __syncthreads();
    if (warp == 0) {
        ws = ss[lane];
        for (int o = 16; o; o >>= 1) ws += __shfl_xor_sync(0xffffffffu, ws, o);
        if (lane == 0) sgsum = ws;
    }

    uint32_t hi = s_hi, lo = 0u;
    for (int it = 0; it < 16 && lo < hi; ++it) {
        const uint32_t mid = (lo + hi + 1u) >> 1;
        const uint32_t m = __ballot_sync(0xffffffffu, (t < n_blocks) && (my_ord >= mid));
        if (lane == 0) wcnt[warp] = __popc(m);
        __syncthreads();
        if (warp == 0) {
            uint32_t c = wcnt[lane];
            for (int o = 16; o; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
            if (lane == 0) s_tot = c;
        }
        __syncthreads();
        if (s_tot >= static_cast<uint32_t>(k_top)) lo = mid; else hi = mid - 1u;
    }
    const uint32_t blk_cut = lo;
    float rmax = -CUDART_INF_F;
    int rcol = 0x7fffffff;
    if (t < n_blocks) {
        if (my_ord >= blk_cut) {
            act_blk[atomicAdd(&n_act_blk, 1u)] = t;
        } else {
            fn_mtp::better(rmax, rcol, my_st.x, __float_as_int(my_st.z));
        }
    }
    __syncthreads();

    const unsigned short* drow = reinterpret_cast<const unsigned short*>(dlogits);
    const int total_act = static_cast<int>(n_act_blk) * 128;
    uint16_t loc_ord[16];
    int loc_col[16];
    int n_loc = 0;
    int ovf = 0;
    for (int idx = t; idx < total_act; idx += 1024) {
        const int i = (act_blk[idx >> 7] << 7) + (idx & 127);
        if (i < V) {
            const uint16_t u = drow[i];
            const uint16_t ord = fn_bf16_ord(u);
            if (ord >= blk_cut) {
                if (n_loc < 16) {
                    loc_ord[n_loc] = ord;
                    loc_col[n_loc] = i;
                    ++n_loc;
                } else {
                    ovf = 1;
                }
            } else {
                fn_mtp::better(rmax, rcol, __uint_as_float(static_cast<uint32_t>(u) << 16), i);
            }
        }
    }
    const int any_ovf = __syncthreads_or(ovf);
    hi = s_hi;
    lo = blk_cut;
    for (int it = 0; it < 16 && lo < hi; ++it) {
        const uint32_t mid = (lo + hi + 1u) >> 1;
        uint32_t cnt = 0;
        if (!any_ovf) {
            for (int j = 0; j < n_loc; ++j) cnt += (loc_ord[j] >= mid);
        } else {
            for (int idx = t; idx < total_act; idx += 1024) {
                const int i = (act_blk[idx >> 7] << 7) + (idx & 127);
                if (i < V) cnt += (fn_bf16_ord(drow[i]) >= mid);
            }
        }
        for (int o = 16; o; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
        if (lane == 0) wcnt[warp] = cnt;
        __syncthreads();
        if (warp == 0) {
            uint32_t c = wcnt[lane];
            for (int o = 16; o; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
            if (lane == 0) s_tot = c;
        }
        __syncthreads();
        if (s_tot >= static_cast<uint32_t>(k_top)) lo = mid; else hi = mid - 1u;
    }
    const uint32_t tok_cut = lo;
    uint32_t gt_cnt = 0;
    if (!any_ovf) {
        for (int j = 0; j < n_loc; ++j) gt_cnt += (loc_ord[j] > tok_cut);
    } else {
        for (int idx = t; idx < total_act; idx += 1024) {
            const int i = (act_blk[idx >> 7] << 7) + (idx & 127);
            if (i < V) gt_cnt += (fn_bf16_ord(drow[i]) > tok_cut);
        }
    }
    for (int o = 16; o; o >>= 1) gt_cnt += __shfl_xor_sync(0xffffffffu, gt_cnt, o);
    if (lane == 0) wcnt[warp] = gt_cnt;
    __syncthreads();
    if (warp == 0) {
        uint32_t c = wcnt[lane];
        for (int o = 16; o; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
        if (lane == 0) {
            s_tot = c;
            n_cand = 0;
            n_eq_taken = 0;
        }
    }
    __syncthreads();
    const uint32_t k_lim = static_cast<uint32_t>(min(k_top, 64));
    const uint32_t eq_lim = (k_lim > s_tot) ? (k_lim - s_tot) : 0u;
    if (!any_ovf) {
        for (int j = 0; j < n_loc; ++j) {
            const int i = loc_col[j];
            const float v = __uint_as_float(static_cast<uint32_t>(drow[i]) << 16);
            const bool take = (loc_ord[j] > tok_cut) ||
                (loc_ord[j] == tok_cut && eq_lim > 0u && atomicAdd(&n_eq_taken, 1u) < eq_lim);
            if (take) {
                const unsigned int slot = atomicAdd(&n_cand, 1u);
                if (slot < k_lim) {
                    po->cand_col[slot] = i;
                    po->cand_old[slot] = v;
                } else {
                    fn_mtp::better(rmax, rcol, v, i);
                }
            } else {
                fn_mtp::better(rmax, rcol, v, i);
            }
        }
    } else {
        for (int idx = t; idx < total_act; idx += 1024) {
            const int i = (act_blk[idx >> 7] << 7) + (idx & 127);
            if (i < V) {
                const uint16_t u = drow[i];
                const uint16_t ord = fn_bf16_ord(u);
                const float v = __uint_as_float(static_cast<uint32_t>(u) << 16);
                const bool take = (ord > tok_cut) ||
                    (ord == tok_cut && eq_lim > 0u && atomicAdd(&n_eq_taken, 1u) < eq_lim);
                if (take) {
                    const unsigned int slot = atomicAdd(&n_cand, 1u);
                    if (slot < k_lim) {
                        po->cand_col[slot] = i;
                        po->cand_old[slot] = v;
                    } else {
                        fn_mtp::better(rmax, rcol, v, i);
                    }
                } else {
                    fn_mtp::better(rmax, rcol, v, i);
                }
            }
        }
    }
    for (int o = 16; o; o >>= 1) {
        fn_mtp::better(rmax, rcol, __shfl_xor_sync(0xffffffffu, rmax, o), __shfl_xor_sync(0xffffffffu, rcol, o));
    }
    if (lane == 0) {
        sv[warp] = rmax;
        si[warp] = rcol;
    }
    __syncthreads();
    if (warp == 0) {
        rmax = sv[lane];
        rcol = si[lane];
        for (int o = 16; o; o >>= 1) {
            fn_mtp::better(rmax, rcol, __shfl_xor_sync(0xffffffffu, rmax, o), __shfl_xor_sync(0xffffffffu, rcol, o));
        }
        if (lane == 0) {
            const int K = static_cast<int>(min(n_cand, k_lim));
            po->gmax = gmax;
            po->gsum = sgsum;
            po->rmax = rmax;
            po->rcol = rcol;
            po->K = K;
            if (K == 0) {
                const float tot_sum = fmaxf(sgsum * expf(gmax - rmax), 1.0f);
                best_out[0] = rmax;
                best_out[1] = rmax + logf(tot_sum);
                best_out[2] = __int_as_float(rcol);
                best_out[3] = 0.0f;
            }
        }
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

extern "C" __global__ void __launch_bounds__(128) fn_w2_rescore_multi(
    const __nv_bfloat16* X,
    const __nv_bfloat16* __restrict__ W_full_bf16,
    const uint32_t* __restrict__ vocab_ids,
    __nv_bfloat16* dlogits,
    W2PickOut* __restrict__ po,
    float* __restrict__ best_out) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
    const int K = po->K;
    const int j = blockIdx.x;
    if (j < K) {
        const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
        const int c = po->cand_col[j];
        const uint32_t gid = vocab_ids[c];
        const uint4* wp = reinterpret_cast<const uint4*>(W_full_bf16 + static_cast<size_t>(gid) * 2560);
        const uint4* xp = reinterpret_cast<const uint4*>(X);
        float sum = 0.0f;
        #pragma unroll
        for (int i = tid; i < 320; i += 128) {
            const uint4 wv = __ldg(wp + i);
            const uint4 xv = __ldg(xp + i);
            sum = fmaf(__uint_as_float((xv.x & 0xFFFFu) << 16), __uint_as_float((wv.x & 0xFFFFu) << 16), sum);
            sum = fmaf(__uint_as_float(xv.x & 0xFFFF0000u),     __uint_as_float(wv.x & 0xFFFF0000u), sum);
            sum = fmaf(__uint_as_float((xv.y & 0xFFFFu) << 16), __uint_as_float((wv.y & 0xFFFFu) << 16), sum);
            sum = fmaf(__uint_as_float(xv.y & 0xFFFF0000u),     __uint_as_float(wv.y & 0xFFFF0000u), sum);
            sum = fmaf(__uint_as_float((xv.z & 0xFFFFu) << 16), __uint_as_float((wv.z & 0xFFFFu) << 16), sum);
            sum = fmaf(__uint_as_float(xv.z & 0xFFFF0000u),     __uint_as_float(wv.z & 0xFFFF0000u), sum);
            sum = fmaf(__uint_as_float((xv.w & 0xFFFFu) << 16), __uint_as_float((wv.w & 0xFFFFu) << 16), sum);
            sum = fmaf(__uint_as_float(xv.w & 0xFFFF0000u),     __uint_as_float(wv.w & 0xFFFF0000u), sum);
        }
        for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        __shared__ float sw[4];
        __shared__ int s_is_last;
        if (lane == 0) sw[warp] = sum;
        __syncthreads();
        if (tid == 0) {
            const float tot = sw[0] + sw[1] + sw[2] + sw[3];
            const __nv_bfloat16 nb = __float2bfloat16_rn(tot);
            dlogits[c] = nb;
            po->cand_new[j] = __bfloat162float(nb);
            __threadfence();
            const unsigned int done = atomicAdd(&po->done_cnt, 1u);
            s_is_last = (done == static_cast<unsigned int>(K - 1)) ? 1 : 0;
            if (s_is_last) po->done_cnt = 0u;
        }
        __syncthreads();
        if (s_is_last && warp == 0) {
            const float gmax = po->gmax;
            const float sgsum = po->gsum;
            float best_v = (lane == 0) ? po->rmax : -CUDART_INF_F;
            int best_c = (lane == 0) ? po->rcol : 0x7fffffff;
            float sub_old = 0.0f;
            int bad_new = 0;
            for (int k = lane; k < K; k += 32) {
                const float nv = po->cand_new[k];
                bad_new |= !isfinite(nv);
                fn_mtp::better(best_v, best_c, nv, po->cand_col[k]);
                sub_old += expf(po->cand_old[k] - gmax);
            }
            const unsigned int any_bad = __ballot_sync(0xffffffffu, bad_new != 0);
            for (int o = 16; o; o >>= 1) {
                fn_mtp::better(best_v, best_c, __shfl_xor_sync(0xffffffffu, best_v, o), __shfl_xor_sync(0xffffffffu, best_c, o));
                sub_old += __shfl_xor_sync(0xffffffffu, sub_old, o);
            }
            const float new_max = __shfl_sync(0xffffffffu, best_v, 0);
            const float rest_sum = fmaxf(sgsum - sub_old, 0.0f) * expf(gmax - new_max);
            float add_new = 0.0f;
            for (int k = lane; k < K; k += 32) {
                add_new += expf(po->cand_new[k] - new_max);
            }
            for (int o = 16; o; o >>= 1) add_new += __shfl_xor_sync(0xffffffffu, add_new, o);
            if (lane == 0) {
                if (any_bad != 0u || !isfinite(new_max)) {
                    best_out[0] = CUDART_NAN_F;
                    best_out[1] = CUDART_NAN_F;
                    best_out[2] = 0.0f;
                    best_out[3] = 0.0f;
                } else {
                    const float tot_sum = fmaxf(rest_sum + add_new, 1.0f);
                    best_out[0] = new_max;
                    best_out[1] = new_max + logf(tot_sum);
                    best_out[2] = __int_as_float(best_c);
                    best_out[3] = 0.0f;
                }
            }
        }
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

// Single-pass vectorized uint4 argmax for greedy verify: zero expf() calls, bit-exact tie-breaking & finiteness check.
extern "C" __global__ void __launch_bounds__(1024) fn_row_argmax(
    const __nv_bfloat16* L,
    int V,
    long long ld,
    float* __restrict__ out) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    using namespace fn_mtp;
    const int r = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const __nv_bfloat16* row = L + static_cast<long long>(r) * ld;
    float v = -CUDART_INF_F;
    int at = 0x7fffffff;
    int bad = 0;
    if ((reinterpret_cast<uintptr_t>(row) & 15u) == 0u && (V & 7) == 0) {
        const uint4* row4 = reinterpret_cast<const uint4*>(row);
        const int v8 = V >> 3;
        for (int c = t; c < v8; c += blockDim.x) {
            const uint4 q = row4[c];
            const uint32_t ws[4] = {q.x, q.y, q.z, q.w};
            const int base = c << 3;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t u0 = ws[k] & 0xFFFFu;
                const uint32_t u1 = ws[k] >> 16;
                bad |= ((u0 & 0x7FFFu) > 0x7F80u) | (u0 == 0x7F80u);
                bad |= ((u1 & 0x7FFFu) > 0x7F80u) | (u1 == 0x7F80u);
                better(v, at, __uint_as_float(u0 << 16), base + 2 * k);
                better(v, at, __uint_as_float(u1 << 16), base + 2 * k + 1);
            }
        }
    } else {
        for (int i = t; i < V; i += blockDim.x) {
            const unsigned short u = reinterpret_cast<const unsigned short*>(row)[i];
            bad |= ((u & 0x7FFFu) > 0x7F80u) | (u == 0x7F80u);
            better(v, at, __uint_as_float(static_cast<uint32_t>(u) << 16), i);
        }
    }
    const int row_bad = __syncthreads_or(bad);
    for (int o = 16; o; o >>= 1) {
        better(v, at, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, at, o));
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    if (lane == 0) {
        sv[warp] = v;
        si[warp] = at;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < (int)(blockDim.x >> 5) ? sv[lane] : -CUDART_INF_F;
        at = lane < (int)(blockDim.x >> 5) ? si[lane] : 0x7fffffff;
        for (int o = 16; o; o >>= 1) {
            better(v, at, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, at, o));
        }
        if (lane == 0) {
            if (row_bad) {
                out[r * 4 + 0] = CUDART_NAN_F;
                out[r * 4 + 1] = CUDART_NAN_F;
                out[r * 4 + 2] = __int_as_float(-1);
                out[r * 4 + 3] = 0.0f;
            } else {
                out[r * 4 + 0] = v;
                out[r * 4 + 1] = v;
                out[r * 4 + 2] = __int_as_float(at);
                out[r * 4 + 3] = 0.0f;
            }
        }
    }
}

// Trial 36: Zero-sync device-side MTP chain controller for CU_GRAPH_COND_TYPE_WHILE.
// Runs the entire greedy MTP draft loop on GPU with 1 graph launch + 1 sync per round.
struct MtpChainCtl {
    double c;
    double ext_enter;
    double ext_stay;
    double ext_tok;
    double p_cliff;
    int start_pos;
    int max_steps;
    int vocab_n;
    int pad;
};

struct MtpChainOut {
    double product;
    double probs[32];
    uint32_t tokens[32];
    int n;
    int drafted;
    int err;
    int pad;
};

__device__ __forceinline__ void mtp_stop_take(
    const MtpChainCtl* ctl, double* product, int j, double p, bool& propose, bool& more) {
    const double c = ctl->c;
    if (c > 0.0) {
        const bool low = (p < c);
        if (low && j > 0) {
            propose = false;
            more = false;
            return;
        }
        propose = true;
        more = !low;
        return;
    }
    if (c < 0.0) {
        const double prod = (*product) * p;
        *product = prod;
        if (j >= 15 && (ctl->ext_stay > 0.0 || ctl->ext_tok > 0.0)) {
            if (prod < ctl->ext_stay || p < ctl->ext_tok) {
                propose = false;
                more = false;
                return;
            }
            propose = true;
            more = true;
            return;
        }
        if (j >= 1 && ctl->p_cliff > 0.0 && p < ctl->p_cliff) {
            propose = false;
            more = false;
            return;
        }
        if (prod < -c) {
            propose = (j == 0);
            more = false;
            return;
        }
        if (j >= 1 && ctl->p_cliff < 0.0 && p < -ctl->p_cliff) {
            propose = true;
            more = false;
            return;
        }
        if (j == 14 && ctl->ext_enter > 0.0 && (prod < ctl->ext_enter || p < ctl->ext_tok)) {
            propose = true;
            more = false;
            return;
        }
    }
    propose = true;
    more = true;
}

extern "C" __global__ void __launch_bounds__(32) fn_mtp_chain_init(
    const float* __restrict__ best,
    const uint32_t* __restrict__ vocab_ids,
    int* __restrict__ s_ids,
    int* __restrict__ s_pos,
    const MtpChainCtl* __restrict__ ctl,
    MtpChainOut* __restrict__ out,
    unsigned long long cond_handle) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    out->product = 1.0;
    out->n = 0;
    out->drafted = 0;
    out->err = 0;
    out->pad = 0;
    const float b0 = best[0];
    const float b1 = best[1];
    const int col = __float_as_int(best[2]);
    if (!isfinite(b0) || !isfinite(b1) || col < 0 || col >= ctl->vocab_n) {
        out->err = 1;
        cudaGraphSetConditional((cudaGraphConditionalHandle)cond_handle, 0u);
        return;
    }
    const uint32_t tok = vocab_ids[col];
    const double prob = exp(static_cast<double>(b0) - static_cast<double>(b1));
    bool propose = false, more = false;
    mtp_stop_take(ctl, &out->product, 0, prob, propose, more);
    if (propose) {
        out->tokens[0] = tok;
        out->probs[0] = prob;
        out->n = 1;
    }
    const bool go = propose && more && (1 < ctl->max_steps);
    if (go) {
        s_ids[0] = static_cast<int>(tok);
        s_pos[0] = ctl->start_pos;
        s_pos[1] = 0;
        s_pos[2] = 0;
        s_pos[3] = 0;
    }
    cudaGraphSetConditional((cudaGraphConditionalHandle)cond_handle, go ? 1u : 0u);
}

extern "C" __global__ void __launch_bounds__(32) fn_mtp_chain_step(
    const float* __restrict__ best,
    const uint32_t* __restrict__ vocab_ids,
    int* __restrict__ s_ids,
    int* __restrict__ s_pos,
    const MtpChainCtl* __restrict__ ctl,
    MtpChainOut* __restrict__ out,
    unsigned long long cond_handle) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const int drafted = out->drafted + 1;
    out->drafted = drafted;
    const float b0 = best[0];
    const float b1 = best[1];
    const int col = __float_as_int(best[2]);
    if (!isfinite(b0) || !isfinite(b1) || col < 0 || col >= ctl->vocab_n) {
        out->err = 1;
        cudaGraphSetConditional((cudaGraphConditionalHandle)cond_handle, 0u);
        return;
    }
    const uint32_t tok = vocab_ids[col];
    const double prob = exp(static_cast<double>(b0) - static_cast<double>(b1));
    bool propose = false, more = false;
    mtp_stop_take(ctl, &out->product, drafted, prob, propose, more);
    if (!propose) {
        cudaGraphSetConditional((cudaGraphConditionalHandle)cond_handle, 0u);
        return;
    }
    const int idx = out->n;
    if (idx < 31) {
        out->tokens[idx] = tok;
        out->probs[idx] = prob;
        out->n = idx + 1;
    }
    const bool go = more && (drafted + 1 < ctl->max_steps);
    if (go) {
        s_ids[0] = static_cast<int>(tok);
        s_pos[0] = ctl->start_pos + drafted;
        s_pos[1] = 0;
        s_pos[2] = 0;
        s_pos[3] = 0;
    }
    cudaGraphSetConditional((cudaGraphConditionalHandle)cond_handle, go ? 1u : 0u);
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

extern "C" __global__ void __launch_bounds__(64) fn_bf16_to_w2(
    const uint16_t* __restrict__ bf16,
    float global_scale,
    uint32_t* __restrict__ W_out,
    uint8_t* __restrict__ S_out,
    int N,
    int K,
    int npad) {
    const int n64 = blockIdx.x;
    const int g = blockIdx.y;
    const int KG = K / 64;
    const size_t tile = static_cast<size_t>(n64) * KG + g;
    const int tid = threadIdx.x;
    const int row = n64 * 64 + tid;
    uint8_t q[64];
    uint8_t sc[2] = {0, 0};
    if (row < N) {
        const uint16_t* rp = bf16 + static_cast<size_t>(row) * K + g * 64;
        const float inv_g = 1.0f / global_scale;
        for (int blk = 0; blk < 2; ++blk) {
            float w[32];
            float sq = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const float f = __uint_as_float(static_cast<uint32_t>(rp[blk * 32 + j]) << 16);
                w[j] = f;
                sq += f * f;
            }
            float s = sqrtf(sq * (1.0f / 32.0f)) * 0.9957f;
            if (s < 1e-12f) {
                sc[blk] = 0;
                #pragma unroll
                for (int j = 0; j < 32; ++j) q[blk * 32 + j] = 0;
            } else {
                for (int iter = 0; iter < 3; ++iter) {
                    float num = 0.0f, den = 0.0f;
                    const float inv = 1.0f / fmaxf(s, 1e-12f);
                    #pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        const float raw = rintf(w[j] * inv + 1.5f);
                        const int qi = raw <= 0.0f ? 0 : (raw >= 3.0f ? 3 : static_cast<int>(raw));
                        const float qc = static_cast<float>(qi) - 1.5f;
                        num += w[j] * qc;
                        den += qc * qc;
                    }
                    s = (den > 0.0f) ? fmaxf(num / den, 0.0f) : 0.0f;
                }
                const float target = fminf(fmaxf(s * inv_g, 0.0f), 448.0f);
                const __nv_fp8_storage_t code = __nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3);
                sc[blk] = static_cast<uint8_t>(code);
                const float sv = __half2float(__half(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3))) * global_scale;
                if (sv > 0.0f) {
                    const float inv = 1.0f / sv;
                    #pragma unroll
                    for (int j = 0; j < 32; ++j) {
                        const float raw = rintf(w[j] * inv + 1.5f);
                        q[blk * 32 + j] = static_cast<uint8_t>(raw <= 0.0f ? 0 : (raw >= 3.0f ? 3 : static_cast<int>(raw)));
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 32; ++j) q[blk * 32 + j] = 0;
                }
            }
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 64; ++j) q[j] = 0;
    }
    S_out[tile * 128 + tid * 2 + 0] = sc[0];
    S_out[tile * 128 + tid * 2 + 1] = sc[1];
    const int jj = tid >> 3;
    const int r_in_jj = tid & 7;
    uint32_t* wtile = W_out + tile * 256 + jj * 32;
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        const int lane = (r_in_jj << 2) | sub;
        uint32_t word = 0u;
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int c = kt * 16 + 2 * sub + (e & 1) + 8 * (e >> 1);
                word |= static_cast<uint32_t>(q[c]) << (kt * 8 + e * 2);
            }
        }
        wtile[lane] = word;
    }
}

extern "C" __global__ void __launch_bounds__(64) fn_bf16_to_w1(
    const uint16_t* __restrict__ bf16,
    float global_scale,
    uint16_t* __restrict__ W_out,
    uint8_t* __restrict__ S_out,
    int N,
    int K,
    int npad) {
    const int n64 = blockIdx.x;
    const int g = blockIdx.y;
    const int KG = K / 64;
    const size_t tile = static_cast<size_t>(n64) * KG + g;
    const int tid = threadIdx.x;
    const int row = n64 * 64 + tid;
    uint8_t b[64];
    uint8_t sc = 0;
    if (row < N) {
        const uint16_t* rp = bf16 + static_cast<size_t>(row) * K + g * 64;
        float sum_abs = 0.0f;
        #pragma unroll
        for (int j = 0; j < 64; ++j) {
            const uint16_t u = rp[j];
            const float f = __uint_as_float(static_cast<uint32_t>(u) << 16);
            b[j] = static_cast<uint8_t>((u >> 15) & 1u);
            sum_abs += fabsf(f);
        }
        const float mean_abs = sum_abs * (1.0f / 64.0f);
        const float target = fminf(fmaxf(mean_abs / global_scale, 0.0f), 448.0f);
        sc = static_cast<uint8_t>(__nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3));
    } else {
        #pragma unroll
        for (int j = 0; j < 64; ++j) b[j] = 0;
    }
    S_out[tile * 64 + tid] = sc;
    const int jj = tid >> 3;
    const int r_in_jj = tid & 7;
    uint16_t* wtile = W_out + tile * 256 + jj * 32;
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        const int lane = (r_in_jj << 2) | sub;
        uint16_t word = 0u;
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int c = kt * 16 + 2 * sub + (e & 1) + 8 * (e >> 1);
                word |= static_cast<uint16_t>(b[c]) << (kt * 4 + e);
            }
        }
        wtile[lane] = word;
    }
}

extern "C" __global__ void __launch_bounds__(1024) fn_head_topk_floor(
    __nv_bfloat16* L,
    int V,
    long long ld,
    int K,
    int* __restrict__ out_c,
    int skip_fill) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    const int r = blockIdx.x, t = threadIdx.x, nt = blockDim.x;
    unsigned short* row = reinterpret_cast<unsigned short*>(L + static_cast<long long>(r) * ld);
    int* oc = out_c + static_cast<long long>(r) * K;
    __shared__ unsigned int hist[256];
    __shared__ unsigned short cand_k[2048];
    __shared__ int cand_c[2048];
    __shared__ unsigned int pick_hi, pick_lo, above, n_gt, n_eq;

    auto okey = [](unsigned short u) -> unsigned int {
        const unsigned short uk = (u == 0x8000u) ? 0u : u;
        return (uk & 0x8000u) ? (~uk & 0xFFFFu) : (uk | 0x8000u);
    };

    const bool aligned8 = ((reinterpret_cast<uintptr_t>(row) & 15u) == 0u) && ((V & 7) == 0);
    int bad = 0;
    unsigned int tmax = 0u;
    if (aligned8) {
        const uint4* row4 = reinterpret_cast<const uint4*>(row);
        const int v8 = V >> 3;
        for (int c = t; c < v8; c += nt) {
            const uint4 q = row4[c];
            const uint32_t ws[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const unsigned short u0 = static_cast<unsigned short>(ws[w] & 0xFFFFu);
                const unsigned short u1 = static_cast<unsigned short>(ws[w] >> 16);
                bad |= ((u0 & 0x7FFFu) > 0x7F80u) | (u0 == 0x7F80u);
                bad |= ((u1 & 0x7FFFu) > 0x7F80u) | (u1 == 0x7F80u);
                const unsigned int k0 = okey(u0);
                const unsigned int k1 = okey(u1);
                if (k0 > tmax) tmax = k0;
                if (k1 > tmax) tmax = k1;
            }
        }
    } else {
        for (int i = t; i < V; i += nt) {
            const unsigned short u = row[i];
            bad |= ((u & 0x7FFFu) > 0x7F80u) | (u == 0x7F80u);
            const unsigned int k = okey(u);
            if (k > tmax) tmax = k;
        }
    }
    const int row_bad = __syncthreads_or(bad);
    if (row_bad) {
        if (t == 0) oc[0] = -1;
        return;
    }

    // Pigeonhole lower bound: K-th largest of the 1024 per-thread maxima <= true K-th largest.
    for (int i = t; i < 256; i += nt) hist[i] = 0;
    __syncthreads();
    atomicAdd(&hist[tmax >> 8], 1u);
    __syncthreads();
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
    const unsigned int lb_hi = pick_hi;
    for (int i = t; i < 256; i += nt) hist[i] = 0;
    __syncthreads();
    if ((tmax >> 8) == lb_hi) atomicAdd(&hist[tmax & 255u], 1u);
    __syncthreads();
    if (t == 0) {
        unsigned int cum = above;
        int b = 255;
        for (; b > 0; --b) {
            if (cum + hist[b] >= static_cast<unsigned int>(K)) break;
            cum += hist[b];
        }
        pick_lo = b;
        n_gt = 0;
        n_eq = 0;
    }
    __syncthreads();
    const unsigned int lb = (lb_hi << 8) | pick_lo;

    // Pass 2: only the ~128 threads with tmax >= lb re-read their L1-cached slice to harvest ok >= lb.
    if (tmax >= lb) {
        if (aligned8) {
            const uint4* row4 = reinterpret_cast<const uint4*>(row);
            const int v8 = V >> 3;
            for (int c = t; c < v8; c += nt) {
                const uint4 q = row4[c];
                const uint32_t ws[4] = {q.x, q.y, q.z, q.w};
                const int base = c << 3;
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const unsigned int k0 = okey(static_cast<unsigned short>(ws[w] & 0xFFFFu));
                    if (k0 >= lb) {
                        const unsigned int slot = atomicAdd(&n_gt, 1u);
                        if (slot < 2048u) {
                            cand_k[slot] = static_cast<unsigned short>(k0);
                            cand_c[slot] = base + 2 * w;
                        }
                    }
                    const unsigned int k1 = okey(static_cast<unsigned short>(ws[w] >> 16));
                    if (k1 >= lb) {
                        const unsigned int slot = atomicAdd(&n_gt, 1u);
                        if (slot < 2048u) {
                            cand_k[slot] = static_cast<unsigned short>(k1);
                            cand_c[slot] = base + 2 * w + 1;
                        }
                    }
                }
            }
        } else {
            for (int i = t; i < V; i += nt) {
                const unsigned int k = okey(row[i]);
                if (k >= lb) {
                    const unsigned int slot = atomicAdd(&n_gt, 1u);
                    if (slot < 2048u) {
                        cand_k[slot] = static_cast<unsigned short>(k);
                        cand_c[slot] = i;
                    }
                }
            }
        }
    }
    __syncthreads();

    if (n_gt <= 2048u) {
        const unsigned int nc = n_gt;
        if (nc <= static_cast<unsigned int>(K)) {
            for (int i = t; i < static_cast<int>(nc); i += nt) oc[i] = cand_c[i];
        } else {
            for (int i = t; i < 256; i += nt) hist[i] = 0;
            __syncthreads();
            for (int i = t; i < static_cast<int>(nc); i += nt) atomicAdd(&hist[cand_k[i] >> 8], 1u);
            __syncthreads();
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
            for (int i = t; i < static_cast<int>(nc); i += nt) {
                if ((cand_k[i] >> 8) == hi) atomicAdd(&hist[cand_k[i] & 255u], 1u);
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
                n_eq = 0;
            }
            __syncthreads();
            const unsigned int kth = (hi << 8) | pick_lo;
            const unsigned int high = above;
            const unsigned int need = static_cast<unsigned int>(K) - high;
            for (int i = t; i < static_cast<int>(nc); i += nt) {
                const unsigned int ck = cand_k[i];
                if (ck > kth) {
                    const unsigned int slot = atomicAdd(&n_eq, 1u);
                    if (slot < high) oc[slot] = cand_c[i];
                } else if (ck == kth) {
                    const int col = cand_c[i];
                    unsigned int rank = 0;
                    for (unsigned int j = 0; j < nc; ++j) {
                        if (cand_k[j] == kth && cand_c[j] < col) ++rank;
                    }
                    if (rank < need) oc[high + rank] = col;
                }
            }
        }
    } else {
        for (int i = t; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        for (int i = t; i < V; i += nt) atomicAdd(&hist[okey(row[i]) >> 8], 1u);
        __syncthreads();
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
            const unsigned int ok = okey(row[i]);
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
        const unsigned int kth = (hi << 8) | pick_lo;
        const unsigned int high = above;
        const unsigned int need = static_cast<unsigned int>(K) - high;
        unsigned int* scan = reinterpret_cast<unsigned int*>(cand_c);
        unsigned int* scan_eq = scan + 1024;
        const int per = (V + nt - 1) / nt, lo = t * per, end = min(V, lo + per);
        unsigned int gt = 0, eq = 0;
        for (int i = lo; i < end; ++i) {
            const unsigned int ok = okey(row[i]);
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
            const unsigned int ok = okey(row[i]);
            if (ok > kth) oc[at++] = i;
            else if (ok == kth && rank < need) { oc[high + rank] = i; ++rank; }
        }
    }
    if (skip_fill) return;
    __syncthreads();
    if (aligned8) {
        const uint4 fill4 = make_uint4(0xC680C680u, 0xC680C680u, 0xC680C680u, 0xC680C680u);
        uint4* row4 = reinterpret_cast<uint4*>(row);
        const int v8 = V >> 3;
        for (int i = t; i < v8; i += nt) row4[i] = fill4;
    } else {
        for (int i = t; i < V; i += nt) row[i] = 0xC680u;
    }
}

extern "C" __global__ void __launch_bounds__(1024) fn_head_rescore_best(
    const __nv_bfloat16* X,
    const __nv_bfloat16* __restrict__ W_bf16,
    const int* cols,
    float* __restrict__ best_out,
    int V,
    int K) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
    const int r = blockIdx.y;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    if (cols[static_cast<long long>(r) * K] < 0) {
        if (t == 0) {
            best_out[r * 4 + 0] = __int_as_float(0x7fc00000);
            best_out[r * 4 + 1] = __int_as_float(0x7fc00000);
            best_out[r * 4 + 2] = __int_as_float(-1);
            best_out[r * 4 + 3] = 0.0f;
        }
#if __CUDA_ARCH__ >= 900
        asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
        return;
    }
    __shared__ float sx[320 * 9];
    const __nv_bfloat16* xr = X + static_cast<size_t>(r) * 2560;
    for (int idx = t; idx < 2560; idx += 1024) {
        sx[(idx >> 3) * 9 + (idx & 7)] = __bfloat162float(xr[idx]);
    }
    __syncthreads();
    float bv = __int_as_float(0xff800000);
    int bc = 0x7fffffff;
    int bad = 0;
    for (int j = warp; j < K; j += 32) {
        const int c = cols[static_cast<long long>(r) * K + j];
        if (c < 0 || c >= V) continue;
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
        if (lane == 0) {
            const __nv_bfloat16 bfv = __float2bfloat16_rn(sum);
            const unsigned short u = *reinterpret_cast<const unsigned short*>(&bfv);
            bad |= ((u & 0x7FFFu) > 0x7F80u) | (u == 0x7F80u);
            const float fv = __uint_as_float(static_cast<uint32_t>(u) << 16);
            if (fv > bv || (fv == bv && c < bc)) {
                bv = fv;
                bc = c;
            }
        }
    }
    __shared__ float sv[32];
    __shared__ int sc[32];
    __shared__ int sb[32];
    if (lane == 0) {
        sv[warp] = bv;
        sc[warp] = bc;
        sb[warp] = bad;
    }
    __syncthreads();
    if (warp == 0) {
        float v = sv[lane];
        int c = sc[lane];
        int b = sb[lane];
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oc = __shfl_xor_sync(0xffffffffu, c, o);
            b |= __shfl_xor_sync(0xffffffffu, b, o);
            if (ov > v || (ov == v && oc < c)) {
                v = ov;
                c = oc;
            }
        }
        if (lane == 0) {
            if (b || c < 0 || c >= V) {
                best_out[r * 4 + 0] = __int_as_float(0x7fc00000);
                best_out[r * 4 + 1] = __int_as_float(0x7fc00000);
                best_out[r * 4 + 2] = __int_as_float(-1);
                best_out[r * 4 + 3] = 0.0f;
            } else {
                best_out[r * 4 + 0] = v;
                best_out[r * 4 + 1] = v;
                best_out[r * 4 + 2] = __int_as_float(c);
                best_out[r * 4 + 3] = 0.0f;
            }
        }
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

extern "C" __global__ void __launch_bounds__(256) fn_head_rescore(
    const __nv_bfloat16* X,
    const __nv_bfloat16* __restrict__ W_bf16,
    const int* cols,
    __nv_bfloat16* __restrict__ out,
    int V,
    int K) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    const int r = blockIdx.y;
    if (cols[static_cast<long long>(r) * K] < 0) return;
    __shared__ float sx[320 * 9];
    const __nv_bfloat16* xr = X + static_cast<size_t>(r) * 2560;
    for (int idx = threadIdx.x; idx < 2560; idx += 256) {
        sx[(idx >> 3) * 9 + (idx & 7)] = __bfloat162float(xr[idx]);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * 8 + warp;
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

// SM121 paced hardware TMA L2 prefetch (cp.async.bulk.prefetch.L2.global + globaltimer pacing, cf. TensorFold l2pace.cu).
// Reads constant weights into L2 only and writes nothing, so model bits are 100% unchanged.
extern "C" __global__ void __launch_bounds__(32) fn_l2_prefetch2_paced(
    const uint8_t* p0,
    unsigned long long b0,
    const uint8_t* p1,
    unsigned long long b1,
    unsigned long long ns_per_piece) {
#if __CUDA_ARCH__ >= 900
    if (threadIdx.x != 0) return;
    constexpr unsigned int CHUNK = 32768u;
    const unsigned long long c = blockIdx.x, g = gridDim.x;
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    unsigned long long k = 0;
    auto pf = [&](const uint8_t* p, unsigned long long bytes) {
        if (!p || !bytes) return;
        for (unsigned long long off = 0; off < bytes; off += CHUNK, ++k) {
            if (k % g != c) continue;
            if (ns_per_piece > 0) {
                const unsigned long long due = t0 + k * ns_per_piece;
                while (true) {
                    unsigned long long now;
                    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                    if (now >= due) break;
                    const unsigned long long left = due - now;
                    __nanosleep(static_cast<unsigned int>(left > 2000ull ? 1000u : (left > 64ull ? left / 2ull : 32u)));
                }
            }
            const unsigned long long left = bytes - off;
            const unsigned int sz = static_cast<unsigned int>((left < CHUNK ? left : CHUNK) & ~15ull);
            if (sz) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(p + off), "r"(sz) : "memory");
        }
    };
    pf(p0, b0);
    pf(p1, b1);
#endif
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
    '#include "common/qmm_frag.cuh"',
    '#include "common/qmm_frag.cuh"\n#include "common/glue_math.cuh"',
)

patch_k(
    "qmmf.cu",
    "enum Mode : int { FP4 = 0, FP8 = 1, MXFP8 = 2, FP8G = 3, BF16 = 4 };",
    """enum Mode : int { FP4 = 0, FP8 = 1, MXFP8 = 2, FP8G = 3, BF16 = 4, BF12 = 5, INT6 = 6, W2 = 7, W1 = 8 };

__device__ __forceinline__ void cp16_256(void* dst, const void* src) {
#if __CUDA_ARCH__ >= 900
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;\\n" :: "r"(tf_common::smem(dst)), "l"(src) : "memory");
#else
    tf_common::cp16(dst, src);
#endif
}

__device__ __forceinline__ uint32_t w1pair(uint32_t word, int s) {
    const uint32_t u = word >> s;
    return 0x3F803F80u | ((u & 1u) << 15) | ((u & 2u) << 30);
}

__device__ __forceinline__ uint32_t int6pair(uint32_t lo4, uint32_t hi2) {
    const uint32_t q0 = (lo4 & 0xFu) | ((hi2 & 0x3u) << 4);
    const uint32_t q1 = ((lo4 >> 4) & 0xFu) | (((hi2 >> 2) & 0x3u) << 4);
    const uint32_t t = 0x42804280u | (q0 << 1) | (q1 << 17);
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\\n" : "=r"(r) : "r"(t), "r"(0x3F803F80u), "r"(0xC2BFC2BFu));
    return r;
}

__device__ __forceinline__ uint32_t w2pair(uint32_t word, int s) {
    const uint32_t w = word >> s;
    const uint32_t t = 0x40004000u | ((w & 0x3u) << 4) | ((w & 0xCu) << 18);
    uint32_t r;
#if __CUDA_ARCH__ >= 900
    asm("sub.rn.bf16x2 %0, %1, %2;\\n" : "=r"(r) : "r"(t), "r"(0x40184018u));
#else
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\\n" : "=r"(r) : "r"(t), "r"(0x3F803F80u), "r"(0xC018C018u));
#endif
    return r;
}

__device__ __noinline__ uint32_t dec12_pair_esc(
        uint32_t x, uint32_t d_pair, uint32_t em,
        const uint8_t* __restrict__ bs, size_t total_tiles, size_t tile_idx, uint32_t key_base) {
    const size_t dir_off = (total_tiles * 64 + 15) & ~static_cast<size_t>(15);
    const uint32_t* dir = reinterpret_cast<const uint32_t*>(bs + dir_off);
    const uint32_t* ents = dir + (total_tiles + 1);
    const uint32_t lo_i = __ldg(dir + tile_idx);
    const uint32_t hi_i = __ldg(dir + tile_idx + 1);
    const uint32_t d0 = d_pair & 0xFu, d1 = (d_pair >> 16) & 0xFu;
    uint16_t lo = static_cast<uint16_t>(((x & 0x80u) << 8) | ((em - d0) << 7) | (x & 0x7Fu));
    if (d0 == 0xFu) {
        lo = 0u;
        for (uint32_t idx = lo_i; idx < hi_i; ++idx) {
            const uint32_t e = __ldg(ents + idx);
            if ((e >> 16) == key_base) { lo = static_cast<uint16_t>(e & 0xFFFFu); break; }
        }
    }
    uint16_t hi = static_cast<uint16_t>((((x >> 16) & 0x80u) << 8) | ((em - d1) << 7) | ((x >> 16) & 0x7Fu));
    if (d1 == 0xFu) {
        hi = 0u;
        for (uint32_t idx = lo_i; idx < hi_i; ++idx) {
            const uint32_t e = __ldg(ents + idx);
            if ((e >> 16) == key_base + 1u) { hi = static_cast<uint16_t>(e & 0xFFFFu); break; }
        }
    }
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}""",
)

patch_k(
    "qmmf.cu",
    """    static constexpr int W = MODE == FP4 ? BN * GS / 2 : MODE == BF16 ? BN * GS * 2 : BN * GS;
    static constexpr int S = MODE == FP4 || MODE == FP8G ? BN * 4 : MODE == MXFP8 ? BN * 2 : 0;
    static constexpr int STAGE = (X + W + S + 127) / 128 * 128;
    static constexpr int PARTIALS = MT * NT * 4 * THREADS * 4;
    static constexpr int BARRIERS = MODE == BF16 && BN > 64 ? 128 : 0;""",
    """    static constexpr int W = MODE == FP4 ? BN * GS / 2 : MODE == W2 ? BN * GS / 4 : MODE == W1 ? BN * GS / 8 : MODE == INT6 ? (BN * GS * 3) / 4 : MODE == BF16 ? BN * GS * 2 : MODE == BF12 ? (BN * GS * 3) / 2 : BN * GS;
    static constexpr int S = MODE == FP4 || MODE == FP8G ? BN * 4 : MODE == MXFP8 || MODE == INT6 || MODE == W2 ? BN * 2 : MODE == BF12 || MODE == W1 ? BN : 0;
    static constexpr int STAGE = (X + W + S + 127) / 128 * 128;
    static constexpr int PARTIALS = MT * NT * 4 * THREADS * 4;
    static constexpr int BARRIERS = (MODE == BF16 || MODE == BF12) && BN > 64 ? 128 : 0;""",
)

patch_k(
    "qmmf.cu",
    """TF_SHARED_DOWN(tf_qmmf_shared64, 64, 64)
#undef TF_SHARED_DOWN""",
    """TF_SHARED_DOWN(tf_qmmf_shared64, 64, 64)
#undef TF_SHARED_DOWN

extern "C" __global__ void __launch_bounds__(64) tf_qmmf_shared_bf12_16_qtr(
        const __nv_bfloat16* x, const unsigned char* w, const uint8_t* bs, float scale, float* out,
        int M, int N, int K, int npad, int ldx, int group, int ldy) {
    qmmf_body<BF12, 16, 16, 1, 2, 3, true, false, false>(
        x, w, bs, scale, out, nullptr, M, N, K, 1, npad, ldx, group, ldy);
}

#define TF_SHARED_DOWN_BF12(NAME, BM, BN) \\
extern "C" __global__ void __launch_bounds__(128) NAME( \\
        const __nv_bfloat16* x, const unsigned char* w, const uint8_t* bs, float scale, float* out, \\
        int M, int N, int K, int npad, int ldx, int group, int ldy) { \\
    qmmf_body<BF12, BM, BN, 1, 4, 3, true, false, false>( \\
        x, w, bs, scale, out, nullptr, M, N, K, 1, npad, ldx, group, ldy); \\
}
TF_SHARED_DOWN_BF12(tf_qmmf_shared_bf12_16_half, 16, 32)
TF_SHARED_DOWN_BF12(tf_qmmf_shared_bf12_16, 16, 64)
TF_SHARED_DOWN_BF12(tf_qmmf_shared_bf12_32, 32, 64)
TF_SHARED_DOWN_BF12(tf_qmmf_shared_bf12_64, 64, 64)
#undef TF_SHARED_DOWN_BF12

// Bit-exact fused BF12 hc.up [4*D, 320] + fn_hc_dense_mix8 (caches [BM,320] act once, streams 20 W tiles with weights-first PDL).
template <int BM, int STAGES>
__device__ __forceinline__ void bf12_hc_upmix_body(
        const __nv_bfloat16* x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        const __nv_bfloat16* norm, __nv_bfloat16* __restrict__ mixed, int M, int D) {
    constexpr int BN = 32, WN = 4, GS = 64, KG = 5, SS = 4, CHUNKS = 8, ROW = 128;
    constexpr int X_ALL = KG * BM * ROW;
    constexpr int W = (BN * GS * 3) / 2, S = BN;
    constexpr int WSTAGE = (W + S + 127) / 128 * 128;
    constexpr int THREADS = WN * 32, MT = BM / 16, NT = (BN / WN) / 8;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, wn = tid >> 5, lane = tid & 31;
    const int m0 = blockIdx.x * BM, d0 = blockIdx.y * BN;
    const int d_tiles = D / 64;
    const size_t total_tiles = static_cast<size_t>(SS * d_tiles) * KG;
    const int part = (d0 % 64) / 32;
    const int w_off = part * W, s_off = part * S;
    auto xbuf = [&](int g) { return buf + g * (BM * ROW); };
    auto wstage = [&](int st) { return buf + X_ALL + st * WSTAGE; };
    auto load_w = [&](int st, int step) {
        const int s = step / KG, g = step - s * KG;
        const size_t tbase = static_cast<size_t>(s * d_tiles + (d0 / 64)) * KG + g;
        unsigned char* pw = wstage(st);
        unsigned char* ps = pw + W;
        for (int c = tid; c < W / 16; c += THREADS) cp16_256(pw + c * 16, w + tbase * 6144 + w_off + c * 16);
        for (int c = tid; c < S / 16; c += THREADS) cp16_256(ps + c * 16, bs + tbase * 64 + s_off + c * 16);
    };
    // Weights-first PDL: start fetching step-0 weights before waiting on fn_hc_reduce_act_s1
    load_w(0, 0);
    qmm_frag::grid_wait();
    for (int c = tid; c < KG * BM * CHUNKS; c += THREADS) {
        const int g = c / (BM * CHUNKS), rem = c % (BM * CHUNKS), r = rem / CHUNKS, ch = rem % CHUNKS;
        cp16z(xbuf(g) + r * ROW + swz<CHUNKS>(r, ch) * 16, x + static_cast<size_t>(min(m0 + r, M - 1)) * (KG * GS) + g * GS + ch * 8, m0 + r < M);
    }
    commit();
    #pragma unroll
    for (int st = 1; st < STAGES - 1; ++st) { load_w(st, st); commit(); }

    float mix_tot[MT][NT][4] = {};
    float acc[MT][NT][4];
    for (int step = 0; step < SS * KG; ++step) {
        const int s = step / KG, g = step - s * KG;
        if (g == 0) {
            #pragma unroll
            for (int i = 0; i < MT; ++i)
                #pragma unroll
                for (int j = 0; j < NT; ++j)
                    acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
        }
        wait<STAGES - 2>();
        __syncthreads();
        const int next = step + STAGES - 1;
        if (next < SS * KG) load_w(next % STAGES, next);
        commit();
        const unsigned char* px = xbuf(g);
        const unsigned char* pw = wstage(step % STAGES);
        const uint8_t* ps = pw + W;
        uint4 wq[NT];
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int jj = wn * NT + j;
            const unsigned char* pjj = pw + jj * 768;
            const uint32_t d01 = reinterpret_cast<const uint32_t*>(pjj + 512)[lane];
            const uint32_t d23 = reinterpret_cast<const uint32_t*>(pjj + 640)[lane];
            const uint32_t f01 = d01 & (d01 >> 1) & (d01 >> 2) & (d01 >> 3) & 0x11111111u;
            const uint32_t f23 = d23 & (d23 >> 1) & (d23 >> 2) & (d23 >> 3) & 0x11111111u;
            const uint32_t em_sh = (static_cast<uint32_t>(ps[jj * 8 + (lane >> 2)]) << 7) * 0x00010001u;
            wq[j] = make_uint4(d01, d23, em_sh, f01 | f23);
        }
        #pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            uint32_t a[MT][4];
            #pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ldmatrix4(a[i], px + r * ROW + swz<CHUNKS>(r, kt * 2 + (lane >> 4)) * 16);
            }
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int jj = wn * NT + j;
                const uint32_t ms = reinterpret_cast<const uint32_t*>(pw + jj * 768)[kt * 32 + lane];
                const uint32_t dlo = kt < 2 ? wq[j].x : wq[j].y;
                const int sh = (kt & 1) * 8;
                const uint32_t d0_ = (dlo >> sh) & 0x000F000Fu, d1_ = (dlo >> (sh + 4)) & 0x000F000Fu;
                const uint32_t x0 = __byte_perm(ms, 0u, 0x4140u), x1 = __byte_perm(ms, 0u, 0x4342u);
                uint32_t b0, b1;
                if (__builtin_expect(wq[j].w == 0u, 1)) {
                    b0 = ((x0 & 0x00800080u) << 8) | (wq[j].z - (d0_ << 7)) | (x0 & 0x007F007Fu);
                    b1 = ((x1 & 0x00800080u) << 8) | (wq[j].z - (d1_ << 7)) | (x1 & 0x007F007Fu);
                } else {
                    const int local_jj = ((d0 & 32) ? 4 : 0) + jj;
                    const size_t tile_idx = static_cast<size_t>(s * d_tiles + (d0 / 64)) * KG + g;
                    const uint32_t kbase = (static_cast<uint32_t>(local_jj) << 9) | (static_cast<uint32_t>(kt) << 7) | (static_cast<uint32_t>(lane) << 2);
                    const uint32_t em = (wq[j].z >> 7) & 0xFFu;
                    b0 = dec12_pair_esc(x0, d0_, em, bs, total_tiles, tile_idx, kbase);
                    b1 = dec12_pair_esc(x1, d1_, em, bs, total_tiles, tile_idx, kbase + 2u);
                }
                #pragma unroll
                for (int i = 0; i < MT; ++i) mma(acc[i][j], a[i], b0, b1);
            }
        }
        if (g == KG - 1) {
            #pragma unroll
            for (int i = 0; i < MT; ++i)
                #pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const int col = d0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
                    #pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int row = m0 + i * 16 + (lane >> 2) + h * 8;
                        if (row < M) {
                            const uint32_t nv = *reinterpret_cast<const uint32_t*>(norm + (static_cast<size_t>(row) * SS + s) * D + col);
                            const float n0f = __uint_as_float((nv & 0xFFFFu) << 16);
                            const float n1f = __uint_as_float(nv & 0xFFFF0000u);
                            const float u0 = tf_glue::rbf(acc[i][j][2 * h]);
                            const float u1 = tf_glue::rbf(acc[i][j][2 * h + 1]);
                            mix_tot[i][j][2 * h]     = __fadd_rn(mix_tot[i][j][2 * h],     tf_glue::rbf(__fmul_rn(tf_glue::rbf(tf_glue::tf_sigmoid(u0)), n0f)));
                            mix_tot[i][j][2 * h + 1] = __fadd_rn(mix_tot[i][j][2 * h + 1], tf_glue::rbf(__fmul_rn(tf_glue::rbf(tf_glue::tf_sigmoid(u1)), n1f)));
                        }
                    }
                }
        }
    }
    wait<0>();
    qmm_frag::grid_launch();
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = d0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + i * 16 + (lane >> 2) + h * 8;
                if (row < M) {
                    const float m0f = tf_glue::rbf(__fdiv_rn(mix_tot[i][j][2 * h], 4.0f));
                    const float m1f = tf_glue::rbf(__fdiv_rn(mix_tot[i][j][2 * h + 1], 4.0f));
                    *reinterpret_cast<__nv_bfloat162*>(mixed + static_cast<size_t>(row) * D + col) = __floats2bfloat162_rn(m0f, m1f);
                }
            }
        }
}

extern "C" __global__ void __launch_bounds__(128) tf_qmmf_bf12_hc_upmix_16(
        const __nv_bfloat16* x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        const __nv_bfloat16* norm, __nv_bfloat16* __restrict__ mixed, int M, int D) {
    bf12_hc_upmix_body<16, 4>(x, w, bs, norm, mixed, M, D);
}
extern "C" __global__ void __launch_bounds__(128) tf_qmmf_bf12_hc_upmix_32(
        const __nv_bfloat16* x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        const __nv_bfloat16* norm, __nv_bfloat16* __restrict__ mixed, int M, int D) {
    bf12_hc_upmix_body<32, 3>(x, w, bs, norm, mixed, M, D);
}
extern "C" __global__ void __launch_bounds__(128) tf_qmmf_bf12_hc_upmix_64(
        const __nv_bfloat16* x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        const __nv_bfloat16* norm, __nv_bfloat16* __restrict__ mixed, int M, int D) {
    bf12_hc_upmix_body<64, 2>(x, w, bs, norm, mixed, M, D);
}""",
)

patch_k(
    "qmmf.cu",
    "BF16_PROMPT(true, true)\n#undef BF16_PROMPT",
    """BF16_PROMPT(true, true)
#undef BF16_PROMPT

// Lossless BF12 decode/half/fused split-K (STAGES=3: 4 CTAs/SM at BM=16, 2 CTAs/SM at BM=64) and 128x128 prompt instances.
#define BF12_INST(BM, BN, WM, WN, ST, F32, CL, FU) \\
    template __global__ void qmmf_kernel<BF12, BM, BN, WM, WN, ST, F32, CL, FU>( \\
        const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, \\
        void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
BF12_INST(16, 64, 1, 4, 3, false, false, false)
BF12_INST(16, 64, 1, 4, 2, false, false, false)
BF12_INST(16, 64, 1, 4, 3, false, true, false)
BF12_INST(16, 64, 1, 4, 3, false, false, true)
BF12_INST(32, 64, 1, 4, 3, false, false, false)
BF12_INST(32, 64, 1, 4, 3, false, true, false)
BF12_INST(32, 64, 1, 4, 3, false, false, true)
BF12_INST(64, 64, 1, 4, 3, false, false, false)
BF12_INST(64, 64, 1, 4, 3, false, true, false)
BF12_INST(64, 64, 1, 4, 3, false, false, true)
BF12_INST(16, 64, 1, 4, 3, true, false, false)
BF12_INST(16, 64, 1, 4, 2, true, false, false)
BF12_INST(16, 64, 1, 4, 3, true, true, false)
BF12_INST(16, 64, 1, 4, 3, true, false, true)
BF12_INST(32, 64, 1, 4, 3, true, false, false)
BF12_INST(32, 64, 1, 4, 3, true, true, false)
BF12_INST(32, 64, 1, 4, 3, true, false, true)
BF12_INST(64, 64, 1, 4, 3, true, false, false)
BF12_INST(64, 64, 1, 4, 3, true, true, false)
BF12_INST(64, 64, 1, 4, 3, true, false, true)
BF12_INST(16, 32, 1, 4, 3, false, false, false)
BF12_INST(16, 32, 1, 4, 3, false, true, false)
BF12_INST(16, 32, 1, 4, 3, true, false, false)
BF12_INST(16, 32, 1, 4, 3, true, true, false)
BF12_INST(32, 32, 1, 4, 3, true, false, false)
BF12_INST(128, 128, 2, 4, 3, false, false, false)
BF12_INST(128, 128, 2, 4, 3, false, false, true)
BF12_INST(128, 128, 2, 4, 3, true, false, false)
BF12_INST(128, 128, 2, 4, 3, true, false, true)
#undef BF12_INST

// 6.25-bit INT6 (MTP dense linears, 30.4-32.3 dB SNR, -24.2% DRAM) and 2.25-bit W2 / 1.125-bit W1 coarse Target lm_head instances.
#define INT6_INST(BM, BN, F32, CL, FU) \\
    template __global__ void qmmf_kernel<INT6, BM, BN, 1, 4, 4, F32, CL, FU>( \\
        const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, \\
        void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
INT6_INST(16, 64, false, false, false)
INT6_INST(16, 64, false, true, false)
INT6_INST(32, 64, false, false, false)
INT6_INST(32, 64, false, true, false)
INT6_INST(64, 64, false, false, false)
INT6_INST(64, 64, false, true, false)
INT6_INST(64, 64, false, false, true)
INT6_INST(16, 64, true, false, false)
INT6_INST(16, 64, true, true, false)
INT6_INST(32, 64, true, false, false)
INT6_INST(32, 64, true, true, false)
INT6_INST(64, 64, true, false, false)
INT6_INST(64, 64, true, true, false)
INT6_INST(64, 64, true, false, true)
INT6_INST(16, 32, false, false, false)
INT6_INST(16, 32, false, true, false)
INT6_INST(16, 32, true, false, false)
INT6_INST(16, 32, true, true, false)
#undef INT6_INST

#define W2_INST(BM, F32, CL, FU) \\
    template __global__ void qmmf_kernel<W2, BM, 64, 1, 4, 4, F32, CL, FU>( \\
        const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, \\
        void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
W2_INST(16, false, false, false)
W2_INST(16, false, true, false)
W2_INST(32, false, false, false)
W2_INST(32, false, true, false)
W2_INST(64, false, false, false)
W2_INST(64, false, true, false)
W2_INST(64, false, false, true)
W2_INST(16, true, false, false)
W2_INST(16, true, true, false)
W2_INST(32, true, false, false)
W2_INST(32, true, true, false)
W2_INST(64, true, false, false)
W2_INST(64, true, true, false)
W2_INST(64, true, false, true)
#undef W2_INST

#define W1_INST(BM, F32, CL, FU) \\
    template __global__ void qmmf_kernel<W1, BM, 64, 1, 4, 4, F32, CL, FU>( \\
        const __nv_bfloat16* __restrict__, const unsigned char* __restrict__, const uint8_t* __restrict__, float, \\
        void* __restrict__, float* __restrict__, int, int, int, int, int, int, int);
W1_INST(16, false, false, false)
W1_INST(16, false, true, false)
W1_INST(32, false, false, false)
W1_INST(32, false, true, false)
W1_INST(64, false, false, false)
W1_INST(64, false, true, false)
W1_INST(64, false, false, true)
W1_INST(16, true, false, false)
W1_INST(16, true, true, false)
W1_INST(32, true, false, false)
W1_INST(32, true, true, false)
W1_INST(64, true, false, false)
W1_INST(64, true, true, false)
W1_INST(64, true, false, true)
#undef W1_INST

// PR #3475 fused split-K at row tiles 16 and 32 for MXFP8, BF16, and INT6.
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
FUSE_SPLIT(INT6, 16, false)
FUSE_SPLIT(INT6, 32, false)
FUSE_SPLIT(INT6, 16, true)
FUSE_SPLIT(INT6, 32, true)
#undef FUSE_SPLIT""",
)

patch_k(
    "qmmf_body.cuh",
    'static_assert(BN == 64 || BN == 32, "a block reads one 64-column tile of words and scales, or its half");',
    'static_assert(BN == 64 || BN == 32 || BN == 16, "a block reads one 64-column tile of words and scales, or its half/quarter");',
)

patch_k(
    "qmmf_body.cuh",
    'static_assert(BN == 64 || BN == 32 || (MODE == BF16 && BN == 128), "packed column tiles");',
    'static_assert(BN == 64 || BN == 32 || ((MODE == BF16 || MODE == BF12) && BN == 128), "packed column tiles");',
)

patch_k(
    "qmmf_body.cuh",
    "        constexpr int TILE_BYTES = MODE == FP4 ? 64 * GS / 2 : MODE == BF16 ? 64 * GS * 2 : 64 * GS;",
    "        constexpr int TILE_BYTES = MODE == FP4 ? 64 * GS / 2 : MODE == W2 ? 64 * GS / 4 : MODE == W1 ? 64 * GS / 8 : MODE == INT6 ? (64 * GS * 3) / 4 : MODE == BF16 ? 64 * GS * 2 : MODE == BF12 ? 6144 : 64 * GS;",
)

patch_k(
    "qmmf_body.cuh",
    """                    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                                 :: "r"(smem(pw + t * TILE_BYTES)), "l"(src), "r"(TILE_BYTES), "r"(bar) : "memory");
                }
            }
        } else""",
    """                    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                                 :: "r"(smem(pw + t * TILE_BYTES)), "l"(src), "r"(TILE_BYTES), "r"(bar) : "memory");
                }
            }
            if constexpr (T::S > 0) {
                unsigned char* ps = pw + T::W;
                constexpr int S_TILE = MODE == BF12 ? 64 : T::S;
                for (int c = tid; c < T::S / 16; c += T::THREADS) {
                    const int st = c / (S_TILE / 16), soff = c % (S_TILE / 16);
                    cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64 + st) * KG + g) * S_TILE + soff * 16);
                }
            }
        } else""",
)

patch_k(
    "qmmf_body.cuh",
    """        if constexpr (BN == 32) {
            // half a packed 64-column tile: its n8 tiles 0-3 or 4-7, contiguous in every format's word order
            const int half = (n0 % 64) / 32;
            for (int c = tid; c < T::W / 16; c += T::THREADS)
                cp16(pw + c * 16, w + (static_cast<size_t>(n0 / 64) * KG + g) * TILE_BYTES + half * (TILE_BYTES / 2) + c * 16);
            if constexpr (T::S > 0) {
                unsigned char* ps = pw + T::W;
                for (int c = tid; c < T::S / 16; c += T::THREADS)
                    cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64) * KG + g) * (2 * T::S) + half * T::S + c * 16);
            }
        } else {
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            cp16(pw + c * 16, w + (static_cast<size_t>(n0 / 64 + t) * KG + g) * TILE_BYTES + off * 16);
        }
        if constexpr (T::S > 0) {
            unsigned char* ps = pw + T::W;
            for (int c = tid; c < T::S / 16; c += T::THREADS)
                cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64) * KG + g) * T::S + c * 16);
        }
        }""",
    """        if constexpr (BN == 16) {
            const int qtr = (n0 % 64) / 16;
            for (int c = tid; c < T::W / 16; c += T::THREADS)
                cp16_256(pw + c * 16, w + (static_cast<size_t>(n0 / 64) * KG + g) * TILE_BYTES + qtr * (TILE_BYTES / 4) + c * 16);
            if constexpr (T::S > 0) {
                unsigned char* ps = pw + T::W;
                for (int c = tid; c < T::S / 16; c += T::THREADS)
                    cp16_256(ps + c * 16, bs + (static_cast<size_t>(n0 / 64) * KG + g) * (4 * T::S) + qtr * T::S + c * 16);
            }
        } else if constexpr (BN == 32) {
            // half a packed 64-column tile: its n8 tiles 0-3 or 4-7, contiguous in every format's word order
            const int half = (n0 % 64) / 32;
            for (int c = tid; c < T::W / 16; c += T::THREADS) {
                if constexpr (MODE == BF12 || MODE == INT6)
                    cp16_256(pw + c * 16, w + (static_cast<size_t>(n0 / 64) * KG + g) * TILE_BYTES + half * (TILE_BYTES / 2) + c * 16);
                else
                    cp16(pw + c * 16, w + (static_cast<size_t>(n0 / 64) * KG + g) * TILE_BYTES + half * (TILE_BYTES / 2) + c * 16);
            }
            if constexpr (T::S > 0) {
                unsigned char* ps = pw + T::W;
                for (int c = tid; c < T::S / 16; c += T::THREADS) {
                    if constexpr (MODE == BF12 || MODE == INT6)
                        cp16_256(ps + c * 16, bs + (static_cast<size_t>(n0 / 64) * KG + g) * (2 * T::S) + half * T::S + c * 16);
                    else
                        cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64) * KG + g) * (2 * T::S) + half * T::S + c * 16);
                }
            }
        } else {
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            if constexpr (BN == 64 && (MODE == BF12 || MODE == W1 || MODE == W2 || MODE == INT6))
                cp16_256(pw + c * 16, w + (static_cast<size_t>(n0 / 64 + t) * KG + g) * TILE_BYTES + off * 16);
            else
                cp16(pw + c * 16, w + (static_cast<size_t>(n0 / 64 + t) * KG + g) * TILE_BYTES + off * 16);
        }
        if constexpr (T::S > 0) {
            unsigned char* ps = pw + T::W;
            constexpr int S_TILE = (MODE == BF12 || MODE == W1) ? 64 : T::S;
            for (int c = tid; c < T::S / 16; c += T::THREADS) {
                const int st = c / (S_TILE / 16), soff = c % (S_TILE / 16);
                if constexpr (BN == 64 && (MODE == BF12 || MODE == W1 || MODE == W2 || MODE == INT6))
                    cp16_256(ps + c * 16, bs + (static_cast<size_t>(n0 / 64 + st) * KG + g) * S_TILE + soff * 16);
                else
                    cp16(ps + c * 16, bs + (static_cast<size_t>(n0 / 64 + st) * KG + g) * S_TILE + soff * 16);
            }
        }
        }""",
)

patch_k(
    "qmmf_body.cuh",
    """            } else if constexpr (MODE == MXFP8) {
                const uint32_t v = *reinterpret_cast<const uint32_t*>(ps + col * 2);
                sq[j][0] = v & 0xFFFFu;
                sq[j][1] = v >> 16;
            }""",
    """            } else if constexpr (MODE == MXFP8 || MODE == INT6 || MODE == W2) {
                const uint32_t v = *reinterpret_cast<const uint32_t*>(ps + col * 2);
                sq[j][0] = v & 0xFFFFu;
                sq[j][1] = v >> 16;
            } else if constexpr (MODE == W1) {
                const uint16_t v = *reinterpret_cast<const uint16_t*>(ps + col);
                sq[j][0] = v & 0xFFu;
                sq[j][1] = v >> 8;
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
            } else if constexpr (MODE == W2) {
                const uint32_t u = reinterpret_cast<const uint32_t*>(pw)[jj * 32 + lane];
                wq[j] = make_uint4(u, 0u, 0u, 0u);
            } else if constexpr (MODE == W1) {
                const uint16_t u = reinterpret_cast<const uint16_t*>(pw)[jj * 32 + lane];
                wq[j] = make_uint4(static_cast<uint32_t>(u), 0u, 0u, 0u);
            } else if constexpr (MODE == INT6) {
                const uint32_t* pjj = reinterpret_cast<const uint32_t*>(pw + jj * 384);
                wq[j] = make_uint4(pjj[lane], pjj[32 + lane], pjj[64 + lane], 0u);
            } else if constexpr (MODE == BF12) {
                const unsigned char* pjj = pw + jj * 768;
                const uint32_t d01 = reinterpret_cast<const uint32_t*>(pjj + 512)[lane];
                const uint32_t d23 = reinterpret_cast<const uint32_t*>(pjj + 640)[lane];
                const uint32_t f01 = d01 & (d01 >> 1) & (d01 >> 2) & (d01 >> 3) & 0x11111111u;
                const uint32_t f23 = d23 & (d23 >> 1) & (d23 >> 2) & (d23 >> 3) & 0x11111111u;
                const uint32_t em_sh = (static_cast<uint32_t>(ps[jj * 8 + (lane >> 2)]) << 7) * 0x00010001u;
                wq[j] = make_uint4(d01, d23, em_sh, f01 | f23);
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
                    if constexpr (MODE == FP8 || MODE == BF16) mma(acc[i][j], a[i], b0, b1);
                    else if (MODE == FP4 || (MODE == MXFP8 && (kt & 1) == 0) || (MODE == FP8G && kt == 0))
                        mma0(d[i][j], a[i], b0, b1);
                    else mma(d[i][j], a[i], b0, b1);
                }
            }
            if constexpr (MODE == FP4 || MODE == MXFP8 || MODE == FP8G) {
                if (MODE == FP4 || (MODE == MXFP8 && (kt & 1)) || (MODE == FP8G && kt == GS / 16 - 1)) {""",
    """                } else if constexpr (MODE == W2) {
                    b0 = w2pair(wq[j].x, kt * 8);
                    b1 = w2pair(wq[j].x, kt * 8 + 4);
                } else if constexpr (MODE == W1) {
                    b0 = w1pair(wq[j].x, kt * 4);
                    b1 = w1pair(wq[j].x, kt * 4 + 2);
                } else if constexpr (MODE == INT6) {
                    const uint32_t lo4 = (kt < 2 ? wq[j].x : wq[j].y) >> ((kt & 1) * 16);
                    const uint32_t hi2 = wq[j].z >> (kt * 8);
                    b0 = int6pair(lo4 & 0xFFu, hi2 & 0xFu);
                    b1 = int6pair((lo4 >> 8) & 0xFFu, (hi2 >> 4) & 0xFu);
                } else if constexpr (MODE == BF12) {
                    const int jj_in_pw = wn * T::NT + j;
                    const unsigned char* pjj = pw + jj_in_pw * 768;
                    const uint32_t ms = reinterpret_cast<const uint32_t*>(pjj)[kt * 32 + lane];
                    const uint32_t dlo = kt < 2 ? wq[j].x : wq[j].y;
                    const int sh = (kt & 1) * 8;
                    const uint32_t d0 = (dlo >> sh) & 0x000F000Fu;
                    const uint32_t d1 = (dlo >> (sh + 4)) & 0x000F000Fu;
                    const uint32_t x0 = __byte_perm(ms, 0u, 0x4140u);
                    const uint32_t x1 = __byte_perm(ms, 0u, 0x4342u);
                    if (__builtin_expect(wq[j].w == 0u, 1)) {
                        b0 = ((x0 & 0x00800080u) << 8) | (wq[j].z - (d0 << 7)) | (x0 & 0x007F007Fu);
                        b1 = ((x1 & 0x00800080u) << 8) | (wq[j].z - (d1 << 7)) | (x1 & 0x007F007Fu);
                    } else {
                        const int local_jj = (BN == 16) ? (((n0 % 64) / 16) * 2 + jj_in_pw) : (BN == 32) ? (((n0 & 32) ? 4 : 0) + jj_in_pw) : (jj_in_pw & 7);
                        const size_t total_tiles = static_cast<size_t>(npad / 64) * KG;
                        const size_t tile_idx = static_cast<size_t>(n0 / 64 + (BN == 128 ? (jj_in_pw >> 3) : 0)) * KG + (g0 + it);
                        const uint32_t kbase = (static_cast<uint32_t>(local_jj) << 9) | (static_cast<uint32_t>(kt) << 7) | (static_cast<uint32_t>(lane) << 2);
                        const uint32_t em = (wq[j].z >> 7) & 0xFFu;
                        b0 = dec12_pair_esc(x0, d0, em, bs, total_tiles, tile_idx, kbase);
                        b1 = dec12_pair_esc(x1, d1, em, bs, total_tiles, tile_idx, kbase + 2u);
                    }
                } else {
                    const uint32_t word = comp(wq[j], kt);
                    b0 = fp8pair(word & 0xFFFFu);
                    b1 = fp8pair(word >> 16);
                }
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    if constexpr (MODE == FP8 || MODE == BF16 || MODE == BF12) mma(acc[i][j], a[i], b0, b1);
                    else if (MODE == FP4 || ((MODE == MXFP8 || MODE == INT6 || MODE == W2) && (kt & 1) == 0) || ((MODE == FP8G || MODE == W1) && kt == 0))
                        mma0(d[i][j], a[i], b0, b1);
                    else mma(d[i][j], a[i], b0, b1);
                }
            }
            if constexpr (MODE == FP4 || MODE == MXFP8 || MODE == FP8G || MODE == INT6 || MODE == W2 || MODE == W1) {
                if (MODE == FP4 || ((MODE == MXFP8 || MODE == INT6 || MODE == W2) && (kt & 1)) || ((MODE == FP8G || MODE == W1) && kt == GS / 16 - 1)) {""",
)

# Fast bit-exact bf16x2 E4M3 scale decode in qmmf_body.cuh (FP4 + INT6 + W2 + W1) and nvfp4_experts.cu (+ mma0)
patch_k(
    "qmmf_body.cuh",
    """                        } else if constexpr (MODE == FP4) {
                            s0 = e4m3f(static_cast<uint8_t>(sq[j][0] >> (8 * blk)));
                            s1 = e4m3f(static_cast<uint8_t>(sq[j][1] >> (8 * blk)));""",
    """                        } else if constexpr (MODE == W1) {
                            const uint32_t b2 = (sq[j][0] & 0xFFu) | ((sq[j][1] & 0xFFu) << 8);
                            const uint32_t sp = tf_common::fp8pair_unit(b2, 0x7B807B80u);
                            s0 = __uint_as_float((sp & 0xFFFFu) << 16);
                            s1 = __uint_as_float(sp & 0xFFFF0000u);
                        } else if constexpr (MODE == FP4 || MODE == INT6 || MODE == W2) {
                            const uint32_t b2 = ((sq[j][0] >> (8 * blk)) & 0xFFu) | (((sq[j][1] >> (8 * blk)) & 0xFFu) << 8);
                            const uint32_t sp = tf_common::fp8pair_unit(b2, 0x7B807B80u);
                            s0 = __uint_as_float((sp & 0xFFFFu) << 16);
                            s1 = __uint_as_float(sp & 0xFFFF0000u);""",
)

patch_k(
    "nvfp4_experts.cu",
    "__device__ __forceinline__ uint4 ld_nc(const uint4* p) { return tf_common::ld_nc(p); }",
    """__device__ __forceinline__ uint4 ld_nc(const uint4* p) {
#if __CUDA_ARCH__ >= 900
  uint4 v;
  asm volatile("ld.global.nc.L1::evict_first.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
#else
  return tf_common::ld_nc(p);
#endif
}""",
)

patch_k(
    "nvfp4_experts.cu",
    """        const uint32_t word = comp(st.w[m], j);
        float p[4] = {0.f, 0.f, 0.f, 0.f};
        mma(p, a0, a1, a2, a3, fp4pair(word, 8 * h), fp4pair(word, 8 * h + 4));
        const uint32_t sw = comp(st.s[m], 2 * h + (j >> 1));
        const int sh = (j & 1) * 16;
        const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);""",
    """        const uint32_t word = comp(st.w[m], j);
        float p[4];
        qmm_frag::mma0(p, a0, a1, a2, a3, fp4pair(word, 8 * h), fp4pair(word, 8 * h + 4));
        const uint32_t sw = comp(st.s[m], 2 * h + (j >> 1));
        const uint32_t sp = tf_common::fp8pair_unit((sw >> ((j & 1) * 16)) & 0xFFFFu, 0x7B807B80u);
        const float s0 = __uint_as_float((sp & 0xFFFFu) << 16), s1 = __uint_as_float(sp & 0xFFFF0000u);""",
)

patch_k(
    "nvfp4_experts.cu",
    """        const uint32_t word = jj ? st.w[m].y : st.w[m].x;
        float p[4] = {0.f, 0.f, 0.f, 0.f};
        mma(p, a0, a1, a2, a3, fp4pair(word, 8 * h), fp4pair(word, 8 * h + 4));
        const int sh = jj * 16;
        const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);""",
    """        const uint32_t word = jj ? st.w[m].y : st.w[m].x;
        float p[4];
        qmm_frag::mma0(p, a0, a1, a2, a3, fp4pair(word, 8 * h), fp4pair(word, 8 * h + 4));
        const uint32_t sp = tf_common::fp8pair_unit((sw >> (jj * 16)) & 0xFFFFu, 0x7B807B80u);
        const float s0 = __uint_as_float((sp & 0xFFFFu) << 16), s1 = __uint_as_float(sp & 0xFFFF0000u);""",
)

# Vectorized float4 coalesced state loads/stores in flashnext_batch.cu (gdn_rows & gdn_rows_late: 226 MB/round)
patch_k(
    "flashnext_batch.cu",
    """    float s[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(row0 + j) * DK + lane * 4 + i];
    float kk[4], qq[4], vv[4], g = 0.0f, beta = 0.0f;""",
    """    float s[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 s4 = *reinterpret_cast<const float4*>(state_in + sbase + (size_t)(row0 + j) * DK + lane * 4);
        s[j][0] = s4.x; s[j][1] = s4.y; s[j][2] = s4.z; s[j][3] = s4.w;
    }
    float kk[4], qq[4], vv[4], g = 0.0f, beta = 0.0f;""",
)

patch_k(
    "flashnext_batch.cu",
    """#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(row0 + j) * DK + lane * 4 + i] = s[j][i];
}

extern "C" __global__ void __launch_bounds__(128) fn_gdn_rows(""",
    """#pragma unroll
    for (int j = 0; j < 4; ++j)
        *reinterpret_cast<float4*>(state_out + sbase + (size_t)(row0 + j) * DK + lane * 4) = make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
}

extern "C" __global__ void __launch_bounds__(128) fn_gdn_rows(""",
)

patch_k(
    "flashnext_batch.cu",
    """    float s[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(row0 + j) * DK + lane * 4 + i];
    const unsigned total = np + (unsigned)rows;
    for (unsigned p = 0; p < total; ++p) {
        if (p == np && (int)np > store_above) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) state_in[sbase + (size_t)(row0 + j) * DK + lane * 4 + i] = s[j][i];
        }
        const size_t lr = (first + p) % (unsigned)log_rows;
        const float4 k4 = *reinterpret_cast<const float4*>(log + (lr * NK + hk) * DK + lane * 4);
        const float4 v4 = *reinterpret_cast<const float4*>(vl + lr * NV * DV + hv * DV + row0);
        const float kk[4] = {k4.x, k4.y, k4.z, k4.w}, vv[4] = {v4.x, v4.y, v4.z, v4.w};
        const float g = gl[(lr * NV + hv) * 2], beta = gl[(lr * NV + hv) * 2 + 1];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float kv = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[j][i] = s[j][i] * g;
                kv = kv + s[j][i] * kk[i];
            }
            kv = warp_sum(kv);
            const float delta = (vv[j] - kv) * beta;
#pragma unroll
            for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
        }
        if (p < np) continue;
        const int r = (int)(p - np);
        const float4 q4 = *reinterpret_cast<const float4*>(qn + ((size_t)r * NK + hk) * DK + lane * 4);
        const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
        float o4[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o4[j] = bf(warp_sum(o));
        }
        if (lane == 0) *reinterpret_cast<float4*>(ys_out + ((size_t)r * NV + hv) * DV + row0) = make_float4(o4[0], o4[1], o4[2], o4[3]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(row0 + j) * DK + lane * 4 + i] = s[j][i];""",
    """    float s[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 s4 = *reinterpret_cast<const float4*>(state_in + sbase + (size_t)(row0 + j) * DK + lane * 4);
        s[j][0] = s4.x; s[j][1] = s4.y; s[j][2] = s4.z; s[j][3] = s4.w;
    }
    const unsigned total = np + (unsigned)rows;
    float4 k4 = make_float4(0.f, 0.f, 0.f, 0.f), v4 = make_float4(0.f, 0.f, 0.f, 0.f);
    float g = 0.0f, beta = 0.0f;
    if (total > 0) {
        const size_t lr0 = first % (unsigned)log_rows;
        k4 = *reinterpret_cast<const float4*>(log + (lr0 * NK + hk) * DK + lane * 4);
        v4 = *reinterpret_cast<const float4*>(vl + lr0 * NV * DV + hv * DV + row0);
        g = gl[(lr0 * NV + hv) * 2];
        beta = gl[(lr0 * NV + hv) * 2 + 1];
    }
    for (unsigned p = 0; p < total; ++p) {
        if (p == np && (int)np > store_above) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                *reinterpret_cast<float4*>(state_in + sbase + (size_t)(row0 + j) * DK + lane * 4) = make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
        }
        float4 nk4 = k4, nv4 = v4;
        float ng = g, nb = beta;
        if (p + 1 < total) {
            const size_t nlr = (first + p + 1) % (unsigned)log_rows;
            nk4 = *reinterpret_cast<const float4*>(log + (nlr * NK + hk) * DK + lane * 4);
            nv4 = *reinterpret_cast<const float4*>(vl + nlr * NV * DV + hv * DV + row0);
            ng = gl[(nlr * NV + hv) * 2];
            nb = gl[(nlr * NV + hv) * 2 + 1];
        }
        const float kk[4] = {k4.x, k4.y, k4.z, k4.w}, vv[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float kv = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[j][i] = s[j][i] * g;
                kv = kv + s[j][i] * kk[i];
            }
            kv = warp_sum(kv);
            const float delta = (vv[j] - kv) * beta;
#pragma unroll
            for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
        }
        if (p >= np) {
            const int r = (int)(p - np);
            const float4 q4 = *reinterpret_cast<const float4*>(qn + ((size_t)r * NK + hk) * DK + lane * 4);
            const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
            float o4[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float o = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
                o4[j] = bf(warp_sum(o));
            }
            if (lane == 0) *reinterpret_cast<float4*>(ys_out + ((size_t)r * NV + hv) * DV + row0) = make_float4(o4[0], o4[1], o4[2], o4[3]);
        }
        k4 = nk4; v4 = nv4; g = ng; beta = nb;
    }
    if (state_out != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            *reinterpret_cast<float4*>(state_out + sbase + (size_t)(row0 + j) * DK + lane * 4) = make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
    }""",
)

patch_k(
    "nvfp4_experts.cu",
    """              const uint32_t sw = comp(sd[m], 2 * h + (j >> 1));
              const int sh = (j & 1) * 16;
              const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);""",
    """              const uint32_t sw = comp(sd[m], 2 * h + (j >> 1));
              const uint32_t sp = tf_common::fp8pair_unit((sw >> ((j & 1) * 16)) & 0xFFFFu, 0x7B807B80u);
              const float s0 = __uint_as_float((sp & 0xFFFFu) << 16), s1 = __uint_as_float(sp & 0xFFFF0000u);""",
)


patch_k(
    "flashnext_batch.cu",
    """extern "C" __global__ void __launch_bounds__(128) fn_gdn_rows_late(
        const float* __restrict__ qn, const float* __restrict__ log, const unsigned* __restrict__ ctl, int log_rows,
        int store_above, float* state_in, float* state_out, int rows, float* __restrict__ ys_out) {
    gdn_rows_late(qn, log, ctl, log_rows, store_above, state_in, state_out, rows, ys_out);
}""",
    """static __device__ __forceinline__ float gdn1t_dot4_g(float* h, const float4* x4, int l, float g) {
    const float4 x = x4[l];
    const float h0 = h[4 * l + 0] * g, h1 = h[4 * l + 1] * g, h2 = h[4 * l + 2] * g, h3 = h[4 * l + 3] * g;
    h[4 * l + 0] = h0; h[4 * l + 1] = h1; h[4 * l + 2] = h2; h[4 * l + 3] = h3;
    float s = 0.0f;
    s = s + h0 * x.x; s = s + h1 * x.y; s = s + h2 * x.z; s = s + h3 * x.w;
    return s;
}

static __device__ __forceinline__ float gdn1t_upd_dot4(float* h, const float4* k4, const float4* q4, int l, float delta) {
    const float4 k = k4[l];
    const float4 q = q4[l];
    const float h0 = h[4 * l + 0] + k.x * delta, h1 = h[4 * l + 1] + k.y * delta;
    const float h2 = h[4 * l + 2] + k.z * delta, h3 = h[4 * l + 3] + k.w * delta;
    h[4 * l + 0] = h0; h[4 * l + 1] = h1; h[4 * l + 2] = h2; h[4 * l + 3] = h3;
    float s = 0.0f;
    s = s + h0 * q.x; s = s + h1 * q.y; s = s + h2 * q.z; s = s + h3 * q.w;
    return s;
}

static __device__ __forceinline__ float4 gdn4t_ld256(const float4* p) {
#if __CUDA_ARCH__ >= 900
    float4 v;
    asm volatile("ld.global.nc.L1::evict_first.L2::256B.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
#else
    return __ldcs(p);
#endif
}

static __device__ __forceinline__ void gdn_rows_late_4t(
        const float* __restrict__ qn, const float* __restrict__ log, const unsigned* __restrict__ ctl, int log_rows,
        int store_above, float* state_in, float* state_out, int rows, float* __restrict__ ys_out) {
    using namespace fn_gdn;
    const int hv = blockIdx.x, blk = blockIdx.y, hk = hv / (NV / NK);
    const int tid = threadIdx.x;
    const int vr_local = tid >> 2;
    const int t4 = tid & 3;
    const int vrow = blk * 32 + vr_local;
    const size_t sbase = (size_t)hv * DV * DK + (size_t)vrow * DK;
    const float* vl = fn_late::vs(log, log_rows);
    const float* gl = fn_late::gbs(log, log_rows);
    const unsigned first = *ctl & 0xffffu, np = *ctl >> 16;

    const float4* row4 = reinterpret_cast<const float4*>(state_in + sbase);
    float4 h[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) h[j] = gdn4t_ld256(row4 + t4 + 4 * j);

    auto store_tile = [&](float* dst_base) {
        float4* out4 = reinterpret_cast<float4*>(dst_base + sbase);
#pragma unroll
        for (int j = 0; j < 8; ++j) out4[t4 + 4 * j] = h[j];
    };

    __shared__ float4 sk4[32], sq4[32];
    const unsigned total = np + (unsigned)rows;
    for (unsigned p = 0; p < total; ++p) {
        if (p == np && (int)np > store_above) store_tile(state_in);
        const size_t lr = (first + p) % (unsigned)log_rows;
        if (tid < 32) {
            sk4[tid] = *reinterpret_cast<const float4*>(log + (lr * NK + hk) * DK + tid * 4);
        }
        if (p >= np && tid >= 32 && tid < 64) {
            const int r = (int)(p - np);
            sq4[tid - 32] = *reinterpret_cast<const float4*>(qn + ((size_t)r * NK + hk) * DK + (tid - 32) * 4);
        }
        const float vv = vl[lr * NV * DV + hv * DV + vrow];
        const float g = gl[(lr * NV + hv) * 2];
        const float beta = gl[(lr * NV + hv) * 2 + 1];
        __syncthreads();

        float4 k[8];
        float d[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            k[j] = sk4[t4 + 4 * j];
            h[j].x *= g; h[j].y *= g; h[j].z *= g; h[j].w *= g;
            d[j] = ((h[j].x * k[j].x + h[j].y * k[j].y) + h[j].z * k[j].z) + h[j].w * k[j].w;
        }
        float s = ((d[0] + d[4]) + (d[2] + d[6])) + ((d[1] + d[5]) + (d[3] + d[7]));
        s = s + __shfl_xor_sync(0xffffffffu, s, 2);
        s = s + __shfl_xor_sync(0xffffffffu, s, 1);
        const float delta = (vv - s) * beta;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            h[j].x += k[j].x * delta; h[j].y += k[j].y * delta; h[j].z += k[j].z * delta; h[j].w += k[j].w * delta;
        }
        if (p >= np) {
            float o[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 q = sq4[t4 + 4 * j];
                o[j] = ((h[j].x * q.x + h[j].y * q.y) + h[j].z * q.z) + h[j].w * q.w;
            }
            float os = ((o[0] + o[4]) + (o[2] + o[6])) + ((o[1] + o[5]) + (o[3] + o[7]));
            os = os + __shfl_xor_sync(0xffffffffu, os, 2);
            os = os + __shfl_xor_sync(0xffffffffu, os, 1);
            if (t4 == 0) {
                const int r = (int)(p - np);
                ys_out[((size_t)r * NV + hv) * DV + vrow] = bf(os);
            }
        }
        __syncthreads();
    }
    if (state_out != nullptr) store_tile(state_out);
}

extern "C" __global__ void __launch_bounds__(128) fn_gdn_rows_late(
        const float* __restrict__ qn, const float* __restrict__ log, const unsigned* __restrict__ ctl, int log_rows,
        int store_above, float* state_in, float* state_out, int rows, float* __restrict__ ys_out) {
    gdn_rows_late_4t(qn, log, ctl, log_rows, store_above, state_in, state_out, rows, ys_out);
}""",
)


patch_k(
    "nvfp4_experts.cu",
    """TF_NVFP4_FACT(tf_nvfp4_fact_swiglu, 2, 2, 4)
TF_NVFP4_FACT(tf_nvfp4_fact_down_f32, 1, 0, 8)""",
    """TF_NVFP4_FACT(tf_nvfp4_fact_swiglu, 2, 2, 4)
TF_NVFP4_FACT(tf_nvfp4_fact_down_f32, 1, 0, 8)

extern "C" __global__ void __launch_bounds__(128) tf_nvfp4_half_coop4(
    const __nv_bfloat16* X, int x_stride, int slots, const uint4* __restrict__ W,
    const float* __restrict__ scale, int KG, int NB, const int* items,
    const int* counts, const int* members, void* __restrict__ out, int N,
    float limit, int skip) {
  qmm_frag::grid_wait();
  constexpr int M = 2;
  constexpr int BLOCK4 = tf_nvfp4_experts::BLOCK4;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = (*counts) * NB * 2;
  const int unit = blockIdx.x * 4 + warp;
  if (unit < units) {
    const int it = unit / (NB * 2), rest = unit - it * NB * 2, cb = rest >> 1, jh = rest & 1;
    const int e = items[3 * it], first = items[3 * it + 1], cnt = items[3 * it + 2];
    if (e != skip) {
      const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
      auto ld4_256 = [](const uint4* p) -> uint4 {
        uint4 v;
#if __CUDA_ARCH__ >= 900
        asm volatile("ld.global.nc.L1::evict_first.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#else
        v = tf_common::ld_nc(p);
#endif
        return v;
      };
      auto ld2_256 = [](const uint2* p) -> uint2 {
        uint2 v;
#if __CUDA_ARCH__ >= 900
        asm volatile("ld.global.nc.L1::evict_first.L2::256B.v2.u32 {%0,%1}, [%2];"
                     : "=r"(v.x), "=r"(v.y) : "l"(p));
#else
        v = tf_common::ld_nc2(p);
#endif
        return v;
      };
      for (int r0 = 0; r0 < cnt; r0 += 16) {
        const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
        const int pr0 = v0 ? members[first + r0 + gq] : 0;
        const int pr1 = v1 ? members[first + r0 + gq + 8] : 0;
        const int x0r = slots ? pr0 / slots : pr0, x1r = slots ? pr1 / slots : pr1;
        const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
        const __nv_bfloat16* x1 = X + (size_t)x1r * x_stride;
        float acc[M][2][4];
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int j = 0; j < 2; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f;
        struct H { uint2 w[M]; uint2 s[M]; uint2 xa[2], xb[2]; };
        auto load = [&](H& st, int g) {
          const uint4* b = blk + (size_t)g * (M * BLOCK4);
#pragma unroll
          for (int m = 0; m < M; ++m) {
            st.w[m] = ld2_256(reinterpret_cast<const uint2*>(b + m * BLOCK4 + lane) + jh);
            const uint4 sf = ld4_256(b + m * BLOCK4 + 32 + t);
            st.s[m] = jh ? make_uint2(sf.y, sf.w) : make_uint2(sf.x, sf.z);
          }
          const uint2 zero = make_uint2(0u, 0u);
          const int k0 = g * 32 + 4 * t;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            st.xa[h] = v0 ? *reinterpret_cast<const uint2*>(x0 + k0 + 16 * h) : zero;
            st.xb[h] = v1 ? *reinterpret_cast<const uint2*>(x1 + k0 + 16 * h) : zero;
          }
        };
        auto comp_s = [&](const H& st) {
#pragma unroll
          for (int m = 0; m < M; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y, a1 = st.xb[h].x, a3 = st.xb[h].y;
              const uint32_t sw = h == 0 ? st.s[m].x : st.s[m].y;
#pragma unroll
              for (int jj = 0; jj < 2; ++jj) {
                const uint32_t word = jj ? st.w[m].y : st.w[m].x;
                float p[4];
                qmm_frag::mma0(p, a0, a1, a2, a3, tf_common::fp4pair(word, 8 * h), tf_common::fp4pair(word, 8 * h + 4));
                const uint32_t sp = tf_common::fp8pair_unit((sw >> (jj * 16)) & 0xFFFFu, 0x7B807B80u);
                const float s0 = __uint_as_float((sp & 0xFFFFu) << 16), s1 = __uint_as_float(sp & 0xFFFF0000u);
                float(&a)[4] = acc[m][jj];
                a[0] = fmaf(p[0], s0, a[0]);
                a[1] = fmaf(p[1], s1, a[1]);
                if (v1) {
                  a[2] = fmaf(p[2], s0, a[2]);
                  a[3] = fmaf(p[3], s1, a[3]);
                }
              }
            }
        };
        H st[2];
        if (0 < KG) load(st[0], 0);
        if (1 < KG) load(st[1], 1);
        for (int g0 = 0; g0 < KG; g0 += 2) {
          if (g0 < KG) { comp_s(st[0]); if (g0 + 2 < KG) load(st[0], g0 + 2); }
          if (g0 + 1 < KG) { comp_s(st[1]); if (g0 + 3 < KG) load(st[1], g0 + 3); }
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const float g = __ldg(scale + e * M + m);
#pragma unroll
          for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[m][j][q] *= g;
        }
        tf_nvfp4_experts::epilogue_half<2, 2>(acc, out, N, cb * COLS + 16 * jh + 2 * t, pr0, pr1, v0, v1, limit);
      }
    }
  }
  qmm_frag::grid_launch();
}

extern "C" __global__ void __launch_bounds__(64) tf_nvfp4_half_coop2(
    const __nv_bfloat16* X, int x_stride, int slots, const uint4* __restrict__ W,
    const float* __restrict__ scale, int KG, int NB, const int* items,
    const int* counts, const int* members, void* __restrict__ out, int N,
    float limit, int skip) {
  qmm_frag::grid_wait();
  constexpr int M = 2;
  constexpr int BLOCK4 = tf_nvfp4_experts::BLOCK4;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = (*counts) * NB * 2;
  const int unit = blockIdx.x * 2 + warp;
  if (unit < units) {
    const int it = unit / (NB * 2), rest = unit - it * NB * 2, cb = rest >> 1, jh = rest & 1;
    const int e = items[3 * it], first = items[3 * it + 1], cnt = items[3 * it + 2];
    if (e != skip) {
      const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
      auto ld4_256 = [](const uint4* p) -> uint4 {
        uint4 v;
#if __CUDA_ARCH__ >= 900
        asm volatile("ld.global.nc.L1::evict_first.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#else
        v = tf_common::ld_nc(p);
#endif
        return v;
      };
      auto ld2_256 = [](const uint2* p) -> uint2 {
        uint2 v;
#if __CUDA_ARCH__ >= 900
        asm volatile("ld.global.nc.L1::evict_first.L2::256B.v2.u32 {%0,%1}, [%2];"
                     : "=r"(v.x), "=r"(v.y) : "l"(p));
#else
        v = tf_common::ld_nc2(p);
#endif
        return v;
      };
      for (int r0 = 0; r0 < cnt; r0 += 16) {
        const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
        const int pr0 = v0 ? members[first + r0 + gq] : 0;
        const int pr1 = v1 ? members[first + r0 + gq + 8] : 0;
        const int x0r = slots ? pr0 / slots : pr0, x1r = slots ? pr1 / slots : pr1;
        const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
        const __nv_bfloat16* x1 = X + (size_t)x1r * x_stride;
        float acc[M][2][4];
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int j = 0; j < 2; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f;
        struct H { uint2 w[M]; uint2 s[M]; uint2 xa[2], xb[2]; };
        auto load = [&](H& st, int g) {
          const uint4* b = blk + (size_t)g * (M * BLOCK4);
#pragma unroll
          for (int m = 0; m < M; ++m) {
            st.w[m] = ld2_256(reinterpret_cast<const uint2*>(b + m * BLOCK4 + lane) + jh);
            const uint4 sf = ld4_256(b + m * BLOCK4 + 32 + t);
            st.s[m] = jh ? make_uint2(sf.y, sf.w) : make_uint2(sf.x, sf.z);
          }
          const uint2 zero = make_uint2(0u, 0u);
          const int k0 = g * 32 + 4 * t;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            st.xa[h] = v0 ? *reinterpret_cast<const uint2*>(x0 + k0 + 16 * h) : zero;
            st.xb[h] = v1 ? *reinterpret_cast<const uint2*>(x1 + k0 + 16 * h) : zero;
          }
        };
        auto comp_s = [&](const H& st) {
#pragma unroll
          for (int m = 0; m < M; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y, a1 = st.xb[h].x, a3 = st.xb[h].y;
              const uint32_t sw = h == 0 ? st.s[m].x : st.s[m].y;
#pragma unroll
              for (int jj = 0; jj < 2; ++jj) {
                const uint32_t word = jj ? st.w[m].y : st.w[m].x;
                float p[4];
                qmm_frag::mma0(p, a0, a1, a2, a3, tf_common::fp4pair(word, 8 * h), tf_common::fp4pair(word, 8 * h + 4));
                const uint32_t sp = tf_common::fp8pair_unit((sw >> (jj * 16)) & 0xFFFFu, 0x7B807B80u);
                const float s0 = __uint_as_float((sp & 0xFFFFu) << 16), s1 = __uint_as_float(sp & 0xFFFF0000u);
                float(&a)[4] = acc[m][jj];
                a[0] = fmaf(p[0], s0, a[0]);
                a[1] = fmaf(p[1], s1, a[1]);
                if (v1) {
                  a[2] = fmaf(p[2], s0, a[2]);
                  a[3] = fmaf(p[3], s1, a[3]);
                }
              }
            }
        };
        H st[2];
        if (0 < KG) load(st[0], 0);
        if (1 < KG) load(st[1], 1);
        for (int g0 = 0; g0 < KG; g0 += 2) {
          if (g0 < KG) { comp_s(st[0]); if (g0 + 2 < KG) load(st[0], g0 + 2); }
          if (g0 + 1 < KG) { comp_s(st[1]); if (g0 + 3 < KG) load(st[1], g0 + 3); }
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
          const float g = __ldg(scale + e * M + m);
#pragma unroll
          for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[m][j][q] *= g;
        }
        tf_nvfp4_experts::epilogue_half<2, 2>(acc, out, N, cb * COLS + 16 * jh + 2 * t, pr0, pr1, v0, v1, limit);
      }
    }
  }
  qmm_frag::grid_launch();
}

// Lean 4-warp contiguous-uint4 r0+=8 gate+up (128 regs, 0 spill, 0 barriers, 226+ GB/s on c=1 & MTP).
extern "C" __global__ void __launch_bounds__(128, 4) tf_nvfp4_full_r8(
    const __nv_bfloat16* X, int x_stride, int slots, const uint4* __restrict__ W,
    const float* __restrict__ scale, int KG, int NB, const int* items,
    const int* counts, const int* members, void* __restrict__ out, int N,
    float limit, int skip) {
  qmm_frag::grid_wait();
  constexpr int BLOCK4 = tf_nvfp4_experts::BLOCK4;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = (*counts) * NB;
  const int unit = blockIdx.x * 4 + warp;
  if (unit < units) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = items[3 * it], first = items[3 * it + 1], cnt = items[3 * it + 2];
    if (e != skip) {
      const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (2 * BLOCK4);
      auto ld4_256 = [](const uint4* p) -> uint4 {
        uint4 v;
#if __CUDA_ARCH__ >= 900
        asm volatile("ld.global.nc.L1::evict_first.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#else
        v = tf_common::ld_nc(p);
#endif
        return v;
      };
      for (int r0 = 0; r0 < cnt; r0 += 8) {
        const bool v0 = r0 + gq < cnt;
        const int pr0 = v0 ? members[first + r0 + gq] : 0;
        const int x0r = slots ? pr0 / slots : pr0;
        const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
        float acc[2][4][2];
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
          for (int j = 0; j < 4; ++j) acc[m][j][0] = acc[m][j][1] = 0.f;
        struct S { uint4 w[2]; uint4 s[2]; };
        auto load = [&](S& st, int g) {
          const uint4* b = blk + (size_t)g * (2 * BLOCK4);
          st.w[0] = ld4_256(b + lane);
          st.s[0] = ld4_256(b + 32 + t);
          st.w[1] = ld4_256(b + BLOCK4 + lane);
          st.s[1] = ld4_256(b + BLOCK4 + 32 + t);
        };
        auto comp_s = [&](const S& st, int g) {
          const uint2 zero = make_uint2(0u, 0u);
          const int k0 = g * 32 + 4 * t;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint2 xa = v0 ? *reinterpret_cast<const uint2*>(x0 + k0 + 16 * h) : zero;
            const uint32_t a0 = xa.x, a2 = xa.y;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
#pragma unroll
              for (int j = 0; j < 4; ++j) {
                const uint32_t word = j == 0 ? st.w[m].x : (j == 1 ? st.w[m].y : (j == 2 ? st.w[m].z : st.w[m].w));
                float p[4];
                qmm_frag::mma0(p, a0, 0u, a2, 0u, tf_common::fp4pair(word, 8 * h), tf_common::fp4pair(word, 8 * h + 4));
                const uint32_t sw = (2 * h + (j >> 1)) == 0 ? st.s[m].x : ((2 * h + (j >> 1)) == 1 ? st.s[m].y : ((2 * h + (j >> 1)) == 2 ? st.s[m].z : st.s[m].w));
                const uint32_t sp = tf_common::fp8pair_unit((sw >> ((j & 1) * 16)) & 0xFFFFu, 0x7B807B80u);
                const float s0 = __uint_as_float((sp & 0xFFFFu) << 16), s1 = __uint_as_float(sp & 0xFFFF0000u);
                acc[m][j][0] = fmaf(p[0], s0, acc[m][j][0]);
                acc[m][j][1] = fmaf(p[1], s1, acc[m][j][1]);
              }
            }
          }
        };
        S st[2];
        if (0 < KG) load(st[0], 0);
        if (1 < KG) load(st[1], 1);
        for (int g0 = 0; g0 < KG; g0 += 2) {
          if (g0 < KG) { comp_s(st[0], g0); if (g0 + 2 < KG) load(st[0], g0 + 2); }
          if (g0 + 1 < KG) { comp_s(st[1], g0 + 1); if (g0 + 3 < KG) load(st[1], g0 + 3); }
        }
        if (v0) {
          const float g0 = __ldg(scale + e * 2), g1 = __ldg(scale + e * 2 + 1);
          const size_t row = (size_t)pr0 * N;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float o0 = swiglu(acc[0][j][0] * g0, acc[1][j][0] * g1, limit);
            const float o1 = swiglu(acc[0][j][1] * g0, acc[1][j][1] * g1, limit);
            *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + cb * COLS + 8 * j + 2 * t) =
                __floats2bfloat162_rn(o0, o1);
          }
        }
      }
    }
  }
  qmm_frag::grid_launch();
}

static __device__ __noinline__ void coop2_u4_wide(
    int warp, int lane, int gq, int t, int e, int cb, int first, int cnt,
    const __nv_bfloat16* X, int x_stride, int slots, const uint4* __restrict__ blk,
    const float* __restrict__ scale, int KG, const int* members,
    void* __restrict__ out, int N, float limit,
    uint2 (*sh_w)[32], uint4 (*sh_s)[4]) {
  constexpr int BLOCK4 = tf_nvfp4_experts::BLOCK4;
  auto ld4_256 = [](const uint4* p) -> uint4 {
    uint4 v;
#if __CUDA_ARCH__ >= 900
    asm volatile("ld.global.nc.L1::evict_first.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#else
    v = tf_common::ld_nc(p);
#endif
    return v;
  };
  const float g0 = __ldg(scale + e * 2), g1 = __ldg(scale + e * 2 + 1);
  for (int r0 = 0; r0 < cnt; r0 += 16) {
    const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
    const int pr0 = v0 ? members[first + r0 + gq] : 0;
    const int pr1 = v1 ? members[first + r0 + gq + 8] : 0;
    const __nv_bfloat16* x0 = X + (size_t)(slots ? pr0 / slots : pr0) * x_stride;
    const __nv_bfloat16* x1 = X + (size_t)(slots ? pr1 / slots : pr1) * x_stride;
    float acc[2][2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int j = 0; j < 2; ++j) acc[m][j][0] = acc[m][j][1] = acc[m][j][2] = acc[m][j][3] = 0.f;
    struct L { uint4 w; uint4 s; uint2 xa[2], xb[2]; };
    auto load = [&](L& st, int g) {
      const uint4* b = blk + (size_t)g * (2 * BLOCK4) + warp * BLOCK4;
      st.w = ld4_256(b + lane);
      if (gq == 0) st.s = ld4_256(b + 32 + t);
      const uint2 zero = make_uint2(0u, 0u);
      const int k0 = g * 32 + 4 * t;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        st.xa[h] = v0 ? *reinterpret_cast<const uint2*>(x0 + k0 + 16 * h) : zero;
        st.xb[h] = v1 ? *reinterpret_cast<const uint2*>(x1 + k0 + 16 * h) : zero;
      }
    };
    auto comp_s = [&](const L& st) {
      if (warp == 0) { sh_w[0][lane] = make_uint2(st.w.z, st.w.w); if (gq == 0) sh_s[0][t] = st.s; }
      else           { sh_w[1][lane] = make_uint2(st.w.x, st.w.y); if (gq == 0) sh_s[1][t] = st.s; }
      __syncthreads();
      uint2 wm[2], sm[2];
      const uint4 sf0 = sh_s[0][t], sf1 = sh_s[1][t];
      if (warp == 0) {
        wm[0] = make_uint2(st.w.x, st.w.y); wm[1] = sh_w[1][lane];
        sm[0] = make_uint2(sf0.x, sf0.z); sm[1] = make_uint2(sf1.x, sf1.z);
      } else {
        wm[0] = sh_w[0][lane]; wm[1] = make_uint2(st.w.z, st.w.w);
        sm[0] = make_uint2(sf0.y, sf0.w); sm[1] = make_uint2(sf1.y, sf1.w);
      }
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y, a1 = st.xb[h].x, a3 = st.xb[h].y;
          const uint32_t sw = h == 0 ? sm[m].x : sm[m].y;
#pragma unroll
          for (int jj = 0; jj < 2; ++jj) {
            const uint32_t word = jj ? wm[m].y : wm[m].x;
            float p[4];
            qmm_frag::mma0(p, a0, a1, a2, a3, tf_common::fp4pair(word, 8 * h), tf_common::fp4pair(word, 8 * h + 4));
            const uint32_t sp = tf_common::fp8pair_unit((sw >> (jj * 16)) & 0xFFFFu, 0x7B807B80u);
            const float s0 = __uint_as_float((sp & 0xFFFFu) << 16), s1 = __uint_as_float(sp & 0xFFFF0000u);
            acc[m][jj][0] = fmaf(p[0], s0, acc[m][jj][0]);
            acc[m][jj][1] = fmaf(p[1], s1, acc[m][jj][1]);
            if (v1) {
              acc[m][jj][2] = fmaf(p[2], s0, acc[m][jj][2]);
              acc[m][jj][3] = fmaf(p[3], s1, acc[m][jj][3]);
            }
          }
        }
      __syncthreads();
    };
    L st[2];
    if (0 < KG) load(st[0], 0);
    if (1 < KG) load(st[1], 1);
    for (int g0i = 0; g0i < KG; g0i += 2) {
      if (g0i < KG) { comp_s(st[0]); if (g0i + 2 < KG) load(st[0], g0i + 2); }
      if (g0i + 1 < KG) { comp_s(st[1]); if (g0i + 3 < KG) load(st[1], g0i + 3); }
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const float g = m == 0 ? g0 : g1;
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[m][j][q] *= g;
    }
    tf_nvfp4_experts::epilogue_half<2, 2>(acc, out, N, cb * COLS + 16 * warp + 2 * t, pr0, pr1, v0, v1, limit);
  }
}

// 2-warp coalesced uint4 D=3 SMEM-exchanged gate+up (88 regs, 0 spill, 640B SMEM, 100% bit-exact).
extern "C" __global__ void __launch_bounds__(64) tf_nvfp4_coop2_u4(
    const __nv_bfloat16* X, int x_stride, int slots, const uint4* __restrict__ W,
    const float* __restrict__ scale, int KG, int NB, const int* items,
    const int* counts, const int* members, void* __restrict__ out, int N,
    float limit, int skip) {
  qmm_frag::grid_wait();
  constexpr int BLOCK4 = tf_nvfp4_experts::BLOCK4;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = (*counts) * NB;
  const int unit = blockIdx.x;
  __shared__ uint2 sh_w[2][32];
  __shared__ uint4 sh_s[2][4];
  if (unit < units) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = items[3 * it], first = items[3 * it + 1], cnt = items[3 * it + 2];
    if (e != skip) {
      const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (2 * BLOCK4);
      if (cnt <= 8) {
        auto ld4_256 = [](const uint4* p) -> uint4 {
          uint4 v;
#if __CUDA_ARCH__ >= 900
          asm volatile("ld.global.nc.L1::evict_first.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
#else
          v = tf_common::ld_nc(p);
#endif
          return v;
        };
        const bool v0 = gq < cnt;
        const int pr0 = v0 ? members[first + gq] : 0;
        const __nv_bfloat16* x0 = X + (size_t)(slots ? pr0 / slots : pr0) * x_stride;
        float acc[2][2][2];
#pragma unroll
        for (int m = 0; m < 2; ++m)
#pragma unroll
          for (int j = 0; j < 2; ++j) acc[m][j][0] = acc[m][j][1] = 0.f;
        struct L { uint4 w; uint4 s; uint2 xa[2]; };
        auto load = [&](L& st, int g) {
          const uint4* b = blk + (size_t)g * (2 * BLOCK4) + warp * BLOCK4;
          st.w = ld4_256(b + lane);
          if (gq == 0) st.s = ld4_256(b + 32 + t);
          const uint2 zero = make_uint2(0u, 0u);
          const int k0 = g * 32 + 4 * t;
          st.xa[0] = v0 ? *reinterpret_cast<const uint2*>(x0 + k0) : zero;
          st.xa[1] = v0 ? *reinterpret_cast<const uint2*>(x0 + k0 + 16) : zero;
        };
        auto comp_s = [&](const L& st) {
          if (warp == 0) { sh_w[0][lane] = make_uint2(st.w.z, st.w.w); if (gq == 0) sh_s[0][t] = st.s; }
          else           { sh_w[1][lane] = make_uint2(st.w.x, st.w.y); if (gq == 0) sh_s[1][t] = st.s; }
          __syncthreads();
          uint2 wm[2], sm[2];
          const uint4 sf0 = sh_s[0][t], sf1 = sh_s[1][t];
          if (warp == 0) {
            wm[0] = make_uint2(st.w.x, st.w.y); wm[1] = sh_w[1][lane];
            sm[0] = make_uint2(sf0.x, sf0.z); sm[1] = make_uint2(sf1.x, sf1.z);
          } else {
            wm[0] = sh_w[0][lane]; wm[1] = make_uint2(st.w.z, st.w.w);
            sm[0] = make_uint2(sf0.y, sf0.w); sm[1] = make_uint2(sf1.y, sf1.w);
          }
#pragma unroll
          for (int m = 0; m < 2; ++m)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y;
              const uint32_t sw = h == 0 ? sm[m].x : sm[m].y;
#pragma unroll
              for (int jj = 0; jj < 2; ++jj) {
                const uint32_t word = jj ? wm[m].y : wm[m].x;
                float p[4];
                qmm_frag::mma0(p, a0, 0u, a2, 0u, tf_common::fp4pair(word, 8 * h), tf_common::fp4pair(word, 8 * h + 4));
                const uint32_t sp = tf_common::fp8pair_unit((sw >> (jj * 16)) & 0xFFFFu, 0x7B807B80u);
                const float s0 = __uint_as_float((sp & 0xFFFFu) << 16), s1 = __uint_as_float(sp & 0xFFFF0000u);
                acc[m][jj][0] = fmaf(p[0], s0, acc[m][jj][0]);
                acc[m][jj][1] = fmaf(p[1], s1, acc[m][jj][1]);
              }
            }
          __syncthreads();
        };
        L st[3];
        if (0 < KG) load(st[0], 0);
        if (1 < KG) load(st[1], 1);
        if (2 < KG) load(st[2], 2);
        for (int g0i = 0; g0i < KG; g0i += 3) {
          if (g0i < KG)     { comp_s(st[0]); if (g0i + 3 < KG) load(st[0], g0i + 3); }
          if (g0i + 1 < KG) { comp_s(st[1]); if (g0i + 4 < KG) load(st[1], g0i + 4); }
          if (g0i + 2 < KG) { comp_s(st[2]); if (g0i + 5 < KG) load(st[2], g0i + 5); }
        }
        if (v0) {
          const float g0 = __ldg(scale + e * 2), g1 = __ldg(scale + e * 2 + 1);
          const size_t row = (size_t)pr0 * N;
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const float o0 = swiglu(acc[0][j][0] * g0, acc[1][j][0] * g1, limit);
            const float o1 = swiglu(acc[0][j][1] * g0, acc[1][j][1] * g1, limit);
            *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + cb * COLS + 16 * warp + 8 * j + 2 * t) =
                __floats2bfloat162_rn(o0, o1);
          }
        }
      } else {
        coop2_u4_wide(warp, lane, gq, t, e, cb, first, cnt, X, x_stride, slots, blk, scale, KG, members, out, N, limit, sh_w, sh_s);
      }
    }
  }
  qmm_frag::grid_launch();
}

// ===== Trial 60: Lossless 256B-aligned 5-bit NVFP4 Expert Scales (S5) =====
static __device__ __forceinline__ uint4 s5_ld4_256(const uint4* p) {
#if __CUDA_ARCH__ >= 900
  uint4 v;
  asm volatile("ld.global.nc.L1::evict_first.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
#else
  return tf_common::ld_nc(p);
#endif
}
static __device__ __forceinline__ uint2 s5_ld2_256(const uint2* p) {
#if __CUDA_ARCH__ >= 900
  uint2 v;
  asm volatile("ld.global.nc.L1::evict_first.L2::256B.v2.u32 {%0,%1}, [%2];"
               : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
#else
  return tf_common::ld_nc2(p);
#endif
}
static __device__ __forceinline__ uint32_t s5_ld_u16_256(const uint16_t* p) {
#if __CUDA_ARCH__ >= 900
  uint32_t v;
  asm volatile("ld.global.nc.L1::evict_first.L2::256B.u16 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
#else
  return static_cast<uint32_t>(__ldg(p));
#endif
}

static __device__ __noinline__ uint32_t s5_lookup_esc_meta(
    const uint2* __restrict__ meta, const uint32_t* __restrict__ esc, int midx, uint32_t key) {
  const uint2 mm = __ldg(meta + midx);
  uint32_t lo = 0u, hi = mm.x >> 8;
  const uint32_t off = mm.y;
  while (lo < hi) {
    const uint32_t mid = (lo + hi) >> 1;
    const uint32_t k = __ldg(esc + off + mid) >> 8;
    if (k < key) lo = mid + 1u; else hi = mid;
  }
  return __ldg(esc + off + lo) & 0xFFu;
}

static __device__ __forceinline__ uint32_t s5_pair_bytes(
    uint32_t lo_byte, uint32_t hi_2bits, uint32_t smin,
    const uint2* __restrict__ meta, const uint32_t* __restrict__ esc, int midx, uint32_t key_base) {
  const uint32_t q0 = (lo_byte & 0xFu) | ((hi_2bits & 1u) << 4);
  const uint32_t q1 = ((lo_byte >> 4) & 0xFu) | ((hi_2bits & 2u) << 3);
  uint32_t b0 = smin + q0;
  uint32_t b1 = smin + q1;
  if (__builtin_expect((q0 == 31u) | (q1 == 31u), 0)) {
    if (q0 == 31u) b0 = s5_lookup_esc_meta(meta, esc, midx, key_base);
    if (q1 == 31u) b1 = s5_lookup_esc_meta(meta, esc, midx, key_base + 1u);
  }
  return (b0 & 0xFFu) | ((b1 & 0xFFu) << 8);
}

// Per-expert 3-array 256B-aligned S5 down_f32 (STACK:0, +2.56% proven on full model in nsys).
static __device__ __noinline__ void s5_down_1w_wide(
    int lane, int gq, int t, int e, int cb, int first, int cnt,
    const __nv_bfloat16* X, int x_stride,
    const uint8_t* __restrict__ base, int KG, int NB,
    float gscale, uint32_t smin, const uint2* __restrict__ meta, const uint32_t* __restrict__ esc,
    const int* members, void* __restrict__ out, int N) {
  const size_t nblk = (size_t)NB * KG;
  for (int r0 = 0; r0 < cnt; r0 += 16) {
    const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
    const int pr0 = v0 ? members[first + r0 + gq] : 0;
    const int pr1 = v1 ? members[first + r0 + gq + 8] : 0;
    const __nv_bfloat16* x0 = X + (size_t)pr0 * x_stride;
    const __nv_bfloat16* x1 = X + (size_t)pr1 * x_stride;
    float acc[1][1][4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[0][0][j][0] = acc[0][0][j][1] = acc[0][0][j][2] = acc[0][0][j][3] = 0.f;
    for (int g = 0; g < KG; ++g) {
      const size_t blk_i = (size_t)cb * KG + g;
      const uint4 wv = s5_ld4_256(reinterpret_cast<const uint4*>(base) + blk_i * 32 + lane);
      const uint2 lov = s5_ld2_256(reinterpret_cast<const uint2*>(base + nblk * 512) + blk_i * 4 + t);
      const uint32_t hiv = s5_ld_u16_256(reinterpret_cast<const uint16_t*>(base + nblk * 544) + blk_i * 4 + t);
      const uint2 zero = make_uint2(0u, 0u);
      const int k0 = g * 32 + 4 * t;
      const uint32_t gbase = ((uint32_t)blk_i * 64u) + (uint32_t)t * 16u;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint2 xa = v0 ? *reinterpret_cast<const uint2*>(x0 + k0 + 16 * h) : zero;
        const uint2 xb = v1 ? *reinterpret_cast<const uint2*>(x1 + k0 + 16 * h) : zero;
        const uint32_t lo_h = h == 0 ? lov.x : lov.y;
        const uint32_t hi_h = (hiv >> (h * 8)) & 0xFFu;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t word = comp(wv, j);
          float p[4];
          qmm_frag::mma0(p, xa.x, xb.x, xa.y, xb.y, tf_common::fp4pair(word, 8 * h), tf_common::fp4pair(word, 8 * h + 4));
          const uint32_t pair_b = s5_pair_bytes((lo_h >> (j * 8)) & 0xFFu, (hi_h >> (j * 2)) & 3u,
                                                smin, meta, esc, e, gbase + (uint32_t)h * 8u + (uint32_t)j * 2u);
          const uint32_t sp = tf_common::fp8pair_unit(pair_b, 0x7B807B80u);
          const float s0 = __uint_as_float((sp & 0xFFFFu) << 16), s1 = __uint_as_float(sp & 0xFFFF0000u);
          float(&a)[4] = acc[0][0][j];
          a[0] = fmaf(p[0], s0, a[0]); a[1] = fmaf(p[1], s1, a[1]);
          if (v1) { a[2] = fmaf(p[2], s0, a[2]); a[3] = fmaf(p[3], s1, a[3]); }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[0][0][j][q] *= gscale;
    epilogue<0, 1, 1>(acc, 0, out, N, cb * 32 + 2 * t, pr0, pr1, v0, v1, 0.f);
  }
}

extern "C" __global__ void __launch_bounds__(32) tf_nvfp4_s5_down_1w(
    const __nv_bfloat16* X, int x_stride, int slots, const uint8_t* __restrict__ W_raw,
    const float* __restrict__ scale, const uint2* __restrict__ meta, const uint32_t* __restrict__ esc,
    int KG, int NB, const int* items, const int* counts, const int* members,
    void* __restrict__ out, int N, float limit, int skip) {
  (void)slots; (void)limit;
  qmm_frag::grid_wait();
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = (*counts) * NB;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = items[3 * it], first = items[3 * it + 1], cnt = items[3 * it + 2];
    if (e == skip) continue;
    const size_t nblk = (size_t)NB * KG;
    const uint8_t* base = W_raw + (size_t)e * (nblk * 552);
    const uint32_t smin = __ldg(meta + e).x & 0xFFu;
    const float gscale = __ldg(scale + e);
    if (cnt <= 8) {
      const bool v0 = gq < cnt;
      const int pr0 = v0 ? members[first + gq] : 0;
      const __nv_bfloat16* x0 = X + (size_t)pr0 * x_stride;
      float acc[4][2];
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j][0] = acc[j][1] = 0.f;
      struct S { uint4 w; uint2 lo; uint32_t hi; uint2 xa[2]; };
      auto load = [&](S& st, int g) {
        const size_t blk_i = (size_t)cb * KG + g;
        st.w = s5_ld4_256(reinterpret_cast<const uint4*>(base) + blk_i * 32 + lane);
        st.lo = s5_ld2_256(reinterpret_cast<const uint2*>(base + nblk * 512) + blk_i * 4 + t);
        st.hi = s5_ld_u16_256(reinterpret_cast<const uint16_t*>(base + nblk * 544) + blk_i * 4 + t);
        const uint2 zero = make_uint2(0u, 0u);
        const int k0 = g * 32 + 4 * t;
        st.xa[0] = v0 ? *reinterpret_cast<const uint2*>(x0 + k0) : zero;
        st.xa[1] = v0 ? *reinterpret_cast<const uint2*>(x0 + k0 + 16) : zero;
      };
      auto comp_s = [&](const S& st, int g) {
        const uint32_t gbase = ((uint32_t)(cb * KG + g) * 64u) + (uint32_t)t * 16u;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y;
          const uint32_t lo_h = h == 0 ? st.lo.x : st.lo.y;
          const uint32_t hi_h = (st.hi >> (h * 8)) & 0xFFu;
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t word = comp(st.w, j);
            float p[4];
            qmm_frag::mma0(p, a0, 0u, a2, 0u, tf_common::fp4pair(word, 8 * h), tf_common::fp4pair(word, 8 * h + 4));
            const uint32_t pair_b = s5_pair_bytes((lo_h >> (j * 8)) & 0xFFu, (hi_h >> (j * 2)) & 3u,
                                                  smin, meta, esc, e, gbase + (uint32_t)h * 8u + (uint32_t)j * 2u);
            const uint32_t sp = tf_common::fp8pair_unit(pair_b, 0x7B807B80u);
            const float s0 = __uint_as_float((sp & 0xFFFFu) << 16), s1 = __uint_as_float(sp & 0xFFFF0000u);
            acc[j][0] = fmaf(p[0], s0, acc[j][0]);
            acc[j][1] = fmaf(p[1], s1, acc[j][1]);
          }
        }
      };
      S st[2];
      if (0 < KG) load(st[0], 0);
      if (1 < KG) load(st[1], 1);
      for (int g0i = 0; g0i < KG; g0i += 2) {
        if (g0i < KG)     { comp_s(st[0], g0i);     if (g0i + 2 < KG) load(st[0], g0i + 2); }
        if (g0i + 1 < KG) { comp_s(st[1], g0i + 1); if (g0i + 3 < KG) load(st[1], g0i + 3); }
      }
      if (v0) {
        const size_t row = (size_t)pr0 * N;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + row + cb * 32 + 8 * j + 2 * t) =
              make_float2(acc[j][0] * gscale, acc[j][1] * gscale);
        }
      }
    } else {
      s5_down_1w_wide(lane, gq, t, e, cb, first, cnt, X, x_stride, base, KG, NB,
                      gscale, smin, meta, esc, members, out, N);
    }
  }
  qmm_frag::grid_launch();
}

// 2-stage SMEM-staged S5 prefill factored down_f32 (STACK:0, 34,048B SMEM)
template <int NT>
static __device__ __forceinline__ void s5_fact_down_tiles(
    const __nv_bfloat16* __restrict__ X, int x_stride,
    const uint8_t* __restrict__ W_raw, const float* __restrict__ scale,
    const uint2* __restrict__ meta, const uint32_t* __restrict__ esc,
    int KG, int NB, const int* __restrict__ members, void* __restrict__ out, int N,
    int cbt, int e, int first, int cnt) {
  constexpr int WN = 8, THREADS = WN * 32, BM = 64, XC = 8, SG = 2;
  constexpr int CW = SG * 32, LW = SG * 2, HW = 1, CBW = CW + LW + HW;
  constexpr int XU = BM * XC, WU = WN * CBW, SU = XU + WU;
  constexpr int XPT = (XU + THREADS - 1) / THREADS;
  auto chunk = [](int r, int c) { return r * XC + (c ^ ((r & 3) << 1)); };
  extern __shared__ uint4 sm[];
  const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
  const int t = lane & 3, gq = lane >> 2, cb = cbt * WN + wn, cbs = min(WN, NB - cbt * WN);
  const __nv_bfloat16* xsrc[XPT];
  int xdst[XPT];
#pragma unroll
  for (int i = 0; i < XPT; ++i) {
    const int q = tid + i * THREADS, r = q / XC, c = q - r * XC;
    xdst[i] = chunk(r, c);
    xsrc[i] = nullptr;
    if (q < XU && r < cnt) {
      const int p = __ldg(members + first + r);
      xsrc[i] = X + (size_t)p * x_stride + 8 * c;
    }
  }
  const size_t nblk_dn = (size_t)NB * KG;
  const uint8_t* e_dn_base = W_raw + (size_t)e * (nblk_dn * 552);
  auto stage = [&](int s, int g0) {
    const int ng = min(SG, KG - g0);
    uint4* xs = sm + s * SU;
#pragma unroll
    for (int i = 0; i < XPT; ++i)
      if (xsrc[i] && ((tid + i * THREADS) % XC) < ng * 4) cp16(xs + xdst[i], xsrc[i] + (size_t)g0 * 32);
    uint4* ws = xs + XU;
    for (int q = tid; q < cbs * CBW; q += THREADS) {
      const int j = q / CBW, o = q - j * CBW;
      const size_t blk0 = (size_t)(cbt * WN + j) * KG + g0;
      if (o < CW) {
        if (o < ng * 32) cp16(ws + q, reinterpret_cast<const uint4*>(e_dn_base) + blk0 * 32 + o);
      } else if (o < CW + LW) {
        const int lo_o = o - CW;
        if (lo_o < ng * 2) cp16(ws + q, reinterpret_cast<const uint4*>(e_dn_base + nblk_dn * 512) + blk0 * 2 + lo_o);
      } else {
        cp16(ws + q, reinterpret_cast<const uint4*>(e_dn_base + nblk_dn * 544) + (blk0 >> 1));
      }
    }
  };
  constexpr int TA = NT > 0 ? NT : 1;
  const bool active = NT > 0 && cb < NB;
  const uint32_t smin = active ? (__ldg(meta + e).x & 0xFFu) : 0u;
  float acc[1][TA][4][4];
#pragma unroll
  for (int r = 0; r < NT; ++r)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[0][r][j][0] = acc[0][r][j][1] = acc[0][r][j][2] = acc[0][r][j][3] = 0.f;
  const int steps = (KG + SG - 1) / SG;
  if (0 < steps) stage(0, 0);
  cp_commit();
  for (int k = 0; k < steps; ++k) {
    cp_wait<0>();
    __syncthreads();
    if (k + 1 < steps) stage((k + 1) & 1, (k + 1) * SG);
    cp_commit();
    if (!active) continue;
    const uint4* xs = sm + (k & 1) * SU;
    const uint4* wcb = xs + XU + wn * CBW;
    const uint2* sm_lo = reinterpret_cast<const uint2*>(wcb + CW);
    const uint16_t* sm_hi = reinterpret_cast<const uint16_t*>(wcb + CW + LW);
#pragma unroll
    for (int gi = 0; gi < SG; ++gi) {
      const int g = k * SG + gi;
      if (g >= KG) break;
      const uint32_t gbase = ((uint32_t)(cb * KG + g) * 64u) + (uint32_t)t * 16u;
      const uint4 wv = wcb[gi * 32 + lane];
      const uint2 lov = sm_lo[gi * 4 + t];
      const uint32_t hiv = sm_hi[gi * 4 + t];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint2 xa[TA], xb[TA];
        const int u = gi * 8 + h * 4 + t;
#pragma unroll
        for (int r = 0; r < NT; ++r) {
          const int r0 = 16 * r + gq;
          xa[r] = reinterpret_cast<const uint2*>(xs + chunk(r0, u >> 1))[u & 1];
          xb[r] = reinterpret_cast<const uint2*>(xs + chunk(r0 + 8, u >> 1))[u & 1];
        }
        const uint32_t lo_h = h == 0 ? lov.x : lov.y;
        const uint32_t hi_h = (hiv >> (h * 8)) & 0xFFu;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t word = comp(wv, j);
          const uint32_t b0 = tf_common::fp4pair(word, 8 * h), b1 = tf_common::fp4pair(word, 8 * h + 4);
          const uint32_t pair_b = s5_pair_bytes((lo_h >> (j * 8)) & 0xFFu, (hi_h >> (j * 2)) & 3u,
                                                smin, meta, esc, e, gbase + (uint32_t)h * 8u + (uint32_t)j * 2u);
          const uint32_t sp_b = tf_common::fp8pair_unit(pair_b, 0x7B807B80u);
          const float s0 = __uint_as_float((sp_b & 0xFFFFu) << 16), s1 = __uint_as_float(sp_b & 0xFFFF0000u);
#pragma unroll
          for (int r = 0; r < NT; ++r) {
            float p[4];
            qmm_frag::mma0(p, xa[r].x, xb[r].x, xa[r].y, xb[r].y, b0, b1);
            float(&a)[4] = acc[0][r][j];
            a[0] = fmaf(p[0], s0, a[0]); a[1] = fmaf(p[1], s1, a[1]);
            a[2] = fmaf(p[2], s0, a[2]); a[3] = fmaf(p[3], s1, a[3]);
          }
        }
      }
    }
  }
  if (!active) return;
  if constexpr (NT > 0) {
    const float g = __ldg(scale + e);
#pragma unroll
    for (int r = 0; r < NT; ++r)
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][r][j][q] *= g;
#pragma unroll
    for (int r = 0; r < NT; ++r) {
      const int m0 = 16 * r + gq, m1 = m0 + 8;
      const bool v0 = m0 < cnt, v1 = m1 < cnt;
      const int p0 = v0 ? __ldg(members + first + m0) : 0, p1 = v1 ? __ldg(members + first + m1) : 0;
      epilogue<0, 1, TA>(acc, r, out, N, cb * 32 + 2 * t, p0, p1, v0, v1, 0.f);
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) tf_nvfp4_s5_fact_down_f32(
    const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint8_t* __restrict__ W_raw,
    const float* __restrict__ scale, const uint2* __restrict__ meta, const uint32_t* __restrict__ esc,
    int KG, int NB, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, void* __restrict__ out, int N, float limit, int skip) {
  (void)slots; (void)limit;
  const int nbt = (NB + 7) / 8, it = blockIdx.x / nbt, cbt = blockIdx.x - it * nbt;
  if (it >= __ldg(counts)) return;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == skip) return;
  const int nt = max(0, min(4, (cnt + 15) >> 4));
  switch (nt) {
    case 4: s5_fact_down_tiles<4>(X, x_stride, W_raw, scale, meta, esc, KG, NB, members, out, N, cbt, e, first, cnt); break;
    case 3: s5_fact_down_tiles<3>(X, x_stride, W_raw, scale, meta, esc, KG, NB, members, out, N, cbt, e, first, cnt); break;
    case 2: s5_fact_down_tiles<2>(X, x_stride, W_raw, scale, meta, esc, KG, NB, members, out, N, cbt, e, first, cnt); break;
    case 1: s5_fact_down_tiles<1>(X, x_stride, W_raw, scale, meta, esc, KG, NB, members, out, N, cbt, e, first, cnt); break;
    default: s5_fact_down_tiles<0>(X, x_stride, W_raw, scale, meta, esc, KG, NB, members, out, N, cbt, e, first, cnt); break;
  }
}""",
)

# 1d. SM121 Programmatic Dependent Launch (PDL): weights-first qmmf/qmmf_shared, coherent nvfp4_experts, and full decode chain
import re

patch_k(
    "qmmf.cu",
    """__device__ __forceinline__ void qmmf_body(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx,
        int group, int ldy) {
#define TF_QMMF_STRIDED
#include "qmmf_body.cuh"
#undef TF_QMMF_STRIDED
}


template <int MODE, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool CLUSTER, bool FUSE = false>
__global__ void __launch_bounds__(WM * WN * 32) qmmf_kernel(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx,
        int group) {
#include "qmmf_body.cuh"
}""",
    """__device__ __forceinline__ void qmmf_body(
        const __nv_bfloat16* x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx,
        int group, int ldy) {
#define TF_QMMF_STRIDED
#define TF_QMMF_WFIRST
#include "qmmf_body.cuh"
#undef TF_QMMF_WFIRST
#undef TF_QMMF_STRIDED
}


template <int MODE, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool CLUSTER, bool FUSE = false>
__global__ void __launch_bounds__(WM * WN * 32) qmmf_kernel(
        const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx,
        int group) {
#define TF_QMMF_WFIRST
#include "qmmf_body.cuh"
#undef TF_QMMF_WFIRST
}""",
)

patch_k(
    "qmmf.cu",
    """template <bool F32>
__global__ void reduce_kernel(const float* __restrict__ part, void* __restrict__ out, long long total, int SK,
                              float scale) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;""",
    """template <bool F32>
__global__ void reduce_kernel(const float* part, void* __restrict__ out, long long total, int SK,
                              float scale) {
    qmm_frag::grid_wait();
    qmm_frag::grid_launch();
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;""",
)
qmmf_p = kdir / "qmmf.cu"
qmmf_p.write_text(
    qmmf_p.read_text(encoding="utf-8").replace(
        "reduce_kernel<false>(const float* __restrict__,", "reduce_kernel<false>(const float*,"
    ).replace(
        "reduce_kernel<true>(const float* __restrict__,", "reduce_kernel<true>(const float*,"
    ),
    encoding="utf-8",
)

patch_k(
    "qmmf_body.cuh",
    """    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
#ifdef TF_QMMF_K32
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * BK + ch * 8,
                  m0 + r < M);
#else
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
#endif
        }
        unsigned char* pw = p + T::X;""",
    """    auto load = [&](int s, int g, int what = 3) {
        unsigned char* p = stage(s);
        if (what & 1)
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
#ifdef TF_QMMF_K32
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * BK + ch * 8,
                  m0 + r < M);
#else
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
#endif
        }
        if (!(what & 2)) return;
        unsigned char* pw = p + T::X;""",
)

patch_k(
    "qmmf_body.cuh",
    """#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load(s, g0 + s);
        commit();
    }""",
    """#ifdef TF_QMMF_WFIRST
    if constexpr (BN <= 64) {
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s)
            if (s < per) load(s, g0 + s, 2);
        qmm_frag::grid_wait();
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < per) load(s, g0 + s, 1);
            commit();
        }
    } else
#endif
    {
#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < per) load(s, g0 + s);
            commit();
        }
    }""",
)

patch_k(
    "qmmf_body.cuh",
    """    wait<0>();
    __syncthreads();
    if constexpr (FUSE) {""",
    """    wait<0>();
    __syncthreads();
#ifdef TF_QMMF_WFIRST
    if constexpr (BN <= 64) qmm_frag::grid_launch();
#endif
    if constexpr (FUSE) {""",
)

# nvfp4_experts.cu: coherent dynamic reads + non-cascading PDL (keep pipelined Stage<M>/Half<M> intact)
nvp = kdir / "nvfp4_experts.cu"
nvtxt = nvp.read_text(encoding="utf-8")
for old_s, new_s in [
    ("st.xa[h] = v0 ? __ldg(reinterpret_cast<const uint2*>(x0 + k0 + 16 * h)) : zero;",
     "st.xa[h] = v0 ? *reinterpret_cast<const uint2*>(x0 + k0 + 16 * h) : zero;"),
    ("st.xb[h] = v1 ? __ldg(reinterpret_cast<const uint2*>(x1 + k0 + 16 * h)) : zero;",
     "st.xb[h] = v1 ? *reinterpret_cast<const uint2*>(x1 + k0 + 16 * h) : zero;"),
]:
    if nvtxt.count(old_s) != 2:
        raise RuntimeError(f"Expected 2 matches for {old_s} in nvfp4_experts.cu, got {nvtxt.count(old_s)}")
    nvtxt = nvtxt.replace(old_s, new_s)

for old_s, new_s in [
    ("""template <int M, int EPI>
__device__ __forceinline__ void unit_full(int unit, int lane, int gq, int t, const __nv_bfloat16* __restrict__ X,
                                          int x_stride, int slots, const uint4* __restrict__ W,
                                          const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                                          const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                                          int skip) {
  const int it = unit / NB, cb = unit - it * NB;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == skip) return;
  const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
  for (int r0 = 0; r0 < cnt; r0 += 16) {
    const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
    const int pr0 = v0 ? __ldg(members + first + r0 + gq) : 0;
    const int pr1 = v1 ? __ldg(members + first + r0 + gq + 8) : 0;
    const int x0r = slots ? pr0 / slots : pr0, x1r = slots ? pr1 / slots : pr1;
    const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
    const __nv_bfloat16* x1 = X + (size_t)x1r * x_stride;
    float acc[M][1][NTW][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
    k_loop<M>(acc, blk, KG, lane, t, x0, x1, v0, v1);
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float g = __ldg(scale + e * M + m);
#pragma unroll
      for (int j = 0; j < NTW; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[m][0][j][q] *= g;
    }
    epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit);
  }
}""",
     """template <int M, int WARPS>
struct SprBox {
  static __device__ __forceinline__ int* ptr(int warp) {
    __shared__ int s[WARPS][18];
    return s[warp];
  }
};
template <int WARPS>
struct SprBox<1, WARPS> {
  static __device__ __forceinline__ int* ptr(int) { return nullptr; }
};

template <int M, int EPI>
__device__ __forceinline__ void unit_full(int unit, int lane, int gq, int t, int* spr, const __nv_bfloat16* X,
                                          int x_stride, int slots, const uint4* __restrict__ W,
                                          const float* __restrict__ scale, int KG, int NB, const int* items,
                                          const int* members, void* __restrict__ out, int N, float limit,
                                          int skip) {
  const int it = unit / NB, cb = unit - it * NB;
  const int e = items[3 * it], first = items[3 * it + 1], cnt = items[3 * it + 2];
  if (e == skip) return;
  if constexpr (M == 2) {
    if (lane == 0) { spr[16] = e; spr[17] = cb; }
  }
  const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
  for (int r0 = 0; r0 < cnt; r0 += 16) {
    const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
    const int pr0 = v0 ? members[first + r0 + gq] : 0;
    const int pr1 = v1 ? members[first + r0 + gq + 8] : 0;
    if constexpr (M == 2) {
      __syncwarp();
      if (t == 0) { spr[gq] = pr0; spr[8 + gq] = pr1; }
      __syncwarp();
    }
    const int x0r = slots ? pr0 / slots : pr0, x1r = slots ? pr1 / slots : pr1;
    const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
    const __nv_bfloat16* x1 = X + (size_t)x1r * x_stride;
    float acc[M][1][NTW][4];
#pragma unroll
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
    k_loop<M>(acc, blk, KG, lane, t, x0, x1, v0, v1);
    int e_ep = e, cb_ep = cb, pr0_ep = pr0, pr1_ep = pr1;
    if constexpr (M == 2) {
      e_ep = spr[16];
      cb_ep = spr[17];
      pr0_ep = spr[gq];
      pr1_ep = spr[8 + gq];
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float g = __ldg(scale + e_ep * M + m);
#pragma unroll
      for (int j = 0; j < NTW; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[m][0][j][q] *= g;
    }
    epilogue<EPI, M, 1>(acc, 0, out, N, cb_ep * COLS + 2 * t, pr0_ep, pr1_ep, v0, v1, limit);
  }
}"""),
    ("""    nvfp4_expert_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                        const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                        const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                        float limit, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS)
    unit_full<M, EPI>(unit, lane, gq, t, X, x_stride, slots, W, scale, KG, NB, items, members, out, N, limit, skip);
}""",
     """    nvfp4_expert_kernel(const __nv_bfloat16* X, int x_stride, int slots, const uint4* __restrict__ W,
                        const float* __restrict__ scale, int KG, int NB, const int* items,
                        const int* counts, const int* members, void* __restrict__ out, int N,
                        float limit, int skip) {
  qmm_frag::grid_wait();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  int* spr = SprBox<M, WARPS>::ptr(warp);
  const int units = (*counts) * NB;
  for (int unit = blockIdx.x * WARPS + warp; unit < units; unit += gridDim.x * WARPS)
    unit_full<M, EPI>(unit, lane, gq, t, spr, X, x_stride, slots, W, scale, KG, NB, items, members, out, N, limit, skip);
  qmm_frag::grid_launch();
}"""),
    ("""    nvfp4_expert_waves(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                       const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                       const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                       float limit, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  if (units == 0) return;  // no items: no wave to divide by
  const int warps = gridDim.x * WARPS, waves = (units + warps - 1) / warps;
  const int run = (units + waves - 1) / waves, w = blockIdx.x * WARPS + (threadIdx.x >> 5);
  if (w >= run) return;
  for (int unit = w; unit < units; unit += run)
    unit_full<M, EPI>(unit, lane, gq, t, X, x_stride, slots, W, scale, KG, NB, items, members, out, N, limit, skip);
}""",
     """    nvfp4_expert_waves(const __nv_bfloat16* X, int x_stride, int slots, const uint4* __restrict__ W,
                       const float* __restrict__ scale, int KG, int NB, const int* items,
                       const int* counts, const int* members, void* __restrict__ out, int N,
                       float limit, int skip) {
  qmm_frag::grid_wait();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  int* spr = SprBox<M, WARPS>::ptr(warp);
  const int units = (*counts) * NB;
  if (units == 0) { qmm_frag::grid_launch(); return; }
  const int warps = gridDim.x * WARPS, waves = (units + warps - 1) / warps;
  const int run = (units + waves - 1) / waves, w = blockIdx.x * WARPS + warp;
  if (w >= run) { qmm_frag::grid_launch(); return; }
  for (int unit = w; unit < units; unit += run)
    unit_full<M, EPI>(unit, lane, gq, t, spr, X, x_stride, slots, W, scale, KG, NB, items, members, out, N, limit, skip);
  qmm_frag::grid_launch();
}"""),
    ("""    nvfp4_expert_half(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                      const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                      const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                      float limit, int skip) {
  constexpr int D = 2;
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB * 2;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / (NB * 2), rest = unit - it * NB * 2, cb = rest >> 1, jh = rest & 1;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
    for (int r0 = 0; r0 < cnt; r0 += 16) {
      const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
      const int pr0 = v0 ? __ldg(members + first + r0 + gq) : 0;
      const int pr1 = v1 ? __ldg(members + first + r0 + gq + 8) : 0;""",
     """    nvfp4_expert_half(const __nv_bfloat16* X, int x_stride, int slots, const uint4* __restrict__ W,
                      const float* __restrict__ scale, int KG, int NB, const int* items,
                      const int* counts, const int* members, void* __restrict__ out, int N,
                      float limit, int skip) {
  qmm_frag::grid_wait();
  constexpr int D = 2;
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = (*counts) * NB * 2;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / (NB * 2), rest = unit - it * NB * 2, cb = rest >> 1, jh = rest & 1;
    const int e = items[3 * it], first = items[3 * it + 1], cnt = items[3 * it + 2];
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
    for (int r0 = 0; r0 < cnt; r0 += 16) {
      const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
      const int pr0 = v0 ? members[first + r0 + gq] : 0;
      const int pr1 = v1 ? members[first + r0 + gq + 8] : 0;"""),
    ("""      epilogue_half<EPI, M>(acc, out, N, cb * COLS + 16 * jh + 2 * t, pr0, pr1, v0, v1, limit);
    }
  }
}""",
     """      epilogue_half<EPI, M>(acc, out, N, cb * COLS + 16 * jh + 2 * t, pr0, pr1, v0, v1, limit);
    }
  }
  qmm_frag::grid_launch();
}"""),
]:
    if nvtxt.count(old_s) != 1:
        raise RuntimeError(f"Expected 1 match in nvfp4_experts.cu, got {nvtxt.count(old_s)}")
    nvtxt = nvtxt.replace(old_s, new_s, 1)

nvtxt = nvtxt.replace(
    "(const __nv_bfloat16* __restrict__, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int* __restrict__, const int* __restrict__, const int* __restrict__, void* __restrict__, int, float, int);",
    "(const __nv_bfloat16*, int, int, const uint4* __restrict__, const float* __restrict__, int, int, const int*, const int*, const int*, void* __restrict__, int, float, int);",
)
nvp.write_text(nvtxt, encoding="utf-8")

# experts.cu: 4-bucket descending-cnt place_items (cnt>8 first, then 3..8, 2, 1), O(1) atomic scatter, e<E bounds
patch_k(
    "experts.cu",
    """__device__ int place_items(int c, int E, int T, int* __restrict__ items, int* __restrict__ counts) {
  __shared__ int wsum[3][32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tiles = (c + T - 1) / T;
  int a = c, b = tiles, u = c > 0;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int ua = __shfl_up_sync(0xffffffffu, a, o), ub = __shfl_up_sync(0xffffffffu, b, o);
    const int uu = __shfl_up_sync(0xffffffffu, u, o);
    if (lane >= o) {
      a += ua;
      b += ub;
      u += uu;
    }
  }
  if (lane == 31) {
    wsum[0][warp] = a;
    wsum[1][warp] = b;
    wsum[2][warp] = u;
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int s = 0; s < 3; ++s) {
      int v = wsum[s][lane];
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int w = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += w;
      }
      wsum[s][lane] = v;
    }
  }
  __syncthreads();
  const int off = (warp ? wsum[0][warp - 1] : 0) + a - c;
  const int ioff = (warp ? wsum[1][warp - 1] : 0) + b - tiles;
  if (tid == 0) {
    counts[0] = wsum[1][31];
    counts[1] = wsum[2][31];
  }
  if (tid < E)
    for (int j = 0; j < tiles; ++j) {
      int* it = items + 3 * (ioff + j);
      it[0] = tid;
      it[1] = off + T * j;
      it[2] = min(T, c - T * j);
    }
  return off;
}

__global__ void __launch_bounds__(PLAN_THREADS)
    plan_kernel(const int* __restrict__ picks, int P, int E, int T, int* __restrict__ members,
                int* __restrict__ items, int* __restrict__ counts) {
  __shared__ int pk[PSMALL];
  __shared__ int cnt[EMAX];
  __shared__ int off[EMAX];
  const int tid = threadIdx.x;
  for (int e = tid; e < EMAX; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = picks[p];
    pk[p] = e;
    atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  const int o = place_items(tid < E ? cnt[tid] : 0, E, T, items, counts);
  if (tid < E) off[tid] = o;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = pk[p];
    int rank = 0;
    for (int q = 0; q < p; ++q) rank += pk[q] == e;
    members[off[e] + rank] = p;
  }
}

// plan_kernel over the first *rows rows' pairs (per a row): a captured bucket's padding rows get no items.
__global__ void __launch_bounds__(PLAN_THREADS)
    plan_rows(const int* __restrict__ picks, int P, int E, int T, int* __restrict__ members,
              int* __restrict__ items, int* __restrict__ counts, const int* __restrict__ rows, int per) {
  P = min(P, *rows * per);
  __shared__ int pk[PSMALL];
  __shared__ int cnt[EMAX];
  __shared__ int off[EMAX];
  const int tid = threadIdx.x;
  for (int e = tid; e < EMAX; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = picks[p];
    pk[p] = e;
    atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  const int o = place_items(tid < E ? cnt[tid] : 0, E, T, items, counts);
  if (tid < E) off[tid] = o;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = pk[p];
    int rank = 0;
    for (int q = 0; q < p; ++q) rank += pk[q] == e;
    members[off[e] + rank] = p;
  }
}

__device__ __forceinline__ void rank_pairs(const int* __restrict__ picks, int P, int E, int* __restrict__ rank,
                                           int* __restrict__ hist) {
  __shared__ int cnt[EMAX];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int e = tid; e < E; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  const int p = blockIdx.x * PLAN_THREADS + tid;
  const bool ok = p < P;
  const int e = ok ? picks[p] : -1 - lane;""",
    """__device__ int place_items(int c, int E, int T, int* __restrict__ items, int* __restrict__ counts) {
  __shared__ int wsum[6][32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tiles = (c + T - 1) / T;
  const int rem = c > 0 ? (c - T * (tiles - 1)) : 0;
  const int b0 = (c > 0 && T > 8) ? ((tiles - 1) + (rem > 8 ? 1 : 0)) : 0;
  const int b1 = (c > 0 && T > 8 && rem >= 3 && rem <= 8) ? 1 : ((c > 0 && T <= 8) ? tiles : 0);
  const int b2 = (c > 0 && T > 8 && rem == 2) ? 1 : 0;
  const int b3 = (c > 0 && T > 8 && rem == 1) ? 1 : 0;
  int a0 = c, a1 = c > 0, a2 = b0, a3 = b1, a4 = b2, a5 = b3;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int u0 = __shfl_up_sync(0xffffffffu, a0, o), u1 = __shfl_up_sync(0xffffffffu, a1, o);
    const int u2 = __shfl_up_sync(0xffffffffu, a2, o), u3 = __shfl_up_sync(0xffffffffu, a3, o);
    const int u4 = __shfl_up_sync(0xffffffffu, a4, o), u5 = __shfl_up_sync(0xffffffffu, a5, o);
    if (lane >= o) { a0 += u0; a1 += u1; a2 += u2; a3 += u3; a4 += u4; a5 += u5; }
  }
  if (lane == 31) {
    wsum[0][warp] = a0; wsum[1][warp] = a1; wsum[2][warp] = a2;
    wsum[3][warp] = a3; wsum[4][warp] = a4; wsum[5][warp] = a5;
  }
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int s = 0; s < 6; ++s) {
      int v = wsum[s][lane];
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int w = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += w;
      }
      wsum[s][lane] = v;
    }
  }
  __syncthreads();
  const int off = (warp ? wsum[0][warp - 1] : 0) + a0 - c;
  const int ioff0 = (warp ? wsum[2][warp - 1] : 0) + a2 - b0;
  const int ioff1 = (warp ? wsum[3][warp - 1] : 0) + a3 - b1;
  const int ioff2 = (warp ? wsum[4][warp - 1] : 0) + a4 - b2;
  const int ioff3 = (warp ? wsum[5][warp - 1] : 0) + a5 - b3;
  const int tot0 = wsum[2][31], tot1 = wsum[3][31], tot2 = wsum[4][31], tot3 = wsum[5][31];
  if (tid == 0) {
    counts[0] = tot0 + tot1 + tot2 + tot3;
    counts[1] = wsum[1][31];
  }
  if (tid < E && c > 0) {
    if (T > 8) {
      for (int j = 0; j < b0; ++j) {
        int* it = items + 3 * (ioff0 + j);
        it[0] = tid; it[1] = off + T * j; it[2] = min(T, c - T * j);
      }
      if (b1) {
        int* it = items + 3 * (tot0 + ioff1);
        it[0] = tid; it[1] = off + T * (tiles - 1); it[2] = rem;
      }
      if (b2) {
        int* it = items + 3 * (tot0 + tot1 + ioff2);
        it[0] = tid; it[1] = off + T * (tiles - 1); it[2] = 2;
      }
      if (b3) {
        int* it = items + 3 * (tot0 + tot1 + tot2 + ioff3);
        it[0] = tid; it[1] = off + T * (tiles - 1); it[2] = 1;
      }
    } else {
      for (int j = 0; j < tiles; ++j) {
        int* it = items + 3 * (ioff1 + j);
        it[0] = tid; it[1] = off + T * j; it[2] = min(T, c - T * j);
      }
    }
  }
  return off;
}

__global__ void __launch_bounds__(PLAN_THREADS)
    plan_kernel(const int* __restrict__ picks, int P, int E, int T, int* __restrict__ members,
                int* __restrict__ items, int* __restrict__ counts) {
  __shared__ int pk[PSMALL];
  __shared__ int cnt[EMAX];
  __shared__ int off[EMAX];
  const int tid = threadIdx.x;
  for (int e = tid; e < E; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = picks[p];
    pk[p] = e;
    if (e >= 0 && e < E) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  const int o = place_items(tid < E ? cnt[tid] : 0, E, T, items, counts);
  if (tid < E) off[tid] = o;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = pk[p];
    if (e >= 0 && e < E) {
      const int pos = atomicAdd(&off[e], 1);
      members[pos] = p;
    }
  }
}

// plan_kernel over the first *rows rows' pairs (per a row): a captured bucket's padding rows get no items.
__global__ void __launch_bounds__(PLAN_THREADS)
    plan_rows(const int* __restrict__ picks, int P, int E, int T, int* __restrict__ members,
              int* __restrict__ items, int* __restrict__ counts, const int* __restrict__ rows, int per) {
  P = min(P, *rows * per);
  __shared__ int pk[PSMALL];
  __shared__ int cnt[EMAX];
  __shared__ int off[EMAX];
  const int tid = threadIdx.x;
  for (int e = tid; e < E; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = picks[p];
    pk[p] = e;
    if (e >= 0 && e < E) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  const int o = place_items(tid < E ? cnt[tid] : 0, E, T, items, counts);
  if (tid < E) off[tid] = o;
  __syncthreads();
  for (int p = tid; p < P; p += PLAN_THREADS) {
    const int e = pk[p];
    if (e >= 0 && e < E) {
      const int pos = atomicAdd(&off[e], 1);
      members[pos] = p;
    }
  }
}

__device__ __forceinline__ void rank_pairs(const int* __restrict__ picks, int P, int E, int* __restrict__ rank,
                                           int* __restrict__ hist) {
  __shared__ int cnt[EMAX];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int e = tid; e < E; e += PLAN_THREADS) cnt[e] = 0;
  __syncthreads();
  const int p = blockIdx.x * PLAN_THREADS + tid;
  const int raw = p < P ? picks[p] : -1;
  const bool ok = p < P && raw >= 0 && raw < E;
  const int e = ok ? raw : -1 - lane;""",
)
patch_k(
    "experts.cu",
    """__global__ void plan_scatter(const int* __restrict__ picks, int P, int E, const int* __restrict__ rank,
                             const int* __restrict__ hist, int* __restrict__ members) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < P) members[hist[(size_t)(p / PLAN_THREADS) * E + picks[p]] + rank[p]] = p;
}

// plan_scatter over the first *rows rows' pairs (per a row).
__global__ void plan_scatter_rows(const int* __restrict__ picks, int P, int E, const int* __restrict__ rank,
                                  const int* __restrict__ hist, int* __restrict__ members, const int* __restrict__ rows,
                                  int per) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < min(P, *rows * per)) members[hist[(size_t)(p / PLAN_THREADS) * E + picks[p]] + rank[p]] = p;
}""",
    """__global__ void plan_scatter(const int* __restrict__ picks, int P, int E, const int* __restrict__ rank,
                             const int* __restrict__ hist, int* __restrict__ members) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < P) {
    const int e = picks[p];
    if (e >= 0 && e < E) members[hist[(size_t)(p / PLAN_THREADS) * E + e] + rank[p]] = p;
  }
}

// plan_scatter over the first *rows rows' pairs (per a row).
__global__ void plan_scatter_rows(const int* __restrict__ picks, int P, int E, const int* __restrict__ rank,
                                  const int* __restrict__ hist, int* __restrict__ members, const int* __restrict__ rows,
                                  int per) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < min(P, *rows * per)) {
    const int e = picks[p];
    if (e >= 0 && e < E) members[hist[(size_t)(p / PLAN_THREADS) * E + e] + rank[p]] = p;
  }
}""",
)

# route_rows.cu + router_mma.cuh: L2::256B weight prefetch (-15.4% router latency, bit-exact) + 1-warp zero-barrier tf_route_topk
patch_k(
    "common/router_mma.cuh",
    "__device__ __forceinline__ void logits(const __nv_bfloat16* __restrict__ X, int x_ld, const __nv_bfloat16* __restrict__ W,",
    "__device__ __forceinline__ void logits(const __nv_bfloat16* X, int x_ld, const __nv_bfloat16* __restrict__ W,",
)
patch_k(
    "common/router_mma.cuh",
    "            qmm_frag::cp16z(&tile[buf][row][8 * (v % (KC / 8))], src, read);",
    """            if (!x_row) {
#if __CUDA_ARCH__ >= 900
                const uint32_t sm = static_cast<uint32_t>(__cvta_generic_to_shared(&tile[buf][row][8 * (v % (KC / 8))]));
                asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16, %2;\\n" :: "r"(sm), "l"(src), "r"(read ? 16 : 0));
#else
                qmm_frag::cp16z(&tile[buf][row][8 * (v % (KC / 8))], src, read);
#endif
            } else {
                qmm_frag::cp16z(&tile[buf][row][8 * (v % (KC / 8))], src, read);
            }""",
)
patch_k(
    "route_rows.cu",
    """extern "C" __global__ void __launch_bounds__(NT) tf_route_topk(const float* __restrict__ L, int NL, int NE, int TOPK,
                                                               int* __restrict__ PICK, float* __restrict__ WTS, int SLOTS,
                                                               int shared) {
    __shared__ float sv[2][NT / 32];
    __shared__ int si[2][NT / 32];
    const int r = blockIdx.x;
    const float* row = L + static_cast<long long>(r) * NL;""",
    """extern "C" __global__ void __launch_bounds__(NT) tf_route_topk(const float* __restrict__ L, int NL, int NE, int TOPK,
                                                               int* __restrict__ PICK, float* __restrict__ WTS, int SLOTS,
                                                               int shared) {
    const int r = blockIdx.x;
    const float* row = L + static_cast<long long>(r) * NL;
    if (blockDim.x == 32 && NE <= 512 && TOPK <= 32) {
        const int lane = threadIdx.x;
        float wv[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int i = lane + j * 32;
            wv[j] = i < NE ? row[i] : -INFINITY;
        }
        float top = 0.0f, total = 0.0f, mine = 0.0f;
        int pick = 0;
        for (int k = 0; k < TOPK; ++k) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int i = lane + j * 32;
                if (wv[j] > bv || (wv[j] == bv && i < bi)) {
                    bv = wv[j];
                    bi = i;
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bi = oi;
                }
            }
            if (k == 0) top = bv;
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (lane + j * 32 == bi) wv[j] = -INFINITY;
            const float ex = tf_exp(__fsub_rn(bv, top));
            total = __fadd_rn(total, ex);
            if (lane == k) {
                mine = ex;
                pick = bi;
            }
        }
        if (lane < TOPK) {
            PICK[static_cast<long long>(r) * SLOTS + lane] = pick;
            WTS[static_cast<long long>(r) * SLOTS + lane] = rbf(__fdiv_rn(mine, total));
        }
        if (shared && lane == 0) {
            PICK[static_cast<long long>(r) * SLOTS + TOPK] = NE;
            WTS[static_cast<long long>(r) * SLOTS + TOPK] = rbf(tf_sigmoid(rbf(row[NE])));
        }
        return;
    }
    __shared__ float sv[2][NT / 32];
    __shared__ int si[2][NT / 32];""",
)

# Trial 48: 1-read-all-4-streams vectorized uint4/float4 hc_writeback_norm (320 threads, 8 cols/thread, 2.25x faster on MoE/slices!)
patch_k(
    "flashnext_hc.cu",
    """    constexpr int D = NC * 256;
    __shared__ float red[NC][8][S];
    __shared__ float rinv[S];
    const int r = blockIdx.x, warp = threadIdx.x >> 5;
    float hv[NC][S];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const int d = c * 256 + threadIdx.x;
        float branch = 0.0f;
        if constexpr (MODE == 4) {
            float acc = BR[static_cast<long long>(r) * D + d];
            for (int k = 1; k < WORLD; ++k) acc = __fadd_rn(acc, BR[k * RS + static_cast<long long>(r) * D + d]);
            branch = rbf(acc);
        } else if constexpr (MODE == 2) {
            float acc = 0.0f;
            for (int k = 0; k <= TOPK; ++k) {
                const long long at = static_cast<long long>(r) * SLOTS + k;
                acc = __fmaf_rn(Y[at * D + d], WTS[at], acc);
            }
            branch = rbf(acc);
        }
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const long long at = static_cast<long long>(r) * S * D + static_cast<long long>(s) * D + d;
            float v = bf(H[at]);
            if constexpr (MODE != 0) {
                v = rbf(__fmaf_rn(branch, bf(INJ[r * S + s]), v));
                HOUT[at] = __float2bfloat16_rn(v);
            }
            hv[c][s] = v;
            const float ss = warp_sum(__fmul_rn(v, v));
            if ((threadIdx.x & 31) == 0) red[c][warp][s] = ss;
        }
    }
    __syncthreads();
    if (threadIdx.x < S) {
        float all = 0.0f;
        for (int c = 0; c < NC; ++c) {
            float total = 0.0f;
            for (int w = 0; w < 8; ++w) total = __fadd_rn(total, red[c][w][threadIdx.x]);
            PSS[(static_cast<long long>(r) * NC + c) * S + threadIdx.x] = total;
            all = __fadd_rn(all, total);
        }
        rinv[threadIdx.x] = tf_common::rms_scale(all, D, eps);
    }
    __syncthreads();
#pragma unroll
    for (int c = 0; c < NC; ++c)
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const int col = s * D + c * 256 + threadIdx.x;
            NORM[static_cast<long long>(r) * S * D + col] =
                __float2bfloat16_rn(__fmul_rn(__fmul_rn(hv[c][s], rinv[s]), SCALE[col]));
        }""",
    """    constexpr int D = NC * 256;
    __shared__ float sw[S][NC];
    __shared__ float rinv[S];
    __shared__ float swts[16];
    const int r = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if constexpr (MODE == 2) {
        if (tid <= TOPK) swts[tid] = WTS[static_cast<long long>(r) * SLOTS + tid];
        __syncthreads();
    }
    float branch[8];
    if constexpr (MODE == 4) {
        const float4* br0 = reinterpret_cast<const float4*>(BR + static_cast<long long>(r) * D);
        float4 a0 = br0[tid * 2];
        float4 a1 = br0[tid * 2 + 1];
        const long long rs4 = RS >> 2;
        for (int k = 1; k < WORLD; ++k) {
            const float4* brk = br0 + static_cast<long long>(k) * rs4;
            const float4 b0 = brk[tid * 2];
            const float4 b1 = brk[tid * 2 + 1];
            a0.x = __fadd_rn(a0.x, b0.x); a0.y = __fadd_rn(a0.y, b0.y);
            a0.z = __fadd_rn(a0.z, b0.z); a0.w = __fadd_rn(a0.w, b0.w);
            a1.x = __fadd_rn(a1.x, b1.x); a1.y = __fadd_rn(a1.y, b1.y);
            a1.z = __fadd_rn(a1.z, b1.z); a1.w = __fadd_rn(a1.w, b1.w);
        }
        branch[0] = rbf(a0.x); branch[1] = rbf(a0.y); branch[2] = rbf(a0.z); branch[3] = rbf(a0.w);
        branch[4] = rbf(a1.x); branch[5] = rbf(a1.y); branch[6] = rbf(a1.z); branch[7] = rbf(a1.w);
    } else if constexpr (MODE == 2) {
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float4* yp0 = reinterpret_cast<const float4*>(Y + static_cast<long long>(r) * SLOTS * D);
        for (int k = 0; k <= TOPK; ++k) {
            const float wk = swts[k];
            if (wk == 0.0f) continue;
            const float4* yk = yp0 + static_cast<long long>(k) * (D / 4);
            const float4 v0 = yk[tid * 2];
            const float4 v1 = yk[tid * 2 + 1];
            acc[0] = __fmaf_rn(v0.x, wk, acc[0]); acc[1] = __fmaf_rn(v0.y, wk, acc[1]);
            acc[2] = __fmaf_rn(v0.z, wk, acc[2]); acc[3] = __fmaf_rn(v0.w, wk, acc[3]);
            acc[4] = __fmaf_rn(v1.x, wk, acc[4]); acc[5] = __fmaf_rn(v1.y, wk, acc[5]);
            acc[6] = __fmaf_rn(v1.z, wk, acc[6]); acc[7] = __fmaf_rn(v1.w, wk, acc[7]);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) branch[i] = rbf(acc[i]);
    }
    const uint4* H4 = reinterpret_cast<const uint4*>(H);
    uint4* HOUT4 = reinterpret_cast<uint4*>(HOUT);
    uint4* NORM4 = reinterpret_cast<uint4*>(NORM);
    float hv[S][8];
#pragma unroll
    for (int s = 0; s < S; ++s) {
        float inj_s = 0.0f;
        if constexpr (MODE != 0) inj_s = bf(INJ[r * S + s]);
        const long long at4 = (static_cast<long long>(r) * S + s) * (D / 8) + tid;
        const uint4 hu_v = H4[at4];
        const uint32_t hu[4] = {hu_v.x, hu_v.y, hu_v.z, hu_v.w};
        uint32_t ho[4];
        float q[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float v0 = __uint_as_float((hu[i] & 0xFFFFu) << 16);
            float v1 = __uint_as_float(hu[i] & 0xFFFF0000u);
            if constexpr (MODE != 0) {
                v0 = rbf(__fmaf_rn(branch[i * 2], inj_s, v0));
                v1 = rbf(__fmaf_rn(branch[i * 2 + 1], inj_s, v1));
                const __nv_bfloat16 b0 = __float2bfloat16_rn(v0);
                const __nv_bfloat16 b1 = __float2bfloat16_rn(v1);
                ho[i] = static_cast<uint32_t>(__bfloat16_as_ushort(b0)) |
                        (static_cast<uint32_t>(__bfloat16_as_ushort(b1)) << 16);
            }
            hv[s][i * 2] = v0;
            hv[s][i * 2 + 1] = v1;
            q[i * 2] = __fmul_rn(v0, v0);
            q[i * 2 + 1] = __fmul_rn(v1, v1);
        }
        if constexpr (MODE != 0) HOUT4[at4] = make_uint4(ho[0], ho[1], ho[2], ho[3]);
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = __fadd_rn(q[k], __shfl_xor_sync(0xffffffffu, q[k], 2));
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = __fadd_rn(q[k], __shfl_xor_sync(0xffffffffu, q[k], 1));
        const float wsum = __fadd_rn(
            __fadd_rn(__fadd_rn(q[0], q[4]), __fadd_rn(q[2], q[6])),
            __fadd_rn(__fadd_rn(q[1], q[5]), __fadd_rn(q[3], q[7])));
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const float rw = __shfl_sync(0xffffffffu, wsum, w * 4);
            if (lane == 0) total = __fadd_rn(total, rw);
        }
        if (lane == 0) {
            PSS[(static_cast<long long>(r) * NC + warp) * S + s] = total;
            sw[s][warp] = total;
        }
    }
    __syncthreads();
    if (tid < S) {
        float all = 0.0f;
        for (int c = 0; c < NC; ++c) all = __fadd_rn(all, sw[tid][c]);
        rinv[tid] = tf_common::rms_scale(all, D, eps);
    }
    __syncthreads();
#pragma unroll
    for (int s = 0; s < S; ++s) {
        const float rv = rinv[s];
        const float4* scp = reinterpret_cast<const float4*>(SCALE + s * D);
        const float4 sc0 = __ldg(scp + tid * 2);
        const float4 sc1 = __ldg(scp + tid * 2 + 1);
        const float sc[8] = {sc0.x, sc0.y, sc0.z, sc0.w, sc1.x, sc1.y, sc1.z, sc1.w};
        uint32_t no[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __nv_bfloat16 n0 = __float2bfloat16_rn(__fmul_rn(__fmul_rn(hv[s][i * 2], rv), sc[i * 2]));
            const __nv_bfloat16 n1 = __float2bfloat16_rn(__fmul_rn(__fmul_rn(hv[s][i * 2 + 1], rv), sc[i * 2 + 1]));
            no[i] = static_cast<uint32_t>(__bfloat16_as_ushort(n0)) |
                    (static_cast<uint32_t>(__bfloat16_as_ushort(n1)) << 16);
        }
        const long long at4 = (static_cast<long long>(r) * S + s) * (D / 8) + tid;
        NORM4[at4] = make_uint4(no[0], no[1], no[2], no[3]);
    }""",
)
for _wbn_fn in (
    "fn_hc_writeback_norm0(",
    "fn_hc_writeback_norm_moe(",
    "fn_hc_writeback_norm_slices(",
):
    patch_k(
        "flashnext_hc.cu",
        f'extern "C" __global__ void __launch_bounds__(256) {_wbn_fn}',
        f'extern "C" __global__ void __launch_bounds__(320) {_wbn_fn}',
    )

# Fast SK<=16 zero-SMEM zero-barrier fn_hc_reduce_act_s1 + fused shared_gu reduce/cat SwiGLU + skip dead XSM in fn_hc_dense_mix8
patch_k(
    "flashnext_hc.cu",
    """extern "C" __global__ void __launch_bounds__(1024) fn_hc_reduce_act(const float* __restrict__ PART, __nv_bfloat16* __restrict__ ACT,""",
    r"""extern "C" __global__ void __launch_bounds__(384) fn_hc_reduce_act_s1(const float* PART, __nv_bfloat16* __restrict__ ACT,
                                                                        float* __restrict__ XS, __nv_bfloat16* __restrict__ INJ,
                                                                        int SK, int M, int S, int LOW, int HAS_INJ, int NDN) {
    (void)XS;
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
    const int r = blockIdx.x, i = threadIdx.x;
    if (i < LOW + (HAS_INJ ? S : 0)) {
        const long long stride = static_cast<long long>(M) * NDN;
        const long long base = static_cast<long long>(r) * NDN + i;
        float acc = PART[base];
        for (int s = 1; s < SK; ++s) acc = __fadd_rn(acc, PART[base + static_cast<long long>(s) * stride]);
        const float v = rbf(__fdiv_rn(rbf(acc), static_cast<float>(S)));
        if (i < LOW) {
            ACT[static_cast<long long>(r) * LOW + i] = __float2bfloat16_rn(bsilu(v));
        } else {
            INJ[static_cast<long long>(r) * S + (i - LOW)] = __float2bfloat16_rn(__fmul_rn(2.0f, bsig(v)));
        }
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

extern "C" __global__ void __launch_bounds__(384) fn_hc_reduce_act_s1_sc(const float* PART, __nv_bfloat16* __restrict__ ACT,
                                                                           float* __restrict__ XS, __nv_bfloat16* __restrict__ INJ,
                                                                           int SK, int M, int S, int LOW, int HAS_INJ, int NDN, float GSCALE) {
    (void)XS;
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
    const int r = blockIdx.x, i = threadIdx.x;
    if (i < LOW + (HAS_INJ ? S : 0)) {
        const long long stride = static_cast<long long>(M) * NDN;
        const long long base = static_cast<long long>(r) * NDN + i;
        float acc = PART[base];
        for (int s = 1; s < SK; ++s) acc = __fadd_rn(acc, PART[base + static_cast<long long>(s) * stride]);
        acc = __fmul_rn(acc, GSCALE);
        const float v = rbf(__fdiv_rn(rbf(acc), static_cast<float>(S)));
        if (i < LOW) {
            ACT[static_cast<long long>(r) * LOW + i] = __float2bfloat16_rn(bsilu(v));
        } else {
            INJ[static_cast<long long>(r) * S + (i - LOW)] = __float2bfloat16_rn(__fmul_rn(2.0f, bsig(v)));
        }
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

extern "C" __global__ void __launch_bounds__(256) fn_shared_reduce_swiglu(const float* PART, __nv_bfloat16* __restrict__ ACT,
                                                                          int M, int W, int SK) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
    const int r = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j < W) {
        const int n2 = 2 * W;
        const long long stride = static_cast<long long>(M) * n2;
        const long long bg = static_cast<long long>(r) * n2 + j;
        const long long bu = bg + W;
        float sg = PART[bg];
        float su = PART[bu];
        for (int s = 1; s < SK; ++s) {
            sg = __fadd_rn(sg, PART[bg + static_cast<long long>(s) * stride]);
            su = __fadd_rn(su, PART[bu + static_cast<long long>(s) * stride]);
        }
        ACT[static_cast<long long>(r) * W + j] = __float2bfloat16_rn(__fmul_rn(bsilu(rbf(sg)), rbf(su)));
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

extern "C" __global__ void __launch_bounds__(256) fn_shared_swiglu_cat(const __nv_bfloat16* GU, __nv_bfloat16* __restrict__ ACT,
                                                                       int W) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
    const int r = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j < W) {
        const long long bg = static_cast<long long>(r) * (2 * W) + j;
        ACT[static_cast<long long>(r) * W + j] = __float2bfloat16_rn(__fmul_rn(bsilu(bf(GU[bg])), bf(GU[bg + W])));
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

extern "C" __global__ void __launch_bounds__(1024) fn_hc_reduce_act(const float* __restrict__ PART, __nv_bfloat16* __restrict__ ACT,""",
)
patch_k(
    "flashnext_hc_dense.cuh",
    """    MIXED[static_cast<long long>(r) * D + d] = __float2bfloat16_rn(mixed);
    row[warp][lane] = mixed;""",
    """    MIXED[static_cast<long long>(r) * D + d] = __float2bfloat16_rn(mixed);
    if (XSM == nullptr) return;
    row[warp][lane] = mixed;""",
)

# Zero-spill fn_shift_window (taps<=3 in scalar regs, eliminating STACK:32) + skip dead xs in fn_gdn_norm & fn_attn_gate + skip 0-weight slots in hc_writeback<2>
patch_k(
    "flashnext_hc.cu",
    """        for (int k = 0; k <= TOPK; ++k) {
            const long long at = static_cast<long long>(r) * SLOTS + k;
            acc = __fmaf_rn(Y[at * D + d], WTS[at], acc);
        }""",
    """        for (int k = 0; k <= TOPK; ++k) {
            const long long at = static_cast<long long>(r) * SLOTS + k;
            const float wk = WTS[at];
            if (wk == 0.0f) continue;
            acc = __fmaf_rn(Y[at * D + d], wk, acc);
        }""",
)
patch_k(
    "flashnext_batch.cu",
    """static __device__ __forceinline__ void shift_window(__nv_bfloat16* __restrict__ win, const __nv_bfloat16* __restrict__ rows,
                                                   long long stride, int n, int taps, int c) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= c) return;
    __nv_bfloat16 next[16];
    for (int j = 0; j < taps; ++j) {
        const int src = j + n;                       // row src of [window (taps rows); new rows]
        next[j] = src < taps ? win[static_cast<long long>(src) * c + ch]
                             : rows[static_cast<long long>(src - taps) * stride + ch];
    }
    for (int j = 0; j < taps; ++j) win[static_cast<long long>(j) * c + ch] = next[j];
}""",
    """static __device__ __noinline__ void shift_window_slow(__nv_bfloat16* __restrict__ win, const __nv_bfloat16* rows,
                                                      long long stride, int n, int taps, int c, int ch) {
    __nv_bfloat16 next[16];
    for (int j = 0; j < taps; ++j) {
        const int src = j + n;
        next[j] = src < taps ? win[static_cast<long long>(src) * c + ch]
                             : rows[static_cast<long long>(src - taps) * stride + ch];
    }
    for (int j = 0; j < taps; ++j) win[static_cast<long long>(j) * c + ch] = next[j];
}

static __device__ __forceinline__ void shift_window(__nv_bfloat16* __restrict__ win, const __nv_bfloat16* rows,
                                                   long long stride, int n, int taps, int c) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= c) return;
    if (taps == 3) {
        const __nv_bfloat16 n0 = n < 3 ? win[static_cast<long long>(n) * c + ch] : rows[static_cast<long long>(n - 3) * stride + ch];
        const __nv_bfloat16 n1 = n + 1 < 3 ? win[static_cast<long long>(n + 1) * c + ch] : rows[static_cast<long long>(n - 2) * stride + ch];
        const __nv_bfloat16 n2 = n + 2 < 3 ? win[static_cast<long long>(n + 2) * c + ch] : rows[static_cast<long long>(n - 1) * stride + ch];
        win[ch] = n0;
        win[static_cast<long long>(c) + ch] = n1;
        win[static_cast<long long>(2) * c + ch] = n2;
        return;
    }
    if (taps == 2) {
        const __nv_bfloat16 n0 = n < 2 ? win[static_cast<long long>(n) * c + ch] : rows[static_cast<long long>(n - 2) * stride + ch];
        const __nv_bfloat16 n1 = n + 1 < 2 ? win[static_cast<long long>(n + 1) * c + ch] : rows[static_cast<long long>(n - 1) * stride + ch];
        win[ch] = n0;
        win[static_cast<long long>(c) + ch] = n1;
        return;
    }
    shift_window_slow(win, rows, stride, n, taps, c, ch);
}""",
)
patch_k(
    "flashnext_batch.cu",
    """    out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
    const float gs = warp_sum(o);
    if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;""",
    """    out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
    if (xs != nullptr) {
        const float gs = warp_sum(o);
        if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
    }""",
)
patch_k(
    "flashnext_attn.cu",
    """    OUT[static_cast<long long>(r) * NQ * HD + h * HD + d] = __float2bfloat16_rn(out);
    const float sum = warp_sum(out);
    if ((d & 31) == 0) XS[static_cast<long long>(r) * (NQ * HD / 32) + h * (HD / 32) + d / 32] = sum;""",
    """    OUT[static_cast<long long>(r) * NQ * HD + h * HD + d] = __float2bfloat16_rn(out);
    if (XS != nullptr) {
        const float sum = warp_sum(out);
        if ((d & 31) == 0) XS[static_cast<long long>(r) * (NQ * HD / 32) + h * (HD / 32) + d / 32] = sum;
    }""",
)

# Instrument remaining decode .cu/.cuh kernels with non-cascading PDL (wait FIRST, then launch_dependents)
PDL_SNIP = "\n#if __CUDA_ARCH__ >= 900\n    asm volatile(\"griddepcontrol.wait;\\n\" ::: \"memory\");\n    asm volatile(\"griddepcontrol.launch_dependents;\\n\" ::: \"memory\");\n#endif\n"

def instrument_pdl(rel: str, expect_min: int = 0, only_prefix: str | None = None, stop_marker: str | None = None) -> int:
    p = kdir / rel
    raw = p.read_text(encoding="utf-8")
    tail = ""
    if stop_marker is not None:
        idx = raw.find(stop_marker)
        if idx < 0:
            raise RuntimeError(f"Missing marker {stop_marker} in {rel}")
        raw, tail = raw[:idx], raw[idx:]
    if only_prefix is None:
        raw = re.sub(r'(\bconst\b[^,;(){}]*?\*)\s*__restrict__\b', r'\1', raw)
    out = []
    pos = 0
    count = 0
    for m in re.finditer(r'\b__global__\b', raw):
        s = m.start()
        if s < pos:
            continue
        # Skip comments
        line_start = raw.rfind('\n', 0, s) + 1
        if '//' in raw[line_start:s]:
            continue
        i = m.end()
        # Walk past return type, optional __launch_bounds__(...), name, and parameter list (...)
        depth = 0
        seen_fn_params = False
        body_brace = -1
        while i < len(raw):
            ch = raw[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    # Look ahead past whitespace for '{', ';', or identifier (if we just finished __launch_bounds__)
                    j = i + 1
                    while j < len(raw) and raw[j] in ' \t\r\n':
                        j += 1
                    if j < len(raw) and raw[j] == '{':
                        body_brace = j
                        seen_fn_params = True
                        break
                    elif j < len(raw) and raw[j] == ';':
                        seen_fn_params = True
                        break
            elif ch == ';' and depth == 0:
                break
            i += 1
        if body_brace >= 0 and seen_fn_params:
            sig = raw[s:body_brace]
            if only_prefix is None or (only_prefix in sig):
                if only_prefix is not None:
                    sig_clean = re.sub(r'(\bconst\b[^,;(){}]*?\*)\s*__restrict__\b', r'\1', sig)
                    out.append(raw[pos:s])
                    out.append(sig_clean)
                    out.append('{')
                    out.append(PDL_SNIP)
                else:
                    out.append(raw[pos:body_brace + 1])
                    out.append(PDL_SNIP)
                pos = body_brace + 1
                count += 1
    out.append(raw[pos:])
    if count < expect_min:
        raise RuntimeError(f"Expected >= {expect_min} __global__ in {rel}, instrumented {count}")
    p.write_text("".join(out) + tail, encoding="utf-8")
    return count

instrument_pdl("experts.cu", expect_min=7, only_prefix="plan_")
instrument_pdl("route_rows.cu", expect_min=4)
patch_k(
    "route_rows.cu",
    """    asm volatile("griddepcontrol.wait;\\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");
#endif

    tf_router_mma::logits<4, 3, 256>(X, x_ld, W, OUT, M, D, E);""",
    """    asm volatile("griddepcontrol.wait;\\n" ::: "memory");
#endif

    tf_router_mma::logits<2, 4, 128>(X, x_ld, W, OUT, M, D, E);
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");
#endif""",
)
patch_k(
    "route_rows.cu",
    'extern "C" __global__ void __launch_bounds__(128) tf_route_logits_few(',
    'extern "C" __global__ void __launch_bounds__(64) tf_route_logits_few(',
)
# Fused 1-CTA topk + plan kernel for M<=32, NE<=512, E_PLAN<=512, SLOTS<=12 (bit-exact picks/wts + 4-bucket descending plan)
(kdir / "route_rows.cu").write_text(
    (kdir / "route_rows.cu").read_text(encoding="utf-8") + r"""

extern "C" __global__ void __launch_bounds__(1024) tf_route_topk_plan(
    const float* L, int NL, int NE, int TOPK, int M, int SLOTS, int shared, int E_PLAN, int T, int RSK,
    int* __restrict__ PICK, float* __restrict__ WTS,
    int* __restrict__ members, int* __restrict__ items, int* __restrict__ counts) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ int sh_cnt[512];
    __shared__ int sh_off[512];
    __shared__ int sh_pick[32 * 12];
    __shared__ int wsum[6][16];
    if (tid < E_PLAN) sh_cnt[tid] = 0;
    __syncthreads();

    if (warp < M) {
        const long long slice_stride = static_cast<long long>(M) * NL;
        const float* row = L + static_cast<long long>(warp) * NL;
        float wv[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const int i = lane + j * 32;
            if (i < NE) {
                float v = row[i];
                for (int s = 1; s < RSK; ++s) v = __fadd_rn(v, row[static_cast<long long>(s) * slice_stride + i]);
                wv[j] = v;
            } else {
                wv[j] = -INFINITY;
            }
        }
        float top = 0.0f, total = 0.0f, mine = 0.0f;
        int pick = 0;
        for (int k = 0; k < TOPK; ++k) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int i = lane + j * 32;
                if (wv[j] > bv || (wv[j] == bv && i < bi)) {
                    bv = wv[j];
                    bi = i;
                }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bi = oi;
                }
            }
            if (k == 0) top = bv;
#pragma unroll
            for (int j = 0; j < 16; ++j)
                if (lane + j * 32 == bi) wv[j] = -INFINITY;
            const float ex = tf_exp(__fsub_rn(bv, top));
            total = __fadd_rn(total, ex);
            if (lane == k) {
                mine = ex;
                pick = bi;
            }
        }
        const int r_slots = shared ? (SLOTS - 1) : SLOTS;
        if (lane < r_slots) {
            if (lane < TOPK) {
                PICK[static_cast<long long>(warp) * SLOTS + lane] = pick;
                WTS[static_cast<long long>(warp) * SLOTS + lane] = rbf(__fdiv_rn(mine, total));
                sh_pick[warp * SLOTS + lane] = pick;
                if (pick >= 0 && pick < E_PLAN) atomicAdd(&sh_cnt[pick], 1);
            } else {
                PICK[static_cast<long long>(warp) * SLOTS + lane] = NE;
                WTS[static_cast<long long>(warp) * SLOTS + lane] = 0.0f;
                sh_pick[warp * SLOTS + lane] = NE;
            }
        }
        if (shared && lane == 0) {
            float sg = row[NE];
            for (int s = 1; s < RSK; ++s) sg = __fadd_rn(sg, row[static_cast<long long>(s) * slice_stride + NE]);
            const int s_slot = SLOTS - 1;
            PICK[static_cast<long long>(warp) * SLOTS + s_slot] = NE;
            WTS[static_cast<long long>(warp) * SLOTS + s_slot] = rbf(tf_sigmoid(rbf(sg)));
            sh_pick[warp * SLOTS + s_slot] = NE;
            if (NE < E_PLAN) atomicAdd(&sh_cnt[NE], 1);
        }
    }
    __syncthreads();

    const int c = (tid < E_PLAN) ? sh_cnt[tid] : 0;
    const int tiles = (c + T - 1) / T;
    const int rem = c > 0 ? (c - T * (tiles - 1)) : 0;
    const int b0 = (c > 0 && T > 8) ? ((tiles - 1) + (rem > 8 ? 1 : 0)) : 0;
    const int b1 = (c > 0 && T > 8 && rem >= 3 && rem <= 8) ? 1 : ((c > 0 && T <= 8) ? tiles : 0);
    const int b2 = (c > 0 && T > 8 && rem == 2) ? 1 : 0;
    const int b3 = (c > 0 && T > 8 && rem == 1) ? 1 : 0;
    int a0 = c, a1 = c > 0, a2 = b0, a3 = b1, a4 = b2, a5 = b3;
    if (warp < 16) {
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u0 = __shfl_up_sync(0xffffffffu, a0, o), u1 = __shfl_up_sync(0xffffffffu, a1, o);
            const int u2 = __shfl_up_sync(0xffffffffu, a2, o), u3 = __shfl_up_sync(0xffffffffu, a3, o);
            const int u4 = __shfl_up_sync(0xffffffffu, a4, o), u5 = __shfl_up_sync(0xffffffffu, a5, o);
            if (lane >= o) { a0 += u0; a1 += u1; a2 += u2; a3 += u3; a4 += u4; a5 += u5; }
        }
        if (lane == 31) {
            wsum[0][warp] = a0; wsum[1][warp] = a1; wsum[2][warp] = a2;
            wsum[3][warp] = a3; wsum[4][warp] = a4; wsum[5][warp] = a5;
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int s = 0; s < 6; ++s) {
            int v = (lane < 16) ? wsum[s][lane] : 0;
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) {
                const int u = __shfl_up_sync(0xffffffffu, v, o);
                if (lane >= o) v += u;
            }
            if (lane < 16) wsum[s][lane] = v;
        }
    }
    __syncthreads();
    const int tot0 = wsum[2][15], tot1 = wsum[3][15], tot2 = wsum[4][15], tot3 = wsum[5][15];
    if (tid == 0) {
        counts[0] = tot0 + tot1 + tot2 + tot3;
        counts[1] = wsum[1][15];
    }
    if (tid < E_PLAN) {
        const int off = (warp ? wsum[0][warp - 1] : 0) + a0 - c;
        sh_off[tid] = off;
        if (c > 0) {
            const int ioff0 = (warp ? wsum[2][warp - 1] : 0) + a2 - b0;
            const int ioff1 = (warp ? wsum[3][warp - 1] : 0) + a3 - b1;
            const int ioff2 = (warp ? wsum[4][warp - 1] : 0) + a4 - b2;
            const int ioff3 = (warp ? wsum[5][warp - 1] : 0) + a5 - b3;
            if (T > 8) {
                for (int j = 0; j < b0; ++j) {
                    int* it = items + 3 * (ioff0 + j);
                    it[0] = tid; it[1] = off + T * j; it[2] = min(T, c - T * j);
                }
                if (b1) {
                    int* it = items + 3 * (tot0 + ioff1);
                    it[0] = tid; it[1] = off + T * (tiles - 1); it[2] = rem;
                }
                if (b2) {
                    int* it = items + 3 * (tot0 + tot1 + ioff2);
                    it[0] = tid; it[1] = off + T * (tiles - 1); it[2] = 2;
                }
                if (b3) {
                    int* it = items + 3 * (tot0 + tot1 + tot2 + ioff3);
                    it[0] = tid; it[1] = off + T * (tiles - 1); it[2] = 1;
                }
            } else {
                for (int j = 0; j < tiles; ++j) {
                    int* it = items + 3 * (ioff1 + j);
                    it[0] = tid; it[1] = off + T * j; it[2] = min(T, c - T * j);
                }
            }
        }
    }
    __syncthreads();
    const int P = M * SLOTS;
    for (int p = tid; p < P; p += blockDim.x) {
        const int e = sh_pick[p];
        if (e >= 0 && e < E_PLAN) {
            const int pos = atomicAdd(&sh_off[e], 1);
            members[pos] = p;
        }
    }
    __syncthreads();
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}
""",
    encoding="utf-8",
)
instrument_pdl("norms.cu", expect_min=10)
instrument_pdl("norms_strided.cuh", expect_min=0)
instrument_pdl("norms_tail.cuh", expect_min=0)
instrument_pdl("norms_l2.cuh", expect_min=0)
instrument_pdl("embed_rows.cu", expect_min=4)
instrument_pdl("embed_affine.cuh", expect_min=0)
instrument_pdl("flashnext_hc.cu", expect_min=14)
# Keep static weight CW __restrict__ intact for Zig's build-time hybrid_ple_source.zig adapter
patch_k("flashnext_hc.cu", "const __nv_bfloat16* CW,", "const __nv_bfloat16* __restrict__ CW,")
# Move griddepcontrol.launch_dependents to epilogue on heavy HC writeback-norm wrappers and PLE gate to avoid SM-hogging PDL cascades
for _hc_call in (
    "hc_writeback_norm<0, 10, 4>(H, nullptr, PSS, nullptr, nullptr, nullptr, nullptr, SCALE, NORM, 0, eps, 0, 1, 1);",
    "hc_writeback_norm<2, 10, 4>(H, HOUT, PSS, nullptr, INJ, Y, WTS, SCALE, NORM, 0, eps, TOPK, SLOTS, 1);",
    "hc_writeback_norm<4, 10, 4>(H, HOUT, PSS, BR, INJ, nullptr, nullptr, SCALE, NORM, RS, eps, 0, 1, WORLD);",
):
    patch_k(
        "flashnext_hc.cu",
        f"""    asm volatile("griddepcontrol.wait;\\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");
#endif

    {_hc_call}""",
        f"""    asm volatile("griddepcontrol.wait;\\n" ::: "memory");
#endif

    {_hc_call}
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");
#endif""",
    )
patch_k(
    "flashnext_hc.cu",
    """extern "C" __global__ void __launch_bounds__(512) fn_ple_gate(const __nv_bfloat16* KEYS, const __nv_bfloat16* VALS,
                                                              const __nv_bfloat16* H, const float* NK,
                                                              const float* NQ, __nv_bfloat16* __restrict__ GATED,
                                                              float* __restrict__ PSS, float eps, int D, int S) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");
#endif""",
    """extern "C" __global__ void __launch_bounds__(512) fn_ple_gate(const __nv_bfloat16* KEYS, const __nv_bfloat16* VALS,
                                                              const __nv_bfloat16* H, const float* NK,
                                                              const float* NQ, __nv_bfloat16* __restrict__ GATED,
                                                              float* __restrict__ PSS, float eps, int D, int S) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\\n" ::: "memory");
#endif""",
)
patch_k(
    "flashnext_hc.cu",
    """        gss = block_sum_serial(gss, scratch);
        if (threadIdx.x == 0) PSS[static_cast<long long>(r) * S + s] = gss;
    }
}""",
    """        gss = block_sum_serial(gss, scratch);
        if (threadIdx.x == 0) PSS[static_cast<long long>(r) * S + s] = gss;
    }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");
#endif
}""",
)
instrument_pdl("flashnext_hc_dense.cuh", expect_min=4)
instrument_pdl("flashnext_batch.cu", expect_min=14, stop_marker="extern \"C\" __global__ void __launch_bounds__(256) fn_bf16_to_nvfp4(")
# Move griddepcontrol.launch_dependents to epilogue on heavy GDN recurrent kernels so fn_gdn_norm (432 CTAs) does not steal SMs from fn_gdn_rows_late (384 CTAs)
for _gdn_call in (
    "gdn_rows<16, 48>(qn, kn, v, gb, state, state, rows, ys_out);",
    "gdn_rows<16, 48>(qn, kn, v, gb, state_in, state_out, rows, ys_out);",
    "gdn_rows_late_4t(qn, log, ctl, log_rows, store_above, state_in, state_out, rows, ys_out);",
    "gdn_rows<8, 24>(qn, kn, v, gb, state, state, rows, ys_out);",
    "gdn_rows<8, 24>(qn, kn, v, gb, state_in, state_out, rows, ys_out);",
):
    patch_k(
        "flashnext_batch.cu",
        f"""    asm volatile("griddepcontrol.wait;\\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");
#endif

    {_gdn_call}""",
        f"""    asm volatile("griddepcontrol.wait;\\n" ::: "memory");
#endif

    {_gdn_call}
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");
#endif""",
    )
instrument_pdl("flashnext_gdn_lanes.cuh", expect_min=4)
instrument_pdl("flashnext_gdn_fused.cuh", expect_min=2)
instrument_pdl("flashnext_gdn_tpr.cuh", expect_min=2)
instrument_pdl("flashnext_attn.cu", expect_min=12)
patch_k(
    "flashnext_attn.cu",
    """template <int NB, bool ONE = false, bool FUSE = false, bool PAGED = false>
__device__ __forceinline__ void attn_chunks(const __nv_bfloat16* Q, const __nv_bfloat16* KC,
                                            const __nv_bfloat16* VC, const int* POS0,
                                            float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                            const int* IDS, const int* NKR, const int* SPR,
                                            int H, int HK, int G, int CH, int NCH, float SCALE, int IDW, int QSA,
                                            Pg pg = {}) {""",
    """template <int NB, bool ONE = false, bool FUSE = false, bool PAGED = false, bool GATE = false>
__device__ __forceinline__ void attn_chunks(const __nv_bfloat16* Q, const __nv_bfloat16* KC,
                                            const __nv_bfloat16* VC, const int* POS0,
                                            float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
                                            const int* IDS, const int* NKR, const int* SPR,
                                            int H, int HK, int G, int CH, int NCH, float SCALE, int IDW, int QSA,
                                            Pg pg = {}, const __nv_bfloat16* PROJ = nullptr, int PW = 0) {""",
)
patch_k(
    "flashnext_attn.cu",
    """    if constexpr (FUSE) {                                  // o / l, rounded once, as fn_attn_merge writes OUT
        __nv_bfloat16* OUT = reinterpret_cast<__nv_bfloat16*>(PO);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int i = g + 8 * h;
            if (i >= G) continue;
            const long long row = (static_cast<long long>(r) * H + hk * G + i) * HD;
#pragma unroll
            for (int q = 0; q < 8; ++q)
#pragma unroll
                for (int e = 0; e < 2; ++e)
                    OUT[row + warp * 64 + q * 8 + 2 * t + e] = __float2bfloat16_rn(__fdiv_rn(oo[q][2 * h + e], ll[h]));
        }
    }""",
    """    if constexpr (FUSE) {                                  // o / l, rounded once, optionally fused with fn_attn_gate (bit-exact)
        __nv_bfloat16* OUT = reinterpret_cast<__nv_bfloat16*>(PO);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int i = g + 8 * h;
            if (i >= G) continue;
            const int head = hk * G + i;
            const long long row = (static_cast<long long>(r) * H + head) * HD;
#pragma unroll
            for (int q = 0; q < 8; ++q)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int d = warp * 64 + q * 8 + 2 * t + e;
                    const __nv_bfloat16 ob = __float2bfloat16_rn(__fdiv_rn(oo[q][2 * h + e], ll[h]));
                    if constexpr (GATE) {
                        const float o = bf(ob);
                        const float gv = bf(PROJ[static_cast<long long>(r) * PW + head * 2 * HD + HD + d]);
                        OUT[row + d] = __float2bfloat16_rn(rbf(__fdiv_rn(o, __fadd_rn(1.0f, tf_exp(-gv)))));
                    } else {
                        OUT[row + d] = ob;
                    }
                }
        }
    }""",
)
(kdir / "flashnext_attn.cu").write_text(
    (kdir / "flashnext_attn.cu").read_text(encoding="utf-8") + r"""

// Double-buffered fused attention for decode single-chunk (eliminates fn_attn_merge + PO/PM/PL round-trip, bit-exact).
extern "C" __global__ void __launch_bounds__(128) fn_attn_fused_ahead(
    const __nv_bfloat16* Q, const __nv_bfloat16* KC, const __nv_bfloat16* VC, const int* POS0,
    float* __restrict__ OUT, const int* IDS, const int* NKR, const int* SPR,
    int H, int HK, int G, int CH, float SCALE, int IDW, int QSA) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    attn_chunks<2, false, true, false>(Q, KC, VC, POS0, OUT, nullptr, nullptr, IDS, NKR, SPR, H, HK, G, CH, 1, SCALE, IDW, QSA);
}

extern "C" __global__ void __launch_bounds__(128) fn_attn_fused_ahead_paged(
    const __nv_bfloat16* Q, const __nv_bfloat16* KC, const __nv_bfloat16* VC, const int* POS0,
    float* __restrict__ OUT, const int* IDS, const int* NKR, const int* SPR,
    int H, int HK, int G, int CH, float SCALE, int IDW, int QSA, fn_attn::Pg pg) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    attn_chunks<2, false, true, true>(Q, KC, VC, POS0, OUT, nullptr, nullptr, IDS, NKR, SPR, H, HK, G, CH, 1, SCALE, IDW, QSA, pg);
}

extern "C" __global__ void __launch_bounds__(128) fn_attn_fused_gate(
    const __nv_bfloat16* Q, const __nv_bfloat16* KC, const __nv_bfloat16* VC, const int* POS0,
    float* __restrict__ OUT, const int* IDS, const int* NKR, const int* SPR,
    int H, int HK, int G, int CH, float SCALE, int IDW, int QSA,
    const __nv_bfloat16* PROJ, int PW) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    attn_chunks<1, true, true, false, true>(Q, KC, VC, POS0, OUT, nullptr, nullptr, IDS, NKR, SPR, H, HK, G, CH, 1, SCALE, IDW, QSA, fn_attn::Pg{}, PROJ, PW);
}

extern "C" __global__ void __launch_bounds__(128) fn_attn_fused_gate_paged(
    const __nv_bfloat16* Q, const __nv_bfloat16* KC, const __nv_bfloat16* VC, const int* POS0,
    float* __restrict__ OUT, const int* IDS, const int* NKR, const int* SPR,
    int H, int HK, int G, int CH, float SCALE, int IDW, int QSA, fn_attn::Pg pg,
    const __nv_bfloat16* PROJ, int PW) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    attn_chunks<1, true, true, true, true>(Q, KC, VC, POS0, OUT, nullptr, nullptr, IDS, NKR, SPR, H, HK, G, CH, 1, SCALE, IDW, QSA, pg, PROJ, PW);
}

extern "C" __global__ void __launch_bounds__(128) fn_attn_fused_ahead_gate(
    const __nv_bfloat16* Q, const __nv_bfloat16* KC, const __nv_bfloat16* VC, const int* POS0,
    float* __restrict__ OUT, const int* IDS, const int* NKR, const int* SPR,
    int H, int HK, int G, int CH, float SCALE, int IDW, int QSA,
    const __nv_bfloat16* PROJ, int PW) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    attn_chunks<2, false, true, false, true>(Q, KC, VC, POS0, OUT, nullptr, nullptr, IDS, NKR, SPR, H, HK, G, CH, 1, SCALE, IDW, QSA, fn_attn::Pg{}, PROJ, PW);
}

extern "C" __global__ void __launch_bounds__(128) fn_attn_fused_ahead_gate_paged(
    const __nv_bfloat16* Q, const __nv_bfloat16* KC, const __nv_bfloat16* VC, const int* POS0,
    float* __restrict__ OUT, const int* IDS, const int* NKR, const int* SPR,
    int H, int HK, int G, int CH, float SCALE, int IDW, int QSA, fn_attn::Pg pg,
    const __nv_bfloat16* PROJ, int PW) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
    attn_chunks<2, false, true, true, true>(Q, KC, VC, POS0, OUT, nullptr, nullptr, IDS, NKR, SPR, H, HK, G, CH, 1, SCALE, IDW, QSA, pg, PROJ, PW);
}
""",
    encoding="utf-8",
)
instrument_pdl("flashnext_attn_lanes.cuh", expect_min=5)
instrument_pdl("flashnext_qsa_tile.cuh", expect_min=1)
instrument_pdl("flashnext_mtp.cu", expect_min=2, stop_marker="extern \"C\" __global__ void __launch_bounds__(128) fn_w2_gemv_fused(")
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

  # 2a2. MTP BF16 -> native SM121 NVFP4 encoder (90.0 MB -> 49.1 MB = -40.9 MB/step = -614 MB/round, 100% bit-exact target!)
  cat > "$fn_dir/cuda_mtp_nvfp4.zig" <<'ZIG'
//! Re-codes clean high-SNR (20.5-26.0 dB) MTP dense BF16 projections to native SM121 NVFP4 at load.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const projection = @import("cuda_projection.zig");

pub fn mode() []const u8 {
    const raw = std.c.getenv("TF_FLASHNEXT_MTP_NVFP4") orelse return "0";
    return std.mem.span(raw);
}

pub fn wantProj(prefix: []const u8) bool {
    const m = mode();
    if (std.mem.eql(u8, m, "all")) return std.mem.startsWith(u8, prefix, "language_model.mtp.");
    if (std.mem.eql(u8, m, "safe")) {
        return std.mem.eql(u8, prefix, "language_model.mtp.fc_hidden") or
            (std.mem.startsWith(u8, prefix, "language_model.mtp.layers.0.") and
                (std.mem.endsWith(u8, prefix, ".o_proj") or std.mem.indexOf(u8, prefix, ".shared_expert.") != null));
    }
    if (std.mem.eql(u8, m, "core")) {
        return std.mem.startsWith(u8, prefix, "language_model.mtp.layers.0.") and
            (std.mem.endsWith(u8, prefix, ".o_proj") or std.mem.indexOf(u8, prefix, ".shared_expert.") != null);
    }
    return false;
}

pub fn wantHcDown(prefix: []const u8) bool {
    if (!std.mem.eql(u8, mode(), "all")) return false;
    // Keep hyper_connection_mixer.down (crest=76.4, 3.38 MB) in MXFP8; attn_hc/mlp_hc.down (crest=14) go to NVFP4.
    return std.mem.startsWith(u8, prefix, "language_model.mtp.layers.0.");
}

pub fn wantHcUp(prefix: []const u8) bool {
    if (!std.mem.eql(u8, mode(), "all")) return false;
    return std.mem.startsWith(u8, prefix, "language_model.mtp.");
}

inline fn bf16ToF32(bytes: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16);
}

inline fn e2m1(v: f32) u8 {
    const sign: u8 = if (std.math.signbit(v)) 8 else 0;
    const a = @abs(v);
    const mag: u8 = if (a < 0.25) 0 else if (a <= 0.75) 1 else if (a < 1.25) 2 else if (a <= 1.75) 3 else if (a < 2.5) 4 else if (a <= 3.5) 5 else if (a < 5.0) 6 else 7;
    return sign | mag;
}

const QuantJob = struct {
    bf16: []const u8,
    codes: []u8,
    scales: []u8,
    global_scale: f32,
    groups: usize,
    per: usize,

    fn worker(ctx: *const QuantJob, tid: usize) void {
        const lo = @min(ctx.groups, tid * ctx.per);
        const hi = @min(ctx.groups, lo + ctx.per);
        const inv_g6 = 1.0 / (ctx.global_scale * 6.0);
        for (lo..hi) |g| {
            const base = g * 16;
            var w: [16]f32 = undefined;
            var mmax: f32 = 0;
            for (0..16) |j| {
                const f = bf16ToF32(ctx.bf16, base + j);
                w[j] = f;
                mmax = @max(mmax, @abs(f));
            }
            const sc_code = core.exlr8.codec.e4m3FromF32(std.math.clamp(mmax * inv_g6, 0.0, 448.0));
            ctx.scales[g] = sc_code;
            const sc_val = core.exlr8.codec.e4m3Value(sc_code) * ctx.global_scale;
            const dst = ctx.codes[g * 8 ..][0..8];
            if (sc_val == 0) {
                @memset(dst, 0);
            } else {
                const inv = 1.0 / sc_val;
                for (0..8) |p| {
                    dst[p] = e2m1(w[2 * p] * inv) | (e2m1(w[2 * p + 1] * inv) << 4);
                }
            }
        }
    }
};

pub fn project(gpa: std.mem.Allocator, driver: *const cuda.Driver, bf16: []const u8, n: usize, k: usize) !projection.Projection {
    if (k % 64 != 0 or bf16.len != n * k * 2) return error.UnexpectedTensor;
    const count = n * k;
    var top_bits: u16 = 0;
    for (0..count) |i| {
        const u = std.mem.readInt(u16, bf16[i * 2 ..][0..2], .little) & 0x7fff;
        if (u <= 0x7f80 and u > top_bits) top_bits = u;
    }
    const max_abs: f32 = @bitCast(@as(u32, top_bits) << 16);
    const global_scale: f32 = if (max_abs > 0) max_abs / (6.0 * 448.0) else 1.0;

    const groups = count / 16;
    const raw_codes = try gpa.alloc(u8, count / 2);
    defer gpa.free(raw_codes);
    const raw_scales = try gpa.alloc(u8, groups);
    defer gpa.free(raw_scales);

    const n_th: usize = if (groups >= 4096) 8 else 1;
    const job: QuantJob = .{
        .bf16 = bf16,
        .codes = raw_codes,
        .scales = raw_scales,
        .global_scale = global_scale,
        .groups = groups,
        .per = (groups + n_th - 1) / n_th,
    };
    if (n_th == 1) {
        QuantJob.worker(&job, 0);
    } else {
        var ths: [8]?std.Thread = @splat(null);
        for (0..n_th) |ti| ths[ti] = std.Thread.spawn(.{}, QuantJob.worker, .{ &job, ti }) catch null;
        for (0..n_th) |ti| {
            if (ths[ti]) |t| t.join() else QuantJob.worker(&job, ti);
        }
    }

    const npad = cuda.nvfp4.padded(n);
    const packed_w = try gpa.alloc(u32, (npad / 64) * (k / 64) * 512);
    defer gpa.free(packed_w);
    const packed_s = try gpa.alloc(u8, npad * (k / 16));
    defer gpa.free(packed_s);
    cuda.nvfp4.packWords(packed_w, raw_codes, n, k);
    cuda.nvfp4.packScales(packed_s, raw_scales, n, k);

    var p = projection.Projection.empty(driver);
    p.w = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(packed_w));
    errdefer p.w.free();
    p.s = try cuda.DeviceBuffer.fromHost(driver, packed_s);
    p.n = n;
    p.k = k;
    p.native = cuda.nvfp4.weight(p.w.ptr, p.s.ptr, global_scale, n, k);
    return p;
}
ZIG

  # 2a3. MTP BF16 -> 6.25-bit symmetric INT6+E4M3 encoder (30.4-32.3 dB SNR, -24.2% DRAM vs MXFP8)
  cat > "$fn_dir/cuda_mtp_int6.zig" <<'ZIG'
//! Re-codes MTP dense BF16 linears and HC mixers to 6.25-bit symmetric INT6 ([-31.5..+31.5]) + g32 E4M3 scales at load.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const projection = @import("cuda_projection.zig");

pub fn mode() []const u8 {
    const raw = std.c.getenv("TF_FLASHNEXT_MTP_INT6") orelse return "0";
    return std.mem.span(raw);
}

pub fn wantProj(prefix: []const u8) bool {
    const m = mode();
    if (!std.mem.startsWith(u8, prefix, "language_model.mtp.")) return false;
    if (std.mem.eql(u8, m, "all")) return true;
    if (std.mem.eql(u8, m, "1")) {
        // Keep fc_hidden in MXFP8 (30.6 dB vs 24.5 dB); all other MTP linears are 28.8-32.3 dB in INT6.
        return !std.mem.eql(u8, prefix, "language_model.mtp.fc_hidden");
    }
    if (std.mem.eql(u8, m, "clean")) {
        // High-SNR clean MTP matrices only: exclude fc_hidden, k_proj (38% outlier), v_proj (55% outlier), indexer.
        return std.mem.eql(u8, prefix, "language_model.mtp.fc_embed") or
            std.mem.endsWith(u8, prefix, ".q_proj") or
            std.mem.endsWith(u8, prefix, ".o_proj") or
            std.mem.indexOf(u8, prefix, ".shared_expert.") != null;
    }
    if (std.mem.eql(u8, m, "qo")) {
        return std.mem.endsWith(u8, prefix, ".q_proj") or
            std.mem.endsWith(u8, prefix, ".o_proj") or
            std.mem.indexOf(u8, prefix, ".shared_expert.") != null;
    }
    if (std.mem.eql(u8, m, "safe")) {
        return std.mem.eql(u8, prefix, "language_model.mtp.fc_embed") or
            std.mem.endsWith(u8, prefix, ".o_proj") or
            std.mem.indexOf(u8, prefix, ".shared_expert.") != null;
    }
    return false;
}

pub fn wantHcDown(prefix: []const u8) bool {
    const m = mode();
    if (!std.mem.startsWith(u8, prefix, "language_model.mtp.")) return false;
    if (std.mem.eql(u8, m, "1") or std.mem.eql(u8, m, "all")) return true;
    if (std.mem.eql(u8, m, "clean") or std.mem.eql(u8, m, "qo") or std.mem.eql(u8, m, "safe")) {
        // Keep hyper_connection_mixer.down (crest=76.4) in MXFP8; attn_hc/mlp_hc.down are clean.
        return std.mem.startsWith(u8, prefix, "language_model.mtp.layers.0.");
    }
    return false;
}

pub fn wantHcUp(prefix: []const u8) bool {
    const m = mode();
    if (!std.mem.startsWith(u8, prefix, "language_model.mtp.")) return false;
    if (std.mem.eql(u8, m, "1") or std.mem.eql(u8, m, "all")) return true;
    if (std.mem.eql(u8, m, "clean")) {
        // Keep mlp_hyper_connection.up (73% outlier) and hyper_connection_mixer.up in MXFP8.
        return std.mem.indexOf(u8, prefix, ".attn_hyper_connection") != null;
    }
    return false;
}

inline fn bf16ToF32(bytes: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16);
}

const PackJob = struct {
    bf16: []const u8,
    w_out: []u32,
    s_out: []u8,
    n: usize,
    k: usize,
    n64_tot: usize,
    kg: usize,
    global_scale: f32,
    per: usize,

    fn worker(ctx: *const PackJob, tid_th: usize) void {
        const lo = @min(ctx.n64_tot, tid_th * ctx.per);
        const hi = @min(ctx.n64_tot, lo + ctx.per);
        const inv_g315: f32 = 1.0 / (ctx.global_scale * 31.5);
        for (lo..hi) |n64| {
            for (0..ctx.kg) |g| {
                const tile = n64 * ctx.kg + g;
                const wtile = ctx.w_out[tile * 768 ..][0..768];
                const stile = ctx.s_out[tile * 128 ..][0..128];
                for (0..64) |tid| {
                    const row = n64 * 64 + tid;
                    var q: [64]u8 = @splat(0);
                    var sc: [2]u8 = .{ 0, 0 };
                    if (row < ctx.n) {
                        const row_base = row * ctx.k + g * 64;
                        for (0..2) |blk| {
                            var w: [32]f32 = undefined;
                            var mmax: f32 = 0.0;
                            for (0..32) |j| {
                                const f = bf16ToF32(ctx.bf16, row_base + blk * 32 + j);
                                w[j] = f;
                                mmax = @max(mmax, @abs(f));
                            }
                            const sc_code = core.exlr8.codec.e4m3FromF32(std.math.clamp(mmax * inv_g315, 0.0, 448.0));
                            sc[blk] = sc_code;
                            const sc_val = core.exlr8.codec.e4m3Value(sc_code) * ctx.global_scale;
                            if (sc_val > 0.0) {
                                const inv = 1.0 / sc_val;
                                for (0..32) |j| {
                                    const raw = @round(w[j] * inv + 31.5);
                                    q[blk * 32 + j] = if (raw <= 0.0) 0 else if (raw >= 63.0) 63 else @intFromFloat(raw);
                                }
                            } else {
                                for (0..32) |j| q[blk * 32 + j] = 32;
                            }
                        }
                    }
                    stile[tid * 2 + 0] = sc[0];
                    stile[tid * 2 + 1] = sc[1];
                    const jj = tid >> 3;
                    const r_in_jj = tid & 7;
                    const wjj = wtile[jj * 96 ..][0..96];
                    for (0..4) |sub| {
                        const lane = (r_in_jj << 2) | sub;
                        var w0: u32 = 0;
                        var w1: u32 = 0;
                        var w2: u32 = 0;
                        for (0..4) |kt| {
                            for (0..4) |e| {
                                const c = kt * 16 + 2 * sub + (e & 1) + 8 * (e >> 1);
                                const qv: u32 = q[c];
                                const lo4 = qv & 0xF;
                                const hi2 = (qv >> 4) & 0x3;
                                if (kt < 2) {
                                    w0 |= lo4 << @intCast(kt * 16 + e * 4);
                                } else {
                                    w1 |= lo4 << @intCast((kt - 2) * 16 + e * 4);
                                }
                                w2 |= hi2 << @intCast(kt * 8 + e * 2);
                            }
                        }
                        wjj[lane] = w0;
                        wjj[32 + lane] = w1;
                        wjj[64 + lane] = w2;
                    }
                }
            }
        }
    }
};

pub fn project(gpa: std.mem.Allocator, driver: *const cuda.Driver, bf16: []const u8, n: usize, k: usize) !projection.Projection {
    if (k % 64 != 0 or bf16.len != n * k * 2) return error.UnexpectedTensor;
    const count = n * k;
    var top_bits: u16 = 0;
    for (0..count) |i| {
        const u = std.mem.readInt(u16, bf16[i * 2 ..][0..2], .little) & 0x7fff;
        if (u <= 0x7f80 and u > top_bits) top_bits = u;
    }
    const max_abs: f32 = @bitCast(@as(u32, top_bits) << 16);
    const global_scale: f32 = if (max_abs > 0.0) max_abs / (31.5 * 448.0) else 1.0;

    const npad = cuda.nvfp4.padded(n);
    const n64_tot = npad / 64;
    const kg = k / 64;
    const tiles = n64_tot * kg;
    const packed_w = try gpa.alloc(u32, tiles * 768);
    defer gpa.free(packed_w);
    const packed_s = try gpa.alloc(u8, tiles * 128);
    defer gpa.free(packed_s);

    const n_th: usize = if (n64_tot >= 8) 8 else 1;
    const job: PackJob = .{
        .bf16 = bf16,
        .w_out = packed_w,
        .s_out = packed_s,
        .n = n,
        .k = k,
        .n64_tot = n64_tot,
        .kg = kg,
        .global_scale = global_scale,
        .per = (n64_tot + n_th - 1) / n_th,
    };
    if (n_th == 1) {
        PackJob.worker(&job, 0);
    } else {
        var ths: [8]?std.Thread = @splat(null);
        for (0..n_th) |ti| ths[ti] = std.Thread.spawn(.{}, PackJob.worker, .{ &job, ti }) catch null;
        for (0..n_th) |ti| {
            if (ths[ti]) |t| t.join() else PackJob.worker(&job, ti);
        }
    }

    var p = projection.Projection.empty(driver);
    p.w = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(packed_w));
    errdefer p.w.free();
    p.s = try cuda.DeviceBuffer.fromHost(driver, packed_s);
    p.n = n;
    p.k = k;
    p.native = .{
        .mode = .int6,
        .codes = p.w.ptr,
        .scales = p.s.ptr,
        .scale = global_scale,
        .n = @intCast(n),
        .k = @intCast(k),
        .npad = @intCast(npad),
    };
    return p;
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
    const s = std.mem.span(raw);
    return std.mem.eql(u8, s, "1") or std.mem.eql(u8, s, "w2") or std.mem.eql(u8, s, "w1");
}

fn useW2() bool {
    const raw = std.c.getenv("TF_FLASHNEXT_HEAD_NVFP4") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "w2");
}

fn useW1() bool {
    const raw = std.c.getenv("TF_FLASHNEXT_HEAD_NVFP4") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "w1");
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

    var bf16_buf = try cuda.DeviceBuffer.fromHost(driver, t.bytes);
    errdefer bf16_buf.free();
    store.release();

    const npad = cuda.nvfp4.padded(n);
    if (useW1()) {
        const global_scale: f32 = if (max_abs > 0) max_abs / 448.0 else 1.0;
        const words_bytes = (npad / 64) * (k / 64) * 512;
        const scales_bytes = (npad / 64) * (k / 64) * 64;
        var w_buf = try cuda.DeviceBuffer.alloc(driver, words_bytes);
        errdefer w_buf.free();
        var s_buf = try cuda.DeviceBuffer.alloc(driver, scales_bytes);
        errdefer s_buf.free();
        errdefer cuda.completion.drain(driver, stream, "head w1 pack did not drain");

        var batch_mod = try cuda.Module.load(driver, cuda.kernels.flashnext_batch);
        defer batch_mod.unload();
        const quant_fn = try batch_mod.function("fn_bf16_to_w1");
        var qa: cuda.Args = .{};
        qa.add(bf16_buf.ptr);
        qa.add(global_scale);
        qa.add(w_buf.ptr);
        qa.add(s_buf.ptr);
        qa.add(@as(c_int, @intCast(n)));
        qa.add(@as(c_int, @intCast(k)));
        qa.add(@as(c_int, @intCast(npad)));
        try cuda.launch.launch(quant_fn, .{ .grid = .{ .x = @intCast(npad / 64), .y = @intCast(k / 64) }, .block = .{ .x = 64 } }, stream, &qa);
        try stream.synchronize();

        return .{
            .w = w_buf,
            .s = s_buf,
            .b = bf16_buf,
            .n = n,
            .k = k,
            .native = .{
                .mode = .w1,
                .codes = w_buf.ptr,
                .scales = s_buf.ptr,
                .scale = global_scale,
                .n = @intCast(n),
                .k = @intCast(k),
                .npad = @intCast(npad),
            },
        };
    }
    if (useW2()) {
        const global_scale: f32 = if (max_abs > 0) max_abs / (1.5 * 448.0) else 1.0;
        const words_bytes = (npad / 64) * (k / 64) * 1024;
        const scales_bytes = (npad / 64) * (k / 64) * 128;
        var w_buf = try cuda.DeviceBuffer.alloc(driver, words_bytes);
        errdefer w_buf.free();
        var s_buf = try cuda.DeviceBuffer.alloc(driver, scales_bytes);
        errdefer s_buf.free();
        errdefer cuda.completion.drain(driver, stream, "head w2 pack did not drain");

        var batch_mod = try cuda.Module.load(driver, cuda.kernels.flashnext_batch);
        defer batch_mod.unload();
        const quant_fn = try batch_mod.function("fn_bf16_to_w2");
        var qa: cuda.Args = .{};
        qa.add(bf16_buf.ptr);
        qa.add(global_scale);
        qa.add(w_buf.ptr);
        qa.add(s_buf.ptr);
        qa.add(@as(c_int, @intCast(n)));
        qa.add(@as(c_int, @intCast(k)));
        qa.add(@as(c_int, @intCast(npad)));
        try cuda.launch.launch(quant_fn, .{ .grid = .{ .x = @intCast(npad / 64), .y = @intCast(k / 64) }, .block = .{ .x = 64 } }, stream, &qa);
        try stream.synchronize();

        return .{
            .w = w_buf,
            .s = s_buf,
            .b = bf16_buf,
            .n = n,
            .k = k,
            .native = .{
                .mode = .w2,
                .codes = w_buf.ptr,
                .scales = s_buf.ptr,
                .scale = 4.0 * global_scale,
                .n = @intCast(n),
                .k = @intCast(k),
                .npad = @intCast(npad),
            },
        };
    }

    const global_scale: f32 = if (max_abs > 0) max_abs / (6.0 * 448.0) else 1.0;
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
    return packRaw(gpa, driver, t.bytes, t.dim(0), t.dim(1));
}

pub fn packRaw(gpa: std.mem.Allocator, driver: *const cuda.Driver, src_bytes: []const u8, n: usize, k: usize) !projection.Projection {
    if (n == 0 or k == 0 or k % 64 != 0) return error.UnexpectedTensor;
    const count = try std.math.mul(usize, n, k);
    if (src_bytes.len != try std.math.mul(usize, count, 2)) return error.UnexpectedTensor;

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
        .src = src_bytes,
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
            .src = src_bytes,
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

  # 3. Fused 2.25-bit E4M3 group-32 MTP proposal head sharing Engine's resident BF16 lm_head buffer
  cat > "$fn_dir/cuda_mtp_w2.zig" <<'ZIG'
//! Fused 2.25-bit symmetric group-32 E4M3 proposal head + single-block top-64 exact BF16 rescore directly to d.best.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");

const kg = wt.dims / 32;
const kg2 = kg / 2;

const PackCtx = struct {
    full_rows: []const u8,
    ids: []const u32,
    codes: []u32,
    scales: []u16,
    globals: []f32,
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
            var g_scales: [kg]f32 = undefined;
            var g_words: [kg]u64 = undefined;
            var rmax: f32 = 0.0;
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
                    g_scales[g] = 0.0;
                    g_words[g] = 0;
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
                g_scales[g] = s;
                g_words[g] = word;
                if (s > rmax) rmax = s;
            }
            const gsc: f32 = if (rmax > 0.0) rmax / 448.0 else 1.0;
            const inv_gsc: f32 = 1.0 / gsc;
            ctx.globals[blk * 32 + lane] = gsc;
            for (0..kg2) |g2| {
                const idx = (blk * kg2 + g2) * 32 + lane;
                const w0 = g_words[g2 * 2];
                const w1 = g_words[g2 * 2 + 1];
                ctx.codes[idx * 4 + 0] = @truncate(w0);
                ctx.codes[idx * 4 + 1] = @truncate(w0 >> 32);
                ctx.codes[idx * 4 + 2] = @truncate(w1);
                ctx.codes[idx * 4 + 3] = @truncate(w1 >> 32);
                const s0 = core.exlr8.codec.e4m3FromF32(std.math.clamp(g_scales[g2 * 2] * inv_gsc, 0.0, 448.0));
                const s1 = core.exlr8.codec.e4m3FromF32(std.math.clamp(g_scales[g2 * 2 + 1] * inv_gsc, 0.0, 448.0));
                ctx.scales[idx] = @as(u16, s0) | (@as(u16, s1) << 8);
            }
        }
    }
};

const W1PackCtx = struct {
    full_rows: []const u8,
    ids: []const u32,
    ws: []u32,
    globals: []f32,
    v: usize,
    per: usize,
    cal: f32,
    g64: bool,

    fn worker(ctx: *const W1PackCtx, tid: usize) void {
        const lo = @min(ctx.v, tid * ctx.per);
        const hi = @min(ctx.v, lo + ctx.per);
        for (lo..hi) |r| {
            const blk = r / 32;
            const lane = r % 32;
            const gid: usize = ctx.ids[r];
            const rb = ctx.full_rows[gid * wt.dims * 2 ..][0 .. wt.dims * 2];
            if (ctx.g64) {
                var g_scales: [40]f32 = undefined;
                var g_w0: [40]u32 = undefined;
                var g_w1: [40]u32 = undefined;
                var rmax: f32 = 0.0;
                for (0..40) |g| {
                    var sum_abs: f32 = 0.0;
                    var w0: u32 = 0;
                    var w1: u32 = 0;
                    for (0..32) |j| {
                        const bits = std.mem.readInt(u16, rb[(g * 64 + j) * 2 ..][0..2], .little);
                        const f: f32 = @bitCast(@as(u32, bits) << 16);
                        sum_abs += @abs(f);
                        if (f < 0.0) w0 |= @as(u32, 1) << @intCast(j);
                    }
                    for (0..32) |j| {
                        const bits = std.mem.readInt(u16, rb[(g * 64 + 32 + j) * 2 ..][0..2], .little);
                        const f: f32 = @bitCast(@as(u32, bits) << 16);
                        sum_abs += @abs(f);
                        if (f < 0.0) w1 |= @as(u32, 1) << @intCast(j);
                    }
                    const s = sum_abs / 64.0;
                    g_scales[g] = s;
                    g_w0[g] = w0;
                    g_w1[g] = w1;
                    if (s > rmax) rmax = s;
                }
                const gsc: f32 = if (rmax > 0.0) rmax / 448.0 else 1.0;
                const inv_gsc: f32 = 1.0 / gsc;
                ctx.globals[blk * 32 + lane] = gsc * ctx.cal;
                for (0..10) |g4| {
                    const base_u4 = blk * 720 + g4 * 72;
                    const idx_a = (base_u4 + lane) * 4;
                    ctx.ws[idx_a + 0] = g_w0[g4 * 4 + 0];
                    ctx.ws[idx_a + 1] = g_w1[g4 * 4 + 0];
                    ctx.ws[idx_a + 2] = g_w0[g4 * 4 + 1];
                    ctx.ws[idx_a + 3] = g_w1[g4 * 4 + 1];
                    const idx_b = (base_u4 + 32 + lane) * 4;
                    ctx.ws[idx_b + 0] = g_w0[g4 * 4 + 2];
                    ctx.ws[idx_b + 1] = g_w1[g4 * 4 + 2];
                    ctx.ws[idx_b + 2] = g_w0[g4 * 4 + 3];
                    ctx.ws[idx_b + 3] = g_w1[g4 * 4 + 3];
                    const s0: u32 = core.exlr8.codec.e4m3FromF32(std.math.clamp(g_scales[g4 * 4 + 0] * inv_gsc, 0.0, 448.0));
                    const s1: u32 = core.exlr8.codec.e4m3FromF32(std.math.clamp(g_scales[g4 * 4 + 1] * inv_gsc, 0.0, 448.0));
                    const s2: u32 = core.exlr8.codec.e4m3FromF32(std.math.clamp(g_scales[g4 * 4 + 2] * inv_gsc, 0.0, 448.0));
                    const s3: u32 = core.exlr8.codec.e4m3FromF32(std.math.clamp(g_scales[g4 * 4 + 3] * inv_gsc, 0.0, 448.0));
                    ctx.ws[(base_u4 + 64) * 4 + lane] = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
                }
            } else {
                var g_scales: [80]f32 = undefined;
                var g_w: [80]u32 = undefined;
                var rmax: f32 = 0.0;
                for (0..80) |g| {
                    var sum_abs: f32 = 0.0;
                    var w0: u32 = 0;
                    for (0..32) |j| {
                        const bits = std.mem.readInt(u16, rb[(g * 32 + j) * 2 ..][0..2], .little);
                        const f: f32 = @bitCast(@as(u32, bits) << 16);
                        sum_abs += @abs(f);
                        if (f < 0.0) w0 |= @as(u32, 1) << @intCast(j);
                    }
                    const s = sum_abs / 32.0;
                    g_scales[g] = s;
                    g_w[g] = w0;
                    if (s > rmax) rmax = s;
                }
                const gsc: f32 = if (rmax > 0.0) rmax / 448.0 else 1.0;
                const inv_gsc: f32 = 1.0 / gsc;
                ctx.globals[blk * 32 + lane] = gsc * ctx.cal;
                for (0..20) |g4| {
                    const base_u4 = blk * 800 + g4 * 40;
                    const idx_w = (base_u4 + lane) * 4;
                    ctx.ws[idx_w + 0] = g_w[g4 * 4 + 0];
                    ctx.ws[idx_w + 1] = g_w[g4 * 4 + 1];
                    ctx.ws[idx_w + 2] = g_w[g4 * 4 + 2];
                    ctx.ws[idx_w + 3] = g_w[g4 * 4 + 3];
                    const s0: u32 = core.exlr8.codec.e4m3FromF32(std.math.clamp(g_scales[g4 * 4 + 0] * inv_gsc, 0.0, 448.0));
                    const s1: u32 = core.exlr8.codec.e4m3FromF32(std.math.clamp(g_scales[g4 * 4 + 1] * inv_gsc, 0.0, 448.0));
                    const s2: u32 = core.exlr8.codec.e4m3FromF32(std.math.clamp(g_scales[g4 * 4 + 2] * inv_gsc, 0.0, 448.0));
                    const s3: u32 = core.exlr8.codec.e4m3FromF32(std.math.clamp(g_scales[g4 * 4 + 3] * inv_gsc, 0.0, 448.0));
                    ctx.ws[(base_u4 + 32) * 4 + lane] = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
                }
            }
        }
    }
};

pub const W2 = struct {
    codes: ?cuda.DeviceBuffer = null,
    scales: ?cuda.DeviceBuffer = null,
    globals: ?cuda.DeviceBuffer = null,
    vocab_ids: ?cuda.DeviceBuffer = null,
    bf16_ptr: u64 = 0,
    scratch: ?cuda.DeviceBuffer = null,
    gemv_fn: ?cuda.Function = null,
    rescore_fn: ?cuda.Function = null,
    pick_fn: ?cuda.Function = null,
    multi_fn: ?cuda.Function = null,
    use_multi: bool = true,
    is_w1: bool = false,
    k_top: c_int = 32,

    pub fn ready(self: *const W2) bool {
        return self.codes != null and self.bf16_ptr != 0;
    }

    pub fn deinit(self: *W2) void {
        if (self.scratch) |*b| b.free();
        if (self.vocab_ids) |*b| b.free();
        if (self.globals) |*b| b.free();
        if (self.scales) |*b| b.free();
        if (self.codes) |*b| b.free();
        self.* = .{};
    }

    pub fn init(gpa: std.mem.Allocator, driver: *cuda.Driver, store: *wt.Store, artifact: *const core.proposal_vocab.Artifact, module: *const cuda.Module) !W2 {
        const raw = std.c.getenv("TF_FLASHNEXT_MTP_W2") orelse return .{};
        const mode_s = std.mem.span(raw);
        const is_w1g64 = std.mem.eql(u8, mode_s, "w1g64");
        const is_w1g32 = std.mem.eql(u8, mode_s, "w1");
        if (!is_w1g64 and !is_w1g32 and !std.mem.eql(u8, mode_s, "1")) return .{};
        const tensor = try store.tensor("language_model.lm_head.weight");
        if (tensor.dtype != .bf16 or !tensor.is(.bf16, &.{ wt.vocab, wt.dims })) return .{};
        const ids = artifact.ids();
        const v = ids.len;
        const nb = (v + 31) / 32;
        const n_blocks = (nb + 3) / 4;

        if (is_w1g64 or is_w1g32) {
            const cal: f32 = if (std.c.getenv("TF_FLASHNEXT_W1_CAL")) |cv| (std.fmt.parseFloat(f32, std.mem.span(cv)) catch 1.15) else 1.15;
            const k_top_v: c_int = if (std.c.getenv("TF_FLASHNEXT_W1_KTOP")) |kv| (std.fmt.parseInt(c_int, std.mem.span(kv), 10) catch 64) else 64;
            const u4_per_blk: usize = if (is_w1g64) 720 else 800;
            const ws_host = try gpa.alloc(u32, nb * u4_per_blk * 4);
            defer gpa.free(ws_host);
            @memset(ws_host, 0);
            const globals_host = try gpa.alloc(f32, nb * 32);
            defer gpa.free(globals_host);
            @memset(globals_host, 0.0);

            const n_threads: usize = 8;
            const ctx: W1PackCtx = .{
                .full_rows = tensor.bytes,
                .ids = ids,
                .ws = ws_host,
                .globals = globals_host,
                .v = v,
                .per = (v + n_threads - 1) / n_threads,
                .cal = cal,
                .g64 = is_w1g64,
            };
            var ths: [n_threads]?std.Thread = @splat(null);
            for (0..n_threads) |ti| ths[ti] = try std.Thread.spawn(.{}, W1PackCtx.worker, .{ &ctx, ti });
            for (&ths) |*th| if (th.*) |t| t.join();
            store.release();

            var ids_buf = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(ids));
            errdefer ids_buf.free();
            var codes_buf = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(ws_host));
            errdefer codes_buf.free();
            var globals_buf = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(globals_host));
            errdefer globals_buf.free();
            const sc_bytes = @max(n_blocks * 16, 16384) + 16384;
            const z = try gpa.alloc(u8, sc_bytes);
            defer gpa.free(z);
            @memset(z, 0);
            var scratch_buf = try cuda.DeviceBuffer.fromHost(driver, z);
            errdefer scratch_buf.free();
            const use_multi = if (std.c.getenv("TF_FLASHNEXT_W2_MULTI")) |mv| !std.mem.eql(u8, std.mem.span(mv), "0") else true;

            return .{
                .codes = codes_buf,
                .scales = null,
                .globals = globals_buf,
                .vocab_ids = ids_buf,
                .scratch = scratch_buf,
                .gemv_fn = try module.function(if (is_w1g64) "fn_w1g64_gemv_fused" else "fn_w1g32_gemv_fused"),
                .rescore_fn = try module.function("fn_w2_rescore_best"),
                .pick_fn = try module.function("fn_w2_pick_cands"),
                .multi_fn = try module.function("fn_w2_rescore_multi"),
                .use_multi = use_multi,
                .is_w1 = true,
                .k_top = k_top_v,
            };
        }

        const total = nb * kg2 * 32;
        const codes_host = try gpa.alloc(u32, total * 4);
        defer gpa.free(codes_host);
        @memset(codes_host, 0);
        const scales_host = try gpa.alloc(u16, total);
        defer gpa.free(scales_host);
        @memset(scales_host, 0);
        const globals_host = try gpa.alloc(f32, nb * 32);
        defer gpa.free(globals_host);
        @memset(globals_host, 0.0);

        const n_threads: usize = 8;
        const ctx: PackCtx = .{
            .full_rows = tensor.bytes,
            .ids = ids,
            .codes = codes_host,
            .scales = scales_host,
            .globals = globals_host,
            .v = v,
            .per = (v + n_threads - 1) / n_threads,
        };
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
        var globals_buf = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(globals_host));
        errdefer globals_buf.free();
        const sc_bytes = @max(n_blocks * 16, 16384) + 16384;
        const z = try gpa.alloc(u8, sc_bytes);
        defer gpa.free(z);
        @memset(z, 0);
        var scratch_buf = try cuda.DeviceBuffer.fromHost(driver, z);
        errdefer scratch_buf.free();
        const use_multi = if (std.c.getenv("TF_FLASHNEXT_W2_MULTI")) |mv| !std.mem.eql(u8, std.mem.span(mv), "0") else true;

        return .{
            .codes = codes_buf,
            .scales = scales_buf,
            .globals = globals_buf,
            .vocab_ids = ids_buf,
            .scratch = scratch_buf,
            .gemv_fn = try module.function("fn_w2_gemv_fused"),
            .rescore_fn = try module.function("fn_w2_rescore_best"),
            .pick_fn = try module.function("fn_w2_pick_cands"),
            .multi_fn = try module.function("fn_w2_rescore_multi"),
            .use_multi = use_multi,
            .is_w1 = false,
            .k_top = 32,
        };
    }

    pub fn projectFused(self: *const W2, stream: cuda.Stream, input: u64, dlogits: u64, best_out: u64, v: usize) !void {
        const nb = (v + 31) / 32;
        const n_blocks = (nb + 3) / 4;
        const sc = self.scratch.?.ptr;
        var a1: cuda.Args = .{};
        a1.add(input);
        a1.add(self.codes.?.ptr);
        if (!self.is_w1) a1.add(self.scales.?.ptr);
        a1.add(self.globals.?.ptr);
        a1.add(dlogits);
        a1.add(sc);
        a1.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(self.gemv_fn.?, .{ .grid = .{ .x = @intCast(n_blocks) }, .block = .{ .x = 128 }, .pdl = true }, stream, &a1);

        if (self.use_multi and n_blocks <= 1024 and self.k_top > 0 and self.k_top <= 64 and self.pick_fn != null and self.multi_fn != null) {
            const po = sc + 16384;
            var ap: cuda.Args = .{};
            ap.add(dlogits);
            ap.add(sc);
            ap.add(@as(c_int, @intCast(n_blocks)));
            ap.add(@as(c_int, @intCast(v)));
            ap.add(self.k_top);
            ap.add(po);
            ap.add(best_out);
            try cuda.launch.launch(self.pick_fn.?, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 }, .pdl = true }, stream, &ap);

            var am: cuda.Args = .{};
            am.add(input);
            am.add(self.bf16_ptr);
            am.add(self.vocab_ids.?.ptr);
            am.add(dlogits);
            am.add(po);
            am.add(best_out);
            try cuda.launch.launch(self.multi_fn.?, .{ .grid = .{ .x = @intCast(self.k_top) }, .block = .{ .x = 128 }, .pdl = true }, stream, &am);
            return;
        }

        var a2: cuda.Args = .{};
        a2.add(input);
        a2.add(self.bf16_ptr);
        a2.add(self.vocab_ids.?.ptr);
        a2.add(dlogits);
        a2.add(sc);
        a2.add(@as(c_int, @intCast(n_blocks)));
        a2.add(@as(c_int, @intCast(v)));
        a2.add(self.k_top);
        a2.add(best_out);
        try cuda.launch.launch(self.rescore_fn.?, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 }, .pdl = true }, stream, &a2);
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

# 5a. zig/src/cuda/qmmf.zig + qmmf_shared.zig: Mode::bf12 + Mode::int6 + Mode::w2 + Mode::w1 + PR #3475 fused split-K
patch_exact(
    cdir, "qmmf.zig",
    "pub const Mode = enum(u3) { fp4 = 0, fp8 = 1, mxfp8 = 2, fp8g = 3, bf16 = 4 };\n\nconst modes = [_]Mode{ .fp4, .fp8, .mxfp8, .fp8g, .bf16 };",
    "pub const Mode = enum(u4) { fp4 = 0, fp8 = 1, mxfp8 = 2, fp8g = 3, bf16 = 4, bf12 = 5, int6 = 6, w2 = 7, w1 = 8 };\n\nconst modes = [_]Mode{ .fp4, .fp8, .mxfp8, .fp8g, .bf16, .bf12, .int6, .w2, .w1 };",
)
patch_exact(
    cdir, "qmmf.zig",
    """        .bf16 => bn * group * 2,
    };
    const s: u32 = if (mode == .fp4 or mode == .fp8g) bn * 4 else if (mode == .mxfp8) bn * 2 else 0;
    const stage = (bm * group * 2 + w + s + 127) / 128 * 128;
    const partials = (bm / 16) * 2 * 4 * threads * 4;
    return @max(4 * stage, partials);""",
    """        .bf16 => bn * group * 2,
        .bf12 => (bn * group * 3) / 2,
        .int6 => (bn * group * 3) / 4,
        .w2 => bn * group / 4,
        .w1 => bn * group / 8,
    };
    const s: u32 = if (mode == .fp4 or mode == .fp8g) bn * 4 else if (mode == .mxfp8 or mode == .int6 or mode == .w2) bn * 2 else if (mode == .bf12 or mode == .w1) bn else 0;
    const stage = (bm * group * 2 + w + s + 127) / 128 * 128;
    const partials = (bm / 16) * 2 * 4 * threads * 4;
    const stages: u32 = if (mode == .bf12) 3 else 4;
    return @max(stages * stage, partials);""",
)
patch_exact(
    cdir, "qmmf.zig",
    """    const name = "_ZN7tf_qmmf11qmmf_kernelILi{d}ELi{d}ELi64ELi1ELi4ELi4ELb{d}ELb{d}ELb{d}" ++
        "EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    const flags = .{ @intFromBool(wide), @intFromBool(cluster), @intFromBool(fused) };
    return std.fmt.comptimePrint(name, .{ @backingInt(mode), bm, flags[0], flags[1], flags[2] });""",
    """    const st: u32 = if (mode == .bf12) 3 else 4;
    const name = "_ZN7tf_qmmf11qmmf_kernelILi{d}ELi{d}ELi64ELi1ELi4ELi{d}ELb{d}ELb{d}ELb{d}" ++
        "EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    const flags = .{ @intFromBool(wide), @intFromBool(cluster), @intFromBool(fused) };
    return std.fmt.comptimePrint(name, .{ @backingInt(mode), bm, st, flags[0], flags[1], flags[2] });""",
)
patch_exact(
    cdir, "qmmf.zig",
    """fn halfSymbol(comptime mode: Mode, comptime cluster: bool, comptime wide: bool) [:0]const u8 {
    const name = "_ZN7tf_qmmf11qmmf_kernelILi{d}ELi16ELi32ELi1ELi4ELi4ELb{d}ELb{d}ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    return std.fmt.comptimePrint(name, .{ @backingInt(mode), @intFromBool(wide), @intFromBool(cluster) });
}

/// Dynamic shared bytes of the half tile: a stage holds half the weights (and scales) of the 64-column tile's.
pub fn halfShared(mode: Mode) u32 {
    const w: u32 = if (mode == .bf16) 32 * group * 2 else 32 * group;
    const s: u32 = if (mode == .mxfp8) 32 * 2 else 0;
    const stage = (16 * group * 2 + w + s + 127) / 128 * 128;
    const partials = 1 * 1 * 4 * threads * 4;
    return @max(4 * stage, partials);
}""",
    """fn halfSymbol(comptime mode: Mode, comptime cluster: bool, comptime wide: bool) [:0]const u8 {
    const st: u32 = if (mode == .bf12) 3 else 4;
    const name = "_ZN7tf_qmmf11qmmf_kernelILi{d}ELi16ELi32ELi1ELi4ELi{d}ELb{d}ELb{d}ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    return std.fmt.comptimePrint(name, .{ @backingInt(mode), st, @intFromBool(wide), @intFromBool(cluster) });
}

/// Dynamic shared bytes of the half tile: a stage holds half the weights (and scales) of the 64-column tile's.
pub fn halfShared(mode: Mode) u32 {
    const w: u32 = if (mode == .bf16) 32 * group * 2 else if (mode == .bf12) (32 * group * 3) / 2 else if (mode == .int6) (32 * group * 3) / 4 else 32 * group;
    const s: u32 = if (mode == .mxfp8 or mode == .int6) 32 * 2 else if (mode == .bf12) 32 else 0;
    const stage = (16 * group * 2 + w + s + 127) / 128 * 128;
    const partials = 1 * 1 * 4 * threads * 4;
    const stages: u32 = if (mode == .bf12) 3 else 4;
    return @max(stages * stage, partials);
}""",
)
# 5a2. zig/src/cuda/experts.zig + experts_waves.zig: TF_NVFP4_DYNAMIC_GRID=1 uncapped 1-warp CTA hardware scheduling
patch_exact(
    cdir, "experts.zig",
    "        const grid = @min((items * nb + e.warps - 1) / e.warps, e.resident[ei]);",
    "        const dyn = e.warps == 1 and (if (std.c.getenv(\"TF_NVFP4_DYNAMIC_GRID\")) |v| std.mem.eql(u8, std.mem.span(v), \"1\") else false);\n        const grid = if (dyn) items * nb else @min((items * nb + e.warps - 1) / e.warps, e.resident[ei]);",
)
patch_exact(
    cdir, "experts_waves.zig",
    "    const fi = pick(gate, items, nb, e.resident[first + 2] * e.warps, classic) orelse return e.run(s, epi, in, l, p, out, items, skip);",
    "    const dyn = e.warps == 1 and (if (std.c.getenv(\"TF_NVFP4_DYNAMIC_GRID\")) |v| std.mem.eql(u8, std.mem.span(v), \"1\") else false);\n    const fi = (if (dyn) (if (gate and 2 * items * nb <= e.resident[first + 2] * e.warps) @as(?usize, first + 2) else null) else pick(gate, items, nb, e.resident[first + 2] * e.warps, classic)) orelse return e.run(s, epi, in, l, p, out, items, skip);",
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
        .int6 => 3,
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
pub const prompt_bf12_shared: u32 = 3 * ((128 * group * 2 + (128 * group * 3) / 2 + 128 + 127) / 128 * 128) + 128;
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
    """    half: [4][2][2]Function, // by halfModeIndex (mxfp8, bf16, bf12, int6), Out, then plain / cluster
    half_below: usize = half_below_default,
    shared_down: [4]Function = undefined,
    shared_down_bf12: [4]Function = undefined,
    hc_upmix_bf12: [3]Function = undefined, // BM=16/32/64 fused BF12 hc.up + dense_mix
    half32_bf12_f32: Function = undefined, // BM=32 BN=32 STAGES=3 f32 (88-CTA hc.down for BM=32 decode)
    bf12_s2: [2]Function = undefined, // Out (bf16/f32), BM=16 BN=64 STAGES=2 plain (5 CTAs/SM for short-K per<=12)
    fuse_s: [4][2][2]Function = undefined, // by halfModeIndex, Out, bm=16/32 (PR #3475)
    fuse_split_n: usize = 0,
    pdl: bool = false,
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
    """        inline for (.{ Mode.mxfp8, Mode.bf16, Mode.bf12, Mode.int6 }, 0..) |mode, mi| inline for (.{ false, true }, 0..) |wide, oi| inline for (.{ false, true }, 0..) |cl, ci| {
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
        inline for (.{ .{ "tf_qmmf_bf12_hc_upmix_16", 23040 }, .{ "tf_qmmf_bf12_hc_upmix_32", 30080 }, .{ "tf_qmmf_bf12_hc_upmix_64", 47360 } }, 0..) |spec, hi| {
            const f = try mod.function(spec[0]);
            try f.allowDynamicShared(spec[1]);
            l.hc_upmix_bf12[hi] = f;
        }
        {
            const f32h = try mod.function("_ZN7tf_qmmf11qmmf_kernelILi5ELi32ELi32ELi1ELi4ELi3ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii");
            try f32h.allowDynamicShared(21888);
            l.half32_bf12_f32 = f32h;
        }
        inline for (.{ Mode.mxfp8, Mode.bf16, Mode.bf12, Mode.int6 }, 0..) |mode, mi| inline for (.{ false, true }, 0..) |wide, oi| inline for (.{ 16, 32 }, 0..) |bm, bi| {
            const f = try mod.function(symbol(mode, bm, false, true, wide));
            try f.allowDynamicShared(sharedBytes(mode, bm));
            l.fuse_s[mi][oi][bi] = f;
        };
        inline for (.{ false, true }, 0..) |wide, oi| {
            const sym = std.fmt.comptimePrint("_ZN7tf_qmmf11qmmf_kernelILi5ELi16ELi64ELi1ELi4ELi2ELb{d}ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", .{@intFromBool(wide)});
            const f = try mod.function(sym);
            try f.allowDynamicShared(2 * 8320);
            l.bf12_s2[oi] = f;
        }
        l.half_below = try halfBelow();
        if (std.c.getenv("TF_FLASHNEXT_FUSED_SPLIT")) |v| {
            if (std.mem.eql(u8, std.mem.span(v), "1")) l.fuse_split_n = 2048;
        }
        l.pdl = capability_major >= 9 and (if (std.c.getenv("TF_FLASHNEXT_PDL")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true);
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
    """    /// matmul() writing y as `out` says: bf16, or fp32.
    pub fn matmulOut(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, out: Out, part: u64) !void {
        const p = planFor(m, w, l.clusters);
        if (p.reduce and part == 0) return error.MissingLanePart;
        const set = (if (out == .f32) l.f32_fns else l.fns)[modeIndex(w.mode)];
        const ti: usize = if (p.bm == 16) 0 else if (p.bm == 32) 1 else 2;
        const prompt = l.usesPrompt(m, w);
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
        }, s, &args);
        if (p.reduce) {
            const total: usize = m * w.n;
            var r: launch_.Args = .{};
            r.add(part);
            r.add(y);
            r.add(@as(i64, @intCast(total)));
            r.add(@as(i32, @intCast(p.sk)));
            r.add(w.scale);
            try launch_.launch(l.reduce[@backingInt(out)], .{
                .grid = .{ .x = @intCast((total + 255) / 256), .y = 1, .z = 1 },
                .block = .{ .x = 256, .y = 1, .z = 1 },
            }, s, &r);
        }
    }""",
    """    /// matmul() writing y as `out` says: bf16, or fp32.
    pub fn matmulOut(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, out: Out, part: u64) !void {
        _ = try l.matmulOutImpl(s, x, ldx, m, w, y, out, part, true, false);
    }

    /// Fused bit-exact BF12 hc.up [4*D, 320] + fn_hc_dense_mix8 (eliminates s.hc_up R+W and 96 launches/round).
    pub fn hcUpmixBf12(l: Lane, s: Stream, act: u64, w: Weight, norm: u64, mixed: u64, m: usize, d: usize) !void {
        const ti: usize = if (m <= 16) 0 else if (m <= 32) 1 else 2;
        const bm: usize = if (ti == 0) 16 else if (ti == 1) 32 else 64;
        const smem: u32 = if (ti == 0) 23040 else if (ti == 1) 30080 else 47360;
        var args: launch_.Args = .{};
        args.add(act);
        args.add(w.codes);
        args.add(w.scales);
        args.add(norm);
        args.add(mixed);
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(d)));
        try launch_.launch(l.hc_upmix_bf12[ti], .{
            .grid = .{ .x = @intCast((m + bm - 1) / bm), .y = @intCast(d / 32), .z = 1 },
            .block = .{ .x = 128, .y = 1, .z = 1 },
            .shared = smem,
            .pdl = l.pdl,
        }, s, &args);
    }

    /// Launch matmul leaving split-K FP32 partials in `part` (skipping reduce_kernel) and returning slice count (1 if already reduced in `y`).
    pub fn matmulRawPart(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, out: Out, part: u64, allow_cluster: bool) !usize {
        return l.matmulOutImpl(s, x, ldx, m, w, y, out, part, allow_cluster, true);
    }

    fn matmulOutImpl(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, out: Out, part: u64, allow_cluster: bool, skip_reduce: bool) !usize {
        const use_out_sk4 = skip_reduce and !allow_cluster and w.mode == .bf12 and w.n == 2560 and w.k >= 4096 and w.sk == 0 and m <= 64 and (if (std.c.getenv("TF_BF12_OUT_SK4")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else false);
        const p = if (use_out_sk4) planSlices(m, 4, false) else planFor(m, w, l.clusters and allow_cluster);
        if (p.reduce and part == 0) return error.MissingLanePart;
        const set = (if (out == .f32) l.f32_fns else l.fns)[modeIndex(w.mode)];
        const ti: usize = if (p.bm == 16) 0 else if (p.bm == 32) 1 else 2;
        const prompt = allow_cluster and l.usesPrompt(m, w);
        const k32 = prompt and w.mode == .bf16 and p.sk == 1 and l.prompt_k32 != null;
        const bm: usize = if (prompt) 128 else p.bm;
        const rows_t = (m + bm - 1) / bm;
        const hm = halfModeIndex(w.mode);
        const fs = l.fuse_split_n != 0 and p.cluster and w.n >= l.fuse_split_n and hm != null;
        const h_thr: usize = if (w.mode == .bf12) 64 else l.half_below;
        const half = hm != null and p.bm == 16 and !p.fused and !fs and rows_t * ((w.n + bn - 1) / bn) * p.sk < h_thr;
        const half32 = w.mode == .bf12 and out == .f32 and p.bm == 32 and !prompt and !p.fused and !p.cluster and !fs and rows_t * ((w.n + bn - 1) / bn) * p.sk < h_thr;
        const s2_max: usize = if (std.c.getenv("TF_BF12_S2_MAX_PER")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 12) else 12;
        const use_s2 = w.mode == .bf12 and !prompt and !half and !half32 and !fs and !p.fused and !p.cluster and p.bm == 16 and (w.k / group) / p.sk <= s2_max;
        const f = if (k32)
            l.prompt_k32.?[@backingInt(out)]
        else if (prompt)
            (if (w.mode == .bf12) l.prompt_bf12.?[@backingInt(out)][@intFromBool(p.fused)] else l.prompt.?[@backingInt(out)][@intFromBool(p.fused)])
        else if (half)
            l.half[hm.?][@backingInt(out)][@intFromBool(p.cluster)]
        else if (half32)
            l.half32_bf12_f32
        else if (fs)
            (if (p.bm == 64) set[6] else l.fuse_s[hm.?][@backingInt(out)][@intFromBool(p.bm == 32)])
        else if (p.fused)
            set[6]
        else if (use_s2)
            l.bf12_s2[@backingInt(out)]
        else
            set[ti * 2 + @intFromBool(p.cluster)];
        const cols: usize = if (prompt) 128 else if (half or half32) 32 else bn;
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
            .shared = if (k32) prompt_k32_shared else if (prompt) (if (w.mode == .bf12) prompt_bf12_shared else prompt_shared) else if (half) halfShared(w.mode) else if (half32) 21888 else if (use_s2) 2 * 8320 else sharedBytes(w.mode, p.bm),
            .cluster = if (p.cluster and !fs) .{ .x = 1, .y = 1, .z = p.sk } else null,
            .pdl = l.pdl and !prompt,
        }, s, &args);
        if (p.reduce) {
            if (skip_reduce) return p.sk;
            const total: usize = m * w.n;
            var r: launch_.Args = .{};
            r.add(part);
            r.add(y);
            r.add(@as(i64, @intCast(total)));
            r.add(@as(i32, @intCast(p.sk)));
            r.add(w.scale);
            try launch_.launch(l.reduce[@backingInt(out)], .{
                .grid = .{ .x = @intCast((total + 255) / 256), .y = 1, .z = 1 },
                .block = .{ .x = 256, .y = 1, .z = 1 },
                .pdl = l.pdl,
            }, s, &r);
        }
        return 1;
    }""",
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
pub const bf12_symbols = [4][:0]const u8{ "tf_qmmf_shared_bf12_16_qtr", "tf_qmmf_shared_bf12_16", "tf_qmmf_shared_bf12_32", "tf_qmmf_shared_bf12_64" };

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
    const qtr = half and w.mode == .bf12;
    const index: usize = if (half) 0 else if (bm == 16) 1 else if (bm == 32) 2 else 3;
    const cols: usize = if (qtr) 16 else if (half) 32 else 64;
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
        .block = .{ .x = if (qtr) 64 else 128 },
        .shared = if (qtr) 3 * 3712 else if (half) q.halfShared(w.mode) else q.sharedBytes(w.mode, @intCast(bm)),
        .pdl = l.pdl,
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

# 5c. cuda_projection_load.zig + cuda_prompt_wt.zig: MTP BF16 projections and MTP HC mixers -> NVFP4 / INT6 / MXFP8
for storage, original in (
    ("t.bytes", "            return projection.dense(a, d, t);"),
    ("bytes", "        return projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = bytes });"),
):
    indent = original[:len(original) - len(original.lstrip())]
    fp4_check = f'if (@import("cuda_mtp_nvfp4.zig").wantProj(prefix))'
    fp4_proj = f'return @import("cuda_mtp_nvfp4.zig").project(a, d, {storage}, n, k);'
    int6_check = f'if (@import("cuda_mtp_int6.zig").wantProj(prefix))'
    int6_proj = f'return @import("cuda_mtp_int6.zig").project(a, d, {storage}, n, k);'
    check = 'if (std.mem.startsWith(u8, prefix, "language_model.mtp."))'
    project = f'return @import("cuda_mtp_mxfp8.zig").project(a, d, {storage}, n, k);'
    patch_exact(root, "cuda_projection_load.zig", original, f"{indent}{fp4_check}\n{indent}    {fp4_proj}\n{indent}{int6_check}\n{indent}    {int6_proj}\n{indent}{check}\n{indent}    {project}\n{original}")

patch_exact(
    root, "cuda_prompt_wt.zig",
    """    var dq = try projection.dense(gpa, driver, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, wide, 1, 1, 1 }, .bytes = data });
    errdefer dq.free();
    const up = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_up.weight", .{prefix}));
    if (!up.is(.bf16, &.{ wide, low })) return error.UnexpectedTensor;
    var uq = try projection.dense(gpa, driver, up);""",
    """    const mtp_mx = std.mem.startsWith(u8, prefix, "language_model.mtp.");
    const mtp_fp4_down = @import("cuda_mtp_nvfp4.zig").wantHcDown(prefix);
    const mtp_fp4_up = @import("cuda_mtp_nvfp4.zig").wantHcUp(prefix);
    const mtp_int6_down = @import("cuda_mtp_int6.zig").wantHcDown(prefix);
    const mtp_int6_up = @import("cuda_mtp_int6.zig").wantHcUp(prefix);
    var dq = if (mtp_fp4_down)
        try @import("cuda_mtp_nvfp4.zig").project(gpa, driver, data, n, wide)
    else if (mtp_int6_down)
        try @import("cuda_mtp_int6.zig").project(gpa, driver, data, n, wide)
    else if (mtp_mx)
        try @import("cuda_mtp_mxfp8.zig").project(gpa, driver, data, n, wide)
    else
        try projection.dense(gpa, driver, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, wide, 1, 1, 1 }, .bytes = data });
    errdefer dq.free();
    const up = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_up.weight", .{prefix}));
    if (!up.is(.bf16, &.{ wide, low })) return error.UnexpectedTensor;
    var uq = if (mtp_fp4_up)
        try @import("cuda_mtp_nvfp4.zig").project(gpa, driver, up.bytes, wide, low)
    else if (mtp_int6_up)
        try @import("cuda_mtp_int6.zig").project(gpa, driver, up.bytes, wide, low)
    else if (mtp_mx)
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
    """        const MtpChainCtl = extern struct {
            c: f64,
            ext_enter: f64,
            ext_stay: f64,
            ext_tok: f64,
            p_cliff: f64,
            start_pos: i32,
            max_steps: i32,
            vocab_n: i32,
            pad: i32,
        };

        const MtpChainOut = extern struct {
            product: f64,
            probs: [32]f64,
            tokens: [32]u32,
            n: i32,
            drafted: i32,
            err: i32,
            pad: i32,
        };

        fn whileEnabled() bool {
            const raw = std.c.getenv("TF_FLASHNEXT_MTP_WHILE") orelse return true;
            return !std.mem.eql(u8, std.mem.span(raw), "0");
        }

        fn buildWhileGraph(o: Self, d: *Draft, s: *parts.Scratch, score_rows: u64, vocab_ids_ptr: u64) !cuda.graph.Exec {
            const e = o.e;
            const h = o.h;
            const st = o.stream();
            var ctx: cuda.abi.Context = null;
            try st.d.check(st.d.api.cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
            var root_g = try cuda.graph.Graph.init(st.d);
            defer root_g.deinit();
            const handle = try root_g.createConditionalHandle(ctx, 0, cuda.abi.graph_cond_assign_default);
            var ia: cuda.Args = .{};
            ia.add(d.best.ptr);
            ia.add(vocab_ids_ptr);
            ia.add(s.ids);
            ia.add(s.pos);
            ia.add(h.while_ctl_dev.ptr);
            ia.add(h.while_out_dev.ptr);
            ia.add(handle);
            const init_node = try root_g.addKernel(&.{}, h.chain_init_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, &ia);
            const while_res = try root_g.addWhileNode(&.{init_node}, handle, ctx);

            var p: batch.Pass(Tri) = .{
                .gpa = e.gpa,
                .io = e.io,
                .stream = st,
                .tri = o.tri,
                .k = &e.batch_k,
                .s = s,
                .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .table_dense = e.table.dense, .layers = (&h.layer)[0..1], .mixer = h.mixer, .head = h.draft, .ngram = &e.ngram },
                .seq = .{ .media = d.media, .conv = &.{}, .state = &.{}, .kc = (&d.kc)[0..1], .vc = (&d.vc)[0..1], .ikc = (&d.ikc)[0..1], .pooled = (&d.pooled)[0..1], .pg = d.pg, .tail = d.kc, .history = &d.unused_history, .cap = d.cap, .scores = score_rows },
                .hooks = .{ .logits = .none },
                .rows = 1,
                .pos0 = d.len,
            };
            const one: pieces.Piece(Self, batch.Pass(Tri)) = .{ .o = o, .p = &p, .d = d, .input = d.out.ptr, .m = 1, .last = true, .score = true };

            try cuda.graph.beginCaptureToGraph(st, while_res.body, .thread_local);
            const cap_err: ?anyerror = blk: {
                one.run() catch |err| break :blk err;
                var sa: cuda.Args = .{};
                sa.add(d.best.ptr);
                sa.add(vocab_ids_ptr);
                sa.add(s.ids);
                sa.add(s.pos);
                sa.add(h.while_ctl_dev.ptr);
                sa.add(h.while_out_dev.ptr);
                sa.add(handle);
                cuda.launch.launch(h.chain_step_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 }, .pdl = false }, st, &sa) catch |err| break :blk err;
                break :blk null;
            };
            if (cap_err) |err| {
                cuda.graph.endCaptureToGraph(st) catch {};
                return err;
            }
            try cuda.graph.endCaptureToGraph(st);

            var exec = try root_g.instantiate();
            errdefer exec.deinit();
            try exec.upload(st);
            return exec;
        }

        fn draftWhile(o: Self, d: *Draft, max_steps: usize, confidence: f64, out: *Chain) !bool {
            const wg = @import("cuda_window_graph.zig");
            const e = o.e;
            if (e.graphs.head_while.off) return false;
            const h = o.h;
            const st = o.stream();
            const s = try e.reserve(plan.reserveRows(1));
            const piece = @max(@as(usize, 1), @min(piece_rows, s.cap / 2));
            const score_rows = try e.reserveScores(d.cap, piece);
            const vocab_ids_ptr = h.w2.vocab_ids.?.ptr;
            const key = wg.HeadWhileKey.init(d, s, score_rows, vocab_ids_ptr, h.while_ctl_dev.ptr, h.while_out_dev.ptr);
            const exec_ptr = e.graphs.head_while.find(key) orelse blk: {
                const exec = o.buildWhileGraph(d, s, score_rows, vocab_ids_ptr) catch |err| {
                    std.log.warn("head_while graphs off: capture failed ({s}), falling back to CPU loop", .{@errorName(err)});
                    e.graphs.head_while.off = true;
                    return false;
                };
                break :blk e.graphs.head_while.put(key, exec);
            };

            const stop_cfg = mtp.Stop.init(confidence);
            const ctl: MtpChainCtl = .{
                .c = stop_cfg.c,
                .ext_enter = stop_cfg.ext_enter,
                .ext_stay = stop_cfg.ext_stay,
                .ext_tok = stop_cfg.ext_tok,
                .p_cliff = stop_cfg.p_cliff,
                .start_pos = @intCast(d.len),
                .max_steps = @intCast(max_steps),
                .vocab_n = @intCast(h.ids.len),
                .pad = 0,
            };
            @memcpy(h.while_ctl_host.bytes[0..@sizeOf(MtpChainCtl)], std.mem.asBytes(&ctl));
            {
                errdefer st.synchronize() catch {};
                try st.d.check(st.d.api.cuMemcpyHtoDAsync_v2(h.while_ctl_dev.ptr, h.while_ctl_host.bytes.ptr, @sizeOf(MtpChainCtl), st.handle), "cuMemcpyHtoDAsync");
                try exec_ptr.launchOn(st);
                try st.d.check(st.d.api.cuMemcpyDtoHAsync_v2(h.while_out_host.bytes.ptr, h.while_out_dev.ptr, @sizeOf(MtpChainOut), st.handle), "cuMemcpyDtoHAsync");
                try st.synchronize();
            }

            const out_host = std.mem.bytesToValue(MtpChainOut, h.while_out_host.bytes[0..@sizeOf(MtpChainOut)]);
            if (out_host.drafted < 0 or out_host.drafted > max_chain) return error.ProposalNonfinite;
            const drafted: usize = @intCast(out_host.drafted);
            d.len += drafted;
            d.drafted += drafted;
            if (out_host.err != 0 or out_host.n < 0 or out_host.n > max_chain) return error.ProposalNonfinite;
            var n: usize = @intCast(out_host.n);
            const ext_min: usize = if (std.c.getenv("TF_FLASHNEXT_EXT_MIN")) |mv| (std.fmt.parseInt(usize, std.mem.span(mv), 10) catch 0) else 0;
            if (ext_min > 15 and n > 15 and n < ext_min) n = 15;
            out.n = n;
            for (0..n) |i| {
                out.tokens[i] = out_host.tokens[i];
                out.probs[i] = out_host.probs[i];
            }
            for (0..@min(drafted, n)) |i| {
                o.e.ngram.warmToken(&d.draft_hist, out.tokens[i]);
            }
            return true;
        }

        pub fn draft(o: Self, d: *Draft, next: []const u32, rows: u64, position: u64, count: usize, sm: ?Sampling, confidence: f64, out: *Chain) !void {
            out.n = 0;
            try o.absorb(d, next, rows, count > 0);
            if (next.len > 0) o.e.ngram.warmToken(&d.draft_hist, next[next.len - 1]);
            const max_steps = @min(count, max_chain);
            const wg = @import("cuda_window_graph.zig");
            if (whileEnabled() and !mtp.isSampled(sm) and o.h.w2.ready() and o.main == null and d.media == null and o.e.collective == null and max_steps > 1 and wg.enabled() and d.len + max_steps - 1 <= d.cap and std.meta.eql(wg.Geometry.at(d.len + 1, d.cap), wg.Geometry.at(d.len + max_steps - 1, d.cap))) {
                if (try o.draftWhile(d, max_steps, confidence, out)) return;
            }
            var stop: mtp.Stop = mtp.Stop.init(confidence);
            var j: usize = 0;
            while (j < max_steps) : (j += 1) {
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

# 5f. cuda_mtp.zig + cuda_mtp_draw.zig + cuda_draw_batch.zig + cuda_batch_hc.zig + cuda_mtp_pieces.zig + cuda_draft_draw.zig + cuda_lanes_load.zig:
#     keep max_chain=15 (.first_copy_rows=16 inside BM=16), zero-expf fn_row_argmax, hc.fuses at n>=48, hook fused 2.25-bit W2 directly to d.best
patch_exact(
    root, "cuda_batch_hc.zig",
    "    return n >= 64 and hc_w.down.native != null",
    "    return n >= 8 and hc_w.down.native != null",
)
patch_exact(
    root, "cuda_batch_hc.zig",
    """    const act = s.act + r0 * low * 2;
    const xs_act = s.xs_act + r0 * (low / 32) * 4;
    const down = hc_w.down.native orelse return error.UnexpectedTensor;
    const up = hc_w.up.native orelse return error.UnexpectedTensor;
    if (n * hc_w.n_down * 4 > s.hc_part_bytes) return error.UnexpectedTensor; // one slice within hc_part (hcPartRows)
    try pass.k.nativeMatmul(pass.stream, norm, wide, n, down, s.hc_part, s.native_part, .f32);
    const inj_ptr = if (hc_w.has_inj == 1) inj else act;
    try pass.tri.hcReduceAct(s.hc_part, act, xs_act, inj_ptr, 1, n, streams, low, hc_w.n_down, hc_w.has_inj);
    try pass.k.nativeMatmul(pass.stream, act, low, n, up, s.hc_up, s.native_part, .bf16);
    try pass.tri.hcDenseMix(s.hc_up, norm, s.mixed + r0 * dims * 2, s.xsm + r0 * (dims / 32) * 4, n, dims, streams);""",
    """    const act = s.act + r0 * low * 2;
    const down = hc_w.down.native orelse return error.UnexpectedTensor;
    const up = hc_w.up.native orelse return error.UnexpectedTensor;
    if (n * hc_w.n_down * 4 > s.hc_part_bytes) return error.UnexpectedTensor; // one slice within hc_part (hcPartRows)
    const hc_sk: usize = if (down.scale == 1.0 or pass.tri.own != null)
        try pass.k.native_lane.matmulRawPart(pass.stream, norm, wide, n, down, s.hc_part, .f32, s.native_part, true)
    else blk: {
        try pass.k.nativeMatmul(pass.stream, norm, wide, n, down, s.hc_part, s.native_part, .f32);
        break :blk 1;
    };
    const part_src = if (hc_sk > 1) s.native_part else s.hc_part;
    const inj_ptr = if (hc_w.has_inj == 1) inj else act;
    if (down.scale != 1.0 and pass.tri.own != null and hc_sk > 1) {
        try pass.tri.own.?.hcReduceActScaled(pass.stream, part_src, act, 0, inj_ptr, hc_sk, n, streams, low, hc_w.n_down, hc_w.has_inj, down.scale);
    } else {
        try pass.tri.hcReduceAct(part_src, act, 0, inj_ptr, hc_sk, n, streams, low, hc_w.n_down, hc_w.has_inj);
    }
    if (hc_w.has_inj == 1 and up.mode == .bf12 and up.n == streams * dims and up.k == low and dims == 2560 and streams == 4 and low == 320 and n <= 32) {
        try pass.k.native_lane.hcUpmixBf12(pass.stream, act, up, norm, s.mixed + r0 * dims * 2, n, dims);
    } else {
        try pass.k.nativeMatmul(pass.stream, act, low, n, up, s.hc_up, s.native_part, .bf16);
        const xsm_ptr: u64 = if (hc_w.has_inj == 1) 0 else s.xsm + r0 * (dims / 32) * 4;
        try pass.tri.hcDenseMix(s.hc_up, norm, s.mixed + r0 * dims * 2, xsm_ptr, n, dims, streams);
    }""",
)
patch_exact(
    root, "cuda_glue.zig",
    "    shared_swiglu: cuda.Function,\n    shared_scatter: cuda.Function,\n    down: cuda.Function,\n    reduce_act: cuda.Function,",
    "    shared_swiglu: cuda.Function,\n    shared_reduce_swiglu: cuda.Function,\n    shared_swiglu_cat: cuda.Function,\n    shared_scatter: cuda.Function,\n    down: cuda.Function,\n    reduce_act: cuda.Function,\n    reduce_act_s1: cuda.Function,\n    reduce_act_s1_sc: cuda.Function,",
)
patch_exact(
    root, "cuda_glue.zig",
    "            .shared_swiglu = try g.hc.function(\"fn_shared_swiglu\"),\n            .shared_scatter = try g.hc.function(\"fn_shared_scatter\"),\n            .down = try g.hc.function(\"fn_hc_down\"),\n            .reduce_act = try g.hc.function(\"fn_hc_reduce_act\"),",
    "            .shared_swiglu = try g.hc.function(\"fn_shared_swiglu\"),\n            .shared_reduce_swiglu = try g.hc.function(\"fn_shared_reduce_swiglu\"),\n            .shared_swiglu_cat = try g.hc.function(\"fn_shared_swiglu_cat\"),\n            .shared_scatter = try g.hc.function(\"fn_shared_scatter\"),\n            .down = try g.hc.function(\"fn_hc_down\"),\n            .reduce_act = try g.hc.function(\"fn_hc_reduce_act\"),\n            .reduce_act_s1 = try g.hc.function(\"fn_hc_reduce_act_s1\"),\n            .reduce_act_s1_sc = try g.hc.function(\"fn_hc_reduce_act_s1_sc\"),",
)
patch_exact(
    root, "cuda_glue.zig",
    """    pub fn sharedSwiGLU(g: *const Glue, s: cuda.Stream, gate: u64, up: u64, act: u64, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        ptrs(&a, &.{ gate, up, act });
        ints(&a, &.{width});
        try go(g.f.shared_swiglu, s, .{ rows, cdiv(width, 256), 1 }, 256, 0, &a);
    }""",
    """    pub fn sharedSwiGLU(g: *const Glue, s: cuda.Stream, gate: u64, up: u64, act: u64, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        ptrs(&a, &.{ gate, up, act });
        ints(&a, &.{width});
        try go(g.f.shared_swiglu, s, .{ rows, cdiv(width, 256), 1 }, 256, 0, &a);
    }

    pub fn sharedReduceSwiGLU(g: *const Glue, s: cuda.Stream, part: u64, act: u64, rows: usize, width: usize, sk: usize) !void {
        var a: cuda.Args = .{};
        ptrs(&a, &.{ part, act });
        ints(&a, &.{ rows, width, sk });
        try go(g.f.shared_reduce_swiglu, s, .{ rows, cdiv(width, 256), 1 }, 256, 0, &a);
    }

    pub fn sharedSwiGLUCat(g: *const Glue, s: cuda.Stream, gu: u64, act: u64, rows: usize, width: usize) !void {
        var a: cuda.Args = .{};
        ptrs(&a, &.{ gu, act });
        ints(&a, &.{width});
        try go(g.f.shared_swiglu_cat, s, .{ rows, cdiv(width, 256), 1 }, 256, 0, &a);
    }""",
)
patch_exact(
    root, "cuda_glue.zig",
    "        try go(g.f.reduce_act, s, .{ rows, 1, 1 }, @intCast(cdiv(@max(cols, low / 32), 32) * 32), 0, &a);\n    }",
    """        try go(if (sk <= 16 and xs == 0 and cols <= 384) g.f.reduce_act_s1 else g.f.reduce_act, s, .{ rows, 1, 1 }, @intCast(cdiv(@max(cols, low / 32), 32) * 32), 0, &a);
    }

    pub fn hcReduceActScaled(g: *const Glue, s: cuda.Stream, part: u64, act: u64, xs: u64, inj: u64, sk: usize, rows: usize, streams: usize, low: usize, cols: usize, has_inj: usize, gscale: f32) !void {
        var a: cuda.Args = .{};
        ptrs(&a, &.{ part, act, xs, inj });
        ints(&a, &.{ sk, rows, streams, low, has_inj, cols });
        a.add(gscale);
        try go(g.f.reduce_act_s1_sc, s, .{ rows, 1, 1 }, @intCast(cdiv(@max(cols, low / 32), 32) * 32), 0, &a);
    }""",
)
patch_exact(
    root, "cuda_glue.zig",
    "                try go(g.f.wbn0, s, grid, 256, 0, &a);",
    "                try go(g.f.wbn0, s, grid, 320, 0, &a);",
)
patch_exact(
    root, "cuda_glue.zig",
    "                try go(g.f.wbn_moe, s, grid, 256, 0, &a);",
    "                try go(g.f.wbn_moe, s, grid, 320, 0, &a);",
)
patch_exact(
    root, "cuda_glue.zig",
    "                try go(g.f.wbn_slices, s, grid, 256, 0, &a);",
    "                try go(g.f.wbn_slices, s, grid, 320, 0, &a);",
)
# Issue #3681: keep clusters for sk<=4 only; run sk=8 (out_proj/o_proj/hc_down) as full-GPU plain split-K + PDL reduce
patch_exact(
    cdir, "qmmf.zig",
    "    const cluster = !fused and sk > 1 and sk <= 8 and clusters;",
    "    const cluster = !fused and sk > 1 and sk <= 4 and clusters;",
)
patch_exact(
    root, "cuda_prompt.zig",
    ".lane_reduce = !e.batch_k.native_lane.clusters };",
    ".lane_reduce = true };",
)
patch_exact(
    root, "cuda_mtp.zig",
    """pub const Stop = struct {
    c: f64,
    product: f64 = 1.0,

    /// Whether draft j (0-based) with probability `p` is proposed, and whether the chain may go on.
    pub fn take(s: *Stop, j: usize, p: f64) struct { propose: bool, more: bool } {
        if (s.c > 0) {
            const low = p < s.c;
            if (low and j > 0) return .{ .propose = false, .more = false };
            return .{ .propose = true, .more = !low };
        }
        if (s.c < 0) {
            s.product *= p;
            // below the line it ends the chain: no later product can reach it again
            if (s.product < -s.c) return .{ .propose = j == 0, .more = false };
        }
        return .{ .propose = true, .more = true };
    }
};""",
    """fn envF64(name: [*:0]const u8, default: f64) f64 {
    const raw = std.c.getenv(name) orelse return default;
    return std.fmt.parseFloat(f64, std.mem.span(raw)) catch default;
}

pub const Stop = struct {
    c: f64,
    product: f64 = 1.0,
    ext_enter: f64 = 0.0,
    ext_stay: f64 = 0.0,
    ext_tok: f64 = 0.0,
    p_cliff: f64 = 0.0,

    pub fn init(c: f64) Stop {
        return .{
            .c = c,
            .ext_enter = envF64("TF_FLASHNEXT_EXT_ENTER", 0.0),
            .ext_stay = envF64("TF_FLASHNEXT_EXT_STAY", 0.0),
            .ext_tok = envF64("TF_FLASHNEXT_EXT_TOK", 0.0),
            .p_cliff = envF64("TF_FLASHNEXT_P_CLIFF", 0.0),
        };
    }

    /// Whether draft j (0-based) with probability `p` is proposed, and whether the chain may go on.
    pub fn take(s: *Stop, j: usize, p: f64) struct { propose: bool, more: bool } {
        if (s.c > 0) {
            const low = p < s.c;
            if (low and j > 0) return .{ .propose = false, .more = false };
            return .{ .propose = true, .more = !low };
        }
        if (s.c < 0) {
            s.product *= p;
            if (j >= 15 and (s.ext_stay > 0 or s.ext_tok > 0)) {
                if (s.product < s.ext_stay or p < s.ext_tok) return .{ .propose = false, .more = false };
                return .{ .propose = true, .more = true };
            }
            if (j >= 1 and s.p_cliff > 0 and p < s.p_cliff) return .{ .propose = false, .more = false };
            // below the line it ends the chain: no later product can reach it again
            if (s.product < -s.c) return .{ .propose = j == 0, .more = false };
            if (j >= 1 and s.p_cliff < 0 and p < -s.p_cliff) return .{ .propose = true, .more = false };
            if (j == 14 and s.ext_enter > 0 and (s.product < s.ext_enter or p < s.ext_tok)) return .{ .propose = true, .more = false };
        }
        return .{ .propose = true, .more = true };
    }
};""",
)
patch_exact(
    root, "cuda_mtp.zig",
    "    topk_fn: cuda.Function, // fn_row_topk: a row's top_k candidates as drawRow ranks them (cuda_mtp_draw.zig)",
    "    topk_fn: cuda.Function,\n    argmax_fn: cuda.Function,\n    chain_init_fn: cuda.Function,\n    chain_step_fn: cuda.Function,\n    while_ctl_dev: cuda.DeviceBuffer,\n    while_out_dev: cuda.DeviceBuffer,\n    while_ctl_host: cuda.HostBuffer,\n    while_out_host: cuda.HostBuffer,\n    w2: @import(\"cuda_mtp_w2.zig\").W2 = .{},",
)
patch_exact(
    root, "cuda_mtp.zig",
    "        h.best_fn = try h.module.function(\"fn_row_best\");\n        h.topk_fn = try h.module.function(\"fn_row_topk\");",
    "        h.best_fn = try h.module.function(\"fn_row_best\");\n        h.topk_fn = try h.module.function(\"fn_row_topk\");\n        h.argmax_fn = try h.module.function(\"fn_row_argmax\");\n        h.chain_init_fn = try h.module.function(\"fn_mtp_chain_init\");\n        h.chain_step_fn = try h.module.function(\"fn_mtp_chain_step\");",
)
patch_exact(
    root, "cuda_mtp.zig",
    """    /// fn_row_best over `rows` rows of `v` scores `ld` apart: [rows, 4] fp32 at `out` (best, log-sum-exp, id bits).
    pub fn best(h: *const Head, stream: cuda.Stream, scores: u64, v: usize, ld: usize, rows: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(scores);
        a.add(cint(v));
        a.add(@as(c_longlong, @intCast(ld)));
        a.add(out);
        try cuda.launch.launch(h.best_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &a);
    }""",
    """    /// fn_row_best over `rows` rows of `v` scores `ld` apart: [rows, 4] fp32 at `out` (best, log-sum-exp, id bits).
    pub fn best(h: *const Head, stream: cuda.Stream, scores: u64, v: usize, ld: usize, rows: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(scores);
        a.add(cint(v));
        a.add(@as(c_longlong, @intCast(ld)));
        a.add(out);
        try cuda.launch.launch(h.best_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &a);
    }

    /// fn_row_argmax over `rows` rows of `v` scores `ld` apart: greedy verify reduction without expf().
    pub fn argmax(h: *const Head, stream: cuda.Stream, scores: u64, v: usize, ld: usize, rows: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(scores);
        a.add(cint(v));
        a.add(@as(c_longlong, @intCast(ld)));
        a.add(out);
        try cuda.launch.launch(h.argmax_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &a);
    }""",
)
patch_exact(
    root, "cuda_mtp_draw.zig",
    "    if (masks.len == 0 and (!mtp.isSampled(sm) or !usable(sm.?))) try h.best(st, d.logits.ptr, vocab, vocab, n, d.best.ptr);",
    "    if (masks.len == 0 and !mtp.isSampled(sm)) try h.argmax(st, d.logits.ptr, vocab, vocab, n, d.best.ptr) else if (masks.len == 0 and !usable(sm.?)) try h.best(st, d.logits.ptr, vocab, vocab, n, d.best.ptr);",
)
patch_exact(
    root, "cuda_draw_batch.zig",
    "        if (greedy) try h.best(st, scores, vocab, vocab, rows, b.device.ptr);",
    "        if (greedy) try h.argmax(st, scores, vocab, vocab, rows, b.device.ptr);",
)
patch_exact(
    root, "cuda_mtp.zig",
    "        h.draw_batch = try @import(\"cuda_draw_batch.zig\").Batch.init(gpa, driver, batch_on, h.ids.len);\n        return h;",
    "        h.draw_batch = try @import(\"cuda_draw_batch.zig\").Batch.init(gpa, driver, batch_on, h.ids.len);\n        errdefer h.draw_batch.deinit(gpa);\n        h.w2 = try @import(\"cuda_mtp_w2.zig\").W2.init(gpa, driver, &store, &h.artifact, &h.module);\n        errdefer h.w2.deinit();\n        h.while_ctl_dev = try cuda.DeviceBuffer.alloc(driver, 128);\n        errdefer h.while_ctl_dev.free();\n        h.while_out_dev = try cuda.DeviceBuffer.alloc(driver, 512);\n        errdefer h.while_out_dev.free();\n        h.while_ctl_host = try cuda.HostBuffer.alloc(driver, 128);\n        errdefer h.while_ctl_host.free();\n        h.while_out_host = try cuda.HostBuffer.alloc(driver, 512);\n        return h;",
)
patch_exact(
    root, "cuda_mtp.zig",
    "    pub fn deinit(h: *Head) void {\n        h.draw_batch.deinit(h.gpa);",
    "    pub fn deinit(h: *Head) void {\n        h.while_out_host.free();\n        h.while_ctl_host.free();\n        h.while_out_dev.free();\n        h.while_ctl_dev.free();\n        h.w2.deinit();\n        h.draw_batch.deinit(h.gpa);",
)
patch_exact(
    root, "cuda_mtp_pieces.zig",
    "            try hh.project(pp, ps.mixed, ps.xsm, r.d.dlogits.ptr, 1);",
    "            if (hh.w2.ready()) try hh.w2.projectFused(pp.stream, ps.mixed, r.d.dlogits.ptr, r.d.best.ptr, hh.ids.len) else try hh.project(pp, ps.mixed, ps.xsm, r.d.dlogits.ptr, 1);",
)
patch_exact(
    root, "cuda_mtp_pieces.zig",
    """            try pp.readout(hh.layer.mlp_hc, 0, tot);
            try pp.moeBlock(&hh.layer);
            pp.pending = 0;
            try pp.writeback(0, tot);
            // each lane's last row: its d.out, and gathered to row i for the draft head (row i <= its last row)
            row0 = 0;
            for (r.group.ds, r.group.nexts, 0..) |d, nx, i| {
                const last = row0 + nx.len - 1;
                try pp.copy(d.out.ptr, ps.h + last * wide * 2, wide * 2);
                if (r.group.scores and last != i) {
                    try pp.copy(ps.h + i * wide * 2, ps.h + last * wide * 2, wide * 2);
                    try pp.copy(ps.pss + i * pss_n * 4, ps.pss + last * pss_n * 4, pss_n * 4);
                }
                row0 += nx.len;
            }
            if (!r.group.scores) return;
            const lanes_n = r.group.ds.len;""",
    """            const lanes_n = r.group.ds.len;
            if (tot > lanes_n) {
                row0 = 0;
                for (r.group.nexts, 0..) |nx, i| {
                    const last = row0 + nx.len - 1;
                    if (last != i) {
                        try pp.copy(ps.h + i * wide * 2, ps.h + last * wide * 2, wide * 2);
                        try pp.copy(ps.pss + i * pss_n * 4, ps.pss + last * pss_n * 4, pss_n * 4);
                    }
                    row0 += nx.len;
                }
                pp.rows = lanes_n;
            }
            try pp.readout(hh.layer.mlp_hc, 0, lanes_n);
            try pp.moeBlock(&hh.layer);
            pp.pending = 0;
            try pp.writeback(0, lanes_n);
            for (r.group.ds, 0..) |d, i| try pp.copy(d.out.ptr, ps.h + i * wide * 2, wide * 2);
            if (!r.group.scores) return;""",
)
patch_exact(
    root, "cuda_draft_draw.zig",
    "    try h.best(st, d.dlogits.ptr, n, n, 1, d.best.ptr);",
    "    if (!h.w2.ready()) try h.best(st, d.dlogits.ptr, n, n, 1, d.best.ptr);",
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
    "    select: cuda.Function,\n    select_lanes: cuda.Function,\n    head_topk_floor: cuda.Function,\n    head_rescore: cuda.Function,\n    head_rescore_best: cuda.Function,",
)
patch_exact(
    root, "cuda_batch_kernels.zig",
    "        k.late_roll = try k.batch.function(\"fn_late_roll\");\n        return k;",
    "        k.late_roll = try k.batch.function(\"fn_late_roll\");\n        k.head_topk_floor = try k.batch.function(\"fn_head_topk_floor\");\n        k.head_rescore = try k.batch.function(\"fn_head_rescore\");\n        k.head_rescore_best = try k.batch.function(\"fn_head_rescore_best\");\n        return k;",
)
patch_exact(
    root, "cuda_batch_head.zig",
    """    if (self.m.head.n == vocab) {
        _ = try self.lane(s.mixed, s.xsm, self.m.head, out, s.dummy, n, 1);
        return;
    }""",
    """    if (self.m.head.n == vocab) {
        _ = try self.lane(s.mixed, s.xsm, self.m.head, out, s.dummy, n, 1);
        if (self.m.head.b.ptr != 0) if (self.m.head.native) |nw| if (nw.mode == .fp4 or nw.mode == .w2 or nw.mode == .w1) {
            const w1_k: usize = if (std.c.getenv("TF_FLASHNEXT_HEAD_W1_KTOP")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 128) else 128;
            const k_top: usize = if (nw.mode == .w1) w1_k else 64;
            if (n * k_top * 4 > s.cap * (wt.experts_n + 1) * 4) return error.UnexpectedTensor;
            const direct = self.hooks.best_out != 0;
            var a1: cuda.Args = .{};
            a1.add(out);
            a1.add(@as(c_int, vocab));
            a1.add(@as(c_longlong, vocab));
            a1.add(@as(c_int, @intCast(k_top)));
            a1.add(s.router_l);
            a1.add(@as(c_int, if (direct) 1 else 0));
            try cuda.launch.launch(self.k.head_topk_floor, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 1024 }, .pdl = true }, self.stream, &a1);
            var a2: cuda.Args = .{};
            a2.add(s.mixed);
            a2.add(self.m.head.b.ptr);
            a2.add(s.router_l);
            if (direct) {
                a2.add(self.hooks.best_out);
                a2.add(@as(c_int, vocab));
                a2.add(@as(c_int, @intCast(k_top)));
                try cuda.launch.launch(self.k.head_rescore_best, .{ .grid = .{ .x = 1, .y = @intCast(n) }, .block = .{ .x = 1024 }, .pdl = true }, self.stream, &a2);
            } else {
                a2.add(out);
                a2.add(@as(c_int, vocab));
                a2.add(@as(c_int, @intCast(k_top)));
                try cuda.launch.launch(self.k.head_rescore, .{ .grid = .{ .x = @intCast((k_top + 7) / 8), .y = @intCast(n) }, .block = .{ .x = 256 }, .pdl = true }, self.stream, &a2);
            }
        };
        return;
    }""",
)

# 5i. cuda_window_graph.zig + cuda_lanes.zig: sound cross-request graph reuse (hash all real pointers/pg instead of seq.gen/d.gen),
#     512-slot 8-way LRU cache, immediate capture on single-stream Key/HeadKey/RollbackKey, and 1024-token pooled cap bucketing.
patch_exact(
    root, "cuda_window_graph.zig",
    "const max_graphs = 128;",
    "const max_graphs = 512;",
)
patch_exact(
    root, "cuda_window_graph.zig",
    "        h.values(&[_]u64{ seq.gen, seq.cap, pieces(seq), s.h, s.cap, scores, model, hooks.logits_out, hooks.streams });",
    "        h.values(&[_]u64{ seq.cap, pieces(seq), s.h, s.cap, scores, model, hooks.logits_out, hooks.best_out, hooks.streams });",
)
patch_exact(
    root, "cuda_window_graph.zig",
    "        h.values(&[_]u64{ d.gen, d.cap, d.kc.ptr, input, d.out.ptr, d.dlogits.ptr, s.h, s.cap, score_rows });",
    "        h.values(&[_]u64{ d.cap, d.kc.ptr, d.vc.ptr, d.ikc.ptr, d.pooled.ptr, d.pg.pt, @as(u64, @bitCast(d.pg.ps)), @as(u64, @as(u32, @bitCast(d.pg.ptok))), input, d.out.ptr, d.dlogits.ptr, d.best.ptr, s.h, s.cap, score_rows });",
)
patch_exact(
    root, "cuda_window_graph.zig",
    "            h.values(&[_]u64{ l.seq.gen, pieces(l.seq), l.seq.cap, l.streams, l.logits });",
    "            h.values(&[_]u64{ pieces(l.seq), l.seq.cap, l.streams, l.logits });",
)
patch_exact(
    root, "cuda_window_graph.zig",
    "            h.values(&[_]u64{ d.gen, d.kc.ptr, d.cap, input, d.out.ptr, d.dlogits.ptr });",
    "            h.values(&[_]u64{ d.kc.ptr, d.vc.ptr, d.ikc.ptr, d.pooled.ptr, d.pg.pt, @as(u64, @bitCast(d.pg.ps)), @as(u64, @as(u32, @bitCast(d.pg.ptok))), d.cap, input, d.out.ptr, d.dlogits.ptr, d.best.ptr });",
)
patch_exact(
    root, "cuda_window_graph.zig",
    "        h.values(&[_]u64{ seq.gen, d.gen, seq.tail.ptr, d.saved.ptr, s.h, s.cap });",
    "        h.values(&[_]u64{ seq.tail.ptr, d.saved.ptr, s.h, s.cap });",
)
patch_exact(
    root, "cuda_lanes.zig",
    """                // a window's rows past the reply's last token (its drafts) still need cache slots
                const cap = @min(ids.len + s.max_new + (if (drafting) mtp.max_rows else 0), prompt.max_tokens);
                const from = self.kv.choose();""",
    """                // a window's rows past the reply's last token (its drafts) still need cache slots
                const raw_cap = @min(ids.len + s.max_new + (if (drafting) mtp.max_rows else 0), prompt.max_tokens);
                const from = self.kv.choose();
                const cap = blk: {
                    if (from == .pool) {
                        if (self.kv.pool) |pool| {
                            const lim = @min(self.context + (if (drafting) mtp.max_rows else 0), prompt.max_tokens);
                            const aligned = @min(((raw_cap + 1023) / 1024) * 1024, lim);
                            if (aligned >= raw_cap and pool.fits(aligned)) break :blk aligned;
                        }
                    }
                    break :blk raw_cap;
                };""",
)

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
        const ways = 8;
        const skip_seen = (K == Key or K == HeadKey or K == RollbackKey);
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
            if (!skip_seen and !c.seen.contains(slot)) {
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

# 5j. Trial 29: shared-expert skip in plan_kernel, paced SM121 TMA L2 prefetch on main BF12 out_proj/o_proj only, 1-warp route_topk, and PDL launches
patch_exact(
    root, "cuda_moe_native.zig",
    "    // the rank's experts and one skip id past them (the shared slot at world 1; every other rank's pick at --tp)\n    const e = owned.count + 1;",
    "    // World-1 NVFP4 skips the shared expert slot (== owned.count) directly inside plan_kernel.\n    const e = if (native.world == 1 and owned.format == .nvfp4) owned.count else owned.count + 1;",
)
# 5l. Trial 37: Fused non-clustered shared_expert gate_up + zero-write greedy lm_head direct-argmax
patch_exact(
    root, "cuda_batch_kernels.zig",
    "    head_topk_floor: cuda.Function,\n    head_rescore: cuda.Function,\n    head_rescore_best: cuda.Function,",
    "    head_topk_floor: cuda.Function,\n    head_rescore: cuda.Function,\n    head_rescore_best: cuda.Function,\n    l2_prefetch2_paced: cuda.Function,\n    l2pf_mb: usize = 10,\n    l2pf_pace_gbps: usize = 165,",
)
patch_exact(
    root, "cuda_batch_kernels.zig",
    "        k.head_topk_floor = try k.batch.function(\"fn_head_topk_floor\");\n        k.head_rescore = try k.batch.function(\"fn_head_rescore\");\n        k.head_rescore_best = try k.batch.function(\"fn_head_rescore_best\");\n        return k;",
    """        k.head_topk_floor = try k.batch.function("fn_head_topk_floor");
        k.head_rescore = try k.batch.function("fn_head_rescore");
        k.head_rescore_best = try k.batch.function("fn_head_rescore_best");
        k.l2_prefetch2_paced = try k.batch.function("fn_l2_prefetch2_paced");
        k.l2pf_mb = if (std.c.getenv("TF_FLASHNEXT_L2PF_MB")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 10) else 10;
        k.l2pf_pace_gbps = if (std.c.getenv("TF_FLASHNEXT_L2PF_PACE_GBPS")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 165) else 165;
        return k;""",
)
patch_exact(
    root, "cuda_batch.zig",
    """        pub fn gdnBlock(self: *Self, lw: *const Layer, layer: usize, li: usize, keep: LayerKeep) !void {
            const s = self.s;
            const n = self.rows;
            const h = lw.heads;
            const g = try self.k.gdnSet(h); // before any launch: the full kernels, or the _h8 twins for a rank's half
            const pw = h.projWidth();
            const cd = h.convDim();
            _ = try self.lane(s.mixed, s.xsm, lw.proj, s.proj, s.dummy, n, 1);""",
    """        fn l2PrefetchOut(self: *Self, lw: *const Layer) !bool {
            if (self.hooks.prompt or self.k.l2pf_mb == 0) return false;
            const w = lw.out.native orelse return false;
            if (w.mode != .bf12) return false; // main target BF12 out_proj/o_proj only (skip short MTP attn)
            const x = self.s.side orelse return false;
            const budget: u64 = @as(u64, self.k.l2pf_mb) << 20;
            const sb: u64 = @min(@as(u64, lw.out.s.len), budget);
            const cb: u64 = @min(@as(u64, lw.out.w.len), budget -| sb);
            if (sb == 0 and cb == 0) return false;
            const ns_per: u64 = if (self.k.l2pf_pace_gbps > 0) @max(@as(u64, 1), (32768 + self.k.l2pf_pace_gbps / 2) / self.k.l2pf_pace_gbps) else 0;
            try x.fork.record(self.stream);
            try x.stream.wait(x.fork);
            var a: cuda.Args = .{};
            a.add(lw.out.s.ptr);
            a.add(sb);
            a.add(lw.out.w.ptr);
            a.add(cb);
            a.add(ns_per);
            try cuda.launch.launch(self.k.l2_prefetch2_paced, .{ .grid = .{ .x = 2 }, .block = .{ .x = 32 }, .pdl = false }, x.stream, &a);
            try x.join.record(x.stream);
            return true;
        }

        pub fn gdnBlock(self: *Self, lw: *const Layer, layer: usize, li: usize, keep: LayerKeep) !void {
            const s = self.s;
            const n = self.rows;
            const h = lw.heads;
            const g = try self.k.gdnSet(h); // before any launch: the full kernels, or the _h8 twins for a rank's half
            const pw = h.projWidth();
            const cd = h.convDim();
            _ = try self.lane(s.mixed, s.xsm, lw.proj, s.proj, s.dummy, n, 1);
            const pf = try self.l2PrefetchOut(lw);""",
)
patch_exact(
    root, "cuda_batch.zig",
    "        out_slices: usize = out_sk,",
    "        out_slices: usize = out_sk,\n        fuse_out_native: bool = false,\n        out_from_native: bool = false,",
)
patch_exact(
    root, "cuda_batch.zig",
    """            if (!fused) try gdn_fused.norm(g.norm, self.stream, s.ys, s.proj, lw.norm.ptr, s.gout, s.gxs, n, h);
            if (self.lanes.len == 0) try self.shiftWindow(self.seq.conv[li].ptr, s.proj, pw, n, gdn.taps - 1, cd);
            self.out_slices = try split_lane.out(self, s.gout, s.gxs, lw);
        }""",
    """            if (!fused) try gdn_fused.norm(g.norm, self.stream, s.ys, s.proj, lw.norm.ptr, s.gout, if (lw.out.native != null) @as(u64, 0) else s.gxs, n, h);
            if (self.lanes.len == 0) try self.shiftWindow(self.seq.conv[li].ptr, s.proj, pw, n, gdn.taps - 1, cd);
            if (self.fuse_out_native) {
                self.out_slices = try self.k.native_lane.matmulRawPart(self.stream, s.gout, lw.out.k, n, lw.out.native.?, s.part, .f32, s.native_part, false);
                self.out_from_native = self.out_slices > 1;
            } else {
                self.out_slices = try split_lane.out(self, s.gout, s.gxs, lw);
                self.out_from_native = false;
            }
            if (pf) try self.stream.wait(self.s.side.?.join);
        }""",
)
patch_exact(
    root, "cuda_batch.zig",
    """            const pn = h.projN();
            _ = try self.lane(s.mixed, s.xsm, lw.proj, pa, s.dummy, n, 1);""",
    """            const pn = h.projN();
            _ = try self.lane(s.mixed, s.xsm, lw.proj, pa, s.dummy, n, 1);
            const pf = try self.l2PrefetchOut(lw);""",
)
patch_exact(
    root, "cuda_batch.zig",
    """            try self.tri.attnGate(s.ao, pa, gated, s.gxs_attn, n, pn, h.q, head_dim);
            self.out_slices = try split_lane.out(self, gated, s.gxs_attn, lw);
        }""",
    """            try self.tri.attnGate(s.ao, pa, gated, if (lw.out.native != null) @as(u64, 0) else s.gxs_attn, n, pn, h.q, head_dim);
            if (self.fuse_out_native) {
                self.out_slices = try self.k.native_lane.matmulRawPart(self.stream, gated, lw.out.k, n, lw.out.native.?, s.part, .f32, s.native_part, false);
                self.out_from_native = self.out_slices > 1;
            } else {
                self.out_slices = try split_lane.out(self, gated, s.gxs_attn, lw);
                self.out_from_native = false;
            }
            if (pf) try self.stream.wait(self.s.side.?.join);
        }""",
)
patch_exact(
    root, "cuda_batch.zig",
    "                self.zero_slices = zs.usable(self.s.part_zero, keep.mix_branch, self.m.collective != null, lw.out.native != null);",
    "                self.zero_slices = zs.usable(self.s.part_zero, keep.mix_branch, self.m.collective != null, lw.out.native != null);\n                self.fuse_out_native = n <= 64 and lw.dense_cut == null and self.m.collective == null and keep.mix_branch == 0 and lw.out.native != null and lw.out.native.?.scale == 1.0 and hc.fuses(self, lw.mlp_hc, n, keep.mlp_hc);",
)
patch_exact(
    root, "cuda_batch.zig",
    "                    try self.tri.hcWritebackNorm(s.h, s.h, s.pss, s.part, s.inj, s.dummy, s.dummy, lw.mlp_hc.scale.ptr, s.norm, eps, sl.stride, n, dims, streams, 4, 1, 1, sl.count);",
    "                    const out_br = if (self.out_from_native) s.native_part else s.part;\n                    const out_st = if (self.out_from_native) n * dims else sl.stride;\n                    const out_cnt = if (self.out_from_native) self.out_slices else sl.count;\n                    self.fuse_out_native = false;\n                    self.out_from_native = false;\n                    try self.tri.hcWritebackNorm(s.h, s.h, s.pss, out_br, s.inj, s.dummy, s.dummy, lw.mlp_hc.scale.ptr, s.norm, eps, out_st, n, dims, streams, 4, 1, 1, out_cnt);",
)
patch_exact(
    root, "cuda_gdn_fused.zig",
    "    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = heads.nv }, .block = .{ .x = gdn.dv } }, stream, &b);",
    "    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = heads.nv }, .block = .{ .x = gdn.dv }, .pdl = true }, stream, &b);",
)
patch_exact(
    cdir, "route_rows.zig",
    "        try launch_.launch(g.topk_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);",
    "        try launch_.launch(g.topk_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = @as(u32, if (experts <= 512 and top_k <= 32) 32 else 128) } }, s, &a);",
)
patch_exact(
    cdir, "experts.zig",
    "        try launch_.launch(e.fns[ei], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = @intCast(e.warps * 32), .y = 1, .z = 1 } }, s, &a);",
    "        try launch_.launch(e.fns[ei], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = @intCast(e.warps * 32), .y = 1, .z = 1 }, .pdl = true }, s, &a);",
)
patch_exact(
    cdir, "experts_waves.zig",
    "    try launch_.launch(e.fns[fi], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = @intCast(e.warps * 32), .y = 1, .z = 1 } }, s, &a);",
    "    try launch_.launch(e.fns[fi], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = @intCast(e.warps * 32), .y = 1, .z = 1 }, .pdl = true }, s, &a);",
)
patch_exact(
    cdir, "grouped.zig",
    "        try launch_.launch(f, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = block } }, s, a);",
    "        try launch_.launch(f, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = block }, .pdl = true }, s, a);",
)
patch_exact(
    root, "cuda_glue.zig",
    "        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = block }, .shared = shared };",
    "        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = block }, .shared = shared, .pdl = true };",
)

# 5k. Trial 36: CUDA 13 conditional WHILE graph ABI + HeadWhileKey/WhileCache for zero-sync GPU MTP drafting
patch_exact(
    cdir, "abi.zig",
    """pub const ExecUpdateResultInfo = extern struct {
    result: ExecUpdateResult,
    error_node: GraphNode,
    error_from_node: GraphNode,
};""",
    """pub const ExecUpdateResultInfo = extern struct {
    result: ExecUpdateResult,
    error_node: GraphNode,
    error_from_node: GraphNode,
};

pub const GraphConditionalHandle = u64;
pub const graph_node_type_conditional: c_uint = 13;
pub const graph_cond_type_while: c_uint = 1;
pub const graph_cond_assign_default: c_uint = 1;

pub const ConditionalNodeParams = extern struct {
    handle: GraphConditionalHandle = 0,
    @"type": c_uint = graph_cond_type_while,
    size: c_uint = 1,
    ph_graph_out: ?[*]Graph = null,
    ctx: Context = null,
};

pub const GraphNodeParamsCond = extern struct {
    @"type": c_uint = graph_node_type_conditional,
    reserved0: [3]c_int = .{ 0, 0, 0 },
    conditional: ConditionalNodeParams = .{},
    pad: [25]c_longlong = @splat(0),
    reserved2: c_longlong = 0,
};""",
)
patch_exact(
    cdir, "abi.zig",
    "    cuStreamBeginCapture_v2: *const fn (Stream, CaptureMode) callconv(.c) R,\n    cuStreamEndCapture: *const fn (Stream, *Graph) callconv(.c) R,",
    "    cuStreamBeginCapture_v2: *const fn (Stream, CaptureMode) callconv(.c) R,\n    cuStreamBeginCaptureToGraph: *const fn (Stream, Graph, ?[*]const GraphNode, ?*const anyopaque, usize, CaptureMode) callconv(.c) R,\n    cuStreamEndCapture: *const fn (Stream, *Graph) callconv(.c) R,",
)
patch_exact(
    cdir, "abi.zig",
    "    cuGraphCreate: *const fn (*Graph, c_uint) callconv(.c) R,\n    cuGraphDestroy: *const fn (Graph) callconv(.c) R,",
    "    cuGraphCreate: *const fn (*Graph, c_uint) callconv(.c) R,\n    cuGraphDestroy: *const fn (Graph) callconv(.c) R,\n    cuGraphConditionalHandleCreate: *const fn (*GraphConditionalHandle, Graph, Context, c_uint, c_uint) callconv(.c) R,\n    cuGraphAddNode_v2: *const fn (*GraphNode, Graph, ?[*]const GraphNode, ?*const anyopaque, usize, *GraphNodeParamsCond) callconv(.c) R,",
)
patch_exact(
    cdir, "abi.zig",
    "    std.debug.assert(@sizeOf(ExecUpdateResultInfo) == 24);",
    "    std.debug.assert(@sizeOf(ExecUpdateResultInfo) == 24);\n    std.debug.assert(@sizeOf(ConditionalNodeParams) == 32);\n    std.debug.assert(@sizeOf(GraphNodeParamsCond) == 256 and @offsetOf(GraphNodeParamsCond, \"conditional\") == 16);",
)
patch_exact(
    cdir, "graph.zig",
    """pub fn captureStatus(stream: Stream) Error!abi.CaptureStatus {
    var s: abi.CaptureStatus = .none;
    try stream.d.check(stream.d.api.cuStreamIsCapturing(stream.handle, &s), "cuStreamIsCapturing");
    return s;
}""",
    """pub fn beginCaptureToGraph(stream: Stream, g: Graph, mode: abi.CaptureMode) Error!void {
    const source = @import("copy_source.zig");
    try source.captureBegin(stream, mode);
    errdefer source.captureEnd(stream);
    try stream.d.check(stream.d.api.cuStreamBeginCaptureToGraph(stream.handle, g.handle, null, null, 0, mode), "cuStreamBeginCaptureToGraph");
}

pub fn endCaptureToGraph(stream: Stream) Error!void {
    const source = @import("copy_source.zig");
    var g: abi.Graph = null;
    errdefer if ((captureStatus(stream) catch .active) == .none) source.captureEnd(stream);
    try stream.d.check(stream.d.api.cuStreamEndCapture(stream.handle, &g), "cuStreamEndCapture");
    source.captureEnd(stream);
}

pub fn captureStatus(stream: Stream) Error!abi.CaptureStatus {
    var s: abi.CaptureStatus = .none;
    try stream.d.check(stream.d.api.cuStreamIsCapturing(stream.handle, &s), "cuStreamIsCapturing");
    return s;
}""",
)
patch_exact(
    cdir, "graph.zig",
    """    pub fn instantiate(self: Graph) Error!Exec {""",
    """    pub const WhileNode = struct { node: Node, body: Graph };

    pub fn createConditionalHandle(self: Graph, ctx: abi.Context, default_val: c_uint, flags: c_uint) Error!abi.GraphConditionalHandle {
        var h: abi.GraphConditionalHandle = 0;
        try self.d.check(self.d.api.cuGraphConditionalHandleCreate(&h, self.handle, ctx, default_val, flags), "cuGraphConditionalHandleCreate");
        return h;
    }

    pub fn addWhileNode(self: Graph, deps: []const Node, handle: abi.GraphConditionalHandle, ctx: abi.Context) Error!WhileNode {
        var np: abi.GraphNodeParamsCond = .{
            .conditional = .{
                .handle = handle,
                .@"type" = abi.graph_cond_type_while,
                .size = 1,
                .ph_graph_out = null,
                .ctx = ctx,
            },
        };
        var n: Node = null;
        try self.d.check(self.d.api.cuGraphAddNode_v2(&n, self.handle, if (deps.len > 0) deps.ptr else null, null, deps.len, &np), "cuGraphAddNode_v2");
        const arr = np.conditional.ph_graph_out orelse return error.CudaFailed;
        return .{ .node = n, .body = .{ .d = self.d, .handle = arr[0] } };
    }

    pub fn instantiate(self: Graph) Error!Exec {""",
)
patch_exact(
    root, "cuda_window_graph.zig",
    """/// The engine's graph caches, one per kind of pass.
pub const Graphs = struct {
    window: Cache(Key) = .{},
    head: Cache(HeadKey) = .{},
    rollback: Cache(RollbackKey) = .{},
    shared: Cache(SharedKey) = .{},
    head_many: Cache(HeadManyKey) = .{},

    pub fn clear(g: *Graphs, gpa: std.mem.Allocator) void {
        g.window.clear(gpa);
        g.head.clear(gpa);
        g.rollback.clear(gpa);
        g.shared.clear(gpa);
        g.head_many.clear(gpa);
    }
};""",
    """/// Key for a cached CUDA 13 CU_GRAPH_COND_TYPE_WHILE greedy MTP chain graph.
pub const HeadWhileKey = struct {
    shape: u64,
    moving: u64,

    pub fn init(d: anytype, s: *const parts.Scratch, score_rows: u64, vocab_ids_ptr: u64, ctl_ptr: u64, out_ptr: u64) HeadWhileKey {
        var h: Hash = .{};
        h.geometry(.at(d.len + 1, d.cap));
        h.values(&[_]u64{
            d.cap, d.kc.ptr, d.vc.ptr, d.ikc.ptr, d.pooled.ptr,
            d.pg.pt, @as(u64, @bitCast(d.pg.ps)), @as(u64, @as(u32, @bitCast(d.pg.ptok))),
            d.out.ptr, d.dlogits.ptr, d.best.ptr,
            s.h, s.cap, s.ids, s.pos, score_rows,
            vocab_ids_ptr, ctl_ptr, out_ptr,
        });
        return .{ .shape = h.shape.final(), .moving = h.moving.final() };
    }
};

pub const WhileCache = struct {
    const cap_entries = 64;
    const Entry = struct { key: HeadWhileKey, exec: cuda.graph.Exec, stamp: u64 };
    items: [cap_entries]?Entry = @splat(null),
    tick: u64 = 0,
    captures: u64 = 0,
    replays: u64 = 0,
    off: bool = false,

    pub fn clear(c: *WhileCache) void {
        for (&c.items) |*slot| {
            if (slot.*) |*e| e.exec.deinit();
            slot.* = null;
        }
    }

    pub fn find(c: *WhileCache, key: HeadWhileKey) ?*cuda.graph.Exec {
        c.tick +%= 1;
        for (&c.items) |*slot| {
            if (slot.*) |*e| {
                if (std.meta.eql(e.key, key)) {
                    e.stamp = c.tick;
                    c.replays += 1;
                    return &e.exec;
                }
            }
        }
        return null;
    }

    pub fn put(c: *WhileCache, key: HeadWhileKey, exec: cuda.graph.Exec) *cuda.graph.Exec {
        c.tick +%= 1;
        c.captures += 1;
        for (&c.items) |*slot| {
            if (slot.* == null) {
                slot.* = .{ .key = key, .exec = exec, .stamp = c.tick };
                return &slot.*.?.exec;
            }
        }
        var lru: usize = 0;
        for (1..cap_entries) |i| {
            if (c.items[i].?.stamp < c.items[lru].?.stamp) lru = i;
        }
        c.items[lru].?.exec.deinit();
        c.items[lru] = .{ .key = key, .exec = exec, .stamp = c.tick };
        return &c.items[lru].?.exec;
    }
};

/// The engine's graph caches, one per kind of pass.
pub const Graphs = struct {
    window: Cache(Key) = .{},
    head: Cache(HeadKey) = .{},
    head_while: WhileCache = .{},
    rollback: Cache(RollbackKey) = .{},
    shared: Cache(SharedKey) = .{},
    head_many: Cache(HeadManyKey) = .{},

    pub fn clear(g: *Graphs, gpa: std.mem.Allocator) void {
        g.window.clear(gpa);
        g.head.clear(gpa);
        g.head_while.clear();
        g.rollback.clear(gpa);
        g.shared.clear(gpa);
        g.head_many.clear(gpa);
    }
};""",
)

# 5m. Trial 37: Fused non-clustered shared_expert gate_up + zero-write greedy lm_head direct-argmax
patch_exact(
    root, "cuda_batch_state.zig",
    "    logits_out: u64 = 0,\n    keep: ?*Keep = null,",
    "    logits_out: u64 = 0,\n    best_out: u64 = 0,\n    keep: ?*Keep = null,",
)
patch_exact(
    root, "cuda_batch_state.zig",
    "        s.shared_gate = try s.add(gpa, d, cap * moe_w * 2);\n        s.shared_up = try s.add(gpa, d, cap * moe_w * 2);",
    "        s.shared_gate = try s.add(gpa, d, cap * 2 * moe_w * 2);\n        s.shared_up = s.shared_gate + cap * moe_w * 2;",
)
patch_exact(
    root, "cuda_mixed_experts.zig",
    """pub const Owned = struct {
    experts: routed.Owned,
    shared: [3]wt.Q4,
    rank: usize,
    world: usize,
    split_shared: bool = false,

    pub fn free(self: *Owned) void {
        self.experts.free();
        for (&self.shared) |*p| p.free();
    }
};""",
    """pub const Owned = struct {
    experts: routed.Owned,
    shared: [3]wt.Q4,
    shared_gu: ?wt.Q4 = null,
    rank: usize,
    world: usize,
    split_shared: bool = false,

    pub fn free(self: *Owned) void {
        self.experts.free();
        if (self.shared_gu) |*g| g.free();
        for (&self.shared) |*p| p.free();
    }
};""",
)
patch_exact(
    root, "cuda_mixed_experts.zig",
    """    var shared: [3]wt.Q4 = @splat(wt.Q4.empty(d));
    errdefer for (&shared) |*p| p.free();
    // The shared branch participates exactly once in the sum across ranks.
    if (rank == 0 or split) {
        var buf: [256]u8 = undefined;
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |name, i| {
            const n = if (i == 2) wt.dims else wt.moe_w;
            const k = if (i == 2) wt.moe_w else wt.dims;
            const key = try std.fmt.bufPrint(&buf, "{s}.shared_expert.{s}", .{ prefix, name });
            shared[i] = if (split) try projections.loadCut(a, d, store, key, n, k, try routed.Cut.init(n, world, rank), .rows) else try projections.load(a, d, store, key, n, k);
            if (shared[i].native == null) return error.UnsupportedQuantization;
        }
    }
    return .{ .experts = e, .shared = shared, .rank = rank, .world = world, .split_shared = split };""",
    """    var shared: [3]wt.Q4 = @splat(wt.Q4.empty(d));
    errdefer for (&shared) |*p| p.free();
    var shared_gu: ?wt.Q4 = null;
    errdefer if (shared_gu) |*g| g.free();
    // The shared branch participates exactly once in the sum across ranks.
    if (rank == 0 or split) {
        var buf: [256]u8 = undefined;
        const bf12 = @import("cuda_bf12_pack.zig");
        if (!split and rank == 0 and bf12.enabled()) {
            const k0 = try std.fmt.bufPrint(&buf, "{s}.shared_expert.gate_proj.weight", .{prefix});
            if (store.has(k0)) {
                const t0 = try store.tensor(k0);
                if (t0.is(.bf16, &.{ wt.moe_w, wt.dims })) {
                    const half_bytes = wt.moe_w * wt.dims * 2;
                    const raw_gu = try a.alloc(u8, half_bytes * 2);
                    defer a.free(raw_gu);
                    @memcpy(raw_gu[0..half_bytes], t0.bytes);
                    const k1 = try std.fmt.bufPrint(&buf, "{s}.shared_expert.up_proj.weight", .{prefix});
                    const t1 = try store.tensor(k1);
                    if (t1.is(.bf16, &.{ wt.moe_w, wt.dims })) {
                        @memcpy(raw_gu[half_bytes .. half_bytes * 2], t1.bytes);
                        shared_gu = try bf12.packRaw(a, d, raw_gu, wt.moe_w * 2, wt.dims);
                    }
                }
            }
        }
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |name, i| {
            if (i < 2 and shared_gu != null) continue;
            const n = if (i == 2) wt.dims else wt.moe_w;
            const k = if (i == 2) wt.moe_w else wt.dims;
            const key = try std.fmt.bufPrint(&buf, "{s}.shared_expert.{s}", .{ prefix, name });
            shared[i] = if (split) try projections.loadCut(a, d, store, key, n, k, try routed.Cut.init(n, world, rank), .rows) else try projections.load(a, d, store, key, n, k);
            if (shared[i].native == null) return error.UnsupportedQuantization;
        }
    }
    return .{ .experts = e, .shared = shared, .shared_gu = shared_gu, .rank = rank, .world = world, .split_shared = split };""",
)
patch_exact(
    root, "cuda_batch_side.zig",
    """    try pass.k.nativeMatmul(st, s.mixed, dims, n, native.shared[0].native.?, s.shared_gate, s.native_part, .bf16);
    try pass.k.nativeMatmul(st, s.mixed, dims, n, native.shared[1].native.?, s.shared_up, s.native_part, .bf16);
    try glue.sharedSwiGLU(st, s.shared_gate, s.shared_up, s.shared_act, n, moe_w);""",
    """    if (native.shared_gu) |gu| {
        const sk = try pass.k.native_lane.matmulRawPart(st, s.mixed, dims, n, gu.native.?, s.shared_gate, .bf16, s.native_part, false);
        if (sk > 1) try glue.sharedReduceSwiGLU(st, s.native_part, s.shared_act, n, moe_w, sk) else try glue.sharedSwiGLUCat(st, s.shared_gate, s.shared_act, n, moe_w);
    } else {
        try pass.k.nativeMatmul(st, s.mixed, dims, n, native.shared[0].native.?, s.shared_gate, s.native_part, .bf16);
        try pass.k.nativeMatmul(st, s.mixed, dims, n, native.shared[1].native.?, s.shared_up, s.native_part, .bf16);
        try glue.sharedSwiGLU(st, s.shared_gate, s.shared_up, s.shared_act, n, moe_w);
    }""",
)
patch_exact(
    root, "cuda_shared_split.zig",
    """    try self.k.native_lane.matmul(self.stream, s.mixed, dims, n, native.shared[0].native.?, s.shared_gate, s.native_part);
    try self.k.native_lane.matmul(self.stream, s.mixed, dims, n, native.shared[1].native.?, s.shared_up, s.native_part);
    try glue.sharedSwiGLU(self.stream, s.shared_gate, s.shared_up, s.shared_act, n, width);""",
    """    if (native.shared_gu) |gu| {
        const sk = try self.k.native_lane.matmulRawPart(self.stream, s.mixed, dims, n, gu.native.?, s.shared_gate, .bf16, s.native_part, false);
        if (sk > 1) try glue.sharedReduceSwiGLU(self.stream, s.native_part, s.shared_act, n, width, sk) else try glue.sharedSwiGLUCat(self.stream, s.shared_gate, s.shared_act, n, width);
    } else {
        try self.k.native_lane.matmul(self.stream, s.mixed, dims, n, native.shared[0].native.?, s.shared_gate, s.native_part);
        try self.k.native_lane.matmul(self.stream, s.mixed, dims, n, native.shared[1].native.?, s.shared_up, s.native_part);
        try glue.sharedSwiGLU(self.stream, s.shared_gate, s.shared_up, s.shared_act, n, width);
    }""",
)
patch_exact(
    root, "cuda_mtp_draft.zig",
    "    history: [2]i64 = .{ 0, 0 },\n    draft_hist: [2]i64 = .{ 0, 0 },\n    unused_history: [2]i64 = .{ 0, 0 },",
    "    history: [2]i64 = .{ 0, 0 },\n    draft_hist: [2]i64 = .{ 0, 0 },\n    unused_history: [2]i64 = .{ 0, 0 },\n    best_direct: bool = false,\n    best_ready: bool = false,",
)
patch_exact(
    root, "cuda_lanes_verify.zig",
    "            for (seqs[0..n], ds[0..n], idss[0..n]) |q, d, ids| try self.ops().verify(q, d, ids);",
    "            for (seqs[0..n], ds[0..n], idss[0..n], drafted[at..][0..n]) |q, d, ids, wi| {\n                d.best_direct = !mtp.isSampled(wins[wi].stream.sampling) and wins[wi].masks.len == 0;\n                try self.ops().verify(q, d, ids);\n            }",
)
patch_exact(
    root, "cuda_mtp_ops.zig",
    """            if (keeps) try copyAsync(o.stream(), d.saved.ptr + d.layout.tail_at, seq.tail.ptr, ple_tail_n * wide * 2);
            try o.e.forward(Tri, o.tri, seq, ids, .{ .logits = .all, .logits_out = d.logits.ptr, .keep = if (keeps) &d.keep else null, .streams = d.streams.ptr, .host_logits = @import("cuda_batch_state.zig").verifyHostLogits() });""",
    """            const vhl = @import("cuda_batch_state.zig").verifyHostLogits();
            const use_best = d.best_direct and !vhl and o.e.head_q.b.ptr != 0 and o.e.head_q.native != null and (o.e.head_q.native.?.mode == .fp4 or o.e.head_q.native.?.mode == .w2 or o.e.head_q.native.?.mode == .w1);
            d.best_direct = false;
            d.best_ready = use_best;
            if (keeps) try copyAsync(o.stream(), d.saved.ptr + d.layout.tail_at, seq.tail.ptr, ple_tail_n * wide * 2);
            try o.e.forward(Tri, o.tri, seq, ids, .{ .logits = .all, .logits_out = d.logits.ptr, .best_out = if (use_best) d.best.ptr else 0, .keep = if (keeps) &d.keep else null, .streams = d.streams.ptr, .host_logits = vhl });""",
)
patch_exact(
    root, "cuda_mtp_ops.zig",
    """                const n = ids.len;
                d.settle(seq);""",
    """                const n = ids.len;
                d.best_direct = false;
                d.best_ready = false;
                d.settle(seq);""",
)
patch_exact(
    root, "cuda_mtp_draw.zig",
    "    if (masks.len == 0 and !mtp.isSampled(sm)) try h.argmax(st, d.logits.ptr, vocab, vocab, n, d.best.ptr) else if (masks.len == 0 and !usable(sm.?)) try h.best(st, d.logits.ptr, vocab, vocab, n, d.best.ptr);",
    "    const ready = d.best_ready;\n    d.best_ready = false;\n    if (masks.len == 0 and !mtp.isSampled(sm)) {\n        if (!ready) try h.argmax(st, d.logits.ptr, vocab, vocab, n, d.best.ptr);\n    } else if (masks.len == 0 and !usable(sm.?)) try h.best(st, d.logits.ptr, vocab, vocab, n, d.best.ptr);",
)


# 5o. Trial 45: Fused 1-CTA topk+plan, lean 4-warp contiguous uint4 tf_nvfp4_full_r8 (c1/MTP) + half_coop2 (c4+), zero-spill 4t GDN, 16-lane paged pool
patch_exact(
    cdir, "route_rows.zig",
    "    wide_fn: Function,\n    topk_fn: Function,",
    "    wide_fn: Function,\n    topk_fn: Function,\n    topk_plan_fn: Function,",
)
patch_exact(
    cdir, "route_rows.zig",
    """        const r: RouteRows = .{ .mod = mod, .logits_fn = try mod.function("tf_route_logits"), .few_fn = try mod.function("tf_route_logits_few"), .wide_fn = try mod.function("tf_route_logits_wide"), .topk_fn = try mod.function("tf_route_topk") };""",
    """        const r: RouteRows = .{ .mod = mod, .logits_fn = try mod.function("tf_route_logits"), .few_fn = try mod.function("tf_route_logits_few"), .wide_fn = try mod.function("tf_route_logits_wide"), .topk_fn = try mod.function("tf_route_topk"), .topk_plan_fn = try mod.function("tf_route_topk_plan") };""",
)
patch_exact(
    cdir, "route_rows.zig",
    """        try launch_.launch(g.topk_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = @as(u32, if (experts <= 512 and top_k <= 32) 32 else 128) } }, s, &a);
    }
};""",
    """        try launch_.launch(g.topk_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = @as(u32, if (experts <= 512 and top_k <= 32) 32 else 128) } }, s, &a);
    }

    /// Fused 1-CTA top-k + 4-bucket plan (with optional SK-slice router partial reduction) for rows<=32, experts<=512, e_plan<=512, slots<=12.
    pub fn topkPlan(g: *const RouteRows, s: Stream, l: u64, ld: usize, experts: usize, top_k: usize, rows: usize, slots: usize, shared: bool, e_plan: usize, tile: usize, rsk: usize, picks: u64, wts: u64, members: u64, items: u64, counts: u64) !void {
        if (rows == 0 or rows > 32 or experts > 512 or e_plan > 512 or slots > 12 or top_k > 32) return error.RowShape;
        var a: launch_.Args = .{};
        a.add(l);
        for ([_]usize{ ld, experts, top_k, rows, slots, @intFromBool(shared), e_plan, tile, rsk }) |v| a.add(@as(i32, @intCast(v)));
        for ([_]u64{ picks, wts, members, items, counts }) |p| a.add(p);
        const blk: u32 = if (rows <= 16) 512 else 1024;
        try launch_.launch(g.topk_plan_fn, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = blk }, .pdl = true }, s, &a);
    }
};""",
)
patch_exact(
    root, "cuda_glue.zig",
    """    /// Each row's ten experts and the shared one: picks and weights [rows, 11] (core route rows).
    pub fn topkMany(g: *const Glue, s: cuda.Stream, logits: u64, pick: u64, wts: u64, rows: usize) !void {
        try g.route.topk(s, logits, wt.experts_n + 1, wt.experts_n, wt.slots - 1, pick, wts, wt.slots, rows, true);
    }""",
    """    /// Each row's ten experts and the shared one: picks and weights [rows, 11] (core route rows).
    pub fn topkMany(g: *const Glue, s: cuda.Stream, logits: u64, pick: u64, wts: u64, rows: usize) !void {
        try g.route.topk(s, logits, wt.experts_n + 1, wt.experts_n, wt.slots - 1, pick, wts, wt.slots, rows, true);
    }

    pub fn topkPlanMany(g: *const Glue, s: cuda.Stream, logits: u64, rsk: usize, pick: u64, wts: u64, rows: usize, e_plan: usize, tile: usize, plan: cuda.grouped.Plan) !void {
        try g.route.topkPlan(s, logits, wt.experts_n + 1, wt.experts_n, wt.slots - 1, rows, wt.slots, true, e_plan, tile, rsk, pick, wts, plan.members, plan.items, plan.counts);
    }

    pub fn topkPlanManyK(g: *const Glue, s: cuda.Stream, logits: u64, rsk: usize, pick: u64, wts: u64, rows: usize, e_plan: usize, tile: usize, plan: cuda.grouped.Plan, top_k: usize) !void {
        try g.route.topkPlan(s, logits, wt.experts_n + 1, wt.experts_n, top_k, rows, wt.slots, true, e_plan, tile, rsk, pick, wts, plan.members, plan.items, plan.counts);
    }""",
)
patch_exact(
    root, "cuda_prompt_load.zig",
    "    router: cuda.DeviceBuffer,\n    linear: bool,",
    "    router: cuda.DeviceBuffer,\n    router_q: ?Q4 = null,\n    linear: bool,\n    is_mtp: bool = false,",
)
patch_exact(
    root, "cuda_prompt_load.zig",
    "        self.router.free();",
    "        self.router.free();\n        if (self.router_q) |*rq| rq.free();",
)
patch_exact(
    root, "cuda_prompt_load.zig",
    "    var router_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(router));\n    errdefer router_b.free();",
    """    var router_b = try cuda.DeviceBuffer.fromHost(driver, std.mem.sliceAsBytes(router));
    errdefer router_b.free();
    var router_q: ?Q4 = null;
    errdefer if (router_q) |*rq| rq.free();
    const use_rbf12 = @import("cuda_bf12_pack.zig").enabled() and (if (std.c.getenv("TF_FLASHNEXT_ROUTER_BF12")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true);
    if (use_rbf12) {
        router_q = try @import("cuda_bf12_pack.zig").packRaw(gpa, driver, std.mem.sliceAsBytes(router), experts_n + 1, dims);
    }""",
)
patch_exact(
    root, "cuda_prompt_load.zig",
    "        .router = router_b,\n        .linear = linear,",
    "        .router = router_b,\n        .router_q = router_q,\n        .linear = linear,\n        .is_mtp = std.mem.startsWith(u8, base, \"language_model.mtp.\"),",
)
patch_exact(
    root, "cuda_batch.zig",
    """            try self.tri.routerMany(s.mixed, lw.router.ptr, s.router_l, dims, n);
            try self.tri.topkMany(s.router_l, s.picks, s.wts, n);
            try self.copy(keep.moe_ids, s.picks, pairs * 4);
            try self.copy(keep.moe_wts, s.wts, pairs * 4);
            const e = experts_n + 1;
            const plan: cuda.grouped.Plan = .{ .members = s.members, .items = s.items, .counts = s.counts, .rank = s.rank, .hist = s.hist };
            if (lw.native_experts) |*native| return moe_native.run(self, native, n, pairs, plan);""",
    """            const e = experts_n + 1;
            const plan: cuda.grouped.Plan = .{ .members = s.members, .items = s.items, .counts = s.counts, .rank = s.rank, .hist = s.hist };
            if (lw.native_experts) |*native| {
                if (native.world == 1 and native.experts.format == .nvfp4 and n <= 32 and keep.moe_ids == 0 and keep.moe_wts == 0 and self.tri.set == null) {
                    if (self.tri.own) |g| {
                        return moe_native.runFused(self, g, lw, native, n, pairs, plan);
                    }
                }
                if (lw.router_q) |rq| {
                    try self.k.nativeMatmul(self.stream, s.mixed, dims, n, rq.native.?, s.router_l, s.native_part, .f32);
                } else {
                    try self.tri.routerMany(s.mixed, lw.router.ptr, s.router_l, dims, n);
                }
                try self.tri.topkMany(s.router_l, s.picks, s.wts, n);
                try self.copy(keep.moe_ids, s.picks, pairs * 4);
                try self.copy(keep.moe_wts, s.wts, pairs * 4);
                return moe_native.run(self, native, n, pairs, plan);
            }
            if (lw.router_q) |rq| {
                try self.k.nativeMatmul(self.stream, s.mixed, dims, n, rq.native.?, s.router_l, s.native_part, .f32);
            } else {
                try self.tri.routerMany(s.mixed, lw.router.ptr, s.router_l, dims, n);
            }
            try self.tri.topkMany(s.router_l, s.picks, s.wts, n);
            try self.copy(keep.moe_ids, s.picks, pairs * 4);
            try self.copy(keep.moe_wts, s.wts, pairs * 4);""",
)
patch_exact(
    root, "cuda_moe_native.zig",
    """/// moeBlockKeep's native layer: each routed pair's fp32 row into y [pairs, dims], the shared slot's on rank 0.
pub fn run(self: anytype, native: *const mixed.Owned, n: usize, pairs: usize, plan: cuda.grouped.Plan) !void {""",
    """/// Optional MTP-only routed-expert top-k cap (target 48 layers always keep full slots-1=10).
fn mtpTopK() usize {
    const std = @import("std");
    const raw = std.c.getenv("TF_FLASHNEXT_MTP_TOPK") orelse return slots - 1;
    const v = std.fmt.parseInt(usize, std.mem.span(raw), 10) catch return slots - 1;
    return @min(v, slots - 1);
}

/// World-1 NVFP4 decode fast path: lossless BF12 router + single-CTA fused reduce+top-k+4-bucket plan skipping global picks round-trip.
pub fn runFused(self: anytype, glue: anytype, lw: anytype, native: *const mixed.Owned, n: usize, pairs: usize, plan: cuda.grouped.Plan) !void {
    const s = self.s;
    const owned = &native.experts;
    const pt = nvfp4_tile.size(self.k, n);
    if (pt != 16) {
        if (lw.router_q) |rq| {
            try self.k.nativeMatmul(self.stream, s.mixed, dims, n, rq.native.?, s.router_l, s.native_part, .f32);
        } else {
            try glue.routerMany(self.stream, s.mixed, lw.router.ptr, s.router_l, dims, n);
        }
        try glue.topkMany(self.stream, s.router_l, s.picks, s.wts, n);
        return run(self, native, n, pairs, plan);
    }
    const e = owned.count;
    const skip: i32 = @intCast(owned.count);
    const active_k: usize = if (lw.is_mtp) mtpTopK() else slots - 1;
    const y_bytes = exchange.clearedRows(native.world, n) * slots * dims * 4;
    if (zs.clearsY(native.world, native.rank)) try self.stream.d.check(self.stream.d.api.cuMemsetD8Async(s.y, 0, y_bytes, self.stream.handle), "cuMemsetD8Async");
    if (lw.router_q) |rq| {
        const rsk = try self.k.native_lane.matmulRawPart(self.stream, s.mixed, dims, n, rq.native.?, s.router_l, .f32, s.native_part, false);
        const r_src = if (rsk > 1) s.native_part else s.router_l;
        try glue.topkPlanManyK(self.stream, r_src, rsk, s.picks, s.wts, n, e, pt, plan, active_k);
    } else {
        try glue.routerMany(self.stream, s.mixed, lw.router.ptr, s.router_l, dims, n);
        try glue.topkPlanManyK(self.stream, s.router_l, 1, s.picks, s.wts, n, e, pt, plan, active_k);
    }
    const side = try shared.fork(self, native, n);
    if (active_k > 0) {
        const active_pairs = n * active_k;
        const items = @min(cuda.grouped.maxItems(pairs, e, pt), cuda.grouped.maxItems(active_pairs, e, pt));
        try nvfp4_tile.run(self.k, pt, self.stream, .up, .{ .x = s.mixed, .stride = dims, .slots = slots }, owned.view(), plan, s.moe_act, items, skip);
        try nvfp4_tile.run(self.k, pt, self.stream, .down_f32, .{ .x = s.moe_act, .stride = owned.width }, owned.view(), plan, s.y, items, skip);
    }
    if (side != null) try shared.join(self, side, native, n) else try @import("cuda_shared_split.zig").run(self, native, n);
}

/// moeBlockKeep's native layer: each routed pair's fp32 row into y [pairs, dims], the shared slot's on rank 0.
pub fn run(self: anytype, native: *const mixed.Owned, n: usize, pairs: usize, plan: cuda.grouped.Plan) !void {""",
)
# Raise max_chain 15->31 while pinning .first_copy_rows=16 so c4/c8/c16 stay inside BM=16
patch_exact(
    root, "cuda_mtp.zig",
    "pub const max_chain: usize = 15;",
    "pub const max_chain: usize = 31;",
)
patch_exact(
    root, "cuda_lanes.zig",
    ".first_copy_rows = if (drafting) mtp.max_chain + 1 else 0,",
    ".first_copy_rows = if (drafting) 16 else 0,\n                    .root_copies = (if (std.c.getenv(\"TF_FLASHNEXT_ROOT_COPIES\")) |v| std.mem.eql(u8, std.mem.span(v), \"1\") else false),",
)
patch_exact(
    root.parent.parent / "core" / "lanes", "root_copy.zig",
    "    if (!e.cfg.root_copies) return requests;",
    "    if (!e.cfg.root_copies or !e.alone) return requests;",
)
patch_exact(
    root, "cuda_lanes_draft.zig",
    "const max_hist = 17;",
    "const max_hist = 33;",
)
patch_exact(
    cdir, "experts.zig",
    """    prompt_fns: [3]Function, // relu^2 up, SwiGLU gate-up, bf16 sums down
    fact_fns: [2]Function, // the decode form's sums on prompt tiles: SwiGLU gate-up, fp32 down
    warps: usize = 4,""",
    """    prompt_fns: [3]Function, // relu^2 up, SwiGLU gate-up, bf16 sums down
    fact_fns: [2]Function, // the decode form's sums on prompt tiles: SwiGLU gate-up, fp32 down
    half4_fn: Function = undefined,
    half2_fn: Function = undefined,
    coop2_u4_fn: Function = undefined,
    full_r8_fn: Function = undefined,
    warps: usize = 4,""",
)
patch_exact(
    cdir, "experts.zig",
    """        var e: Nvfp4Experts = .{ .mod = mod, .fns = undefined, .resident = undefined, .prompt_fns = undefined, .fact_fns = undefined, .warps = warps };""",
    """        var e: Nvfp4Experts = .{ .mod = mod, .fns = undefined, .resident = undefined, .prompt_fns = undefined, .fact_fns = undefined, .half4_fn = try mod.function("tf_nvfp4_half_coop4"), .half2_fn = try mod.function("tf_nvfp4_half_coop2"), .coop2_u4_fn = try mod.function("tf_nvfp4_coop2_u4"), .full_r8_fn = try mod.function("tf_nvfp4_full_r8"), .warps = warps };""",
)
patch_exact(
    cdir, "experts_waves.zig",
    """    const dyn = e.warps == 1 and (if (std.c.getenv("TF_NVFP4_DYNAMIC_GRID")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false);
    const fi = (if (dyn) (if (gate and 2 * items * nb <= e.resident[first + 2] * e.warps) @as(?usize, first + 2) else null) else pick(gate, items, nb, e.resident[first + 2] * e.warps, classic)) orelse return e.run(s, epi, in, l, p, out, items, skip);""",
    """    const dyn = e.warps == 1 and (if (std.c.getenv("TF_NVFP4_DYNAMIC_GRID")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false);
    if (dyn and gate and !l.relu2) {
        const h_min: usize = if (std.c.getenv("TF_NVFP4_HALF_MIN")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 1) else 1;
        const fr8_max: usize = if (std.c.getenv("TF_NVFP4_FULL_R8_MAX")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 192) else 192;
        const h_max: usize = if (std.c.getenv("TF_NVFP4_HALF4_MAX")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 320) else 320;
        const h_warps: usize = if (std.c.getenv("TF_NVFP4_HALF_WARPS")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 2) else 2;
        if (items >= h_min and items <= fr8_max) {
            const kg = l.dims / 32;
            const grid = (items * nb + 3) / 4;
            if (grid >= 1) {
                var a: launch_.Args = .{};
                a.add(in.x);
                a.add(@as(i32, @intCast(in.stride)));
                a.add(@as(i32, @intCast(in.slots)));
                a.add(l.up);
                a.add(l.up_scale);
                a.add(@as(i32, @intCast(kg)));
                a.add(@as(i32, @intCast(nb)));
                for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
                a.add(@as(i32, @intCast(l.width)));
                a.add(l.limit);
                a.add(skip);
                return launch_.launch(e.full_r8_fn, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 }, .pdl = true }, s, &a);
            }
        }
        if (items >= h_min and items <= h_max) {
            const kg = l.dims / 32;
            const use2 = h_warps == 2;
            const grid = if (use2) items * nb else (items * nb * 2 + 3) / 4;
            if (grid >= 1) {
                var a: launch_.Args = .{};
                a.add(in.x);
                a.add(@as(i32, @intCast(in.stride)));
                a.add(@as(i32, @intCast(in.slots)));
                a.add(l.up);
                a.add(l.up_scale);
                a.add(@as(i32, @intCast(kg)));
                a.add(@as(i32, @intCast(nb)));
                for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
                a.add(@as(i32, @intCast(l.width)));
                a.add(l.limit);
                a.add(skip);
                const use_u4 = use2 and (if (std.c.getenv("TF_NVFP4_COOP_U4")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true);
                const hfn = if (use_u4) e.coop2_u4_fn else if (use2) e.half2_fn else e.half4_fn;
                return launch_.launch(hfn, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = if (use2) 64 else 128, .y = 1, .z = 1 }, .pdl = true }, s, &a);
            }
        }
    }
    const fi = (if (dyn) (if (gate and 2 * items * nb <= e.resident[first + 2] * e.warps) @as(?usize, first + 2) else null) else pick(gate, items, nb, e.resident[first + 2] * e.warps, classic)) orelse return e.run(s, epi, in, l, p, out, items, skip);""",
)
patch_exact(
    root, "cuda_batch_late.zig",
    "    try cuda.launch.launch(k.chain_late, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 16 }, .block = .{ .x = 128 } }, stream, &a);",
    "    try cuda.launch.launch(k.chain_late, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 32 }, .block = .{ .x = 128 } }, stream, &a);",
)
patch_exact(
    root, "cuda_mtp_ops.zig",
    "                try cuda.launch.launch(o.e.batch_k.chain_late, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 16 }, .block = .{ .x = 128 } }, st, &a);",
    "                try cuda.launch.launch(o.e.batch_k.chain_late, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 32 }, .block = .{ .x = 128 } }, st, &a);",
)
patch_exact(
    root, "cuda_lanes.zig",
    """            own.kv.pool = try kvl.Pool.init(own.gpa, own.engine, own.kv.layout.?, own.context, own.kv.head, streams, count);
            own.kv.paged_only = kvl.pagedOnly() and streams > 0;""",
    """            const paged_only = kvl.pagedOnly() and streams > 0;
            own.kv.pool = try kvl.Pool.init(own.gpa, own.engine, own.kv.layout.?, own.context, own.kv.head, @as(usize, streams) + @intFromBool(paged_only), count);
            own.kv.paged_only = paged_only;""",
)

# 5n. Trial 40: Zero-state_out GDN verify (-113.4 MB/round dead writes on c1/c4/c8/c16, zero seq.state swaps on settle)
patch_exact(
    root, "cuda_mtp_draft.zig",
    """            lk.state_out = d.spare[li].ptr;""",
    """            lk.state_out = if (d.log != null) 0 else d.spare[li].ptr;""",
)
patch_exact(
    root, "cuda_mtp_draft.zig",
    """    pub fn settle(d: *Draft, seq: *prompt.Seq) void {
        if (!d.open) return;
        for (seq.state, &d.spare) |*a, *b| std.mem.swap(cuda.DeviceBuffer, a, b);""",
    """    pub fn settle(d: *Draft, seq: *prompt.Seq) void {
        if (!d.open) return;
        if (d.log != null) {
            d.keptLate(d.rows);
            d.open = false;
            return;
        }
        for (seq.state, &d.spare) |*a, *b| std.mem.swap(cuda.DeviceBuffer, a, b);""",
)
patch_exact(
    root, "cuda_batch_late.zig",
    "    if (keep.state_out == 0) return error.UnexpectedTensor;\n",
    "",
)
patch_exact(
    root, "cuda_batch_gdn_lanes.zig",
    "        if (lk.log != 0 and lk.state_out == 0) return error.UnexpectedTensor;\n",
    "",
)

import re as _re

def add_pdl_to_launches(base: Path, rel: str, cond: str, expect_min: int = 1) -> None:
    p = base / rel
    txt = p.read_text(encoding="utf-8")
    n_rep = 0
    def repl(m):
        nonlocal n_rep
        s = m.group(0)
        if ".pdl" in s:
            return s
        # Insert `, .pdl = <cond>` right before the closing ` }` of the Config literal
        idx = s.rfind("}")
        n_rep += 1
        return s[:idx].rstrip() + f", .pdl = {cond} " + s[idx:]
    # Match single-line launch calls whose Config literal is `.{ ... }`
    new_txt = _re.sub(r'(?:cuda\.launch\.launch|launch_\.launch)\([^;\n]*?\.\{[^{;\n]*(?:\.\{[^{;\n]*\}[^{;\n]*)*\}', repl, txt)
    if n_rep < expect_min:
        raise RuntimeError(f"add_pdl_to_launches({rel}): expected >= {expect_min}, patched {n_rep}")
    p.write_text(new_txt, encoding="utf-8")

add_pdl_to_launches(cdir, "route_rows.zig", "rows <= 64", expect_min=3)
add_pdl_to_launches(cdir, "norms.zig", "rows <= 64", expect_min=10)
add_pdl_to_launches(cdir, "embed_rows.zig", "rows <= 64", expect_min=4)
add_pdl_to_launches(root, "cuda_glue.zig", "true", expect_min=3)
add_pdl_to_launches(root, "cuda_batch.zig", "true", expect_min=4)
add_pdl_to_launches(root, "cuda_batch_late.zig", "true", expect_min=2)
add_pdl_to_launches(root, "cuda_batch_moe.zig", "true", expect_min=1)
add_pdl_to_launches(root, "cuda_batch_ops.zig", "true", expect_min=3)
add_pdl_to_launches(root, "cuda_batch_gdn_lanes.zig", "true", expect_min=4)
add_pdl_to_launches(root, "cuda_batch_lanes.zig", "true", expect_min=1)
add_pdl_to_launches(root, "cuda_mtp.zig", "true", expect_min=3)
add_pdl_to_launches(root, "cuda_mtp_ops.zig", "true", expect_min=1)
add_pdl_to_launches(root, "cuda_mtp_roll.zig", "true", expect_min=4)

# 5p. Trial 54: 33-CTA route_w2_s4_k128 + single-chunk fused decode attention (fn_attn_fused_ahead_paged)
patch_exact(
    cdir, "route_rows.zig",
    "pub const few_shared: u32 = 3 * 48 * (256 + 8) * 2;",
    "pub const few_shared: u32 = 4 * 32 * (128 + 8) * 2;",
)
patch_exact(
    cdir, "route_rows.zig",
    """        try launch_.launch(if (few) g.few_fn else g.logits_fn, .{ .grid = .{ .x = @intCast((rows + 15) / 16), .y = @intCast((e + 31) / 32), .z = 1 }, .block = .{ .x = 128 }, .shared = if (few) few_shared else many_shared, .pdl = rows <= 64 }, s, &a);""",
    """        try launch_.launch(if (few) g.few_fn else g.logits_fn, .{ .grid = .{ .x = @intCast((rows + 15) / 16), .y = @intCast(if (few) (e + 15) / 16 else (e + 31) / 32), .z = 1 }, .block = .{ .x = if (few) 64 else 128 }, .shared = if (few) few_shared else many_shared, .pdl = rows <= 64 }, s, &a);""",
)
patch_exact(
    root, "cuda_glue.zig",
    "    fused_paged: cuda.Function,\n};",
    "    fused_paged: cuda.Function,\n    fused_ahead: cuda.Function,\n    fused_ahead_paged: cuda.Function,\n    fused_gate: cuda.Function,\n    fused_gate_paged: cuda.Function,\n    fused_ahead_gate: cuda.Function,\n    fused_ahead_gate_paged: cuda.Function,\n};",
)
patch_exact(
    root, "cuda_glue.zig",
    "            .fused_paged = try g.attn.function(\"fn_attn_fused_paged\"),\n        };",
    "            .fused_paged = try g.attn.function(\"fn_attn_fused_paged\"),\n            .fused_ahead = try g.attn.function(\"fn_attn_fused_ahead\"),\n            .fused_ahead_paged = try g.attn.function(\"fn_attn_fused_ahead_paged\"),\n            .fused_gate = try g.attn.function(\"fn_attn_fused_gate\"),\n            .fused_gate_paged = try g.attn.function(\"fn_attn_fused_gate_paged\"),\n            .fused_ahead_gate = try g.attn.function(\"fn_attn_fused_ahead_gate\"),\n            .fused_ahead_gate_paged = try g.attn.function(\"fn_attn_fused_ahead_gate_paged\"),\n        };",
)
patch_exact(
    root, "cuda_glue.zig",
    "        try g.f.fused_paged.allowDynamicShared(oneSmem());\n        return g;",
    "        try g.f.fused_paged.allowDynamicShared(oneSmem());\n        try g.f.fused_ahead.allowDynamicShared(chunkSmem(2));\n        try g.f.fused_ahead_paged.allowDynamicShared(chunkSmem(2));\n        try g.f.fused_gate.allowDynamicShared(oneSmem());\n        try g.f.fused_gate_paged.allowDynamicShared(oneSmem());\n        try g.f.fused_ahead_gate.allowDynamicShared(chunkSmem(2));\n        try g.f.fused_ahead_gate_paged.allowDynamicShared(chunkSmem(2));\n        return g;",
)
patch_exact(
    root, "cuda_glue.zig",
    "        try go(if (pg.pt != 0) g.f.fused_paged else g.f.fused, s, .{ rows, kv, 1 }, 128, oneSmem(), &a);\n    }",
    """        const ahead = rows <= ahead_rows;
        const f = if (ahead) g.f.fused_ahead else g.f.fused;
        const paged = if (ahead) g.f.fused_ahead_paged else g.f.fused_paged;
        try go(if (pg.pt != 0) paged else f, s, .{ rows, kv, 1 }, 128, if (ahead) chunkSmem(2) else oneSmem(), &a);
    }

    pub fn attnFusedGate(g: *const Glue, s: cuda.Stream, q: u64, kc: u64, vc: u64, pos: u64, out: u64, ids: u64, nk: u64, sparse: u64, rows: usize, h: Heads, pg: Pg, proj: u64, pw: usize) !void {
        const kv: usize = h.kv;
        const group = h.q / kv;
        if (group > tile_heads) return error.RowShape;
        var a: cuda.Args = .{};
        ptrs(&a, &.{ q, kc, vc, pos, out, ids, nk, sparse });
        ints(&a, &.{ h.q, kv, group, 512 });
        a.add(@as(f32, 0.0625));
        ints(&a, &.{ wt.index_idw, 1 });
        if (pg.pt != 0) a.add(pg);
        a.add(proj);
        a.add(@as(i32, @intCast(pw)));
        const ahead = rows <= ahead_rows;
        const f = if (ahead) g.f.fused_ahead_gate else g.f.fused_gate;
        const paged = if (ahead) g.f.fused_ahead_gate_paged else g.f.fused_gate_paged;
        try go(if (pg.pt != 0) paged else f, s, .{ rows, kv, 1 }, 128, if (ahead) chunkSmem(2) else oneSmem(), &a);
    }""",
)
patch_exact(
    root, "cuda_batch.zig",
    """            if (self.lanes_attn) {
                try batch_lanes.attn(self, lw, ai, pa);
            } else if (self.lanes.len > 0) {
                // shared rounds: each lane's rope, cache writes, selection and attention on its own stream's caches
                const own_seq = self.seq;
                const own_pos0 = self.pos0;
                defer {
                    self.seq = own_seq;
                    self.pos0 = own_pos0;
                }
                var row0: usize = 0;
                for (self.lanes, 0..) |l, i| {
                    self.seq = l.seq;
                    self.pos0 = l.pos;
                    try self.attnRows(lw, ai, pa + row0 * pn * 2, l.rows, .{ .pos = s.pos + i * 16, .q = s.q + row0 * h.q * head_dim * 2, .iq = s.iq + row0 * index_heads * index_dim * 2, .ao = s.ao + row0 * h.q * head_dim * 2 });
                    row0 += l.rows;
                }
            } else try self.attnRows(lw, ai, pa, n, .{ .pos = s.pos, .q = s.q, .iq = s.iq, .ao = s.ao });
            const gated = if (keep.gated != 0) keep.gated else s.gated;
            try self.tri.attnGate(s.ao, pa, gated, if (lw.out.native != null) @as(u64, 0) else s.gxs_attn, n, pn, h.q, head_dim);""",
    """            const gated = if (keep.gated != 0) keep.gated else s.gated;
            const inline_gate = !self.lanes_attn and lw.out.native != null and self.tri.fusedAttentionOwned() and self.tri.own != null;
            if (self.lanes_attn) {
                try batch_lanes.attn(self, lw, ai, pa);
            } else if (self.lanes.len > 0) {
                // shared rounds: each lane's rope, cache writes, selection and attention on its own stream's caches
                const own_seq = self.seq;
                const own_pos0 = self.pos0;
                defer {
                    self.seq = own_seq;
                    self.pos0 = own_pos0;
                }
                var row0: usize = 0;
                for (self.lanes, 0..) |l, i| {
                    self.seq = l.seq;
                    self.pos0 = l.pos;
                    try self.attnRows(lw, ai, pa + row0 * pn * 2, l.rows, .{ .pos = s.pos + i * 16, .q = s.q + row0 * h.q * head_dim * 2, .iq = s.iq + row0 * index_heads * index_dim * 2, .ao = s.ao + row0 * h.q * head_dim * 2, .gated = if (inline_gate) gated + row0 * h.q * head_dim * 2 else 0, .pn = pn });
                    row0 += l.rows;
                }
            } else try self.attnRows(lw, ai, pa, n, .{ .pos = s.pos, .q = s.q, .iq = s.iq, .ao = s.ao, .gated = if (inline_gate) gated else 0, .pn = pn });
            if (!inline_gate) {
                try self.tri.attnGate(s.ao, pa, gated, if (lw.out.native != null) @as(u64, 0) else s.gxs_attn, n, pn, h.q, head_dim);
            }""",
)
patch_exact(
    root, "cuda_batch.zig",
    "        const RowsAt = struct { pos: u64, q: u64, iq: u64, ao: u64 };",
    "        const RowsAt = struct { pos: u64, q: u64, iq: u64, ao: u64, gated: u64 = 0, pn: usize = 0 };",
)
patch_exact(
    root, "cuda_batch.zig",
    """                    if (!fused) {
                        const chunks = cdiv(@min(ends, wt.dense_keys), 512);
                        try self.tri.attnChunks(at.q + r0 * qr, kc, vc, pos, s.po, s.pm, s.pl, s.qids, s.qnk, s.qspr, m, h, chunks, true, pg);
                        try self.tri.attnMerge(s.po, s.pm, s.pl, pos, at.ao + r0 * qr, s.qnk, s.qspr, m, h, true);
                    }
                }
                if (fused) try self.tri.attnFused(at.q + g0 * qr, kc, vc, at.pos + (g0 / att_rows) * 16, at.ao + g0 * qr, s.qids, s.qnk, s.qspr, gm, h, pg);""",
    """                    if (!fused) {
                        const chunks = cdiv(@min(ends, wt.dense_keys), 512);
                        if (chunks == 1 and self.tri.fusedAttentionOwned()) {
                            if (at.gated != 0) {
                                try self.tri.own.?.attnFusedGate(self.stream, at.q + r0 * qr, kc, vc, pos, at.gated + r0 * qr, s.qids, s.qnk, s.qspr, m, h, pg, pa + r0 * at.pn * 2, at.pn);
                            } else {
                                try self.tri.attnFused(at.q + r0 * qr, kc, vc, pos, at.ao + r0 * qr, s.qids, s.qnk, s.qspr, m, h, pg);
                            }
                        } else {
                            try self.tri.attnChunks(at.q + r0 * qr, kc, vc, pos, s.po, s.pm, s.pl, s.qids, s.qnk, s.qspr, m, h, chunks, true, pg);
                            try self.tri.attnMerge(s.po, s.pm, s.pl, pos, at.ao + r0 * qr, s.qnk, s.qspr, m, h, true);
                            if (at.gated != 0) try self.tri.attnGate(at.ao + r0 * qr, pa + r0 * at.pn * 2, at.gated + r0 * qr, 0, m, at.pn, h.q, head_dim);
                        }
                    }
                }
                if (fused) {
                    if (at.gated != 0) {
                        try self.tri.own.?.attnFusedGate(self.stream, at.q + g0 * qr, kc, vc, at.pos + (g0 / att_rows) * 16, at.gated + g0 * qr, s.qids, s.qnk, s.qspr, gm, h, pg, pa + g0 * at.pn * 2, at.pn);
                    } else {
                        try self.tri.attnFused(at.q + g0 * qr, kc, vc, at.pos + (g0 / att_rows) * 16, at.ao + g0 * qr, s.qids, s.qnk, s.qspr, gm, h, pg);
                    }
                }""",
)

# 5q. Trial 60: Lossless 256B-aligned 5-bit NVFP4 Expert Scales (S5) for down_f32 (decode + prefill, STACK:0, +2.56% on down_f32, -0.94 GB VRAM)
patch_exact(
    cdir, "experts.zig",
    """    shared_up: u64 = 0,
    shared_down: u64 = 0,
};""",
    """    shared_up: u64 = 0,
    shared_down: u64 = 0,
    down_s5_meta: u64 = 0,
    down_s5_esc: u64 = 0,
};""",
)
patch_exact(
    cdir, "experts.zig",
    """    coop2_u4_fn: Function = undefined,
    full_r8_fn: Function = undefined,
    warps: usize = 4,""",
    """    coop2_u4_fn: Function = undefined,
    full_r8_fn: Function = undefined,
    s5_down_1w_fn: Function = undefined,
    s5_fact_down_fn: Function = undefined,
    warps: usize = 4,""",
)
patch_exact(
    cdir, "experts.zig",
    """        var e: Nvfp4Experts = .{ .mod = mod, .fns = undefined, .resident = undefined, .prompt_fns = undefined, .fact_fns = undefined, .half4_fn = try mod.function("tf_nvfp4_half_coop4"), .half2_fn = try mod.function("tf_nvfp4_half_coop2"), .coop2_u4_fn = try mod.function("tf_nvfp4_coop2_u4"), .full_r8_fn = try mod.function("tf_nvfp4_full_r8"), .warps = warps };""",
    """        var e: Nvfp4Experts = .{
            .mod = mod,
            .fns = undefined,
            .resident = undefined,
            .prompt_fns = undefined,
            .fact_fns = undefined,
            .half4_fn = try mod.function("tf_nvfp4_half_coop4"),
            .half2_fn = try mod.function("tf_nvfp4_half_coop2"),
            .coop2_u4_fn = try mod.function("tf_nvfp4_coop2_u4"),
            .full_r8_fn = try mod.function("tf_nvfp4_full_r8"),
            .s5_down_1w_fn = try mod.function("tf_nvfp4_s5_down_1w"),
            .s5_fact_down_fn = try mod.function("tf_nvfp4_s5_fact_down_f32"),
            .warps = warps,
        };
        try e.s5_fact_down_fn.allowDynamicShared(34048);""",
)
patch_exact(
    cdir, "experts.zig",
    """    pub fn run(e: *const Nvfp4Experts, s: Stream, epi: Epi, in: Rows, l: Layer, p: Plan, out: u64, items: usize, skip: i32) !void {
        const gate = epi == .up;
        const kg = (if (gate) l.dims else l.width) / 32;
        const nb = (if (gate) l.width else l.dims) / cols;""",
    """    pub fn run(e: *const Nvfp4Experts, s: Stream, epi: Epi, in: Rows, l: Layer, p: Plan, out: u64, items: usize, skip: i32) !void {
        const gate = epi == .up;
        const kg = (if (gate) l.dims else l.width) / 32;
        const nb = (if (gate) l.width else l.dims) / cols;
        if (!gate and epi == .down_f32 and l.down_s5_meta != 0) {
            const grid = items * nb;
            if (grid < 1) return;
            var a: launch_.Args = .{};
            a.add(in.x);
            a.add(@as(i32, @intCast(in.stride)));
            a.add(@as(i32, 0));
            a.add(l.down);
            a.add(l.down_scale);
            a.add(l.down_s5_meta);
            a.add(l.down_s5_esc);
            a.add(@as(i32, @intCast(kg)));
            a.add(@as(i32, @intCast(nb)));
            for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
            a.add(@as(i32, @intCast(l.dims)));
            a.add(@as(f32, 0.0));
            a.add(skip);
            return launch_.launch(e.s5_down_1w_fn, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 }, .pdl = true }, s, &a);
        }""",
)
patch_exact(
    cdir, "experts_waves.zig",
    """    if (dyn and gate and !l.relu2) {""",
    """    if (!gate and epi == .down_f32 and l.down_s5_meta != 0) {
        return e.run(s, epi, in, l, p, out, items, skip);
    }
    if (dyn and gate and !l.relu2) {""",
)
patch_exact(
    cdir, "experts_fact.zig",
    """    if (items < 1) return;
    var a: launch_.Args = .{};""",
    """    if (items < 1) return;
    if (!gate and l.down_s5_meta != 0) {
        var a: launch_.Args = .{};
        a.add(in.x);
        a.add(@as(i32, @intCast(in.stride)));
        a.add(@as(i32, 0));
        a.add(l.down);
        a.add(l.down_scale);
        a.add(l.down_s5_meta);
        a.add(l.down_s5_esc);
        a.add(@as(i32, @intCast(l.width / 32)));
        a.add(@as(i32, @intCast(nb)));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(l.dims)));
        a.add(@as(f32, 0.0));
        a.add(skip);
        const wn = Nvfp4Experts.fact_wn[1];
        const cfg: launch_.Config = .{ .grid = .{ .x = @intCast(items * ((nb + wn - 1) / wn)), .y = 1, .z = 1 }, .block = .{ .x = @intCast(wn * 32), .y = 1, .z = 1 }, .shared = 34048 };
        return launch_.launch(e.s5_fact_down_fn, cfg, s, &a);
    }
    var a: launch_.Args = .{};""",
)
patch_exact(
    root, "cuda_nvfp4_tile.zig",
    """    if (pt != 16) if (k.expert_tile) |*tile| if (tile.takes(epi)) return tile.run(s, epi, in, l, p, out, items, skip);""",
    """    if (pt != 16 and (epi == .up or l.down_s5_meta == 0)) if (k.expert_tile) |*tile| if (tile.takes(epi)) return tile.run(s, epi, in, l, p, out, items, skip);""",
)
patch_exact(
    root, "cuda_modelopt_experts.zig",
    """    /// .nvfp4 (global scales a projection) or .fp8g (f32 128x128 block scales, cuda_fp8_experts_load.zig).
    format: cuda.experts.Format = .nvfp4,

    pub fn free(self: *Owned) void {
        self.up.free();
        self.down.free();
        self.up_scale.free();
        self.down_scale.free();
        self.* = undefined;
    }

    pub fn view(self: *const Owned) cuda.experts.Layer {
        return .{ .format = self.format, .up = self.up.ptr, .down = self.down.ptr, .up_scale = self.up_scale.ptr, .down_scale = self.down_scale.ptr, .width = self.width, .dims = self.dims, .experts = self.count };
    }""",
    """    /// .nvfp4 (global scales a projection) or .fp8g (f32 128x128 block scales, cuda_fp8_experts_load.zig).
    format: cuda.experts.Format = .nvfp4,
    down_s5_meta: ?cuda.DeviceBuffer = null,
    down_s5_esc: ?cuda.DeviceBuffer = null,

    pub fn free(self: *Owned) void {
        self.up.free();
        self.down.free();
        self.up_scale.free();
        self.down_scale.free();
        if (self.down_s5_meta) |*b| b.free();
        if (self.down_s5_esc) |*b| b.free();
        self.* = undefined;
    }

    pub fn view(self: *const Owned) cuda.experts.Layer {
        return .{
            .format = self.format,
            .up = self.up.ptr,
            .down = self.down.ptr,
            .up_scale = self.up_scale.ptr,
            .down_scale = self.down_scale.ptr,
            .width = self.width,
            .dims = self.dims,
            .experts = self.count,
            .down_s5_meta = if (self.down_s5_meta) |b| b.ptr else 0,
            .down_s5_esc = if (self.down_s5_esc) |b| b.ptr else 0,
        };
    }""",
)
patch_exact(
    root, "cuda_modelopt_threads.zig",
    """/// A layer's planes borrowed with their pages read ahead, then packed on `n` threads a chunk of experts at a time.
pub fn load(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, dims: usize, width: usize, count: usize, n: usize) !experts.Owned {
    return loadChunked(a, d, store, prefix, dims, width, count, n, staging_bytes);
}""",
    """fn s5Enabled() bool {
    const raw = std.c.getenv("TF_NVFP4_S5") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "1");
}

fn bestSmin(scales: []const u8) u8 {
    var hist: [256]u32 = @splat(0);
    for (scales) |b| hist[b] += 1;
    var cur: u32 = 0;
    for (0..31) |i| cur += hist[i];
    var best_sum: u32 = cur;
    var best_s: u8 = 0;
    for (1..226) |s| {
        cur = cur + hist[s + 30] - hist[s - 1];
        if (cur > best_sum) {
            best_sum = cur;
            best_s = @intCast(s);
        }
    }
    return best_s;
}

fn packDownS5(
    out_bytes: []u8,
    codes: []const u8,
    scales: []const u8,
    n: usize,
    k: usize,
    esc_out: *std.ArrayListUnmanaged(u32),
    alloc: std.mem.Allocator,
) !u8 {
    const nb = n / ex.cols;
    const kg = k / 32;
    const nblk = nb * kg;
    const smin = bestSmin(scales);
    for (0..nb) |cb| for (0..kg) |g| {
        const blk_i = cb * kg + g;
        const block_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, out_bytes[blk_i * 512 ..][0..512]));
        for (0..8) |gq| for (0..4) |t| for (0..4) |j| {
            const row = cb * ex.cols + j * 8 + gq;
            var w: u32 = 0;
            for (0..2) |h| for (0..4) |q| {
                const input = g * 32 + h * 16 + t * 4 + q;
                const byte = codes[row * (k / 2) + input / 2];
                const nib: u32 = if (input & 1 == 0) byte & 0xF else byte >> 4;
                w |= nib << ex.slot(h, q);
            };
            block_u32[(gq * 4 + t) * 4 + j] = w;
        };
        const lo_dst = out_bytes[nblk * 512 + blk_i * 32 ..][0..32];
        const hi_dst = out_bytes[nblk * 544 + blk_i * 8 ..][0..8];
        for (0..4) |t| {
            var hi16: u16 = 0;
            for (0..2) |h| for (0..4) |j| {
                var lo_byte: u8 = 0;
                for (0..2) |c| {
                    const u = (h * 4 + j) * 2 + c;
                    const row = cb * ex.cols + j * 8 + t * 2 + c;
                    const raw = scales[row * (k / 16) + g * 2 + h];
                    const s_idx = t * 16 + u;
                    const key: u32 = @intCast(blk_i * 64 + s_idx);
                    const q: u8 = if (raw >= smin and @as(usize, raw) <= @as(usize, smin) + 30)
                        raw - smin
                    else blk: {
                        try esc_out.append(alloc, (key << 8) | raw);
                        break :blk 31;
                    };
                    lo_byte |= (q & 0xF) << @intCast(c * 4);
                    if ((q & 16) != 0) hi16 |= @as(u16, 1) << @intCast(u);
                }
                lo_dst[t * 8 + h * 4 + j] = lo_byte;
            };
            std.mem.writeInt(u16, hi_dst[t * 2 ..][0..2], hi16, .little);
        }
    };
    return smin;
}

const S5DownRec = struct {
    smin_d: u8 = 0,
    esc_d: std.ArrayListUnmanaged(u32) = .empty,
};

const WorkS5Down = struct {
    views: []const View,
    up_words: []u32,
    down_bytes_buf: []u8,
    recs: []S5DownRec,
    up_words_per: usize,
    down_stride: usize,
    width: usize,
    dims: usize,
    lo: usize,
    hi: usize,
    failed: ?(error{OutOfMemory} || core.nvfp4_csf.Error) = null,

    fn run(w: *WorkS5Down) void {
        w.pack() catch |err| {
            w.failed = err;
        };
    }

    fn pack(w: *WorkS5Down) !void {
        const each = @max(w.width * (w.dims / 16), w.dims * (w.width / 16));
        const bufs = try std.heap.page_allocator.alloc(u8, 3 * each);
        defer std.heap.page_allocator.free(bufs);
        const alloc = std.heap.page_allocator;
        for (w.lo..w.hi) |i| {
            const g = w.views[3 * i];
            const u = w.views[3 * i + 1];
            const o = w.views[3 * i + 2];
            const gs = try scalesOf(g, w.width, w.dims, bufs[0..each]);
            const us = try scalesOf(u, w.width, w.dims, bufs[each..][0..each]);
            const os = try scalesOf(o, w.dims, w.width, bufs[2 * each ..][0..each]);
            ex.packGateUp(w.up_words[i * w.up_words_per ..][0..w.up_words_per], .{ g.codes, gs }, .{ u.codes, us }, w.width, w.dims);
            const dbuf = w.down_bytes_buf[i * w.down_stride ..][0..w.down_stride];
            w.recs[i].smin_d = try packDownS5(dbuf, o.codes, os, w.dims, w.width, &w.recs[i].esc_d, alloc);
        }
    }
};

fn packS5Down(views: []const View, up_words: []u32, down_buf: []u8, recs: []S5DownRec, down_stride: usize, width: usize, dims: usize, n: usize) !void {
    const count = views.len / 3;
    const up_words_per = up_words.len / count;
    const t_n = @max(1, @min(n, max_threads, count));
    const per = (count + t_n - 1) / t_n;
    var works: [max_threads]WorkS5Down = undefined;
    var handles: [max_threads]?std.Thread = @splat(null);
    for (0..t_n) |t| {
        works[t] = .{
            .views = views,
            .up_words = up_words,
            .down_bytes_buf = down_buf,
            .recs = recs,
            .up_words_per = up_words_per,
            .down_stride = down_stride,
            .width = width,
            .dims = dims,
            .lo = @min(count, t * per),
            .hi = @min(count, (t + 1) * per),
        };
        handles[t] = std.Thread.spawn(.{}, WorkS5Down.run, .{&works[t]}) catch blk: {
            works[t].run();
            break :blk null;
        };
    }
    for (handles[0..t_n]) |h| if (h) |th| th.join();
    for (works[0..t_n]) |w| if (w.failed) |err| return err;
}

/// A layer's planes borrowed with their pages read ahead, then packed on `n` threads a chunk of experts at a time.
pub fn load(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, dims: usize, width: usize, count: usize, n: usize) !experts.Owned {
    return loadChunked(a, d, store, prefix, dims, width, count, n, staging_bytes);
}""",
)
patch_exact(
    root, "cuda_modelopt_threads.zig",
    """pub fn loadMode(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, dims: usize, width: usize, count: usize, n: usize, stage_cap: usize, comptime direct: bool) !experts.Owned {
    const up_bytes = try experts.packedBytes(width, dims, 2);
    const down_bytes = try experts.packedBytes(dims, width, 1);""",
    """pub fn loadMode(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, dims: usize, width: usize, count: usize, n: usize, stage_cap: usize, comptime direct: bool) !experts.Owned {
    const use_s5 = s5Enabled() and width % 32 == 0 and dims % 32 == 0 and ((dims / 32) * (width / 32)) % 32 == 0;
    const up_bytes = try experts.packedBytes(width, dims, 2);
    const down_bytes = if (use_s5) (dims / 32) * (width / 32) * 552 else try experts.packedBytes(dims, width, 1);""",
)
patch_exact(
    root, "cuda_modelopt_threads.zig",
    """    var lo: usize = 0;
    var k: usize = 0;
    while (lo < count) : ({
        lo += chunk;
        k += 1;
    }) {
        const m = @min(chunk, count - lo);
        // the next chunk's gather runs while this one packs; the main thread allocates nothing from `a` meanwhile
        defer pair.wait() catch {};
        if (sources != null and lo + chunk < count) {
            marker.k = k + 1;
            const hook: core.direct_gather.Hook = .{ .ctx = &marker, .run = @TypeOf(marker).run };
            const ahead = planes(views[3 * (lo + chunk) ..][0 .. 3 * @min(chunk, count - lo - chunk)], ptrs);
            try pair.start(sources.?, ahead, hook);
        }
        const up_words = std.mem.bytesAsSlice(u32, stage[0 .. m * up_bytes]);
        const down_words: []u32 = @alignCast(std.mem.bytesAsSlice(u32, stage[m * up_bytes ..][0 .. m * down_bytes]));
        try pack(views[3 * lo ..][0 .. 3 * m], up_words, down_words, width, dims, n);""",
    """    const s5_recs: []S5DownRec = if (use_s5) try a.alloc(S5DownRec, count) else &.{};
    defer {
        for (s5_recs) |*r| r.esc_d.deinit(std.heap.page_allocator);
        if (use_s5) a.free(s5_recs);
    }
    for (s5_recs) |*r| r.* = .{};
    var lo: usize = 0;
    var k: usize = 0;
    while (lo < count) : ({
        lo += chunk;
        k += 1;
    }) {
        const m = @min(chunk, count - lo);
        // the next chunk's gather runs while this one packs; the main thread allocates nothing from `a` meanwhile
        defer pair.wait() catch {};
        if (sources != null and lo + chunk < count) {
            marker.k = k + 1;
            const hook: core.direct_gather.Hook = .{ .ctx = &marker, .run = @TypeOf(marker).run };
            const ahead = planes(views[3 * (lo + chunk) ..][0 .. 3 * @min(chunk, count - lo - chunk)], ptrs);
            try pair.start(sources.?, ahead, hook);
        }
        const up_words = std.mem.bytesAsSlice(u32, stage[0 .. m * up_bytes]);
        if (use_s5) {
            try packS5Down(views[3 * lo ..][0 .. 3 * m], up_words, stage[m * up_bytes ..][0 .. m * down_bytes], s5_recs[lo..][0..m], down_bytes, width, dims, n);
        } else {
            const down_words: []u32 = @alignCast(std.mem.bytesAsSlice(u32, stage[m * up_bytes ..][0 .. m * down_bytes]));
            try pack(views[3 * lo ..][0 .. 3 * m], up_words, down_words, width, dims, n);
        }""",
)
patch_exact(
    root, "cuda_modelopt_threads.zig",
    """    var us = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(up_globals));
    errdefer us.free();
    const ds = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(down_globals));
    return .{ .up = up, .down = down, .up_scale = us, .down_scale = ds, .width = width, .dims = dims, .count = count };""",
    """    var us = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(up_globals));
    errdefer us.free();
    var ds = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(down_globals));
    errdefer ds.free();
    if (use_s5) {
        const dn_meta = try a.alloc([2]u32, count);
        defer a.free(dn_meta);
        var dn_esc_tot: usize = 0;
        for (s5_recs) |r| dn_esc_tot += r.esc_d.items.len;
        const dn_esc = try a.alloc(u32, @max(1, dn_esc_tot));
        defer a.free(dn_esc);
        dn_esc[0] = 0;
        var doff: usize = 0;
        for (s5_recs, 0..) |r, i| {
            dn_meta[i] = .{ @as(u32, r.smin_d) | (@as(u32, @intCast(r.esc_d.items.len)) << 8), @intCast(doff) };
            @memcpy(dn_esc[doff..][0..r.esc_d.items.len], r.esc_d.items);
            doff += r.esc_d.items.len;
        }
        var dm = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(dn_meta));
        errdefer dm.free();
        const de = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(dn_esc));
        return .{ .up = up, .down = down, .up_scale = us, .down_scale = ds, .width = width, .dims = dims, .count = count, .down_s5_meta = dm, .down_s5_esc = de };
    }
    return .{ .up = up, .down = down, .up_scale = us, .down_scale = ds, .width = width, .dims = dims, .count = count };""",
)
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
    export TF_FLASHNEXT_MTP_W2="w1g64"
    export TF_FLASHNEXT_W1_CAL="1.15"
    export TF_FLASHNEXT_W1_KTOP="64"
    export TF_FLASHNEXT_MTP_EXPERTS="nvfp4"
    export TF_FLASHNEXT_MTP_NVFP4="0"
    export TF_FLASHNEXT_MTP_INT6="clean"
    export TF_NVFP4_S5="1"
    export TF_FLASHNEXT_HEAD_NVFP4="w1"
    export TF_FLASHNEXT_HEAD_W1_KTOP="128"
    export TF_FLASHNEXT_MTP_TOPK="10"
    export TF_BF12_OUT_SK4="0"
    export TF_FLASHNEXT_TRUNK_BF12="1"
    export TF_FLASHNEXT_FUSED_SPLIT="0"
    export TF_FLASHNEXT_L2PF_MB="0"
    export TF_FLASHNEXT_L2PF_PACE_GBPS="165"
    export TF_NVFP4_DECODE_WARPS="1"
    export TF_NVFP4_DYNAMIC_GRID="1"
    export TF_FLASHNEXT_ATTN_LANES="1"
    export TF_FLASHNEXT_WARM_AHEAD="1"
    export TF_FLASHNEXT_GREEDY_WIDE="1"
    export TF_FLASHNEXT_MTP_WHILE="1"
    if [[ "$PROFILE" == "guard" || "$PROFILE" == "nvidia" ]]; then
      export TENSORFOLD_MEMORY_RESERVE_GIB=6
      export TF_FLASHNEXT_DRAFTS="6"
      export TF_FLASHNEXT_DRAFTS_GREEDY="14"
      export TF_FLASHNEXT_CONFIDENCE="-0.35"
      CTX="$GUARD_CONTEXT"
      PAR="$GUARD_PARALLEL"
      THINK="--thinking"
    else
      export TF_FLASHNEXT_DRAFTS_GREEDY="15"
      export TF_FLASHNEXT_CONFIDENCE="-0.28"
      export TF_FLASHNEXT_EXT_ENTER="0.0"
      export TF_FLASHNEXT_EXT_STAY="0.0"
      export TF_FLASHNEXT_EXT_TOK="0.0"
      export TF_FLASHNEXT_EXT_MIN="0"
      export TF_FLASHNEXT_ROOT_COPIES="0"
      export TF_FLASHNEXT_P_CLIFF="0.0"
      export TF_FLASHNEXT_W2_MULTI="0"
      export TF_FLASHNEXT_ROUTER_BF12="0"
      export TF_FLASHNEXT_KV_PAGES="256"
      export TF_FLASHNEXT_KV_PAGES_ALL="1"
      export TF_NVFP4_HALF_WARPS="2"
      export TF_NVFP4_COOP_U4="1"
      export TF_NVFP4_HALF_MIN="1"
      export TF_NVFP4_FULL_R8_MAX="0"
      export TF_NVFP4_HALF4_MAX="320"
      export TF_BF12_S2_MAX_PER="12"
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
