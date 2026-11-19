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
BUILD_FLAVOR="mtp-w2f-nvfp4-g4-head4-bf12s3-dynexp-t24"
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

  # 1a. Append fused 2.25-bit E4M3 W2 proposal GEMV + single-block top-64 exact BF16 rescore-to-best kernels to flashnext_mtp.cu
  cat >> "$kern_dir/flashnext_mtp.cu" <<'CU'

extern "C" __global__ void __launch_bounds__(128) fn_w2_gemv_fused(
    const __nv_bfloat16* __restrict__ X,
    const uint4* __restrict__ W,
    const uint16_t* __restrict__ S,
    const float* __restrict__ G,
    __nv_bfloat16* __restrict__ out,
    float4* __restrict__ stats,
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
    const int row = blk * 32 + lane;
    float fv = -CUDART_INF_F;
    if (blk * 32 < V) {
        const uint4* wp = W + static_cast<size_t>(blk) * 40 * 32 + lane;
        const uint16_t* sp = S + static_cast<size_t>(blk) * 40 * 32 + lane;
        float acc = 0.0f;
        for (int g2 = 0; g2 < 40; ++g2) {
            const uint4 w4 = __ldg(wp + g2 * 32);
            const uint32_t s2 = __ldg(sp + g2 * 32);
            const uint32_t sx2 = (s2 & 0xFFu) | ((s2 & 0xFF00u) << 8);
            const uint32_t ts = ((sx2 & 0x007F007Fu) << 4) | ((sx2 & 0x00800080u) << 8);
            uint32_t bf2;
            asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(bf2) : "r"(ts), "r"(0x7B807B80u), "r"(0x80008000u));
            const float sc0 = __uint_as_float((bf2 & 0xFFFFu) << 16);
            const float sc1 = __uint_as_float(bf2 & 0xFFFF0000u);
            const int g0 = g2 * 2;
            const float* xg0 = sx + g0 * 32;
            float dot0 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) dot0 = fmaf(xg0[j], static_cast<float>((w4.x >> (2 * j)) & 3u), dot0);
            #pragma unroll
            for (int j = 0; j < 16; ++j) dot0 = fmaf(xg0[16 + j], static_cast<float>((w4.y >> (2 * j)) & 3u), dot0);
            acc = fmaf(sc0, dot0 - sg_sum[g0], acc);
            const float* xg1 = sx + (g0 + 1) * 32;
            float dot1 = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) dot1 = fmaf(xg1[j], static_cast<float>((w4.z >> (2 * j)) & 3u), dot1);
            #pragma unroll
            for (int j = 0; j < 16; ++j) dot1 = fmaf(xg1[16 + j], static_cast<float>((w4.w >> (2 * j)) & 3u), dot1);
            acc = fmaf(sc1, dot1 - sg_sum[g0 + 1], acc);
        }
        acc *= __ldg(G + row);
        if (row < V) {
            const __nv_bfloat16 bv = __float2bfloat16_rn(acc);
            out[row] = bv;
            fv = __bfloat162float(bv);
        }
    }
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
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W_full_bf16,
    const uint32_t* __restrict__ vocab_ids,
    __nv_bfloat16* __restrict__ dlogits,
    const float4* __restrict__ stats,
    int n_blocks,
    int V,
    float* __restrict__ best_out) {
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
            unsigned int cum = 0;
            int bc = 127;
            for (int b = 0; b <= 126; ++b) {
                if (cum + hist[b] >= 32u) {
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
        unsigned int cum = 0;
        int bc = 127;
        for (int b = 0; b <= 126; ++b) {
            if (cum + hist[b] >= 32u) {
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
            s_eq_need = 32u - cum;
        }
        n_cand = 0;
        n_eq_taken = 0;
    }
    __syncthreads();

    const int sc = s_strict_cut, bc = s_b_cut;
    const unsigned int eq_lim = s_eq_need;
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
                    if (slot < 32u) {
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
                if (slot < 32u) {
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

    const int K = static_cast<int>(min(n_cand, 32u));
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

// Single-pass vectorized uint4 argmax for greedy verify: zero expf() calls, bit-exact tie-breaking & finiteness check.
extern "C" __global__ void __launch_bounds__(1024) fn_row_argmax(
    const __nv_bfloat16* __restrict__ L,
    int V,
    long long ld,
    float* __restrict__ out) {
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

extern "C" __global__ void __launch_bounds__(256) fn_head_rescore(
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
    """    static constexpr int W = MODE == FP4 ? BN * GS / 2 : MODE == BF16 ? BN * GS * 2 : MODE == BF12 ? (BN * GS * 3) / 2 : BN * GS;
    static constexpr int S = MODE == FP4 || MODE == FP8G ? BN * 4 : MODE == MXFP8 ? BN * 2 : MODE == BF12 ? BN : 0;
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
#undef TF_SHARED_DOWN_BF12""",
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
BF12_INST(16, 64, 1, 4, 3, false, true, false)
BF12_INST(16, 64, 1, 4, 3, false, false, true)
BF12_INST(32, 64, 1, 4, 3, false, false, false)
BF12_INST(32, 64, 1, 4, 3, false, true, false)
BF12_INST(32, 64, 1, 4, 3, false, false, true)
BF12_INST(64, 64, 1, 4, 3, false, false, false)
BF12_INST(64, 64, 1, 4, 3, false, true, false)
BF12_INST(64, 64, 1, 4, 3, false, false, true)
BF12_INST(16, 64, 1, 4, 3, true, false, false)
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
                    if constexpr (MODE == FP8 || MODE == BF16) mma(acc[i][j], a[i], b0, b1);""",
    """                } else if constexpr (MODE == BF12) {
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
                        const int local_jj = (BN == 32) ? (((n0 & 32) ? 4 : 0) + jj_in_pw) : (jj_in_pw & 7);
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
                    if constexpr (MODE == FP8 || MODE == BF16 || MODE == BF12) mma(acc[i][j], a[i], b0, b1);""",
)

# Fast bit-exact bf16x2 E4M3 scale decode in qmmf_body.cuh (FP4) and nvfp4_experts.cu (+ mma0)
patch_k(
    "qmmf_body.cuh",
    """                        } else if constexpr (MODE == FP4) {
                            s0 = e4m3f(static_cast<uint8_t>(sq[j][0] >> (8 * blk)));
                            s1 = e4m3f(static_cast<uint8_t>(sq[j][1] >> (8 * blk)));""",
    """                        } else if constexpr (MODE == FP4) {
                            const uint32_t b2 = ((sq[j][0] >> (8 * blk)) & 0xFFu) | (((sq[j][1] >> (8 * blk)) & 0xFFu) << 8);
                            const uint32_t sp = tf_common::fp8pair_unit(b2, 0x7B807B80u);
                            s0 = __uint_as_float((sp & 0xFFFFu) << 16);
                            s1 = __uint_as_float(sp & 0xFFFF0000u);""",
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
#pragma unroll
    for (int j = 0; j < 4; ++j)
        *reinterpret_cast<float4*>(state_out + sbase + (size_t)(row0 + j) * DK + lane * 4) = make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);""",
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

  # 2a2. Selective MTP BF16 -> native SM121 NVFP4 encoder (o_proj + shared_expert only)
  cat > "$fn_dir/cuda_mtp_nvfp4.zig" <<'ZIG'
//! Re-codes selected MTP layer-0 BF16 projections (o_proj, shared_expert) to native SM121 NVFP4 at load.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const projection = @import("cuda_projection.zig");

pub fn enabled() bool {
    const raw = std.c.getenv("TF_FLASHNEXT_MTP_NVFP4") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "1");
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

pub const W2 = struct {
    codes: ?cuda.DeviceBuffer = null,
    scales: ?cuda.DeviceBuffer = null,
    globals: ?cuda.DeviceBuffer = null,
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
        if (self.globals) |*b| b.free();
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
        const n_blocks = (nb + 3) / 4;
        var scratch_buf = try cuda.DeviceBuffer.alloc(driver, @max(n_blocks * 16, 16384));
        errdefer scratch_buf.free();

        return .{
            .codes = codes_buf,
            .scales = scales_buf,
            .globals = globals_buf,
            .vocab_ids = ids_buf,
            .scratch = scratch_buf,
            .gemv_fn = try module.function("fn_w2_gemv_fused"),
            .rescore_fn = try module.function("fn_w2_rescore_best"),
        };
    }

    pub fn projectFused(self: *const W2, stream: cuda.Stream, input: u64, dlogits: u64, best_out: u64, v: usize) !void {
        const nb = (v + 31) / 32;
        const n_blocks = (nb + 3) / 4;
        const sc = self.scratch.?.ptr;
        var a1: cuda.Args = .{};
        a1.add(input);
        a1.add(self.codes.?.ptr);
        a1.add(self.scales.?.ptr);
        a1.add(self.globals.?.ptr);
        a1.add(dlogits);
        a1.add(sc);
        a1.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(self.gemv_fn.?, .{ .grid = .{ .x = @intCast(n_blocks) }, .block = .{ .x = 128 } }, stream, &a1);

        var a2: cuda.Args = .{};
        a2.add(input);
        a2.add(self.bf16_ptr);
        a2.add(self.vocab_ids.?.ptr);
        a2.add(dlogits);
        a2.add(sc);
        a2.add(@as(c_int, @intCast(n_blocks)));
        a2.add(@as(c_int, @intCast(v)));
        a2.add(best_out);
        try cuda.launch.launch(self.rescore_fn.?, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &a2);
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
    const s: u32 = if (mode == .fp4 or mode == .fp8g) bn * 4 else if (mode == .mxfp8) bn * 2 else 0;
    const stage = (bm * group * 2 + w + s + 127) / 128 * 128;
    const partials = (bm / 16) * 2 * 4 * threads * 4;
    return @max(4 * stage, partials);""",
    """        .bf16 => bn * group * 2,
        .bf12 => (bn * group * 3) / 2,
    };
    const s: u32 = if (mode == .fp4 or mode == .fp8g) bn * 4 else if (mode == .mxfp8) bn * 2 else if (mode == .bf12) bn else 0;
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
    const w: u32 = if (mode == .bf16) 32 * group * 2 else if (mode == .bf12) (32 * group * 3) / 2 else 32 * group;
    const s: u32 = if (mode == .mxfp8) 32 * 2 else if (mode == .bf12) 32 else 0;
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

# 5c. cuda_projection_load.zig + cuda_prompt_wt.zig: selective MTP o_proj/shared_expert -> NVFP4, other MTP BF16 + HC -> MXFP8
patch_exact(
    root, "cuda_projection_load.zig",
    "            return projection.dense(a, d, t);",
    """            if (std.mem.startsWith(u8, prefix, "language_model.mtp.layers.0.") and (std.mem.endsWith(u8, prefix, ".o_proj") or std.mem.indexOf(u8, prefix, ".shared_expert.") != null) and @import("cuda_mtp_nvfp4.zig").enabled())
                return @import("cuda_mtp_nvfp4.zig").project(a, d, t.bytes, n, k);
            if (std.mem.startsWith(u8, prefix, "language_model.mtp."))
                return @import("cuda_mtp_mxfp8.zig").project(a, d, t.bytes, n, k);
            return projection.dense(a, d, t);""",
)
patch_exact(
    root, "cuda_projection_load.zig",
    "        return projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = bytes });",
    """        if (std.mem.startsWith(u8, prefix, "language_model.mtp."))
            return @import("cuda_mtp_mxfp8.zig").project(a, d, bytes, n, k);
        return projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = bytes });""",
)

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
            var stop: mtp.Stop = mtp.Stop.init(confidence);
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

# 5f. cuda_mtp.zig + cuda_mtp_draw.zig + cuda_draw_batch.zig + cuda_batch_hc.zig + cuda_mtp_pieces.zig + cuda_draft_draw.zig + cuda_lanes_load.zig:
#     keep max_chain=15 (.first_copy_rows=16 inside BM=16), zero-expf fn_row_argmax, hc.fuses at n>=48, hook fused 2.25-bit W2 directly to d.best
patch_exact(
    root, "cuda_batch_hc.zig",
    "    return n >= 64 and hc_w.down.native != null",
    "    return n >= 48 and hc_w.down.native != null",
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

    pub fn init(c: f64) Stop {
        return .{
            .c = c,
            .ext_enter = envF64("TF_FLASHNEXT_EXT_ENTER", 0.0),
            .ext_stay = envF64("TF_FLASHNEXT_EXT_STAY", 0.0),
            .ext_tok = envF64("TF_FLASHNEXT_EXT_TOK", 0.0),
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
            // below the line it ends the chain: no later product can reach it again
            if (s.product < -s.c) return .{ .propose = j == 0, .more = false };
            if (j == 14 and s.ext_enter > 0 and s.product < s.ext_enter) return .{ .propose = true, .more = false };
        }
        return .{ .propose = true, .more = true };
    }
};""",
)
patch_exact(
    root, "cuda_mtp.zig",
    "    topk_fn: cuda.Function, // fn_row_topk: a row's top_k candidates as drawRow ranks them (cuda_mtp_draw.zig)",
    "    topk_fn: cuda.Function,\n    argmax_fn: cuda.Function,\n    w2: @import(\"cuda_mtp_w2.zig\").W2 = .{},",
)
patch_exact(
    root, "cuda_mtp.zig",
    "        h.best_fn = try h.module.function(\"fn_row_best\");\n        h.topk_fn = try h.module.function(\"fn_row_topk\");",
    "        h.best_fn = try h.module.function(\"fn_row_best\");\n        h.topk_fn = try h.module.function(\"fn_row_topk\");\n        h.argmax_fn = try h.module.function(\"fn_row_argmax\");",
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
    "        h.draw_batch = try @import(\"cuda_draw_batch.zig\").Batch.init(gpa, driver, batch_on, h.ids.len);\n        errdefer h.draw_batch.deinit(gpa);\n        h.w2 = try @import(\"cuda_mtp_w2.zig\").W2.init(gpa, driver, &store, &h.artifact, &h.module);\n        return h;",
)
patch_exact(
    root, "cuda_mtp.zig",
    "    pub fn deinit(h: *Head) void {\n        h.draw_batch.deinit(h.gpa);",
    "    pub fn deinit(h: *Head) void {\n        h.w2.deinit();\n        h.draw_batch.deinit(h.gpa);",
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
            const k_top: usize = 64;
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
            try cuda.launch.launch(self.k.head_rescore, .{ .grid = .{ .x = @intCast((k_top + 7) / 8), .y = @intCast(n) }, .block = .{ .x = 256 } }, self.stream, &a2);
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
    export TF_FLASHNEXT_MTP_NVFP4="1"
    export TF_FLASHNEXT_MTP_EXPERTS="nvfp4"
    export TF_FLASHNEXT_HEAD_NVFP4="1"
    export TF_FLASHNEXT_TRUNK_BF12="1"
    export TF_FLASHNEXT_FUSED_SPLIT="1"
    export TF_NVFP4_DECODE_WARPS="1"
    export TF_NVFP4_DYNAMIC_GRID="1"
    export TF_FLASHNEXT_ATTN_LANES="1"
    export TF_FLASHNEXT_WARM_AHEAD="1"
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
      export TF_FLASHNEXT_DRAFTS_GREEDY="15"
      export TF_FLASHNEXT_CONFIDENCE="-0.38"
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
