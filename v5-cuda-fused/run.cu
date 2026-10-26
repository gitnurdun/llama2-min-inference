/* Llama-2-7B inference on CUDA (Tesla T4, sm_75).
 *
 * v5: int4 (Q4_0-like) block-32 weights + dp4a kernels + fused operators +
 *     CUDA graph replay.  (An optional int8 KV cache is compiled off by
 *     default; see INT8_KV below.)
 *
 * Changes from v4:
 *  1. The M=1 int4 GEMM now loads 16 bytes (uint4 = 32 int4) per lane per
 *     iteration instead of 4, and owns a whole 32-block; K that is not a
 *     multiple of 1024 is finished with one masked wide iteration.  Measured
 *     ~226 -> ~245 GB/s on the w1/w3 shape.
 *  2. Operators are fused to cut kernel count and memory passes:
 *       rmsnorm + residual-add + activation-quantize   (one kernel)
 *       silu*up  + activation-quantize                 (one kernel)
 *       attention + output quantize                    (one kernel)
 *       rope + per-head quantize of K (int8 KV only)
 *     The FFN residual add is folded into the *next* layer's attention norm.
 *  3. The whole per-token forward is captured once into a CUDA graph.  The
 *     token id and absolute position are read from device memory (d_meta), so
 *     the graph is static and only a tiny 8-byte H2D copy changes per step.
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <cuda_runtime.h>
#ifndef _WIN32
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#define QBLOCK 32
#define QMAGIC 0x344C4C51u
#define QVERSION 2u

// int8 KV cache (1 byte per element + one fp32 scale per head) vs fp32 KV.
// Default OFF: measured at n=128/256 the int8 attention is latency-bound (one
// warp per head = 32 warps total) so its dequant ALU costs more than the 4x
// bandwidth saving, and it adds ~0.4% per-vector error.  Enable with
// -DINT8_KV=1 only for very long contexts; see lesson.md.
#ifndef INT8_KV
#define INT8_KV 0
#endif

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(EXIT_FAILURE); } } while (0)

// ----------------------------------------------------------------------------
// model

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct {         // device pointers
    uint8_t* q;          // (rows, cols/2) packed int4
    float* s;            // (rows, nb) block scales
    int rows, cols, nb;
} DQWeight;

typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    DQWeight wq, wk, wv, wo;
    float* rms_ffn_weight;
    DQWeight w1, w2, w3;
    float* rms_final_weight;
    DQWeight wcls;
} DTransformer;

typedef struct {
    int maxT;
    float *x;          // residual stream [dim]
    float *q;          // queries [dim]
    float *hb, *hb2;   // ffn intermediates [hidden]
    float *kf, *vf;    // float K/V row before int8 quantize [kv_dim]
    float *wo_out;     // attention output projection [dim]
    float *ffn_out;    // ffn output projection [dim]
    float *logits;     // [vocab]
    int8_t* actq;      // quantized activations [maxK]
    float* acts;       // activation scales [maxK/32]
    int8_t* outq;      // quantized attention output [dim]
    float* outscale;   // attention output scales [dim/32]
#if INT8_KV
    // int8 KV cache: one scale per (position, kv-head)
    int8_t* key_cache;   // [n_layers][maxT][kv_dim]
    int8_t* value_cache; // [n_layers][maxT][kv_dim]
    float* key_scale;    // [n_layers][maxT][n_kv_heads]
    float* value_scale;  // [n_layers][maxT][n_kv_heads]
#else
    float* key_cache;    // [n_layers][maxT][kv_dim]
    float* value_cache;  // [n_layers][maxT][kv_dim]
#endif
    int* d_meta;         // [token, pos]
} DState;

static cudaStream_t g_stream = 0;
static cudaGraph_t g_graph = 0;
static cudaGraphExec_t g_exec = 0;

// ----------------------------------------------------------------------------
// kernels

__global__ void embed_kernel(const float* __restrict__ emb, const int* __restrict__ meta,
                             float* __restrict__ x, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) x[i] = emb[(size_t)meta[0] * dim + i];
}

// rmsnorm + optional residual add + activate to int8 blocks.
//   v = x[i] + (resid ? resid[i] : 0)
//   if resid: xout[i] = v           (persist the residual stream)
//   q = w * v * rsqrt(mean(v^2)+eps), block-32 int8
__global__ void rmsnorm_quant_kernel(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     const float* __restrict__ resid,
                                     float* __restrict__ xout,
                                     int8_t* __restrict__ Xq,
                                     float* __restrict__ Xs, int n) {
    extern __shared__ float smem[];
    float* xs = smem;
    float* wsum = smem + n;               // one partial sum per warp
    int t = threadIdx.x, nt = blockDim.x;
    int warp = t >> 5, lane = t & 31;
    float ss = 0.0f;
    for (int i = t; i < n; i += nt) {
        float v = x[i] + (resid ? resid[i] : 0.0f);
        xs[i] = v;
        ss += v * v;
    }
    // warp-shuffle reduction (2 syncs total instead of log2(nt))
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    if (lane == 0) wsum[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        int nw = nt >> 5;
        ss = (lane < nw) ? wsum[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
        if (lane == 0) wsum[0] = ss;
    }
    __syncthreads();
    float inv = rsqrtf(wsum[0] / (float)n + 1e-5f);
    if (resid != NULL && xout != NULL) {
        for (int i = t; i < n; i += nt) xout[i] = xs[i];
    }
    int nb = n / QBLOCK;
    for (int b = t; b < nb; b += nt) {
        const float* xb = xs + b * QBLOCK;
        const float* wb = w + b * QBLOCK;
        float amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < QBLOCK; j++) {
            float a = fabsf(wb[j] * inv * xb[j]);
            if (a > amax) amax = a;
        }
        float s = amax / 127.0f;
        if (s == 0.0f) s = 1e-8f;
        int8_t* qo = Xq + b * QBLOCK;
        #pragma unroll
        for (int j = 0; j < QBLOCK; j++) {
            int q = __float2int_rn(wb[j] * inv * xb[j] / s);
            q = max(-127, min(127, q));
            qo[j] = (int8_t)q;
        }
        Xs[b] = s;
    }
}

// silu(hb) * hb2, quantized to int8 blocks.  One warp per 32-block so the grid
// is wide enough to use the whole GPU (the old one-thread-per-block version had
// a 2-block grid and sat on 2 SMs).
__global__ void swiglu_quant_kernel(const float* __restrict__ hb,
                                    const float* __restrict__ hb2,
                                    int8_t* __restrict__ Xq,
                                    float* __restrict__ Xs, int n) {
    int nb = n / QBLOCK;
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= nb) return;
    int idx = warp * QBLOCK + lane;
    float v = hb[idx];
    float g = (v / (1.0f + expf(-v))) * hb2[idx];
    float a = fabsf(g);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float s = a / 127.0f;
    if (s == 0.0f) s = 1e-8f;
    int q = __float2int_rn(g / s);
    q = max(-127, min(127, q));
    Xq[idx] = (int8_t)q;
    if (lane == 0) Xs[warp] = s;
}

// One warp per kv-head: optional RoPE then per-head symmetric int8 quantize
// straight into the cache row for the current position.
__global__ void kv_quant_kernel(const float* __restrict__ in,
                                int8_t* __restrict__ cache,   // [maxT][kv_dim]
                                float* __restrict__ scale,    // [maxT][n_kv_heads]
                                const int* __restrict__ meta,
                                int head_size, int n_kv_heads, int has_rope) {
    int h = blockIdx.x;
    int lane = threadIdx.x;
    int per = head_size / 32;              // elements per lane (assume mult. of 32)
    int np = per / 2;
    int pos = meta[1];
    int kv_dim = n_kv_heads * head_size;
    float vals[8];
    if (has_rope) {
        for (int t = 0; t < np; t++) {
            int p = lane * np + t;         // pair index within head
            int i = 2 * p;                 // element index within head (even)
            float freq = 1.0f / powf(10000.0f, (float)i / (float)head_size);
            float ang = (float)pos * freq;
            float c = cosf(ang), s = sinf(ang);
            float v0 = in[h * head_size + i];
            float v1 = in[h * head_size + i + 1];
            vals[2 * t]     = v0 * c - v1 * s;
            vals[2 * t + 1] = v0 * s + v1 * c;
        }
    } else {
        for (int t = 0; t < per; t++) {
            vals[t] = in[h * head_size + lane * per + t];
        }
    }
    float amax = 0.0f;
    for (int t = 0; t < per; t++) {
        float a = fabsf(vals[t]);
        if (a > amax) amax = a;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float s = amax / 127.0f;
    if (s == 0.0f) s = 1e-8f;
    int8_t* cr = cache + (size_t)pos * kv_dim + h * head_size;
    for (int t = 0; t < per; t++) {
        int q = __float2int_rn(vals[t] / s);
        q = max(-127, min(127, q));
        cr[lane * per + t] = (int8_t)q;
    }
    if (lane == 0) scale[(size_t)pos * n_kv_heads + h] = s;
}

// Unpack four packed int4 words into eight sign-extended int8 groups.
__device__ __forceinline__ void unpack4(uint4 w, int g[8]) {
    uint32_t l0 = __vsub4((w.x & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
    uint32_t h0 = __vsub4(((w.x >> 4) & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
    uint32_t l1 = __vsub4((w.y & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
    uint32_t h1 = __vsub4(((w.y >> 4) & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
    uint32_t l2 = __vsub4((w.z & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
    uint32_t h2 = __vsub4(((w.z >> 4) & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
    uint32_t l3 = __vsub4((w.w & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
    uint32_t h3 = __vsub4(((w.w >> 4) & 0x0f0f0f0fu) ^ 0x08080808u, 0x08080808u);
    g[0] = (int)__byte_perm(l0, h0, 0x5140);
    g[1] = (int)__byte_perm(l0, h0, 0x7362);
    g[2] = (int)__byte_perm(l1, h1, 0x5140);
    g[3] = (int)__byte_perm(l1, h1, 0x7362);
    g[4] = (int)__byte_perm(l2, h2, 0x5140);
    g[5] = (int)__byte_perm(l2, h2, 0x7362);
    g[6] = (int)__byte_perm(l3, h3, 0x5140);
    g[7] = (int)__byte_perm(l3, h3, 0x7362);
}

// Single-token (M=1) int4 GEMM: one warp per output row, 32 lanes split K.
// Each lane loads one uint4 = 32 int4 = exactly one 32-block, so a single block
// scale applies and the lane issues 8 dp4a.  K that is not a multiple of 1024
// is finished by one *masked* wide iteration (24 active lanes for hidden=11008)
// instead of dropping to narrow 4-byte loads.
__global__ void matmul_q4_vec_kernel(const int8_t* __restrict__ Xq,
                                     const float* __restrict__ Xs,
                                     const uint8_t* __restrict__ Wq,
                                     const float* __restrict__ Ws,
                                     float* __restrict__ C, int N, int K) {
    int n = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (n >= N) return;
    const uint8_t* wr = Wq + (size_t)n * (K / 2);
    const float* wsr = Ws + (size_t)n * (K / 32);
    float acc = 0.0f;
    int k0 = 0;
    for (; k0 + 1024 <= K; k0 += 1024) {
        uint4 w = *reinterpret_cast<const uint4*>(wr + (k0 >> 1) + 16 * lane);
        int g[8];
        unpack4(w, g);
        const int8_t* xp = Xq + k0 + 32 * lane;
        int4 a = *reinterpret_cast<const int4*>(xp);
        int4 b = *reinterpret_cast<const int4*>(xp + 16);
        int d = 0;
        d = __dp4a(g[0], a.x, d);
        d = __dp4a(g[1], a.y, d);
        d = __dp4a(g[2], a.z, d);
        d = __dp4a(g[3], a.w, d);
        d = __dp4a(g[4], b.x, d);
        d = __dp4a(g[5], b.y, d);
        d = __dp4a(g[6], b.z, d);
        d = __dp4a(g[7], b.w, d);
        int blk = (k0 >> 5) + lane;
        acc += wsr[blk] * Xs[blk] * (float)d;
    }
    int rem = K - k0;                  // < 1024, a multiple of 32
    if (lane < (rem >> 5)) {
        uint4 w = *reinterpret_cast<const uint4*>(wr + (k0 >> 1) + 16 * lane);
        int g[8];
        unpack4(w, g);
        const int8_t* xp = Xq + k0 + 32 * lane;
        int4 a = *reinterpret_cast<const int4*>(xp);
        int4 b = *reinterpret_cast<const int4*>(xp + 16);
        int d = 0;
        d = __dp4a(g[0], a.x, d);
        d = __dp4a(g[1], a.y, d);
        d = __dp4a(g[2], a.z, d);
        d = __dp4a(g[3], a.w, d);
        d = __dp4a(g[4], b.x, d);
        d = __dp4a(g[5], b.y, d);
        d = __dp4a(g[6], b.z, d);
        d = __dp4a(g[7], b.w, d);
        int blk = (k0 >> 5) + lane;
        acc += wsr[blk] * Xs[blk] * (float)d;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) C[n] = acc;
}

__global__ void rope_kernel(float* __restrict__ vec, int vec_dim,
                            int head_size, const int* __restrict__ meta) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vec_dim / 2) return;
    int i = 2 * (int)idx;
    int hd = i % head_size;
    int t = meta[1];
    float freq = 1.0f / powf(10000.0f, (float)hd / (float)head_size);
    float val = (float)t * freq;
    float c = cosf(val), s = sinf(val);
    float v0 = vec[i], v1 = vec[i + 1];
    vec[i]     = v0 * c - v1 * s;
    vec[i + 1] = v0 * s + v1 * c;
}

// One warp per head.  Reads int8 K/V (with per-(pos,kv-head) scales), fuses
// softmax, and writes the output already quantized to block-32 int8 for the
// following output projection.
__global__ void attention_cache_kernel(const float* __restrict__ q,
                                       const int8_t* __restrict__ kcache,
                                       const int8_t* __restrict__ vcache,
                                       const float* __restrict__ ksbase,
                                       const float* __restrict__ vsbase,
                                       int8_t* __restrict__ outq,
                                       float* __restrict__ outscale,
                                       const int* __restrict__ meta,
                                       int n_heads, int n_kv_heads,
                                       int head_size, int kv_mul) {
    int h = blockIdx.x;
    int lane = threadIdx.x;
    int pos = meta[1];
    extern __shared__ float sc[];
    int kv_dim = n_kv_heads * head_size;
    const float* qh = q + h * head_size;
    int kh = h / kv_mul;
    float scale = rsqrtf((float)head_size);
    for (int s = 0; s <= pos; s++) {
        const int8_t* kp = kcache + (size_t)s * kv_dim + kh * head_size;
        float part = 0.0f;
        for (int j = lane; j < head_size; j += 32) part += qh[j] * (float)kp[j];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
        if (lane == 0) sc[s] = part * scale * ksbase[(size_t)s * n_kv_heads + kh];
    }
    __syncwarp();
    float m = -1e30f;
    for (int s = lane; s <= pos; s += 32) m = fmaxf(m, sc[s]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    m = __shfl_sync(0xffffffffu, m, 0);
    float local = 0.0f;
    for (int s = lane; s <= pos; s += 32) { float e = expf(sc[s] - m); sc[s] = e; local += e; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(0xffffffffu, local, o);
    local = __shfl_sync(0xffffffffu, local, 0);
    float inv = 1.0f / local;
    int per = head_size / 32;
    int j0 = lane * per;
    int vh = kh * head_size;
    float oacc[8];
    for (int k = 0; k < per; k++) oacc[k] = 0.0f;
    for (int s = 0; s <= pos; s++) {
        float pv = sc[s] * vsbase[(size_t)s * n_kv_heads + kh];
        const int8_t* vp = vcache + (size_t)s * kv_dim + vh + j0;
        for (int k = 0; k < per; k++) oacc[k] += pv * (float)vp[k];
    }
    float amax = 0.0f;
    for (int k = 0; k < per; k++) { float a = fabsf(oacc[k]); if (a > amax) amax = a; }
    #pragma unroll
    for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float scl = amax / 127.0f;
    if (scl == 0.0f) scl = 1e-8f;
    for (int k = 0; k < per; k++) {
        int qv = __float2int_rn(oacc[k] * inv / scl);
        qv = max(-127, min(127, qv));
        outq[h * head_size + j0 + k] = (int8_t)qv;
    }
    if (lane % 8 == 0) outscale[h * (head_size / 32) + lane / 8] = scl;
}

// fp32 KV variants (used when INT8_KV=0 for the quality comparison).
__global__ void kv_store_kernel(const float* __restrict__ in,
                                float* __restrict__ cache,
                                const int* __restrict__ meta,
                                int head_size, int n_kv_heads, int has_rope) {
    int h = blockIdx.x;
    int lane = threadIdx.x;
    int per = head_size / 32;
    int np = per / 2;
    int pos = meta[1];
    int kv_dim = n_kv_heads * head_size;
    float* cr = cache + (size_t)pos * kv_dim + h * head_size;
    if (has_rope) {
        for (int t = 0; t < np; t++) {
            int p = lane * np + t;
            int i = 2 * p;
            float freq = 1.0f / powf(10000.0f, (float)i / (float)head_size);
            float ang = (float)pos * freq;
            float c = cosf(ang), s = sinf(ang);
            float v0 = in[h * head_size + i];
            float v1 = in[h * head_size + i + 1];
            cr[lane * per + 2 * t]     = v0 * c - v1 * s;
            cr[lane * per + 2 * t + 1] = v0 * s + v1 * c;
        }
    } else {
        for (int t = 0; t < per; t++) cr[lane * per + t] = in[h * head_size + lane * per + t];
    }
}

__global__ void attention_f32_kernel(const float* __restrict__ q,
                                     const float* __restrict__ kcache,
                                     const float* __restrict__ vcache,
                                     int8_t* __restrict__ outq,
                                     float* __restrict__ outscale,
                                     const int* __restrict__ meta,
                                     int n_heads, int n_kv_heads,
                                     int head_size, int kv_mul) {
    int h = blockIdx.x;
    int lane = threadIdx.x;
    int pos = meta[1];
    extern __shared__ float sc[];
    int kv_dim = n_kv_heads * head_size;
    const float* qh = q + h * head_size;
    int kh = h / kv_mul;
    float scale = rsqrtf((float)head_size);
    for (int s = 0; s <= pos; s++) {
        const float* kp = kcache + (size_t)s * kv_dim + kh * head_size;
        float part = 0.0f;
        for (int j = lane; j < head_size; j += 32) part += qh[j] * kp[j];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
        if (lane == 0) sc[s] = part * scale;
    }
    __syncwarp();
    float m = -1e30f;
    for (int s = lane; s <= pos; s += 32) m = fmaxf(m, sc[s]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    m = __shfl_sync(0xffffffffu, m, 0);
    float local = 0.0f;
    for (int s = lane; s <= pos; s += 32) { float e = expf(sc[s] - m); sc[s] = e; local += e; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(0xffffffffu, local, o);
    local = __shfl_sync(0xffffffffu, local, 0);
    float inv = 1.0f / local;
    int per = head_size / 32;
    int j0 = lane * per;
    int vh = kh * head_size;
    float oacc[8];
    for (int k = 0; k < per; k++) oacc[k] = 0.0f;
    for (int s = 0; s <= pos; s++) {
        float pv = sc[s];
        const float* vp = vcache + (size_t)s * kv_dim + vh + j0;
        for (int k = 0; k < per; k++) oacc[k] += pv * vp[k];
    }
    float amax = 0.0f;
    for (int k = 0; k < per; k++) { float a = fabsf(oacc[k]); if (a > amax) amax = a; }
    #pragma unroll
    for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float scl = amax / 127.0f;
    if (scl == 0.0f) scl = 1e-8f;
    for (int k = 0; k < per; k++) {
        int qv = __float2int_rn(oacc[k] * inv / scl);
        qv = max(-127, min(127, qv));
        outq[h * head_size + j0 + k] = (int8_t)qv;
    }
    if (lane % 8 == 0) outscale[h * (head_size / 32) + lane / 8] = scl;
}

// ----------------------------------------------------------------------------
// host helpers

static DQWeight dslice(const DQWeight* w, int l, int layer_rows) {
    DQWeight r;
    r.q = w->q + (size_t)l * layer_rows * (w->cols / 2); // int4: 2 values/byte
    r.s = w->s + (size_t)l * layer_rows * w->nb;
    r.rows = layer_rows;
    r.cols = w->cols;
    r.nb = w->nb;
    return r;
}

static float* dalloc_f32(uint8_t** ptr, size_t n) {
    float* d = NULL;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, *ptr, n * sizeof(float), cudaMemcpyHostToDevice));
    *ptr += n * sizeof(float);
    return d;
}

static void dalloc_q4(uint8_t** ptr, DQWeight* w, long rows, int cols) {
    long nb = cols / QBLOCK;
    size_t packed = (size_t)rows * (cols / 2);
    CUDA_CHECK(cudaMalloc((void**)&w->q, packed));
    CUDA_CHECK(cudaMemcpy(w->q, *ptr, packed, cudaMemcpyHostToDevice));
    *ptr += packed;
    w->s = dalloc_f32(ptr, (size_t)rows * nb);
    w->rows = rows;
    w->cols = cols;
    w->nb = nb;
}

static void load_weights(DTransformer* w, Config* p, uint8_t* base) {
    uint8_t* ptr = base + 2 * sizeof(uint32_t) + sizeof(Config);
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    unsigned long long L = p->n_layers;
    w->token_embedding_table = dalloc_f32(&ptr, (size_t)p->vocab_size * p->dim);
    w->rms_att_weight = dalloc_f32(&ptr, L * p->dim);
    dalloc_q4(&ptr, &w->wq, L * p->dim, p->dim);
    dalloc_q4(&ptr, &w->wk, L * kv_dim, p->dim);
    dalloc_q4(&ptr, &w->wv, L * kv_dim, p->dim);
    dalloc_q4(&ptr, &w->wo, L * p->dim, p->dim);
    w->rms_ffn_weight = dalloc_f32(&ptr, L * p->dim);
    dalloc_q4(&ptr, &w->w1, L * p->hidden_dim, p->dim);
    dalloc_q4(&ptr, &w->w2, L * p->dim, p->hidden_dim);
    dalloc_q4(&ptr, &w->w3, L * p->hidden_dim, p->dim);
    w->rms_final_weight = dalloc_f32(&ptr, p->dim);
    dalloc_q4(&ptr, &w->wcls, p->vocab_size, p->dim);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void alloc_state(DState* s, Config* p, int maxT) {
    int dim = p->dim, hidden = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int maxK = hidden > dim ? hidden : dim;
    size_t f = sizeof(float);
    CUDA_CHECK(cudaMalloc(&s->x,      (size_t)dim * f));
    CUDA_CHECK(cudaMalloc(&s->q,      (size_t)dim * f));
    CUDA_CHECK(cudaMalloc(&s->hb,     (size_t)hidden * f));
    CUDA_CHECK(cudaMalloc(&s->hb2,    (size_t)hidden * f));
    CUDA_CHECK(cudaMalloc(&s->kf,     (size_t)kv_dim * f));
    CUDA_CHECK(cudaMalloc(&s->vf,     (size_t)kv_dim * f));
    CUDA_CHECK(cudaMalloc(&s->wo_out, (size_t)dim * f));
    CUDA_CHECK(cudaMalloc(&s->ffn_out,(size_t)dim * f));
    CUDA_CHECK(cudaMalloc(&s->logits, (size_t)p->vocab_size * f));
    CUDA_CHECK(cudaMalloc(&s->actq,   (size_t)maxK * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&s->acts,   (size_t)(maxK / QBLOCK) * f));
    CUDA_CHECK(cudaMalloc(&s->outq,   (size_t)dim * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&s->outscale, (size_t)(dim / QBLOCK) * f));
#if INT8_KV
    CUDA_CHECK(cudaMalloc(&s->key_cache,   (size_t)p->n_layers * maxT * kv_dim * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&s->value_cache, (size_t)p->n_layers * maxT * kv_dim * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&s->key_scale,   (size_t)p->n_layers * maxT * p->n_kv_heads * f));
    CUDA_CHECK(cudaMalloc(&s->value_scale, (size_t)p->n_layers * maxT * p->n_kv_heads * f));
#else
    CUDA_CHECK(cudaMalloc(&s->key_cache,   (size_t)p->n_layers * maxT * kv_dim * f));
    CUDA_CHECK(cudaMalloc(&s->value_cache, (size_t)p->n_layers * maxT * kv_dim * f));
#endif
    CUDA_CHECK(cudaMalloc(&s->d_meta, 2 * sizeof(int)));
    s->maxT = maxT;
}

static void launch_matmul(float* C, const int8_t* Xq, const float* Xs, DQWeight W) {
    dim3 block(32, 8);
    dim3 grid((W.rows + block.y - 1) / block.y);
    matmul_q4_vec_kernel<<<grid, block, 0, g_stream>>>(Xq, Xs, W.q, W.s, C, W.rows, W.cols);
}

static void launch_rmsnorm_quant(const float* x, const float* w, const float* resid,
                                 float* xout, int8_t* Xq, float* Xs, int n) {
    int th = 256;
    size_t sh = ((size_t)n + 32) * sizeof(float);   // x scratch + 32 warp sums
    rmsnorm_quant_kernel<<<1, th, sh, g_stream>>>(x, w, resid, xout, Xq, Xs, n);
}

// ----------------------------------------------------------------------------
// forward (launches only; used for both capture and replay)

static void forward_launch(DTransformer* w, DState* s, Config* p) {
    int dim = p->dim, hidden = p->hidden_dim;
    int n_heads = p->n_heads, n_kv_heads = p->n_kv_heads;
    int head_size = dim / n_heads;
    int kv_dim = n_kv_heads * head_size;
    int kv_mul = n_heads / n_kv_heads;
    int maxT = s->maxT;

    embed_kernel<<<(dim + 255) / 256, 256, 0, g_stream>>>(
        w->token_embedding_table, s->d_meta, s->x, dim);

    for (int l = 0; l < p->n_layers; l++) {
        // attention norm; the FFN residual of the previous layer is added here
        // (and persisted into x, so the residual stream stays correct)
        const float* resid = (l == 0) ? NULL : s->ffn_out;
        float* xout = (l == 0) ? NULL : s->x;
        launch_rmsnorm_quant(s->x, w->rms_att_weight + (size_t)l * dim,
                             resid, xout, s->actq, s->acts, dim);

        launch_matmul(s->q,  s->actq, s->acts, dslice(&w->wq, l, dim));
        launch_matmul(s->kf, s->actq, s->acts, dslice(&w->wk, l, kv_dim));
        launch_matmul(s->vf, s->actq, s->acts, dslice(&w->wv, l, kv_dim));

        rope_kernel<<<(dim / 2 + 255) / 256, 256, 0, g_stream>>>(
            s->q, dim, head_size, s->d_meta);

#if INT8_KV
        int8_t* kcache = s->key_cache + (size_t)l * maxT * kv_dim;
        int8_t* vcache = s->value_cache + (size_t)l * maxT * kv_dim;
        float* ksbase = s->key_scale + (size_t)l * maxT * n_kv_heads;
        float* vsbase = s->value_scale + (size_t)l * maxT * n_kv_heads;
        kv_quant_kernel<<<n_kv_heads, 32, 0, g_stream>>>(
            s->kf, kcache, ksbase, s->d_meta, head_size, n_kv_heads, 1);
        kv_quant_kernel<<<n_kv_heads, 32, 0, g_stream>>>(
            s->vf, vcache, vsbase, s->d_meta, head_size, n_kv_heads, 0);
        attention_cache_kernel<<<n_heads, 32, (size_t)maxT * sizeof(float), g_stream>>>(
            s->q, kcache, vcache, ksbase, vsbase, s->outq, s->outscale,
            s->d_meta, n_heads, n_kv_heads, head_size, kv_mul);
#else
        float* kcache = s->key_cache + (size_t)l * maxT * kv_dim;
        float* vcache = s->value_cache + (size_t)l * maxT * kv_dim;
        kv_store_kernel<<<n_kv_heads, 32, 0, g_stream>>>(
            s->kf, kcache, s->d_meta, head_size, n_kv_heads, 1);
        kv_store_kernel<<<n_kv_heads, 32, 0, g_stream>>>(
            s->vf, vcache, s->d_meta, head_size, n_kv_heads, 0);
        attention_f32_kernel<<<n_heads, 32, (size_t)maxT * sizeof(float), g_stream>>>(
            s->q, kcache, vcache, s->outq, s->outscale,
            s->d_meta, n_heads, n_kv_heads, head_size, kv_mul);
#endif

        launch_matmul(s->wo_out, s->outq, s->outscale, dslice(&w->wo, l, dim));

        // ffn norm; attention output projection is the residual add
        launch_rmsnorm_quant(s->x, w->rms_ffn_weight + (size_t)l * dim,
                             s->wo_out, s->x, s->actq, s->acts, dim);
        launch_matmul(s->hb,  s->actq, s->acts, dslice(&w->w1, l, hidden));
        launch_matmul(s->hb2, s->actq, s->acts, dslice(&w->w3, l, hidden));

        int nb = hidden / QBLOCK;
        int nthreads = nb * 32;                 // one warp per 32-block
        swiglu_quant_kernel<<<(nthreads + 255) / 256, 256, 0, g_stream>>>(
            s->hb, s->hb2, s->actq, s->acts, hidden);
        launch_matmul(s->ffn_out, s->actq, s->acts, dslice(&w->w2, l, dim));
    }

    // final norm; the last FFN output is the pending residual
    launch_rmsnorm_quant(s->x, w->rms_final_weight, s->ffn_out, s->x,
                         s->actq, s->acts, dim);
    launch_matmul(s->logits, s->actq, s->acts, w->wcls);
}

static void build_graph(DTransformer* w, DState* s, Config* p) {
    int meta[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(s->d_meta, meta, sizeof(meta), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamCreate(&g_stream));
    CUDA_CHECK(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal));
    forward_launch(w, s, p);
    CUDA_CHECK(cudaStreamEndCapture(g_stream, &g_graph));
    CUDA_CHECK(cudaGraphInstantiate(&g_exec, g_graph, NULL, NULL, 0));
}

// Run one token at absolute position `pos`; optionally fetch the logits.
static void run_token(DTransformer* w, DState* s, Config* p, int token, int pos,
                      int want_logits, float* h_logits) {
    int meta[2] = { token, pos };
    CUDA_CHECK(cudaMemcpyAsync(s->d_meta, meta, sizeof(meta),
                               cudaMemcpyHostToDevice, g_stream));
    CUDA_CHECK(cudaGraphLaunch(g_exec, g_stream));
    if (want_logits) {
        CUDA_CHECK(cudaMemcpyAsync(h_logits, s->logits,
                                   (size_t)p->vocab_size * sizeof(float),
                                   cudaMemcpyDeviceToHost, g_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    (void)w;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer

typedef struct { char *str; int id; } TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { exit(EXIT_FAILURE); }
        if (fread(&len, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) { return; }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    char* str_buffer = (char*)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) { str_len = 0; }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) { continue; }
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }
    while (1) {
        float best_score = -1e10;
        int best_id = -1, best_idx = -1;
        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) { break; }
        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) { tokens[i] = tokens[i+1]; }
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

// ----------------------------------------------------------------------------
// Sampler (CPU)

typedef struct { float prob; int index; } ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) { return i; }
    }
    return n - 1;
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) { last_idx = i; break; }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) { return probindex[i].index; }
    }
    return probindex[last_idx].index;
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) { if (x[i] > max_val) max_val = x[i]; }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) { x[i] /= sum; }
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = (ProbIndex*)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// chat loop

int read_stdin(const char* guide, char* buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) == NULL) return 0;
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') { buffer[len - 1] = '\0'; }
    return 1;
}

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation (one token per forward, int8 KV cache, CUDA graph)

void generate_gpu(DTransformer* w, DState* s, Config* p, Tokenizer* tk, Sampler* sm,
                  char* prompt, int steps) {
    char* empty_prompt = "";
    if (prompt == NULL) prompt = empty_prompt;

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tk, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    float* h_logits = (float*)malloc(p->vocab_size * sizeof(float));
    long start = 0;
    int next = 0;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        int last_prompt = (pos >= num_prompt_tokens - 1);
        run_token(w, s, p, token, pos, last_prompt, h_logits);
        if (!last_prompt) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sm, h_logits);
        }
        pos++;
        if (next == 1) break;
        char* piece = decode(tk, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;
        if (start == 0) start = time_in_ms();
    }
    printf("\n");
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }
    free(prompt_tokens);
    free(h_logits);
}

void chat_gpu(DTransformer* w, DState* s, Config* p, Tokenizer* tk, Sampler* sm,
              char* cli_user_prompt, char* cli_system_prompt, int steps) {
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    float* h_logits = (float*)malloc(p->vocab_size * sizeof(float));

    int user_idx = 0;
    int8_t user_turn = 1;
    int next = 0;
    int token = 0;
    int pos = 0;
     
    while (pos < steps) {
        if (user_turn) {
            if (pos == 0) {
                if (cli_system_prompt == NULL) {
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            if (pos == 0 && cli_user_prompt != NULL) {
                strcpy(user_prompt, cli_user_prompt);
            } else if (!read_stdin("User: ", user_prompt, sizeof(user_prompt))) {
                break;
            }
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            encode(tk, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0;
            user_turn = 0;
            printf("Assistant: ");
        }

        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }
        if (token == 2) { user_turn = 1; }

        int last_prompt = (user_idx >= num_prompt_tokens);
        run_token(w, s, p, token, pos, 1, h_logits);
        next = sample(sm, h_logits);
        pos++;

        if (last_prompt && next != 2) {
            char* piece = decode(tk, token, next);
            safe_printf(piece);
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
    free(h_logits);
}

// ----------------------------------------------------------------------------
// main

void error_usage() {
    fprintf(stderr, "Usage:   run_cuda <checkpoint> [options]\n");
    fprintf(stderr, "Example: run_cuda llama2-7b-q4.bin -n 64 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature, default 1.0\n");
    fprintf(stderr, "  -p <float>  top-p, default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps, default 64\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> tokenizer path\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default generate\n");
    fprintf(stderr, "  -y <string> system prompt (chat)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 64;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = "generate";
    char *system_prompt = NULL;

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) { fprintf(stderr, "no CUDA device\n"); exit(EXIT_FAILURE); }
    int best_dev = -1;
    size_t best_free = 0;
    for (int i = 0; i < ndev; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) == cudaSuccess && f > best_free) { best_free = f; best_dev = i; }
    }
    CUDA_CHECK(cudaSetDevice(best_dev));
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    fprintf(stderr, "using GPU %d (%.1f GB free / %.1f GB)\n", best_dev, f / 1073741824.0, t / 1073741824.0);

    FILE* file = fopen(checkpoint_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open %s\n", checkpoint_path); exit(EXIT_FAILURE); }
    uint32_t magic = 0, version = 0;
    if (fread(&magic, 4, 1, file) != 1 || fread(&version, 4, 1, file) != 1) exit(EXIT_FAILURE);
    if (magic != QMAGIC || version != QVERSION) { fprintf(stderr, "bad checkpoint\n"); exit(EXIT_FAILURE); }
    Config config;
    if (fread(&config, sizeof(Config), 1, file) != 1) exit(EXIT_FAILURE);
    int shared_weights = config.vocab_size > 0;
    config.vocab_size = abs(config.vocab_size);
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fclose(file);

    int fd = open(checkpoint_path, O_RDONLY);
    if (fd == -1) { fprintf(stderr, "open failed\n"); exit(EXIT_FAILURE); }
    uint8_t* host_data = (uint8_t*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (host_data == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); exit(EXIT_FAILURE); }

    if (shared_weights) { fprintf(stderr, "shared weights unsupported\n"); exit(EXIT_FAILURE); }

    DTransformer w;
    load_weights(&w, &config, host_data);

    if (steps == 0 || steps > config.seq_len) steps = config.seq_len;

    DState s;
    alloc_state(&s, &config, steps);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, config.vocab_size);
    Sampler sampler;
    build_sampler(&sampler, config.vocab_size, temperature, topp, rng_seed);

    build_graph(&w, &s, &config);

    if (strcmp(mode, "generate") == 0) {
        generate_gpu(&w, &s, &config, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat_gpu(&w, &s, &config, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    return 0;
}
