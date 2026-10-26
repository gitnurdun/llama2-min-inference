/* Llama-2-7B inference on CUDA (Tesla T4, sm_75).
 *
 * v4: int4 (Q4_0-like) block-32 weights + dp4a kernels + KV cache.
 * v3 was already bandwidth bound (~242 GB/s of the T4's ~320), so the only way
 * to go faster is fewer bytes: int4 halves the weight stream.  Activations
 * stay int8.  Packed int4 is unpacked to int8 with byte tricks and fed to
 * __dp4a; the extra ALU work is far below the memory time, so int4 is still
 * bandwidth bound.
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
    float *x, *xb, *xb2, *q, *hb, *hb2, *logits;
    float* key_cache;    // [n_layers][maxT][kv_dim]
    float* value_cache;  // [n_layers][maxT][kv_dim]
    int8_t* actq;
    float* acts;
    int* d_tokens;
} DState;

// ----------------------------------------------------------------------------
// kernels

__global__ void embed_kernel(const float* __restrict__ emb, const int* __restrict__ tok,
                             float* __restrict__ x, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (i < dim) {
        x[(size_t)t * dim + i] = emb[(size_t)tok[t] * dim + i];
    }
}

__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w,
                               float* __restrict__ o, int n) {
    int t = blockIdx.x;
    const float* xt = x + (size_t)t * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    __shared__ float red[256];
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / n + 1e-5f);
    float* ot = o + (size_t)t * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        ot[i] = w[i] * (inv * xt[i]);
    }
}

__global__ void quantize_act_kernel(const float* __restrict__ X, int K,
                                    int8_t* __restrict__ Xq, float* __restrict__ Xs,
                                    int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int nb = K / QBLOCK;
    int m = idx / nb;
    int b = idx % nb;
    const float* xb = X + (size_t)m * K + b * QBLOCK;
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < QBLOCK; j++) {
        float a = fabsf(xb[j]);
        if (a > amax) amax = a;
    }
    float s = amax / 127.0f;
    if (s == 0.0f) s = 1e-8f;
    int8_t* qo = Xq + (size_t)m * K + b * QBLOCK;
    #pragma unroll
    for (int j = 0; j < QBLOCK; j++) {
        int q = __float2int_rn(xb[j] / s);
        q = max(-127, min(127, q));
        qo[j] = (int8_t)q;
    }
    Xs[m * nb + b] = s;
}

// weights are stored signed in device memory (converted once at load), so no
// per-element un-bias here; int4 gives 16-byte coalesced loads.
__global__ void matmul_q8_kernel(const int8_t* __restrict__ Xq,
                                 const float* __restrict__ Xs,
                                 const int8_t* __restrict__ Wq,
                                 const float* __restrict__ Ws,
                                 float* __restrict__ C,
                                 int T, int N, int K) {
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || m >= T) return;
    int nb = K / QBLOCK;
    const int8_t* wr = Wq + (size_t)n * K;
    const float* wsr = Ws + (size_t)n * nb;
    const int8_t* xr = Xq + (size_t)m * K;
    const float* xsr = Xs + (size_t)m * nb;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        const int4* wp = reinterpret_cast<const int4*>(wr + b * QBLOCK);
        const int4* xp = reinterpret_cast<const int4*>(xr + b * QBLOCK);
        int4 w0 = wp[0], x0 = xp[0], w1 = wp[1], x1 = xp[1];
        int d0 = 0, d1 = 0;
        d0 = __dp4a(w0.x, x0.x, d0);
        d0 = __dp4a(w0.y, x0.y, d0);
        d0 = __dp4a(w0.z, x0.z, d0);
        d0 = __dp4a(w0.w, x0.w, d0);
        d1 = __dp4a(w1.x, x1.x, d1);
        d1 = __dp4a(w1.y, x1.y, d1);
        d1 = __dp4a(w1.z, x1.z, d1);
        d1 = __dp4a(w1.w, x1.w, d1);
        sum += wsr[b] * xsr[b] * (float)(d0 + d1);
    }
    C[(size_t)m * N + n] = sum;
}

// Single-token (M=1) int4 GEMM: one warp per output row, 32 lanes split K.
// Each lane loads 4 packed bytes = 8 int4, sign-extends the nibbles with
// byte tricks and does two dp4a against 8 int8 activations.  A lane's 8 int4
// all live in one 32-block, so the block scale is applied right away.
__global__ void matmul_q4_vec_kernel(const int8_t* __restrict__ Xq,
                                     const float* __restrict__ Xs,
                                     const uint8_t* __restrict__ Wq,
                                     const float* __restrict__ Ws,
                                     float* __restrict__ C, int N, int K) {
    int n = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (n >= N) return;
    int nb = K / 32;
    const uint8_t* wr = Wq + (size_t)n * (K / 2);
    const float* wsr = Ws + (size_t)n * nb;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 256) {
        uint32_t w = *reinterpret_cast<const uint32_t*>(wr + (k0 >> 1) + 4 * lane);
        uint32_t lo = w & 0x0f0f0f0fu;
        uint32_t hi = (w >> 4) & 0x0f0f0f0fu;
        // sign-extend packed 4-bit two's-complement values to 8-bit bytes
        lo = __vsub4(lo ^ 0x08080808u, 0x08080808u);
        hi = __vsub4(hi ^ 0x08080808u, 0x08080808u);
        int g0 = (int)__byte_perm(lo, hi, 0x5140); // q0 q1 q2 q3
        int g1 = (int)__byte_perm(lo, hi, 0x7362); // q4 q5 q6 q7
        const int8_t* xp = Xq + k0 + 8 * lane;
        int x0 = *reinterpret_cast<const int*>(xp);
        int x1 = *reinterpret_cast<const int*>(xp + 4);
        int d = __dp4a(g0, x0, 0) + __dp4a(g1, x1, 0);
        int b = (k0 >> 5) + (lane >> 2);
        acc += wsr[b] * Xs[b] * (float)d;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) C[n] = acc;
}

// Single-token attention over the KV cache: one warp per head.
__global__ void attention_cache_kernel(const float* __restrict__ q,
                                       const float* __restrict__ kcache,
                                       const float* __restrict__ vcache,
                                       float* __restrict__ out,
                                       int pos, int n_heads, int head_size, int kv_mul) {
    int h = blockIdx.x;
    int lane = threadIdx.x;
    extern __shared__ float sc[];
    int kv_dim = (n_heads / kv_mul) * head_size;
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
    int vh = kh * head_size;
    for (int j = lane; j < head_size; j += 32) {
        float acc = 0.0f;
        for (int s = 0; s <= pos; s++) acc += sc[s] * vcache[(size_t)s * kv_dim + vh + j];
        out[h * head_size + j] = acc * inv;
    }
}

__global__ void rope_kernel(float* __restrict__ vec, int T, int pos0,
                            int vec_dim, int head_size) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)T * (vec_dim / 2);
    if (idx >= total) return;
    int half = vec_dim / 2;
    int local = (int)(idx / half);
    int t = pos0 + local;
    int p = idx % half;
    int i = 2 * p;
    int hd = i % head_size;
    float freq = 1.0f / powf(10000.0f, (float)hd / (float)head_size);
    float val = (float)t * freq;
    float c = cosf(val), s = sinf(val);
    float* v = vec + (size_t)local * vec_dim;
    float v0 = v[i], v1 = v[i + 1];
    v[i]     = v0 * c - v1 * s;
    v[i + 1] = v0 * s + v1 * c;
}

// one warp (32 threads) per (token, head); causal attention + softmax fused
__global__ void attention_kernel(const float* __restrict__ q, const float* __restrict__ k,
                                 const float* __restrict__ v, float* __restrict__ out,
                                 int T, int n_heads, int head_size, int kv_mul) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    int lane = threadIdx.x;
    extern __shared__ float sc[];
    int dim = n_heads * head_size;
    int kv_dim = (n_heads / kv_mul) * head_size;
    const float* qh = q + (size_t)t * dim + h * head_size;
    int kh = h / kv_mul;
    float scale = rsqrtf((float)head_size);
    for (int s = 0; s <= t; s++) {
        const float* khp = k + (size_t)s * kv_dim + kh * head_size;
        float part = 0.0f;
        for (int j = lane; j < head_size; j += 32) part += qh[j] * khp[j];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
        if (lane == 0) sc[s] = part * scale;
    }
    __syncwarp();
    float m = -1e30f;
    for (int s = lane; s <= t; s += 32) m = fmaxf(m, sc[s]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    m = __shfl_sync(0xffffffffu, m, 0);
    float local = 0.0f;
    for (int s = lane; s <= t; s += 32) {
        float e = expf(sc[s] - m);
        sc[s] = e;
        local += e;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(0xffffffffu, local, o);
    local = __shfl_sync(0xffffffffu, local, 0);
    float inv = 1.0f / local;
    int vh = kh * head_size;
    for (int j = lane; j < head_size; j += 32) {
        float acc = 0.0f;
        for (int s = 0; s <= t; s++) {
            acc += sc[s] * v[(size_t)s * kv_dim + vh + j];
        }
        out[(size_t)t * dim + h * head_size + j] = acc * inv;
    }
}

__global__ void add_kernel(float* __restrict__ x, const float* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void swiglu_kernel(float* __restrict__ hb, const float* __restrict__ hb2, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = hb[i];
        hb[i] = (v / (1.0f + expf(-v))) * hb2[i];
    }
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

// Each tensor is copied into its own 256-byte aligned device buffer straight
// from the mmap, so the q8 weights are suitably aligned for int4 loads and the
// whole file never has to be resident on the device at once.
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
    // single-token activations
    CUDA_CHECK(cudaMalloc(&s->x,    (size_t)dim * f));
    CUDA_CHECK(cudaMalloc(&s->xb,   (size_t)dim * f));
    CUDA_CHECK(cudaMalloc(&s->xb2,  (size_t)dim * f));
    CUDA_CHECK(cudaMalloc(&s->q,    (size_t)dim * f));
    CUDA_CHECK(cudaMalloc(&s->hb,   (size_t)hidden * f));
    CUDA_CHECK(cudaMalloc(&s->hb2,  (size_t)hidden * f));
    CUDA_CHECK(cudaMalloc(&s->logits, (size_t)p->vocab_size * f));
    CUDA_CHECK(cudaMalloc(&s->actq, (size_t)maxK * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&s->acts, (size_t)(maxK / QBLOCK) * f));
    CUDA_CHECK(cudaMalloc(&s->d_tokens, (size_t)maxT * sizeof(int)));
    // KV cache
    CUDA_CHECK(cudaMalloc(&s->key_cache,   (size_t)p->n_layers * maxT * kv_dim * f));
    CUDA_CHECK(cudaMalloc(&s->value_cache, (size_t)p->n_layers * maxT * kv_dim * f));
    s->maxT = maxT;
}

static void launch_quant(const float* X, int T, int K, int8_t* Xq, float* Xs) {
    int total = T * (K / QBLOCK);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    quantize_act_kernel<<<blocks, threads>>>(X, K, Xq, Xs, total);
}

// single-token (M=1) GEMM: one warp per output row
static void launch_matmul(float* C, const int8_t* Xq, const float* Xs, DQWeight W) {
    dim3 block(32, 8);
    dim3 grid((W.rows + block.y - 1) / block.y);
    matmul_q4_vec_kernel<<<grid, block>>>(Xq, Xs, W.q, W.s, C, W.rows, W.cols);
}

// ----------------------------------------------------------------------------
// forward

// One token at absolute position `pos`: appends its K/V to the cache and
// attends over positions 0..pos.  Returns the logits for that position.
static float* forward_step(DTransformer* w, DState* s, Config* p, int token, int pos) {
    int dim = p->dim, hidden = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int n_heads = p->n_heads;
    int head_size = dim / n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int maxT = s->maxT;

    CUDA_CHECK(cudaMemcpy(s->d_tokens, &token, sizeof(int), cudaMemcpyHostToDevice));
    embed_kernel<<<dim3((dim + 255) / 256, 1), dim3(256)>>>(
        w->token_embedding_table, s->d_tokens, s->x, dim);

    int rms_threads = 256;
    for (int l = 0; l < p->n_layers; l++) {
        // attention rmsnorm
        rmsnorm_kernel<<<1, rms_threads>>>(s->x, w->rms_att_weight + (size_t)l * dim, s->xb, dim);
        launch_quant(s->xb, 1, dim, s->actq, s->acts);
        launch_matmul(s->q, s->actq, s->acts, dslice(&w->wq, l, dim));

        // K/V go straight into the cache at [l][pos]
        float* krow = s->key_cache   + ((size_t)l * maxT + pos) * kv_dim;
        float* vrow = s->value_cache + ((size_t)l * maxT + pos) * kv_dim;
        launch_matmul(krow, s->actq, s->acts, dslice(&w->wk, l, kv_dim));
        launch_matmul(vrow, s->actq, s->acts, dslice(&w->wv, l, kv_dim));

        // RoPE at the absolute position
        rope_kernel<<<(dim / 2 + 255) / 256, 256>>>(s->q, 1, pos, dim, head_size);
        rope_kernel<<<(kv_dim / 2 + 255) / 256, 256>>>(krow, 1, pos, kv_dim, head_size);

        // attention over the cache
        const float* kbase = s->key_cache   + (size_t)l * maxT * kv_dim;
        const float* vbase = s->value_cache + (size_t)l * maxT * kv_dim;
        attention_cache_kernel<<<n_heads, 32, (size_t)(pos + 1) * sizeof(float)>>>(
            s->q, kbase, vbase, s->xb, pos, n_heads, head_size, kv_mul);

        // output projection + residual
        launch_quant(s->xb, 1, dim, s->actq, s->acts);
        launch_matmul(s->xb2, s->actq, s->acts, dslice(&w->wo, l, dim));
        add_kernel<<<(dim + 255) / 256, 256>>>(s->x, s->xb2, dim);

        // ffn
        rmsnorm_kernel<<<1, rms_threads>>>(s->x, w->rms_ffn_weight + (size_t)l * dim, s->xb, dim);
        launch_quant(s->xb, 1, dim, s->actq, s->acts);
        launch_matmul(s->hb, s->actq, s->acts, dslice(&w->w1, l, hidden));
        launch_matmul(s->hb2, s->actq, s->acts, dslice(&w->w3, l, hidden));
        swiglu_kernel<<<(hidden + 255) / 256, 256>>>(s->hb, s->hb2, hidden);
        launch_quant(s->hb, 1, hidden, s->actq, s->acts);
        launch_matmul(s->xb, s->actq, s->acts, dslice(&w->w2, l, dim));
        add_kernel<<<(dim + 255) / 256, 256>>>(s->x, s->xb, dim);
    }

    rmsnorm_kernel<<<1, rms_threads>>>(s->x, w->rms_final_weight, s->xb, dim);
    launch_quant(s->xb, 1, dim, s->actq, s->acts);
    launch_matmul(s->logits, s->actq, s->acts, w->wcls);
    return s->logits;
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
// generation (one token per forward, KV cache)

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
        float* d_logits = forward_step(w, s, p, token, pos);
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            CUDA_CHECK(cudaMemcpy(h_logits, d_logits, (size_t)p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
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

        float* d_logits = forward_step(w, s, p, token, pos);
        CUDA_CHECK(cudaMemcpy(h_logits, d_logits, (size_t)p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        next = sample(sm, h_logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
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
    fprintf(stderr, "Example: run_cuda llama2-7b-q8.bin -n 64 -i \"Once upon a time\"\n");
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

    // pick the device with the most free memory
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

    // read config from the header, mmap the file, copy it to the device
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
