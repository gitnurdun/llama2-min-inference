/* Llama-2-7B inference on CUDA (Tesla T4, sm_75).
 *
 * v2: int8 block-32 weights (same format as v1), custom dp4a kernels, NO KV
 * cache.  Each generation step runs the whole sequence through the network as a
 * batch with a causal attention mask, then samples the last position.  Weights
 * stay int8 in device memory (7.96 GB); activations are quantized per-token.
 *
 * Bias note: quantization stores weights unsigned (value + 128) to match the
 * CPU VPDPBUSD scheme.  On the GPU __dp4a is signed x signed, so we un-bias by
 * XOR 0x80808080 when loading, which maps the biased byte back to its signed
 * two's-complement value exactly.
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
#define QMAGIC 0x384C4C51u
#define QVERSION 1u

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(EXIT_FAILURE); } } while (0)

// ----------------------------------------------------------------------------
// model

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct {         // device pointers
    int8_t* q;           // (rows, cols) signed int8
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
    float *x, *xb, *xb2, *q, *k, *v, *hb, *hb2, *logits;
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

// convert biased-unsigned weights (value+128) to signed two's complement once
__global__ void xor_signed_kernel(int8_t* __restrict__ p, long n16) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n16) {
        uint4 v = reinterpret_cast<uint4*>(p)[i];
        v.x ^= 0x80808080u; v.y ^= 0x80808080u;
        v.z ^= 0x80808080u; v.w ^= 0x80808080u;
        reinterpret_cast<uint4*>(p)[i] = v;
    }
}

__global__ void rope_kernel(float* __restrict__ vec, int T, int vec_dim, int head_size) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)T * (vec_dim / 2);
    if (idx >= total) return;
    int half = vec_dim / 2;
    int t = idx / half;
    int p = idx % half;
    int i = 2 * p;
    int hd = i % head_size;
    float freq = 1.0f / powf(10000.0f, (float)hd / (float)head_size);
    float val = (float)t * freq;
    float c = cosf(val), s = sinf(val);
    float* v = vec + (size_t)t * vec_dim;
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
    r.q = w->q + (size_t)l * layer_rows * w->cols;
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

static void dalloc_q8(uint8_t** ptr, DQWeight* w, long rows, int cols) {
    long nb = cols / QBLOCK;
    CUDA_CHECK(cudaMalloc((void**)&w->q, (size_t)rows * cols));
    CUDA_CHECK(cudaMemcpy(w->q, *ptr, (size_t)rows * cols, cudaMemcpyHostToDevice));
    *ptr += (size_t)rows * cols;
    w->s = dalloc_f32(ptr, (size_t)rows * nb);
    w->rows = rows;
    w->cols = cols;
    w->nb = nb;
    long n16 = (long)rows * cols / 16;
    xor_signed_kernel<<<(n16 + 255) / 256, 256>>>((int8_t*)w->q, n16);
}

static void load_weights(DTransformer* w, Config* p, uint8_t* base) {
    uint8_t* ptr = base + 2 * sizeof(uint32_t) + sizeof(Config);
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    unsigned long long L = p->n_layers;
    w->token_embedding_table = dalloc_f32(&ptr, (size_t)p->vocab_size * p->dim);
    w->rms_att_weight = dalloc_f32(&ptr, L * p->dim);
    dalloc_q8(&ptr, &w->wq, L * p->dim, p->dim);
    dalloc_q8(&ptr, &w->wk, L * kv_dim, p->dim);
    dalloc_q8(&ptr, &w->wv, L * kv_dim, p->dim);
    dalloc_q8(&ptr, &w->wo, L * p->dim, p->dim);
    w->rms_ffn_weight = dalloc_f32(&ptr, L * p->dim);
    dalloc_q8(&ptr, &w->w1, L * p->hidden_dim, p->dim);
    dalloc_q8(&ptr, &w->w2, L * p->dim, p->hidden_dim);
    dalloc_q8(&ptr, &w->w3, L * p->hidden_dim, p->dim);
    w->rms_final_weight = dalloc_f32(&ptr, p->dim);
    dalloc_q8(&ptr, &w->wcls, p->vocab_size, p->dim);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void alloc_state(DState* s, Config* p) {
    int dim = p->dim, hidden = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int maxK = hidden > dim ? hidden : dim;
    int T = p->seq_len;
    size_t f = sizeof(float);
    CUDA_CHECK(cudaMalloc(&s->x,    (size_t)T * dim * f));
    CUDA_CHECK(cudaMalloc(&s->xb,   (size_t)T * dim * f));
    CUDA_CHECK(cudaMalloc(&s->xb2,  (size_t)T * dim * f));
    CUDA_CHECK(cudaMalloc(&s->q,    (size_t)T * dim * f));
    CUDA_CHECK(cudaMalloc(&s->k,    (size_t)T * kv_dim * f));
    CUDA_CHECK(cudaMalloc(&s->v,    (size_t)T * kv_dim * f));
    CUDA_CHECK(cudaMalloc(&s->hb,   (size_t)T * hidden * f));
    CUDA_CHECK(cudaMalloc(&s->hb2,  (size_t)T * hidden * f));
    CUDA_CHECK(cudaMalloc(&s->logits, (size_t)p->vocab_size * f));
    CUDA_CHECK(cudaMalloc(&s->actq, (size_t)T * maxK * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&s->acts, (size_t)T * (maxK / QBLOCK) * f));
    CUDA_CHECK(cudaMalloc(&s->d_tokens, (size_t)T * sizeof(int)));
    s->maxT = T;
}

static void launch_quant(const float* X, int T, int K, int8_t* Xq, float* Xs) {
    int total = T * (K / QBLOCK);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    quantize_act_kernel<<<blocks, threads>>>(X, K, Xq, Xs, total);
}

static void launch_matmul(float* C, const int8_t* Xq, const float* Xs,
                          DQWeight W, int T) {
    dim3 block(32, 16);
    dim3 grid((T + block.x - 1) / block.x, (W.rows + block.y - 1) / block.y);
    matmul_q8_kernel<<<grid, block>>>(Xq, Xs, W.q, W.s, C, T, W.rows, W.cols);
}

// ----------------------------------------------------------------------------
// forward

static float* forward_gpu(DTransformer* w, DState* s, Config* p, int T) {
    int dim = p->dim, hidden = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int n_heads = p->n_heads;
    int head_size = dim / n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;

    dim3 eblock(256);
    dim3 egrid((dim + 255) / 256, T);
    embed_kernel<<<egrid, eblock>>>(w->token_embedding_table, s->d_tokens, s->x, dim);

    int rms_threads = 256;
    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm_kernel<<<T, rms_threads>>>(s->x, w->rms_att_weight + (size_t)l * dim, s->xb, dim);
        launch_quant(s->xb, T, dim, s->actq, s->acts);
        launch_matmul(s->q, s->actq, s->acts, dslice(&w->wq, l, dim), T);
        launch_matmul(s->k, s->actq, s->acts, dslice(&w->wk, l, kv_dim), T);
        launch_matmul(s->v, s->actq, s->acts, dslice(&w->wv, l, kv_dim), T);

        rope_kernel<<<(T * (dim / 2) + 255) / 256, 256>>>(s->q, T, dim, head_size);
        rope_kernel<<<(T * (kv_dim / 2) + 255) / 256, 256>>>(s->k, T, kv_dim, head_size);

        dim3 ablock(32);
        dim3 agrid(T, n_heads);
        attention_kernel<<<agrid, ablock, (size_t)T * sizeof(float)>>>(
            s->q, s->k, s->v, s->xb, T, n_heads, head_size, kv_mul);

        launch_quant(s->xb, T, dim, s->actq, s->acts);
        launch_matmul(s->xb2, s->actq, s->acts, dslice(&w->wo, l, dim), T);
        add_kernel<<<(T * dim + 255) / 256, 256>>>(s->x, s->xb2, (long)T * dim);

        rmsnorm_kernel<<<T, rms_threads>>>(s->x, w->rms_ffn_weight + (size_t)l * dim, s->xb, dim);
        launch_quant(s->xb, T, dim, s->actq, s->acts);
        launch_matmul(s->hb, s->actq, s->acts, dslice(&w->w1, l, hidden), T);
        launch_matmul(s->hb2, s->actq, s->acts, dslice(&w->w3, l, hidden), T);
        swiglu_kernel<<<(T * hidden + 255) / 256, 256>>>(s->hb, s->hb2, (long)T * hidden);

        launch_quant(s->hb, T, hidden, s->actq, s->acts);
        launch_matmul(s->xb, s->actq, s->acts, dslice(&w->w2, l, dim), T);
        add_kernel<<<(T * dim + 255) / 256, 256>>>(s->x, s->xb, (long)T * dim);
    }

    // final norm + classifier for the last position only
    rmsnorm_kernel<<<T, rms_threads>>>(s->x, w->rms_final_weight, s->xb, dim);
    launch_quant(s->xb + (size_t)(T - 1) * dim, 1, dim, s->actq, s->acts);
    launch_matmul(s->logits, s->actq, s->acts, w->wcls, 1);
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
// generation (GPU, whole-sequence / no KV cache)

// One generation step from the current token sequence: run the whole sequence
// through the net (no KV cache), sample the next token, return it.  Used both
// for the prompt prefill and the autoregressive loop.
static int step_gpu(DTransformer* w, DState* s, Config* p, Sampler* sm,
                    int* tokens, int T, float* h_logits) {
    CUDA_CHECK(cudaMemcpy(s->d_tokens, tokens, (size_t)T * sizeof(int), cudaMemcpyHostToDevice));
    float* d_logits = forward_gpu(w, s, p, T);
    CUDA_CHECK(cudaMemcpy(h_logits, d_logits, (size_t)p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return sample(sm, h_logits);
}

void generate_gpu(DTransformer* w, DState* s, Config* p, Tokenizer* tk, Sampler* sm,
                  char* prompt, int steps) {
    char* empty_prompt = "";
    if (prompt == NULL) prompt = empty_prompt;

    int num_prompt_tokens = 0;
    int* tokens = (int*)malloc((strlen(prompt) + 3 + steps) * sizeof(int));
    encode(tk, prompt, 1, 0, tokens, &num_prompt_tokens);

    float* h_logits = (float*)malloc(p->vocab_size * sizeof(float));
    int T = num_prompt_tokens;
    int token = tokens[T - 1];
    long t0 = time_in_ms();

    while (T < steps) {
        int next = step_gpu(w, s, p, sm, tokens, T, h_logits);
        if (next == 1) break;
        char* piece = decode(tk, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;
        tokens[T] = next;
        T++;
    }
    printf("\n");
    long t1 = time_in_ms();
    int gen = T - num_prompt_tokens;
    if (gen > 0) fprintf(stderr, "achieved tok/s: %f\n", gen / (double)(t1 - t0) * 1000);
    free(tokens);
    free(h_logits);
    (void)s;
}

// With no KV cache each turn is generated independently from its rendered
// template (multi-turn context would require re-encoding the whole history,
// which is exactly what the KV cache will fix later).
void chat_gpu(DTransformer* w, DState* s, Config* p, Tokenizer* tk, Sampler* sm,
              char* cli_user_prompt, char* cli_system_prompt, int steps) {
    char system_prompt[512] = "";
    char user_prompt[512];
    int* tokens = (int*)malloc((1152 + steps) * sizeof(int));
    float* h_logits = (float*)malloc(p->vocab_size * sizeof(float));
    int first = 1;

    while (1) {
        if (first && cli_system_prompt != NULL) {
            strcpy(system_prompt, cli_system_prompt);
        } else if (first) {
            read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
        }
        if (first && cli_user_prompt != NULL) {
            strcpy(user_prompt, cli_user_prompt);
        } else if (!read_stdin("User: ", user_prompt, sizeof(user_prompt))) {
            break;
        }

        char turn[1152];
        if (first && system_prompt[0] != '\0') {
            sprintf(turn, "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]", system_prompt, user_prompt);
        } else {
            sprintf(turn, "[INST] %s [/INST]", user_prompt);
        }

        int T = 0;
        encode(tk, turn, 1, 0, tokens, &T);
        if (T >= steps) T = steps;
        printf("Assistant: ");
        int token = tokens[T - 1];
        while (T < steps) {
            int next = step_gpu(w, s, p, sm, tokens, T, h_logits);
            if (next == 2) break;
            char* piece = decode(tk, token, next);
            safe_printf(piece);
            fflush(stdout);
            token = next;
            tokens[T] = next;
            T++;
        }
        printf("\n");
        first = 0;
    }
    free(tokens);
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
    alloc_state(&s, &config);

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
