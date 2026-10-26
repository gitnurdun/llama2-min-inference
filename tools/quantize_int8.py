#!/usr/bin/env python3
"""Quantize a llama2.c v0 fp32 checkpoint to int8 (Q8_0-like) block-32 weights.

This is the canonical copy of the tool used by v1 (CPU AVX-512 VNNI) and by the
CUDA builds v2/v3.  Run it from the repository root with the pytorch_env conda
environment:

    conda run -n pytorch_env python tools/quantize_int8.py \
        models/llama2-7b.bin models/llama2-7b-q8.bin

Output layout (little-endian):
    uint32 magic   = 0x384C4C51
    uint32 version = 1
    int32  config[7]  (dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                       vocab_size, seq_len  -- vocab_size sign kept)

    fp32 token_embedding_table [vocab, dim]
    fp32 rms_att_weight        [L, dim]

    q8   wq   [L*dim,     dim]
    q8   wk   [L*kv_dim,  dim]
    q8   wv   [L*kv_dim,  dim]
    q8   wo   [L*dim,     dim]

    fp32 rms_ffn_weight        [L, dim]

    q8   w1   [L*hidden,  dim]
    q8   w2   [L*dim,     hidden]
    q8   w3   [L*hidden,  dim]

    fp32 rms_final_weight      [dim]

    q8   wcls [vocab, dim]     (only when weights are unshared)

q8 block layout, block size 32:
    uint8 weights[rows, cols]      (signed value + 128, i.e. biased to unsigned)
    fp32  scales [rows, cols/32]

Why "biased to unsigned"?  The CPU kernel uses VPDPBUSD = unsigned x signed.
The GPU kernel uses __dp4a = signed x signed and un-biases with XOR 0x80.
"""
import numpy as np
import struct
import sys
import time

BLOCK = 32
MAGIC = 0x384C4C51
VERSION = 1


def quantize(w):
    """w: (rows, cols) float32 -> (uint8 biased weights, fp32 scales)."""
    rows, cols = w.shape
    assert cols % BLOCK == 0
    wb = w.reshape(rows, cols // BLOCK, BLOCK)
    amax = np.abs(wb).max(axis=2)
    scale = amax / 127.0
    scale[scale == 0.0] = 1e-8
    q = np.rint(wb / scale[:, :, None])
    np.clip(q, -127.0, 127.0, out=q)
    qu = (q + 128.0).astype(np.uint8)
    return qu.reshape(rows, cols), scale.astype(np.float32)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "models/llama2-7b.bin"
    dst = sys.argv[2] if len(sys.argv) > 2 else "models/llama2-7b-q8.bin"
    t0 = time.time()

    with open(src, "rb") as f:
        cfg = struct.unpack("7i", f.read(28))
    dim, hidden, L, nh, nkv, vocab, seq = cfg
    vocab_abs = abs(vocab)
    shared = vocab > 0
    kv_dim = dim * nkv // nh
    head_size = dim // nh
    print(f"dim={dim} hidden={hidden} L={L} nh={nh} nkv={nkv} "
          f"vocab={vocab_abs} seq={seq} shared={shared}", flush=True)

    mm = np.memmap(src, dtype=np.float32, mode="r")
    p = 7
    emb = (p, vocab_abs * dim); p += vocab_abs * dim
    rms_att = (p, L * dim); p += L * dim
    wq = (p, L * dim, dim); p += L * dim * dim
    wk = (p, L * kv_dim, dim); p += L * dim * kv_dim
    wv = (p, L * kv_dim, dim); p += L * dim * kv_dim
    wo = (p, L * dim, dim); p += L * dim * dim
    rms_ffn = (p, L * dim); p += L * dim
    w1 = (p, L * hidden, dim); p += L * dim * hidden
    w2 = (p, L * dim, hidden); p += L * hidden * dim
    w3 = (p, L * hidden, dim); p += L * dim * hidden
    rms_final = (p, dim); p += dim
    p += seq * head_size // 2   # freq_cis_real (skipped, not written)
    p += seq * head_size // 2   # freq_cis_imag (skipped, not written)
    wcls = (p, vocab_abs, dim) if not shared else None

    def put_f32(out, pos):
        off, n = pos
        out.write(np.asarray(mm[off:off + n], dtype=np.float32).tobytes())

    def put_q8(out, pos):
        off, rows, cols = pos
        w = np.asarray(mm[off:off + rows * cols], dtype=np.float32).reshape(rows, cols)
        q, s = quantize(w)
        out.write(q.tobytes())
        out.write(s.tobytes())
        del w, q, s

    with open(dst, "wb") as out:
        out.write(struct.pack("I", MAGIC))
        out.write(struct.pack("I", VERSION))
        out.write(struct.pack("7i", *cfg))
        put_f32(out, emb)
        put_f32(out, rms_att)
        put_q8(out, wq)
        put_q8(out, wk)
        put_q8(out, wv)
        put_q8(out, wo)
        put_f32(out, rms_ffn)
        put_q8(out, w1)
        put_q8(out, w2)
        put_q8(out, w3)
        put_f32(out, rms_final)
        if wcls is not None:
            put_q8(out, wcls)

    print(f"wrote {dst} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
