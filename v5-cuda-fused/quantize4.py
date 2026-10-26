#!/usr/bin/env python3
"""Quantize llama2.c v0 fp32 checkpoint to int4 (Q4_0-like) block-32 weights.

Output layout (little-endian):
    uint32 magic   = 0x344C4C51
    uint32 version = 2
    int32  config[7]  (vocab_size sign kept)

    fp32 token_embedding_table [vocab, dim]
    fp32 rms_att_weight        [L, dim]

    q4   wq, wk, wv, wo        (int4, block 32, 2 per byte + per-block scale)
    fp32 rms_ffn_weight        [L, dim]
    q4   w1, w2, w3
    fp32 rms_final_weight      [dim]
    q4   wcls

q4 block layout, block size 32:
    uint8 weights[rows, cols/2]   (2 signed int4 per byte, even elem = low nibble)
    fp32  scales [rows, cols/32]

A 32-value block uses scale = absmax/8 and q in [-8,7], packed two's complement.
"""
import numpy as np
import struct
import time

SRC = "../models/llama2-7b.bin"
DST = "../models/llama2-7b-q4.bin"
MAGIC = 0x344C4C51
VERSION = 2
BLOCK = 32


def quantize_q4(w):
    rows, cols = w.shape
    assert cols % BLOCK == 0
    wb = w.reshape(rows, cols // BLOCK, BLOCK)
    amax = np.abs(wb).max(axis=2)
    scale = amax / 8.0
    scale[scale == 0.0] = 1e-8
    q = np.rint(wb / scale[:, :, None])
    np.clip(q, -8.0, 7.0, out=q)
    qi = (q.astype(np.int16) & 0xF).astype(np.uint8)      # 4-bit two's complement
    even = qi[:, :, 0::2]
    odd = qi[:, :, 1::2]
    packed = (even | (odd << 4)).astype(np.uint8)
    return packed.reshape(rows, cols // 2), scale.astype(np.float32)


def main():
    t0 = time.time()
    with open(SRC, "rb") as f:
        cfg = struct.unpack("7i", f.read(28))
    dim, hidden, L, nh, nkv, vocab, seq = cfg
    vocab_abs = abs(vocab)
    shared = vocab > 0
    kv_dim = dim * nkv // nh
    head_size = dim // nh
    print(f"dim={dim} hidden={hidden} L={L} vocab={vocab_abs}", flush=True)

    mm = np.memmap(SRC, dtype=np.float32, mode="r")
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
    p += seq * head_size // 2
    p += seq * head_size // 2
    wcls = (p, vocab_abs, dim) if not shared else None

    def put_f32(out, pos):
        off, n = pos
        out.write(np.asarray(mm[off:off + n], dtype=np.float32).tobytes())

    def put_q4(out, pos):
        off, rows, cols = pos
        w = np.asarray(mm[off:off + rows * cols], dtype=np.float32).reshape(rows, cols)
        q, s = quantize_q4(w)
        out.write(q.tobytes())
        out.write(s.tobytes())
        del w, q, s

    with open(DST, "wb") as out:
        out.write(struct.pack("I", MAGIC))
        out.write(struct.pack("I", VERSION))
        out.write(struct.pack("7i", *cfg))
        put_f32(out, emb)
        put_f32(out, rms_att)
        put_q4(out, wq)
        put_q4(out, wk)
        put_q4(out, wv)
        put_q4(out, wo)
        put_f32(out, rms_ffn)
        put_q4(out, w1)
        put_q4(out, w2)
        put_q4(out, w3)
        put_f32(out, rms_final)
        if wcls is not None:
            put_q4(out, wcls)

    print(f"done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
