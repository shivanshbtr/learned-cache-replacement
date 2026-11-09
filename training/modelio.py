"""Model file formats shared with the C++ simulator (simulator/learned.h).

MLP file (.mlp, little-endian):  magic "ACMLP001"; u32 n_num, emb_dim, n_ids(=K+1), n_layers; u32 dims[n_layers+1];
  u64 vocab_pcs[K]; f32 mean[n_num], std[n_num]; f32 emb_last[n_ids*emb_dim], emb_cur[n_ids*emb_dim];
  per layer: f32 W[out*in] (row-major), f32 b[out].  Hidden layers ReLU, last layer linear (one score per candidate).
LSTM file (.lstm, little-endian): magic "ACLSTM01"; u32 n_num(9), emb, n_ids(=K+1), hist, hidden, dpos, dc, dh; u64 vocab_pcs[K];
  f32 in this order: mean[n_num], std[n_num], emb[n_ids*emb], w_ih[4*hidden*S], w_hh[4*hidden*hidden], b[4*hidden], pos[hist*dpos],
  a1w[dc*F], a1b[dc], qw[K*dc], qb[K], bw[dh*(dc+hidden)], bb[dh], vw[dh], vb[1]     (S = emb+5, F = n_num+2*emb, K = hidden+dpos)
  LSTM gate order i,f,g,o (PyTorch); b = b_ih + b_hh.  See simulator/learned.h (LstmModel) for the forward pass.
GBDT file (.gbdt, text): "ACGBDT 1", "vocab K p1 p2 ...", a line "---", then the LightGBM text model (raw score = sum of trees).
"""
import struct
import numpy as np


def write_mlp(path, vocab, mean, std, emb_last, emb_cur, layers):
    """layers: list of (W [out,in], b [out]) numpy arrays."""
    n_ids, emb = emb_last.shape
    dims = [layers[0][0].shape[1]] + [w.shape[0] for w, _ in layers]
    assert dims[0] == len(mean) + 2 * emb and dims[-1] == 1 and n_ids == len(vocab) + 1
    with open(path, "wb") as f:
        f.write(b"ACMLP001")
        f.write(struct.pack("<4I", len(mean), emb, n_ids, len(layers)))
        f.write(struct.pack("<%dI" % len(dims), *dims))
        f.write(np.asarray(vocab, dtype="<u8").tobytes())
        for a in (mean, std, emb_last, emb_cur):
            f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())
        for w, b in layers:
            f.write(np.ascontiguousarray(w, dtype="<f4").tobytes()); f.write(np.ascontiguousarray(b, dtype="<f4").tobytes())


def read_mlp(path):
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:8] == b"ACMLP001"
    n_num, emb, n_ids, n_layers = struct.unpack_from("<4I", raw, 8)
    off = 24
    dims = list(struct.unpack_from("<%dI" % (n_layers + 1), raw, off)); off += 4 * (n_layers + 1)
    vocab = np.frombuffer(raw, dtype="<u8", count=n_ids - 1, offset=off).copy(); off += 8 * (n_ids - 1)
    def take(n):
        nonlocal off
        a = np.frombuffer(raw, dtype="<f4", count=n, offset=off).copy(); off += 4 * n; return a
    mean, std = take(n_num), take(n_num)
    emb_last, emb_cur = take(n_ids * emb).reshape(n_ids, emb), take(n_ids * emb).reshape(n_ids, emb)
    layers = []
    for l in range(n_layers):
        w = take(dims[l + 1] * dims[l]).reshape(dims[l + 1], dims[l]); b = take(dims[l + 1]); layers.append((w, b))
    assert off == len(raw), "trailing bytes in %s" % path
    return dict(vocab=vocab, mean=mean, std=std, emb_last=emb_last, emb_cur=emb_cur, layers=layers, dims=dims, emb_dim=emb, n_num=n_num)


def mlp_params(m):
    return int(m["emb_last"].size + m["emb_cur"].size + sum(w.size + b.size for w, b in m["layers"]) + 2 * m["n_num"])


def predict_mlp(m, num, pcid, curid):
    """numpy float32 forward pass identical to the C++ one. Returns scores [N,W]."""
    N, W, _ = num.shape
    x = ((num - m["mean"]) / m["std"]).astype(np.float32)
    cur = np.repeat(m["emb_cur"][curid][:, None, :], W, axis=1)
    h = np.concatenate([x, m["emb_last"][pcid], cur], axis=-1).reshape(N * W, -1)
    for i, (w, b) in enumerate(m["layers"]):
        h = h @ w.T + b
        if i < len(m["layers"]) - 1:
            h = np.maximum(h, 0)
    return h.reshape(N, W)


def write_gbdt(path, vocab, model_text):
    with open(path, "w") as f:
        f.write("ACGBDT 1\nvocab %d %s\n---\n" % (len(vocab), " ".join(str(int(v)) for v in vocab)))
        f.write(model_text)


def read_gbdt(path):
    txt = open(path).read()
    head, body = txt.split("\n---\n", 1)
    parts = head.split("\n")[1].split()
    k = int(parts[1])
    vocab = np.array([int(v) for v in parts[2:2 + k]], dtype=np.uint64)
    return vocab, body


# ---------------------------------------------------------------- LSTM + attention
_LSTM_ORDER = ["mean", "std", "emb", "w_ih", "w_hh", "b", "pos", "a1w", "a1b", "qw", "qb", "bw", "bb", "vw", "vb"]


def _lstm_shapes(n_num, emb, n_ids, hist, hidden, dpos, dc, dh):
    S, F, K = emb + 5, n_num + 2 * emb, hidden + dpos
    return {"mean": (n_num,), "std": (n_num,), "emb": (n_ids, emb), "w_ih": (4 * hidden, S), "w_hh": (4 * hidden, hidden), "b": (4 * hidden,),
            "pos": (hist, dpos), "a1w": (dc, F), "a1b": (dc,), "qw": (K, dc), "qb": (K,), "bw": (dh, dc + hidden), "bb": (dh,), "vw": (dh,), "vb": (1,)}


def write_lstm(path, vocab, w, hist, hidden, dpos, dc, dh):
    """w: dict with the arrays named in _LSTM_ORDER (shapes as in the format description)."""
    n_num, emb = len(w["mean"]), w["emb"].shape[1]
    n_ids = w["emb"].shape[0]
    assert n_ids == len(vocab) + 1
    shp = _lstm_shapes(n_num, emb, n_ids, hist, hidden, dpos, dc, dh)
    with open(path, "wb") as f:
        f.write(b"ACLSTM01")
        f.write(struct.pack("<8I", n_num, emb, n_ids, hist, hidden, dpos, dc, dh))
        f.write(np.asarray(vocab, dtype="<u8").tobytes())
        for k in _LSTM_ORDER:
            a = np.asarray(w[k], dtype="<f4").reshape(shp[k])
            f.write(np.ascontiguousarray(a).tobytes())


def read_lstm(path):
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:8] == b"ACLSTM01", "not an ACLSTM01 file"
    n_num, emb, n_ids, hist, hidden, dpos, dc, dh = struct.unpack_from("<8I", raw, 8)
    off = 8 + 32
    vocab = np.frombuffer(raw, dtype="<u8", count=n_ids - 1, offset=off).copy(); off += 8 * (n_ids - 1)
    shp = _lstm_shapes(n_num, emb, n_ids, hist, hidden, dpos, dc, dh)
    m = dict(vocab=vocab, hist=hist, hidden=hidden, dpos=dpos, dc=dc, dh=dh, emb_dim=emb, n_num=n_num)
    for k in _LSTM_ORDER:
        n = int(np.prod(shp[k]))
        m[k] = np.frombuffer(raw, dtype="<f4", count=n, offset=off).copy().reshape(shp[k]).astype(np.float32); off += 4 * n
    assert off == len(raw), "trailing bytes in %s" % path
    return m


def lstm_params(m):
    return int(sum(m[k].size for k in _LSTM_ORDER))


def _sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def predict_lstm(m, num, pcid, curid, hpc, hnum, chunk=4096):
    """numpy float32 forward pass identical to the C++ one (LstmModel). num [N,W,9], pcid [N,W], curid [N], hpc [N,H], hnum [N,H,5]. Returns scores [N,W]."""
    N, W, _ = num.shape
    H, hid, K = m["hist"], m["hidden"], m["hidden"] + m["dpos"]
    inv = np.float32(1.0 / np.sqrt(K))
    out = np.zeros((N, W), dtype=np.float32)
    pos = m["pos"]
    for s in range(0, N, chunk):
        e = min(N, s + chunk); B = e - s
        step = np.concatenate([m["emb"][hpc[s:e].astype(np.int64)], hnum[s:e].astype(np.float32)], axis=-1)   # [B,H,S]
        h = np.zeros((B, hid), np.float32); c = np.zeros((B, hid), np.float32); Hs = np.zeros((B, H, hid), np.float32)
        for t in range(H):
            z = step[:, t] @ m["w_ih"].T + h @ m["w_hh"].T + m["b"]
            ig, fg, gg, og = _sigmoid(z[:, :hid]), _sigmoid(z[:, hid:2 * hid]), np.tanh(z[:, 2 * hid:3 * hid]), _sigmoid(z[:, 3 * hid:])
            c = fg * c + ig * gg; h = og * np.tanh(c); Hs[:, t] = h
        x = ((num[s:e] - m["mean"]) / m["std"]).astype(np.float32)
        cur = np.repeat(m["emb"][curid[s:e].astype(np.int64)][:, None, :], W, axis=1)
        x = np.concatenate([x, m["emb"][pcid[s:e].astype(np.int64)], cur], axis=-1)                              # [B,W,F]
        cv = np.maximum(x @ m["a1w"].T + m["a1b"], 0)                                                          # [B,W,dc]
        u = cv @ m["qw"].T + m["qb"]                                                                            # [B,W,K]
        keys = np.concatenate([Hs, np.broadcast_to(pos, (B, H, m["dpos"]))], axis=-1)                           # [B,H,K]
        att = np.einsum("bwk,bhk->bwh", u, keys) * inv
        att = np.exp(att - att.max(-1, keepdims=True)); att = att / att.sum(-1, keepdims=True)
        g = np.einsum("bwh,bhd->bwd", att, Hs)
        zz = np.maximum(np.concatenate([cv, g], axis=-1) @ m["bw"].T + m["bb"], 0)
        out[s:e] = (zz @ m["vw"] + m["vb"][0]).astype(np.float32)
    return out
