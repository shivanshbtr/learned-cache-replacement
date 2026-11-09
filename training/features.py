"""Per-candidate features for the eviction models (same definition in C++: simulator/learned.h).

For one eviction decision (current access `seq`, set with W resident lines) and candidate way w, using ONLY past information:
  x0 = log2(1 + seq - last_idx[w])            time since the line's last access
  x1 = log2(1 + seq - insert_idx[w])          age since insertion
  x2 = log2(1 + count[w])                     accesses since insertion
  x3 = 1 if count >= 2 else 0                 line was re-used while resident
  x4 = log2(1 + gap between its last two accesses) if x3 else 0
  x5 = (number of ways accessed more recently than w) / (W - 1)       recency rank, 0 = most recent
  x6 = x0 - mean over the set of x0           time since last access relative to the set
  x7 = 1 if the line's last access was a store
  x8 = 1 if the current (missing) access is a store
  pc_id  = id of the PC of the line's last access     (vocabulary of the most frequent PCs in training data, 0 = unknown)
  cur_id = id of the PC of the current access
GBDT uses [x0..x8, pc_id, cur_id] as 11 numeric columns; the MLP embeds the two ids.

History features for the LSTM model (same definition in C++: simulator/learned.h). For a decision at access index t and
history length H, step j = 0..H-1 is access i = t-H+1+j (the last step is the current access):
  pc id of access i (0 for padding)  +  5 numbers:  sign(line_i - line_{i-1}), log2(1+|delta|)/20, is_write_i,
  same_set_as_current (set of line_i == set of the current line), valid (1 if i >= 0).  delta = 0 for i = 0. Padding steps (i < 0) are all 0.
"""
import numpy as np
from dataset import NO_NEXT_USE, mask_to_bits, measured, informative

NUM_FEATS = 9
HIST_FEATS = 5
RD_CAP = 1 << 21          # reuse distance cap (accesses) for the auxiliary target
FEAT_NAMES = ["since_last", "age", "count", "reused", "last_gap", "recency_rank", "since_last_rel", "last_is_write", "cur_is_write"]


def build_vocab(datasets, k):
    """Most frequent k PCs over the traces of `datasets` (list of dicts from dataset.load_npz). Returns uint64 array, id = index + 1."""
    tot = {}
    for d in datasets:
        u, c = np.unique(d["trace_pc"], return_counts=True)
        for a, b in zip(u.tolist(), c.tolist()):
            tot[a] = tot.get(a, 0) + b
    top = sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return np.array([p for p, _ in top], dtype=np.uint64)


def pc_ids(pcs, vocab):
    pcs = np.asarray(pcs, dtype=np.uint64)
    if len(vocab) == 0:
        return np.zeros(pcs.shape, dtype=np.int32)
    sorter = np.argsort(vocab)
    pos = np.searchsorted(vocab, pcs, sorter=sorter)
    cand = sorter[np.clip(pos, 0, len(vocab) - 1)]
    ok = vocab[cand] == pcs
    return np.where(ok, cand + 1, 0).astype(np.int32)


def prev_use(d):
    """prev_use[i] = index of the previous access to the same line, or -1 (cached inside d)."""
    if "_prev" not in d:
        line = d["trace_line"]
        order = np.argsort(line, kind="stable")
        sl = line[order]
        same = sl[1:] == sl[:-1]
        prev = np.full(len(line), -1, dtype=np.int64)
        prev[order[1:][same]] = order[:-1][same]
        d["_prev"] = prev
    return d["_prev"]


def build(d, vocab, sel=None):
    """Returns (num float32 [N,W,9], pcid int32 [N,W], curid int32 [N], bits bool [N,W]) for decisions `sel` (indices or None = all)."""
    idx = np.arange(len(d["seq"])) if sel is None else np.asarray(sel)
    seq = d["seq"][idx].astype(np.int64)
    last = d["last_idx"][idx].astype(np.int64)
    ins = d["insert_idx"][idx].astype(np.int64)
    cnt = d["count"][idx].astype(np.int64)
    W = last.shape[1]
    x = np.zeros(last.shape + (NUM_FEATS,), dtype=np.float64)
    ts = (seq[:, None] - last).astype(np.float64)
    x[..., 0] = np.log2(1.0 + ts)
    x[..., 1] = np.log2(1.0 + (seq[:, None] - ins).astype(np.float64))
    x[..., 2] = np.log2(1.0 + cnt.astype(np.float64))
    has_prev = cnt >= 2
    x[..., 3] = has_prev
    pg = last - prev_use(d)[last]
    x[..., 4] = np.where(has_prev, np.log2(1.0 + pg.astype(np.float64)), 0.0)
    rank = (last[:, :, None] < last[:, None, :]).sum(-1)
    x[..., 5] = rank / float(W - 1) if W > 1 else 0.0
    x[..., 6] = x[..., 0] - x[..., 0].mean(1, keepdims=True)
    x[..., 7] = d["trace_write"][last]
    x[..., 8] = d["trace_write"][seq][:, None]
    pcid = pc_ids(d["trace_pc"][last], vocab)
    curid = pc_ids(d["trace_pc"][seq], vocab)
    bits = mask_to_bits(d["label_mask"][idx], W)
    return x.astype(np.float32), pcid, curid, bits


def select(d, informative_only=True, measured_only=True, max_rows=None, seed=0, keep=None):
    """keep: optional bool mask over decisions (merged multi-application files: only the decisions of one application)."""
    ok = np.ones(len(d["seq"]), dtype=bool)
    if keep is not None: ok &= keep
    if measured_only: ok &= measured(d)
    if informative_only: ok &= informative(d)
    sel = np.nonzero(ok)[0]
    if max_rows is not None and len(sel) > max_rows:
        sel = np.sort(np.random.default_rng(seed).choice(sel, max_rows, replace=False))
    return sel


def gbdt_matrix(num, pcid, curid):
    """[N,W,9] + ids -> float32 matrix [N*W, 11] (row n*W+w)."""
    N, W, _ = num.shape
    cur = np.repeat(curid[:, None], W, axis=1)
    return np.concatenate([num.reshape(N * W, NUM_FEATS), pcid.reshape(-1, 1).astype(np.float32), cur.reshape(-1, 1).astype(np.float32)], axis=1)


def build_hist(d, vocab, sel, H):
    """History inputs for decisions `sel`: (hpc int16 [N,H], hnum float32 [N,H,5]). Past information only (accesses seq-H+1 .. seq)."""
    idx = np.arange(len(d["seq"])) if sel is None else np.asarray(sel)
    seq = d["seq"][idx].astype(np.int64)
    pos = seq[:, None] + np.arange(-H + 1, 1, dtype=np.int64)[None, :]
    valid = pos >= 0
    pc = np.clip(pos, 0, None)
    line = d.get("trace_line_plain", d["trace_line"])   # merged multi-application files: plain line addresses (the simulator strips the application bits)
    li = line[pc].astype(np.int64)
    lp = line[np.clip(pos - 1, 0, None)].astype(np.int64)
    delta = np.where(pos >= 1, li - lp, 0)
    mask = np.uint64(int(d["meta"]["sets"]) - 1)
    same = (line[pc] & mask) == (line[seq] & mask)[:, None]
    num = np.zeros(pos.shape + (HIST_FEATS,), dtype=np.float64)
    num[..., 0] = np.sign(delta)
    num[..., 1] = np.log2(1.0 + np.abs(delta).astype(np.float64)) / 20.0
    num[..., 2] = d["trace_write"][pc]
    num[..., 3] = same
    num[..., 4] = 1.0
    num *= valid[..., None]
    hpc = np.where(valid, pc_ids(d["trace_pc"][pc], vocab), 0).astype(np.int16)
    return hpc, num.astype(np.float32)


def reuse_target(d, sel):
    """Auxiliary training target (label information, never a model input): log2(1 + reuse distance)/21 per candidate, capped at RD_CAP."""
    idx = np.arange(len(d["seq"])) if sel is None else np.asarray(sel)
    seq = d["seq"][idx].astype(np.int64)
    nu = d["next_use"][idx].astype(np.int64)
    dist = np.where(nu == NO_NEXT_USE, RD_CAP, np.minimum(nu - seq[:, None], RD_CAP))
    return (np.log2(1.0 + dist.astype(np.float64)) / 21.0).astype(np.float32)
