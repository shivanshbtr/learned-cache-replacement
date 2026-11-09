"""Dataset loading for the Belady-labeled eviction decisions (see docs/dataset_format.md).

A dataset file (.npz) holds ONE LLC trace plus the decisions the labeler recorded on it:
  per decision d (N of them):  seq[d], set[d], label_mask[d], driver_victim[d]
  per decision and way w:      last_idx[d,w], insert_idx[d,w], count[d,w], next_use[d,w]   (next_use = label info only)
  whole trace (T accesses):    trace_line[T] (line address), trace_pc[T], trace_write[T], trace_icount[T]
Model inputs must use only information from the past: line/pc of the resident lines come from trace_*[last_idx],
"time since last access" = seq - last_idx, "age" = seq - insert_idx, "count" = count. NEVER feed next_use to a model.
"""
import json
import numpy as np

NO_NEXT_USE = 0xFFFFFFFF
TRACE_DTYPE = np.dtype([("addr", "<u8"), ("pc", "<u8"), ("icount", "<u4"), ("size", "u1"), ("flags", "u1"), ("pad", "<u2")])
DEC_HEADER = np.dtype([("magic", "S8"), ("version", "<u4"), ("ways", "<u4"), ("rec_words", "<u4"), ("stride", "<u4"),
                       ("sets", "<u4"), ("line_bytes", "<u4"), ("num_records", "<u8"), ("trace_len", "<u8"),
                       ("warmup", "<u8"), ("driver", "S8")])


def read_trace(path):
    """Read an LLC trace written by cachesim (header 64 B + 24 B records) into a structured array."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw[:7].tobytes() != b"ACTRACE":
        raise ValueError("not an ACTRACE file: %s" % path)
    return np.frombuffer(raw[64:].tobytes(), dtype=TRACE_DTYPE)


def read_dec(path):
    """Read a .dec file written by the labeler. Returns (header dict, uint32 array [N, rec_words])."""
    with open(path, "rb") as f:
        h = np.frombuffer(f.read(64), dtype=DEC_HEADER)[0]
        if not h["magic"].startswith(b"ACDEC") or h["version"] != 1:
            raise ValueError("not an ACDEC v1 file: %s" % path)
        data = np.fromfile(f, dtype="<u4")
    rw, n = int(h["rec_words"]), int(h["num_records"])
    if data.size != rw * n:
        raise ValueError("size mismatch in %s: %d words, expected %d" % (path, data.size, rw * n))
    hd = {k: (h[k].decode().rstrip("\0") if h[k].dtype.kind == "S" else int(h[k])) for k in DEC_HEADER.names}
    return hd, data.reshape(n, rw)


def build_npz(trace_path, dec_path, out_path, meta_extra=None):
    """Pack trace + decisions into one compressed .npz. Returns the dict of arrays (without compression)."""
    tr = read_trace(trace_path)
    hd, rec = read_dec(dec_path)
    if hd["trace_len"] != len(tr):
        raise ValueError("trace length %d does not match labeler header %d" % (len(tr), hd["trace_len"]))
    ways = hd["ways"]
    body = rec[:, 4:].reshape(len(rec), ways, 4)
    meta = dict(hd, trace=str(trace_path), **(meta_extra or {}))
    arrays = dict(
        seq=rec[:, 0].copy(), set=rec[:, 1].copy(), label_mask=rec[:, 2].copy(), driver_victim=rec[:, 3].copy(),
        last_idx=np.ascontiguousarray(body[:, :, 0]), insert_idx=np.ascontiguousarray(body[:, :, 1]),
        count=np.ascontiguousarray(body[:, :, 2]), next_use=np.ascontiguousarray(body[:, :, 3]),
        trace_line=(tr["addr"] >> np.uint64(int(np.log2(hd["line_bytes"])))).astype(np.uint64),
        trace_pc=tr["pc"].copy(), trace_write=(tr["flags"] & 1).astype(np.uint8), trace_icount=tr["icount"].copy(),
        meta=np.array(json.dumps(meta)),
    )
    np.savez_compressed(out_path, **arrays)
    return arrays


def load_npz(path):
    """Load a dataset file into a plain dict (meta decoded to a dict)."""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def measured(d):
    """Boolean mask of decisions after the warm-up part of the trace (use these for evaluation)."""
    return d["seq"] >= d["meta"]["warmup"]


def mask_to_bits(mask, ways):
    """label_mask [N] -> boolean matrix [N, ways] (True = that way is a Belady victim)."""
    return ((mask[:, None] >> np.arange(ways, dtype=np.uint32)[None, :]) & 1).astype(bool)


def informative(d):
    """Decisions where at least one way is NOT a Belady victim. Decisions whose label_mask covers every way (all lines
    never used again) carry no learning signal; drop them for training and report results with and without them."""
    full = (np.uint64(1) << np.uint64(d["meta"]["ways"])) - np.uint64(1)
    return d["label_mask"].astype(np.uint64) != full


def offline_baselines(d):
    """Top-1 agreement with the Belady victim set for simple rules, on measured decisions.
    A rule's victim counts as correct if it is in the (possibly tied) Belady set. Returns a dict of fractions."""
    m = measured(d)
    ways = d["meta"]["ways"]
    bits = mask_to_bits(d["label_mask"][m], ways)
    n = len(bits)
    if n == 0:
        return {}
    idx = np.arange(n)
    last, ins = d["last_idx"][m].astype(np.int64), d["insert_idx"][m].astype(np.int64)
    res = {
        "random (expected)": float((bits.sum(1) / ways).mean()),
        "lru (oldest last access)": float(bits[idx, last.argmin(1)].mean()),
        "mru (newest last access)": float(bits[idx, last.argmax(1)].mean()),
        "fifo (oldest insertion)": float(bits[idx, ins.argmin(1)].mean()),
        "driver policy": float(bits[idx, d["driver_victim"][m]].mean()),
        "tie fraction": float((bits.sum(1) > 1).mean()),
        "all-tied (no signal)": float((bits.sum(1) == ways).mean()),
    }
    return res


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        d = load_npz(p)
        m = measured(d)
        print("%s: %d decisions (%d measured), ways=%d, trace=%d accesses, driver=%s, stride=%d"
              % (p, len(d["seq"]), int(m.sum()), d["meta"]["ways"], len(d["trace_line"]), d["meta"]["driver"], d["meta"]["stride"]))
        for k, v in offline_baselines(d).items():
            print("   %-28s %.3f" % (k, v))
