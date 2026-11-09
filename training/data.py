"""Select dataset files and build training/eval matrices."""
import glob, json, os
import numpy as np
import dataset, features


def list_files(root, split, workload=None, drivers=None):
    """.npz files under root/<split>/. workload: 'hash', 'graph', ... or None/'all'. drivers: list like ['lru','belady'] or None."""
    out = []
    for f in sorted(glob.glob(os.path.join(root, split, "*.npz"))):
        b = os.path.basename(f)
        if workload and workload != "all" and not (b.startswith("%s__%s_" % (split, workload)) or b.startswith("%s__mix" % split)):
            continue   # solo files of the workload, plus merged multi-application files (mixed-stream DAgger; their view is chosen in load())
        if drivers and not any(b.endswith(".%s.npz" % d) for d in drivers):
            continue
        out.append(f)
    return out


def workload_of(path):
    return os.path.basename(path).split("__", 1)[1].split("_", 1)[0]


PC_APP_SHIFT = 56   # merged traces (multisim merge): application id in PC bits 56+ and in line bits 48+


def is_mixed(d):
    return "apps" in d["meta"]


def mix_view(d, workload):
    """View of a MERGED multi-application dataset for the model of `workload` ('all'/None = the generic model).
    The model of a workload sees the PCs of its own application(s) only; the PC of every other application's access is shown as 0 (unknown),
    exactly as multisim shows it. Only decisions requested by its own application are training rows. The generic model sees all PCs and all
    decisions. Application bits are removed from the PCs; the line addresses keep them (a line of application X is a different line)."""
    pc = d["trace_pc"]
    app = (pc >> np.uint64(PC_APP_SHIFT)).astype(np.int64)
    plain = pc & np.uint64((1 << PC_APP_SHIFT) - 1)
    d2 = dict(d)
    d2["trace_line_plain"] = d["trace_line"] & np.uint64((1 << 48) - 1)   # history features (LSTM) use plain line addresses, as the simulator does
    if workload in (None, "all"):
        d2["trace_pc"] = plain
        return d2
    names = d["meta"]["apps"]
    mine = np.array([n == workload for n in names] + [False] * (256 - len(names)))[app]
    d2["trace_pc"] = np.where(mine, plain, np.uint64(0))
    d2["_keep"] = mine[d["seq"].astype(np.int64)]     # decision d is requested by application app[seq[d]]
    return d2


def load(path, workload=None):
    d = dataset.load_npz(path)
    return mix_view(d, workload) if is_mixed(d) else d


def load_rows(files, vocab, max_per_file=60000, seed=0, informative_only=True, measured_only=True, hist=0, workload=None):
    """Concatenate features of all files. Returns dict(num, pcid, curid, bits, file_idx); with hist=H also hpc, hnum (LSTM history) and rd (reuse target).
    workload selects the view of merged multi-application files (see mix_view)."""
    parts = []
    for k, f in enumerate(files):
        d = load(f, workload)
        sel = features.select(d, informative_only, measured_only, max_per_file, seed + k, keep=d.get("_keep"))
        if len(sel) == 0:
            continue
        num, pcid, curid, bits = features.build(d, vocab, sel)
        part = [num, pcid, curid, bits, np.full(len(sel), k, dtype=np.int32)]
        if hist:
            hpc, hnum = features.build_hist(d, vocab, sel, hist)
            part += [hpc, hnum, features.reuse_target(d, sel)]
        parts.append(part)
    if not parts:
        raise SystemExit("no usable decisions in: %s" % files)
    names = ["num", "pcid", "curid", "bits", "file_idx"] + (["hpc", "hnum", "rd"] if hist else [])
    out = {}
    for i, k in enumerate(names):
        out[k] = np.concatenate([p[i] for p in parts])
        for p in parts: p[i] = None      # free the per-file pieces as soon as they are joined (halves the peak)
    return out


def mean_std(num, chunk=2_000_000):
    """Per-feature mean and std over all rows and ways of num [N,W,F]; float64 accumulation in chunks (no full-size temporaries: generic models load millions of rows).
    Returns float32 (mean, max(std, 1e-6))."""
    F = num.shape[-1]
    flat = num.reshape(-1, F); n = len(flat)
    s = np.zeros(F)
    for i in range(0, n, chunk): s += flat[i:i + chunk].sum(0, dtype=np.float64)
    mean = s / n
    v = np.zeros(F)
    for i in range(0, n, chunk): v += ((flat[i:i + chunk].astype(np.float64) - mean) ** 2).sum(0)
    return mean.astype(np.float32), np.maximum(np.sqrt(v / n), 1e-6).astype(np.float32)


def normalize(num, mean, std, chunk=100_000, inplace=False):
    """(num - mean) / std as float32, computed in chunks (same values as the one-shot expression, without three full-size temporaries).
    inplace=True overwrites `num` (float32) and returns it: no second copy."""
    out = num if inplace else np.empty(num.shape, dtype=np.float32)
    for i in range(0, len(num), chunk): out[i:i + chunk] = (num[i:i + chunk] - mean) / std
    return out


def vocab_from_files(files, k, workload=None):
    """k most frequent PCs (same counts as features.build_vocab over the loaded files). PC 0 (= 'unknown', what other applications' accesses
    are shown as) never takes a slot. Streams the files and reads only the arrays it needs: loading all files at once took >11 GB for the generic models."""
    tot = {}
    for f in files:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        d = {"meta": meta, "trace_pc": z["trace_pc"]}
        if "apps" in meta:
            d["trace_line"] = z["trace_line"]; d["seq"] = z["seq"]
            d = mix_view(d, workload)
        u, c = np.unique(d["trace_pc"], return_counts=True)
        for a, b in zip(u.tolist(), c.tolist()):
            tot[a] = tot.get(a, 0) + b
        del z, d
    top = sorted(tot.items(), key=lambda kv: (-kv[1], kv[0]))[:k + 1]
    v = np.array([p for p, _ in top], dtype=np.uint64)
    return v[v != 0][:k]


def top1(scores, bits):
    """Fraction of decisions whose argmax-score way is a Belady victim (ties resolved to the lowest way)."""
    pick = scores.argmax(1)
    return float(bits[np.arange(len(bits)), pick].mean())
