#!/usr/bin/env python3
"""LSTM + attention check. (1) A model with RANDOM weights is written with modelio.write_lstm and run by the C++ simulator as the eviction policy
(labeler --driver lstm:<file>); every victim it chose must equal the argmax of the numpy reference forward pass (modelio.predict_lstm) on the
recorded state: this validates the C++ history ring, the step inputs, the LSTM, the attention and the file format. (2) If PyTorch is installed:
train 2 epochs with train_lstm.py (which itself checks torch == numpy scores) and repeat the C++ parity check on the trained model."""
import json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
import dataset, features, modelio

LABELER, CACHESIM = os.path.join(ROOT, "build", "labeler"), os.path.join(ROOT, "build", "cachesim")
CFG = os.path.join(ROOT, "configs", "small_test.cfg")
tmp = tempfile.mkdtemp()
errors = []
def check(c, msg):
    if not c: errors.append(msg)
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: errors.append("command failed: %s\n%s%s" % (" ".join(cmd), r.stdout[-500:], r.stderr[-1500:]))
    return r

def make_trace(path, seed, n=40000):
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, 3, n)
    hot = rng.integers(0, 700, n); scan = 5000 + (np.arange(n) % 4500); cold = 100000 + rng.integers(0, 30000, n)
    lines = np.where(pick == 0, hot, np.where(pick == 1, scan, cold)).astype(np.uint64)
    rec = np.zeros(n, dtype=dataset.TRACE_DTYPE)
    rec["addr"] = lines << np.uint64(6); rec["pc"] = 0x400000 + (pick.astype(np.uint64) << np.uint64(4)); rec["icount"] = 3
    rec["size"] = 64; rec["flags"] = (rng.integers(0, 4, n) == 0).astype(np.uint8)
    hdr = np.zeros(1, dtype=[("magic", "S8"), ("v", "<u4"), ("rs", "<u4"), ("n", "<u8"), ("ic", "<u8"), ("res", "S32")])
    hdr["magic"] = b"ACTRACE"; hdr["v"] = 1; hdr["rs"] = 24; hdr["n"] = n; hdr["ic"] = 3 * n
    with open(path, "wb") as f: f.write(hdr.tobytes()); f.write(rec.tobytes())

def parity(model_path, trace, label):
    dec, npz = os.path.join(tmp, "m.dec"), os.path.join(tmp, "m.npz")
    run([LABELER, "--config", CFG, "--trace", trace, "--driver", "lstm:%s" % model_path, "--out", dec])
    dataset.build_npz(trace, dec, npz)
    d = dataset.load_npz(npz)
    N = len(d["seq"]); check(N > 3000, "%s: too few decisions (%d)" % (label, N))
    m = modelio.read_lstm(model_path)
    num, pcid, curid, bits = features.build(d, m["vocab"])
    hpc, hnum = features.build_hist(d, m["vocab"], None, m["hist"])
    sc = modelio.predict_lstm(m, num, pcid, curid, hpc, hnum)
    pick = sc.argmax(1)
    srt = np.sort(sc, 1); tied = (srt[:, -1] - srt[:, -2]) < 1e-4
    mism = int((pick != d["driver_victim"]).sum()); hard = int(((pick != d["driver_victim"]) & ~tied).sum())
    print("lstm parity (%s): %d decisions, %d victims differ from numpy argmax (%d outside numeric ties)" % (label, N, mism, hard))
    check(hard == 0, "%s: %d C++ victims differ from the numpy model" % (label, hard))
    # the policy must also run under cachesim and report its stats
    out = os.path.join(tmp, "r.json")
    run([CACHESIM, "run", "--config", CFG, "--trace", trace, "--policies", "lru,lstm:%s,belady" % model_path, "--out", out])
    res = json.load(open(out)); p = {x["name"]: x for x in res["policies"]}
    key = [k for k in p if k.startswith("lstm:")][0]
    st = p[key].get("model", {})
    check(st.get("kind") == "lstm" and st.get("params") == modelio.lstm_params(m) and st.get("lat_mean_ns", 0) > 0, "%s: model stats wrong: %s" % (label, st))
    check(st.get("model_bytes") == os.path.getsize(model_path), "%s: model_bytes != file size" % label)
    check(p["belady"]["misses"] <= p[key]["misses"], "%s: model beat Belady (bug)" % label)
    print("   %s misses %d, lru %d, belady %d, stats %s" % (key, p[key]["misses"], p["lru"]["misses"], p["belady"]["misses"], json.dumps(st)))

def random_model(path, seed, hist, hidden, emb, dpos, dc, dh):
    rng = np.random.default_rng(seed)
    vocab = np.array([0x400000, 0x400010, 0x400020, 0x400030], dtype=np.uint64)   # three PCs of the synthetic trace + one unused
    n_ids = len(vocab) + 1; n_num = features.NUM_FEATS
    shp = modelio._lstm_shapes(n_num, emb, n_ids, hist, hidden, dpos, dc, dh)
    w = {k: rng.normal(0, 0.5, s).astype(np.float32) for k, s in shp.items()}
    w["mean"] = rng.normal(0, 1, shp["mean"]).astype(np.float32); w["std"] = rng.uniform(0.5, 2.0, shp["std"]).astype(np.float32)
    modelio.write_lstm(path, vocab, w, hist, hidden, dpos, dc, dh)

tt = os.path.join(tmp, "test.llc.trace"); make_trace(tt, 99)
for i, (hist, hidden, emb, dpos, dc, dh) in enumerate([(8, 6, 3, 2, 5, 4), (16, 12, 4, 4, 10, 8), (1, 4, 2, 1, 3, 3)]):
    mp = os.path.join(tmp, "rand%d.lstm" % i)
    random_model(mp, 10 + i, hist, hidden, emb, dpos, dc, dh)
    m = modelio.read_lstm(mp)
    check(modelio.lstm_params(m) > 0 and m["hist"] == hist, "read_lstm round trip failed")
    parity(mp, tt, "random weights, H=%d hidden=%d" % (hist, hidden))

try:
    import torch  # noqa: F401
    have_torch = True
except Exception:
    have_torch = False
if have_torch:
    root = os.path.join(tmp, "ds")
    for split, seed in (("train", 1), ("train", 2), ("val", 3)):
        os.makedirs(os.path.join(root, split), exist_ok=True)
        label = "%s__synth_%d" % (split, seed)
        t = os.path.join(tmp, label + ".llc.trace"); make_trace(t, seed)
        for drv in ("lru", "belady"):
            dec = os.path.join(tmp, "x.dec")
            run([LABELER, "--config", CFG, "--trace", t, "--driver", drv, "--out", dec])
            dataset.build_npz(t, dec, os.path.join(root, split, "%s.%s.npz" % (label, drv)))
    lp = os.path.join(tmp, "synth.lstm")
    r = run([sys.executable, os.path.join(ROOT, "training", "train_lstm.py"), "--data", root, "--workload", "synth", "--out", lp,
             "--epochs", "2", "--hist", "8", "--hidden", "8", "--dc", "8", "--dh", "8", "--max-per-file", "6000"])
    print("\n".join(r.stdout.strip().splitlines()[-3:]))
    if r.returncode == 0:
        check("export self-check" in r.stdout, "train_lstm did not run its export self-check")
        parity(lp, tt, "trained with PyTorch")
else:
    print("PyTorch not installed: training script not exercised (test_lstm part 2 SKIPPED)")
print("test_lstm: %s" % ("FAILED" if errors else "OK (C++ LSTM policy reproduces the numpy model decision by decision)"))
for e in errors: print("  ", e)
sys.exit(1 if errors else 0)
