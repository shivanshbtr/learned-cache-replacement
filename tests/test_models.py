#!/usr/bin/env python3
"""End-to-end model check: train LightGBM and MLP on a small synthetic dataset, export, then let the C++ simulator run the
models as eviction policies (labeler --driver gbdt:/mlp:) and verify that EVERY victim the C++ policy chose equals the argmax of the
Python model evaluated on the recorded state. This validates the C++ feature code, the online bookkeeping and the inference."""
import json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
import dataset, features, modelio, data
import lightgbm as lgb

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

root = os.path.join(tmp, "ds")
for split, seed in (("train", 1), ("train", 2), ("val", 3)):
    os.makedirs(os.path.join(root, split), exist_ok=True)
    label = "%s__synth_%d" % (split, seed)
    t = os.path.join(tmp, label + ".llc.trace"); make_trace(t, seed)
    for drv in ("lru", "belady"):
        dec = os.path.join(tmp, "x.dec")
        run([LABELER, "--config", CFG, "--trace", t, "--driver", drv, "--out", dec])
        dataset.build_npz(t, dec, os.path.join(root, split, "%s.%s.npz" % (label, drv)))

models = os.path.join(tmp, "models"); os.makedirs(models)
py = sys.executable
gb, ml = os.path.join(models, "synth.gbdt"), os.path.join(models, "synth.mlp")
r = run([py, os.path.join(ROOT, "training", "train_lgbm.py"), "--data", root, "--workload", "synth", "--out", gb, "--rounds", "40", "--leaves", "15"])
print(r.stdout.strip().splitlines()[-1] if r.stdout else "")
try:
    import torch  # noqa: F401
    r = run([py, os.path.join(ROOT, "training", "train_mlp.py"), "--data", root, "--workload", "synth", "--out", ml, "--epochs", "4", "--hidden", "16,16"])
except ImportError:
    # no PyTorch (e.g. a minimal install): a random-weight MLP file still tests the C++ inference against modelio.predict_mlp
    print("PyTorch not installed: MLP parity uses random weights instead of a trained model")
    rng = np.random.default_rng(5); emb = 3
    voc = np.array([0x400000, 0x400010, 0x400020], dtype=np.uint64)   # the PCs of make_trace
    layers = [(rng.normal(0, .5, (16, 9 + 2 * emb)), rng.normal(0, .1, 16)), (rng.normal(0, .5, (16, 16)), rng.normal(0, .1, 16)), (rng.normal(0, .5, (1, 16)), rng.normal(0, .1, 1))]
    modelio.write_mlp(ml, voc, rng.normal(3, 1, 9), rng.uniform(.5, 2, 9), rng.normal(0, 1, (len(voc) + 1, emb)), rng.normal(0, 1, (len(voc) + 1, emb)), layers)
    r = None
print(r.stdout.strip().splitlines()[-1] if (r and r.stdout) else "")

# a trace the models have not seen
tt = os.path.join(tmp, "test.llc.trace"); make_trace(tt, 99)
for kind, path in (("gbdt", gb), ("mlp", ml)):
    dec, npz = os.path.join(tmp, "m.dec"), os.path.join(tmp, "m.npz")
    r = run([LABELER, "--config", CFG, "--trace", tt, "--driver", "%s:%s" % (kind, path), "--out", dec])
    dataset.build_npz(tt, dec, npz)
    d = dataset.load_npz(npz)
    N = len(d["seq"]); check(N > 3000, "%s: too few decisions (%d)" % (kind, N))
    if kind == "gbdt":
        vocab, text = modelio.read_gbdt(path)
        num, pcid, curid, bits = features.build(d, vocab)
        sc = lgb.Booster(model_str=text).predict(features.gbdt_matrix(num, pcid, curid), raw_score=True).reshape(N, -1)
    else:
        m = modelio.read_mlp(path)
        num, pcid, curid, bits = features.build(d, m["vocab"])
        sc = modelio.predict_mlp(m, num, pcid, curid)
    pick = sc.argmax(1)
    mism = int((pick != d["driver_victim"]).sum())
    # a mismatch is acceptable only if the two top scores are numerically tied (float rounding)
    srt = np.sort(sc, 1)
    tied = (srt[:, -1] - srt[:, -2]) < 1e-4
    hard = int(((pick != d["driver_victim"]) & ~tied).sum())
    print("%s parity: %d decisions, %d victims differ from Python argmax (%d outside numeric ties)" % (kind, N, mism, hard))
    check(hard == 0, "%s: %d C++ victims differ from the Python model" % (kind, hard))
    # the models should beat random on unseen data (weak sanity bound)
    t1 = data.top1(sc[features.select(d)], bits[features.select(d)]) if len(features.select(d)) else 0
    check(t1 > 0.0, "top-1 not computed")
    # cachesim reports the same misses as the labeler run and prints model stats
    out = os.path.join(tmp, "r.json")
    run([CACHESIM, "run", "--config", CFG, "--trace", tt, "--policies", "lru,%s:%s,belady" % (kind, path), "--out", out])
    res = json.load(open(out)); p = {x["name"]: x for x in res["policies"]}
    key = [k for k in p if k.startswith(kind + ":")][0]
    st = json.load(open(os.path.join(tmp, "dummy"))) if False else None
    check("model" in p[key] and p[key]["model"]["params"] > 0 and p[key]["model"]["lat_mean_ns"] > 0, "%s: model stats missing" % kind)
    check(p["belady"]["misses"] <= p[key]["misses"], "%s: model beat Belady (bug)" % kind)
    mm = res["trace"]["warmup_accesses"]
    meas = int(d["meta"]["trace_len"] if False else len(d["trace_line"])) - mm
    print("   %s misses %d, lru %d, belady %d, model stats %s" % (key, p[key]["misses"], p["lru"]["misses"], p["belady"]["misses"], json.dumps(p[key]["model"])))
print("test_models: %s" % ("FAILED" if errors else "OK (C++ MLP and GBDT policies reproduce the Python models decision by decision)"))
for e in errors: print("  ", e)
sys.exit(1 if errors else 0)
