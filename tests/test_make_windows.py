#!/usr/bin/env python3
"""scripts/make_windows.py end to end on small fake data (no PyTorch needed; random-weight MLP, LSTM and GBDT files):
 - builds a fake results tree (4 test traces, inputs.tsv, model_set.json, models_ext with mix1 and mix1b but NO mix2),
 - runs make_windows.py (4 single-application windows + the shared window),
 - checks: 5 output files, expected policy labels (mix2 skipped with a note), hits + misses = window length, the window starts after the warm-up,
   inspector records exist for the two inspected policies of each file, learned candidates carry scores, a missing model set never aborts the run."""
import json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
import dataset, modelio
import lightgbm as lgb

tmp = tempfile.mkdtemp(); errors = []
def check(c, m):
    if not c: errors.append(m)

W = ["hash", "graph", "matrix", "conv"]
ARGS = {"hash": "18 100000 1500000 1 5", "graph": "100000 8 0 5 4", "matrix": "700 0 1", "conv": "832 3 6 1"}
tsv = os.path.join(tmp, "inputs.tsv")
with open(tsv, "w") as f:
    for w in W: f.write("test\t%s\t0\t%s\n" % (w, ARGS[w]))
trd = os.path.join(tmp, "traces"); os.makedirs(os.path.join(trd, "test"))

def make_trace(path, seed, n, pc0, base):
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, 3, n)
    hot = rng.integers(0, 700, n); scan = 5000 + (np.arange(n) % 4500); cold = 100000 + rng.integers(0, 30000, n)
    lines = (np.where(pick == 0, hot, np.where(pick == 1, scan, cold)) + base).astype(np.uint64)
    rec = np.zeros(n, dtype=dataset.TRACE_DTYPE)
    rec["addr"] = lines << np.uint64(6); rec["pc"] = pc0 + (pick.astype(np.uint64) << np.uint64(4)); rec["icount"] = 3
    rec["size"] = 64; rec["flags"] = (rng.integers(0, 4, n) == 0).astype(np.uint8)
    hdr = np.zeros(1, dtype=[("magic", "S8"), ("v", "<u4"), ("rs", "<u4"), ("n", "<u8"), ("ic", "<u8"), ("res", "S32")])
    hdr["magic"] = b"ACTRACE"; hdr["v"] = 1; hdr["rs"] = 24; hdr["n"] = n; hdr["ic"] = 3 * n
    with open(path, "wb") as f: f.write(hdr.tobytes()); f.write(rec.tobytes())

for i, w in enumerate(W):
    make_trace(os.path.join(trd, "test", "test__%s%s.llc.trace" % (w, "".join("_" + x for x in ARGS[w].split()))), 20 + i, 24000 + 3000 * i, 0x400000 + 0x1000 * (i % 2), 7000 * i)

vocab = np.array([0x400000, 0x400010, 0x400020, 0x401000, 0x401010], dtype=np.uint64)
def mlp(path, seed, hidden=8):
    rng = np.random.default_rng(seed); emb = 3
    layers = [(rng.normal(0, .5, (hidden, 9 + 2 * emb)), rng.normal(0, .1, hidden)), (rng.normal(0, .5, (1, hidden)), rng.normal(0, .1, 1))]
    modelio.write_mlp(path, vocab, rng.normal(3, 1, 9), rng.uniform(.5, 2, 9), rng.normal(0, 1, (len(vocab) + 1, emb)), rng.normal(0, 1, (len(vocab) + 1, emb)), layers)
def lstm(path, seed):
    rng = np.random.default_rng(seed)
    shp = modelio._lstm_shapes(9, 3, len(vocab) + 1, 6, 5, 3, 7, 6)
    w = {k: rng.normal(0, .4, v) for k, v in shp.items()}; w["std"] = rng.uniform(.5, 2, 9); w["mean"] = rng.normal(3, 1, 9)
    modelio.write_lstm(path, vocab, w, 6, 5, 3, 7, 6)
def gbdt(path, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (600, 11)); y = (X[:, 0] + rng.normal(0, .5, 600) > 0).astype(np.float32)
    b = lgb.train(dict(objective="binary", num_leaves=7, verbose=-1, seed=seed), lgb.Dataset(X, y), num_boost_round=6)
    modelio.write_gbdt(path, vocab, b.model_to_string())

md = os.path.join(tmp, "models"); me = os.path.join(tmp, "models_ext"); os.makedirs(md); os.makedirs(me)
mk = {"mlp": mlp, "gbdt": gbdt, "lstm": lstm}
stem = {"mlp": "mlp", "gbdt": "lgbm", "lstm": "lstm"}
files = {}; seed = 100
for fam in ("mlp", "gbdt", "lstm"):
    files[fam] = {}
    for w in W + ["all"]:
        seed += 1
        p = os.path.join(md, "%s_%s.%s" % (w, stem[fam], fam)); mk[fam](p, seed); files[fam][w] = p
        for ver in ("mix1", "mix1b"):
            seed += 1; mk[fam](os.path.join(me, "%s_%s_%s.%s" % (w, stem[fam], ver, fam)), seed)
setj = os.path.join(tmp, "model_set.json")
json.dump({"version": 1, "selected": {"mlp": "", "gbdt": "", "lstm": ""}, "files": files}, open(setj, "w"))

out = os.path.join(tmp, "win")
r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_windows.py"), "--len", "1500", "--out", out, "--cfg", os.path.join(ROOT, "configs", "small_test.cfg"),
                    "--set", setj, "--inputs", tsv, "--traces", trd, "--models", me, "--bin", os.path.join(ROOT, "build"), "--slice", "300"], capture_output=True, text=True)
if r.returncode != 0: errors.append("make_windows.py failed:\n%s\n%s" % (r.stdout[-1500:], r.stderr[-2000:]))
check("skipped gen: mix2 MLP" in r.stdout, "the missing mix2 set was not reported as skipped")

want_single = ["LRU", "DRRIP", "Belady (oracle)", "MLP specialized", "MLP generic", "GBDT specialized", "GBDT generic", "LSTM specialized", "LSTM generic"]
want_shared = ["LRU", "DRRIP", "Belady (oracle)", "gen: frozen MLP", "gen: mix1 MLP", "gen: mix1b MLP", "gen: mix1b GBDT", "gen: mix1b LSTM",
               "req: frozen MLP", "req: mix1b MLP", "req: mix1b LSTM", "own: frozen MLP", "own: mix1b MLP"]
def verify(fn, want, insp_labels):
    p = os.path.join(out, fn)
    if not os.path.exists(p): errors.append("missing " + fn); return
    J = json.load(open(p)); L = J["len"]
    check(J["start"] >= J["warmup"], fn + ": window starts inside the warm-up")
    got = [x["label"] for x in J["policies"]]
    check(got == want, "%s: labels %s != %s" % (fn, got, want))
    for x in J["policies"]:
        check(x["window_hits"] + x["window_misses"] == L, "%s %s: hits+misses != len" % (fn, x["label"]))
        if x["label"] in insp_labels:
            d = x.get("decisions", [])
            check(len(d) > 0, "%s %s: no inspector decisions" % (fn, x["label"]))
            check(all(len(c) == 16 for dd in d for c in dd["cand"]), "%s %s: candidates are not 16 fields" % (fn, x["label"]))
        else:
            check("decisions" not in x, "%s %s: unexpected decisions" % (fn, x["label"]))
    return J
for w in W: verify("single_%s.json" % w, want_single, ["MLP specialized", "MLP generic"])
J = verify("shared_default.json", want_shared, ["gen: mix1b MLP", "req: mix1b MLP"])
if J: check(J["apps"] == W, "shared apps %s" % J["apps"])
check(not os.path.exists(os.path.join(out, "_merged.llc.trace")), "temporary merged trace not removed")
print("OK" if not errors else "\n".join(errors))
sys.exit(1 if errors else 0)
