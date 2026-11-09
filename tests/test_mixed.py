#!/usr/bin/env python3
"""Mixed-stream DAgger plumbing (no torch needed):
 (1) multisim merge + labeler with the composite driver req=/own=/gen= gives EXACTLY the measured misses of `multisim run` (same schedule, models, warm-up);
 (2) a labeled merged trace exported to .npz: application views (other applications' PCs are 0, only own decisions are rows, vocab has no PC 0,
     the generic view sees all PCs); a LightGBM model trains on solo + merged files and loads in the simulator."""
import json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
import dataset, modelio, data, features

B = lambda n: os.path.join(ROOT, "build", n)
CFG = os.path.join(ROOT, "configs", "small_test.cfg")
tmp = tempfile.mkdtemp(); errors = []
def check(c, m):
    if not c: errors.append(m)
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: errors.append("command failed: %s\n%s%s" % (" ".join(cmd), r.stdout[-400:], r.stderr[-1200:]))
    return r

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

names = ["hash", "graph", "matrix"]
tr = {}
for i, (nm, n) in enumerate(zip(names, (30000, 24000, 36000))):
    tr[nm] = os.path.join(tmp, nm + ".llc.trace"); make_trace(tr[nm], 10 + i, n, 0x400000 + 0x1000 * (i % 2), 0 if i < 2 else 7000)
vocab = np.array([0x400000, 0x400010, 0x400020, 0x401000, 0x401010], dtype=np.uint64)
def rand_mlp(path, seed, hidden=8):
    rng = np.random.default_rng(seed); emb = 3
    layers = [(rng.normal(0, .5, (hidden, 9 + 2 * emb)), rng.normal(0, .1, hidden)), (rng.normal(0, .5, (1, hidden)), rng.normal(0, .1, 1))]
    modelio.write_mlp(path, vocab, rng.normal(3, 1, 9), rng.uniform(.5, 2, 9), rng.normal(0, 1, (len(vocab) + 1, emb)), rng.normal(0, 1, (len(vocab) + 1, emb)), layers)
def rand_lstm(path, seed):
    rng = np.random.default_rng(seed)
    shp = modelio._lstm_shapes(9, 3, len(vocab) + 1, 6, 5, 3, 7, 6)
    w = {k: rng.normal(0, .4, v) for k, v in shp.items()}; w["std"] = rng.uniform(.5, 2, 9); w["mean"] = rng.normal(3, 1, 9)
    modelio.write_lstm(path, vocab, w, 6, 5, 3, 7, 6)
mf = {}; lf = {}
for i, nm in enumerate(names + ["gen"]):
    mf[nm] = os.path.join(tmp, nm + ".mlp"); rand_mlp(mf[nm], 50 + i)
    lf[nm] = os.path.join(tmp, nm + ".lstm"); rand_lstm(lf[nm], 70 + i)
apps = []
for nm in names: apps += ["--app", "%s=%s" % (nm, tr[nm])]

# (1) equality with multisim ---------------------------------------------------------------------------------------------
for S in (300, 5000):
    merged, meta = os.path.join(tmp, "m%d.llc.trace" % S), os.path.join(tmp, "m%d.json" % S)
    run([B("multisim"), "merge", "--config", CFG] + apps + ["--slice-instr", str(S), "--out", merged, "--meta", meta])
    sets = ",".join("%s:%s" % (nm, mf[nm]) for nm in names)
    res = os.path.join(tmp, "r%d.json" % S)
    run([B("multisim"), "run", "--config", CFG] + apps + ["--slice-instr", str(S), "--policies", "lru,req:s,own:s,gen:g,belady", "--set", "s=" + sets, "--gen", "g=" + mf["gen"], "--out", res])
    want = {r["policy"]: r["misses"] for r in json.load(open(res))["runs"]}
    drv = {"lru": "lru", "req:s": "req=" + "+".join("mlp:" + mf[n] for n in names), "own:s": "own=" + "+".join("mlp:" + mf[n] for n in names),
           "gen:g": "gen=mlp:" + mf["gen"], "belady": "belady"}
    for pol, d in drv.items():
        st = os.path.join(tmp, "st.json")
        run([B("labeler"), "--config", CFG, "--trace", merged, "--driver", d, "--out", os.path.join(tmp, "x.dec"), "--stats", st])
        got = json.load(open(st))["measured_misses"]
        print("  slice %d %-7s labeler %d  multisim %d" % (S, pol, got, want[pol])) if os.environ.get("VERBOSE") else None
        check(got == want[pol], "slice %d %s: labeler on merged trace %d misses, multisim %d" % (S, pol, got, want[pol]))
check(json.load(open(os.path.join(tmp, "m300.json")))["apps"] == names, "meta apps")
# the same equality for the LSTM (global history ring: plain lines, other applications' PCs as 0) and for the bigger slice
for S in (300,):
    merged = os.path.join(tmp, "m%d.llc.trace" % S)
    sets = ",".join("%s:%s" % (nm, lf[nm]) for nm in names)
    res = os.path.join(tmp, "rl%d.json" % S)
    run([B("multisim"), "run", "--config", CFG] + apps + ["--slice-instr", str(S), "--policies", "req:s,own:s,gen:g", "--set", "s=" + sets, "--gen", "g=" + lf["gen"], "--out", res])
    want = {r["policy"]: r["misses"] for r in json.load(open(res))["runs"]}
    drv = {"req:s": "req=" + "+".join("lstm:" + lf[n] for n in names), "own:s": "own=" + "+".join("lstm:" + lf[n] for n in names), "gen:g": "gen=lstm:" + lf["gen"]}
    for pol, d in drv.items():
        st = os.path.join(tmp, "st.json")
        run([B("labeler"), "--config", CFG, "--trace", merged, "--driver", d, "--out", os.path.join(tmp, "x.dec"), "--stats", st])
        got = json.load(open(st))["measured_misses"]
        check(got == want[pol], "LSTM slice %d %s: labeler on merged trace %d misses, multisim %d" % (S, pol, got, want[pol]))

# (2) datasets and views -------------------------------------------------------------------------------------------------
merged, meta = os.path.join(tmp, "m300.llc.trace"), os.path.join(tmp, "m300.json")
ds = os.path.join(tmp, "ds"); os.makedirs(ds + "/train"); os.makedirs(ds + "/val")
def label(trace, out, drv, metaj=None):
    run([B("labeler"), "--config", CFG, "--trace", trace, "--driver", drv, "--out", out + ".dec", "--stride", "1"])
    cmd = [sys.executable, os.path.join(ROOT, "training", "export_npz.py"), "--trace", trace, "--dec", out + ".dec", "--out", out, "--delete-dec"]
    run(cmd + (["--meta-json", metaj] if metaj else []))
for split in ("train", "val"):
    for nm in names:   # solo files: lru and belady drivers (a stand-in for the existing datasets)
        for drv in ("lru", "belady"): label(tr[nm], "%s/%s/%s__%s_x.%s.npz" % (ds, split, split, nm, drv), drv)
    label(merged, "%s/%s/%s__mix0.mixreq1mlp.npz" % (ds, split, split), "req=" + "+".join("mlp:" + mf[n] for n in names), meta)
mfile = "%s/train/train__mix0.mixreq1mlp.npz" % ds
d = dataset.load_npz(mfile)
check(d["meta"].get("apps") == names, "npz meta has apps")
app_of = (d["trace_pc"] >> np.uint64(56)).astype(int)
check(set(np.unique(app_of)) == {0, 1, 2}, "all three applications present in the merged trace")
v = data.load(mfile, "graph")
check(set(np.unique(v["trace_pc"][app_of != 1])) == {0}, "graph view hides other applications' PCs")
check(np.all(v["trace_pc"][app_of == 1] == (d["trace_pc"][app_of == 1] & np.uint64((1 << 56) - 1))), "graph view keeps its own PCs (app bits removed)")
check(np.all(app_of[d["seq"][v["_keep"]]] == 1) and v["_keep"].sum() > 0, "graph view keeps only graph-requested decisions")
g = data.load(mfile, "all")
check("_keep" not in g and np.all(g["trace_pc"] < (1 << 56)), "generic view: all decisions, plain PCs")
check(len(np.unique(g["trace_line"] >> np.uint64(48))) == 3, "lines keep the application bits")
for w in names + ["all"]:
    files = data.list_files(ds, "train", w, ["lru", "belady", "mixreq1mlp"])
    voc = data.vocab_from_files(files, 16, w)
    check(0 not in voc.tolist() and len(voc) > 0, "vocab for %s has no PC 0" % w)
    R = data.load_rows(files, voc, 20000, 1, workload=w)
    check(R["bits"].shape[0] > 0 and R["num"].shape[1] == 16 and np.isfinite(R["num"]).all(), "rows for %s" % w)
    # mixed rows: curid of a per-app model never refers to another application's PC
    m = R["file_idx"] == [i for i, f in enumerate(files) if "mix0" in f][0]
    if w != "all": check(m.sum() > 0 and (R["curid"][m] > 0).mean() > 0.5, "own-requester rows for %s have known current PCs" % w)
# LSTM history features of merged files use PLAIN line addresses (deltas between accesses of different applications are small numbers, not ~2^48)
Rh = data.load_rows(data.list_files(ds, "train", "graph", ["lru", "mixreq1mlp"]), data.vocab_from_files(data.list_files(ds, "train", "graph", ["lru", "mixreq1mlp"]), 16, "graph"), 5000, 1, hist=6, workload="graph")
check(float(Rh["hnum"][..., 1].max()) < 1.0 and Rh["hnum"].shape[1] == 6, "history deltas use plain lines (max log-delta feature %.2f)" % float(Rh["hnum"][..., 1].max()))
# an app that does not appear in the merged file's apps list gets no rows from it
check(not data.load(mfile, "conv")["_keep"].any(), "workload absent from the mixture: no rows")

try:
    import lightgbm
    out = os.path.join(tmp, "graph_gbdt_mix1.gbdt")
    r = run([sys.executable, os.path.join(ROOT, "training", "train_lgbm.py"), "--data", ds, "--workload", "graph", "--drivers", "lru,belady,mixreq1mlp",
             "--out", out, "--rounds", "20", "--vocab", "16"])
    check(os.path.exists(out), "train_lgbm on solo + merged files")
    if os.path.exists(out):
        res = os.path.join(tmp, "g.json")
        run([B("multisim"), "run", "--config", CFG] + apps + ["--slice-instr", "300", "--policies", "req:s,lru", "--set", "s=hash:%s,graph:%s,matrix:%s" % (mf["hash"], out, mf["matrix"]), "--out", res])
except ImportError:
    print("lightgbm not installed: skipping gbdt training check")

if errors:
    print("FAILED:\n" + "\n".join(errors)); sys.exit(1)
print("test_mixed OK")
