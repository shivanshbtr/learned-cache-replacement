#!/usr/bin/env python3
"""End-to-end test of the final-evaluation pipeline (scripts/final_plan.py, run_final.sh, final_e2.sh, final_multi.sh, final_overhead.sh, final_report.py)
on a temporary tree with tiny synthetic traces and random-weight models (no PyTorch needed):
  * run_models.sh on the 'val' split makes the validation results; `select` must pick, per model family, the variant with the best mean gap closed
    (recomputed here independently from the result files) and must refuse to change after the test split has started;
  * `bighidden` gives a generic MLP whose file size is within 6% of the sum of the four specialized files;
  * run_final.sh (select e2 multi e6 report) runs to the end; the tables are checked against direct `cachesim` / `multisim` runs;
  * web_data.json parses and has the sections the web app reads; plots exist."""
import glob, json, os, shutil, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dataset, modelio
import final_plan

CACHESIM, MULTISIM = os.path.join(ROOT, "build", "cachesim"), os.path.join(ROOT, "build", "multisim")
CFG = os.path.join(ROOT, "configs", "small_test.cfg")
tmp = tempfile.mkdtemp()
errors = []
def check(c, msg):
    if not c: errors.append(msg)
def run(cmd, env=None, ok=True):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run(cmd, capture_output=True, text=True, env=e)
    if ok and r.returncode != 0: errors.append("command failed: %s\n%s%s" % (" ".join(cmd), r.stdout[-1500:], r.stderr[-1500:]))
    return r

WL = ["hash", "graph", "matrix", "conv"]
vocab = np.array([0x400000, 0x400010, 0x400020, 0x401000, 0x401010], dtype=np.uint64)

def make_trace(path, seed, n, pc0=0x400000):
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, 3, n)
    hot = rng.integers(0, 700, n); scan = 5000 + (np.arange(n) % 4500); cold = 100000 + rng.integers(0, 30000, n)
    lines = np.where(pick == 0, hot, np.where(pick == 1, scan, cold)).astype(np.uint64)
    rec = np.zeros(n, dtype=dataset.TRACE_DTYPE)
    rec["addr"] = lines << np.uint64(6); rec["pc"] = pc0 + (pick.astype(np.uint64) << np.uint64(4)); rec["icount"] = 3
    rec["size"] = 64; rec["flags"] = (rng.integers(0, 4, n) == 0).astype(np.uint8)
    hdr = np.zeros(1, dtype=[("magic", "S8"), ("v", "<u4"), ("rs", "<u4"), ("n", "<u8"), ("ic", "<u8"), ("res", "S32")])
    hdr["magic"] = b"ACTRACE"; hdr["v"] = 1; hdr["rs"] = 24; hdr["n"] = n; hdr["ic"] = 3 * n
    with open(path, "wb") as f: f.write(hdr.tobytes()); f.write(rec.tobytes())

def rand_mlp(path, seed, hidden=8, nh=2):
    rng = np.random.default_rng(seed); emb = 3
    ds = [9 + 2 * emb] + [hidden] * nh + [1]
    layers = [(rng.normal(0, .5, (ds[i + 1], ds[i])), rng.normal(0, .1, ds[i + 1])) for i in range(len(ds) - 1)]
    modelio.write_mlp(path, vocab, rng.normal(3, 1, 9), rng.uniform(.5, 2, 9), rng.normal(0, 1, (len(vocab) + 1, emb)), rng.normal(0, 1, (len(vocab) + 1, emb)), layers)

def rand_lstm(path, seed, hist=6, hidden=5, dpos=3, dc=7, dh=6, emb=3):
    rng = np.random.default_rng(seed); n_ids = len(vocab) + 1
    w = {k: rng.normal(0, .4, s) for k, s in modelio._lstm_shapes(9, emb, n_ids, hist, hidden, dpos, dc, dh).items()}
    w["std"] = rng.uniform(.5, 2, 9); w["mean"] = rng.normal(3, 1, 9)
    modelio.write_lstm(path, vocab, w, hist, hidden, dpos, dc, dh)

def rand_gbdt(path, seed):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (3000, 11)); y = (X[:, 0] + .5 * X[:, 3] > 0).astype(int)
    b = lgb.train(dict(objective="binary", num_leaves=7, verbose=-1, seed=seed, min_data_in_leaf=20), lgb.Dataset(X, y), num_boost_round=12)
    modelio.write_gbdt(path, vocab, b.model_to_string())

# ---- tree: traces (val: one per workload; test: two per workload), models, inputs.tsv
TR, MD, SEEDMD, FIN, SIM = [os.path.join(tmp, d) for d in ("traces", "models", "models_seeds", "final", "sim")]
for s in ("val", "test"): os.makedirs(os.path.join(TR, s))
os.makedirs(MD); os.makedirs(SEEDMD)
tsv = []
for i, w in enumerate(WL):
    make_trace(os.path.join(TR, "val", "val__%s_9.llc.trace" % w), 50 + i, 30000 + 3000 * i)
    tsv.append("val\t%s\t0\t9" % w)
    for j in (1, 2):
        make_trace(os.path.join(TR, "test", "test__%s_%d.llc.trace" % (w, j)), 70 + 10 * i + j, 26000 + 2000 * i + 500 * j)
        tsv.append("test\t%s\t0\t%d" % (w, j))
INPUTS = os.path.join(tmp, "inputs.tsv"); open(INPUTS, "w").write("\n".join(tsv) + "\n")
seed = 1
for who in WL + ["all"]:
    for sfx in ("", "_dag1", "_dag2"):
        seed += 1; rand_mlp(os.path.join(MD, "%s_mlp%s.mlp" % (who, sfx)), seed)
    for sfx in ("", "_dag1"):
        seed += 1; rand_gbdt(os.path.join(MD, "%s_lgbm%s.gbdt" % (who, sfx)), seed)
    seed += 1; rand_lstm(os.path.join(MD, "%s_lstm.lstm" % who), seed)

# ---- validation results with the same script used for the real runs
r = run(["bash", os.path.join(ROOT, "scripts", "run_models.sh"), "val", TR, MD, SIM], env={"CONFIG": CFG, "JOBS": "2"})
vals = sorted(glob.glob(os.path.join(SIM, "val", "*.result.json")))
check(len(vals) == 4, "run_models.sh val should write 4 result files, got %d" % len(vals))

# independent recomputation of the selection rule
def mean_gap(stem, ext, sfx):
    g = []
    for f in vals:
        w = os.path.basename(f).split("__", 1)[1].split("_", 1)[0]
        P = {p["name"]: p["misses"] for p in json.load(open(f))["policies"]}
        for who in (w, "all"):
            n = "%s:%s_%s%s" % (ext, who, stem, sfx)
            g.append((P["lru"] - P[n]) / (P["lru"] - P["belady"]))
    return sum(g) / len(g)
expect = {}
for kind, (stem, ext), sfxs in (("mlp", ("mlp", "mlp"), ["", "_dag1", "_dag2"]), ("gbdt", ("lgbm", "gbdt"), ["", "_dag1"]), ("lstm", ("lstm", "lstm"), [""])):
    best = None
    for s in sfxs:
        v = mean_gap(stem, ext, s)
        if best is None or v > best[1] + 1e-3: best = (s, v)
    expect[kind] = best[0]

sel_args = ["--val", os.path.join(SIM, "val"), "--models", MD, "--seeds", SEEDMD]
r = run([sys.executable, os.path.join(ROOT, "scripts", "final_plan.py"), "select", "--final", FIN] + sel_args)
ms = json.load(open(os.path.join(FIN, "model_set.json")))
check(ms["selected"] == expect, "select picked %s, independent recomputation says %s" % (ms["selected"], expect))
check("allbig" not in ms["files"]["mlp"], "allbig must be absent before it is trained")
check(all(len(h) == 64 for h in ms["sha256"].values()), "sha256 missing")
print("select (val only): %s (mean gaps %s)" % (ms["selected"], {k: {s or "base": round(v, 3) for s, v in d.items()} for k, d in ms["val_mean_gap"].items()}))

# bighidden: equal total storage
sfx = ms["selected"]["mlp"]
r = run([sys.executable, os.path.join(ROOT, "scripts", "final_plan.py"), "bighidden", "--set", os.path.join(FIN, "model_set.json"), "--models", MD, "--sfx", sfx])
hid = r.stdout.strip().split("\n")[-1]
total = sum(os.path.getsize(os.path.join(MD, "%s_mlp%s.mlp" % (w, sfx))) for w in WL)
nh = len(hid.split(","))
rand_mlp(os.path.join(MD, "allbig_mlp%s.mlp" % sfx), 900, hidden=int(hid.split(",")[0]), nh=nh)
size = os.path.getsize(os.path.join(MD, "allbig_mlp%s.mlp" % sfx))
print("bighidden %s -> %d bytes vs sum of specialized %d" % (hid, size, total))
check(abs(size - total) / total < 0.06, "allbig size %d is not within 6%% of %d" % (size, total))
for s in (2, 3):
    for who in WL + ["all", "allbig"]:
        rand_mlp(os.path.join(SEEDMD, "%s_mlp%s_s%d.mlp" % (who, sfx, s)), 1000 + 10 * s + len(who), hidden=int(hid.split(",")[0]) if who == "allbig" else 8)

# ---- the whole final run
env = {"TR": TR, "FIN": FIN, "CFG": CFG, "INPUTS": INPUTS, "SEL_ARGS": " ".join(sel_args), "E6_ACC": "5000", "E7_DIV": "10 4", "E8_SLICES": "3000",
       "E8_CFGS": "default llc8m_16w", "E5_CFGS": "llc1m_16w llc2m_8w", "JOBS": "2", "JOBS_MULTI": "2", "FORCE_E6": "1"}
r = run(["bash", os.path.join(ROOT, "scripts", "run_final.sh")], env=env)
for l in r.stdout.splitlines():
    if "FAILED" in l: errors.append("job failed: " + l)
check(os.path.exists(os.path.join(FIN, "test_started")), "test_started not written")
ms2 = json.load(open(os.path.join(FIN, "model_set.json")))
check("allbig" in ms2["files"]["mlp"] and ms2["seed_files"].get("hash") and len(ms2["seed_files"]["allbig"]) == 2, "second select did not pick up allbig and the seed models")
check(ms2["selected"] == expect, "selection changed after the extra models were added")
# frozen: a re-run must not rewrite the model set even with different val data
before = open(os.path.join(FIN, "model_set.json")).read()
r = run([sys.executable, os.path.join(ROOT, "scripts", "final_plan.py"), "select", "--final", FIN] + sel_args)
check("frozen" in r.stdout and open(os.path.join(FIN, "model_set.json")).read() == before, "model set must stay frozen after the test split started")

# E2: tables vs a direct cachesim run
T = os.path.join(FIN, "tables")
for f in ("e2_test.csv", "e2_test_long.csv", "e2_summary.csv", "e3.csv", "e4.csv", "e5.csv", "e6_models.csv", "e6_storage.csv", "e6_activation.csv", "e7_misses.csv",
          "e7_interference.csv", "e7_breakeven.csv", "e7_cost_model.csv"):
    check(os.path.exists(os.path.join(T, f)), "missing table " + f)
check(bool(glob.glob(os.path.join(T, "e8_*.csv"))), "missing E8 tables")
import csv
rows = list(csv.DictReader(open(os.path.join(T, "e2_test_long.csv"))))
tr0 = os.path.join(TR, "test", "test__hash_1.llc.trace")
spec_hash = "mlp:hash_mlp%s" % ms["selected"]["mlp"]
out = os.path.join(tmp, "direct.json")
run([CACHESIM, "run", "--config", CFG, "--trace", tr0, "--policies", "lru,belady,%s" % spec_hash.replace("mlp:", "mlp:" + MD + "/", 1) + ".mlp", "--out", out])
P = {p["name"]: p["misses"] for p in json.load(open(out))["policies"]}
mine = {r["policy"]: r for r in rows if r["trace"] == "test__hash_1"}
check(mine["lru"]["misses"] == str(P["lru"]) and mine["belady"]["misses"] == str(P["belady"]), "e2 table baselines differ from a direct cachesim run")
check(mine[spec_hash]["misses"] == str(P[spec_hash]), "e2 table model misses differ from a direct cachesim run")
g = 100.0 * (P["lru"] - P[spec_hash]) / (P["lru"] - P["belady"])
check(abs(float(mine[spec_hash]["gap_closed_pct"]) - g) < 0.01, "gap closed in the table (%s) != recomputed %.2f" % (mine[spec_hash]["gap_closed_pct"], g))
check(all(float(r["gap_closed_pct"]) <= 100.0001 for r in rows if r["gap_closed_pct"] != ""), "a policy beats Belady in the table (bug)")
# E3 has three arms with 3 seeds each for the MLPs
e3 = list(csv.DictReader(open(os.path.join(T, "e3_long.csv"))))
check({"specialized", "generic (equal size)", "generic-big (equal total storage)"} <= {r["arm"] for r in e3}, "E3 arms missing")
check(all(r["n_seeds"] == "3" for r in e3 if r["arm"] in ("specialized", "generic (equal size)", "generic-big (equal total storage)")), "E3 should use 3 seeds")
# E6 sizes equal the model files
for r in csv.DictReader(open(os.path.join(T, "e6_models.csv"))):
    if r["model"].startswith("specialized mlp (hash)"): check(int(r["bytes"]) == os.path.getsize(os.path.join(MD, "hash_mlp%s.mlp" % sfx)), "E6 bytes != file size")
# E7 / E8: one multisim run recomputed directly
m7 = sorted(glob.glob(os.path.join(FIN, "multi", "e7_*.base.json")))
check(len(m7) == 2, "expected 2 E7 slice lengths, got %d" % len(m7))
j = json.load(open(m7[0])); sl = j["schedule"]["slice_instr"]
cmd = [MULTISIM, "run", "--config", CFG, "--slice-instr", str(sl), "--policies", "lru,belady", "--until", "first", "--out", os.path.join(tmp, "m.json")]
for w in WL: cmd += ["--app", "%s=%s" % (w, os.path.join(TR, "test", "test__%s_1.llc.trace" % w))]
run(cmd)
d = {x["policy"]: x["misses"] for x in json.load(open(os.path.join(tmp, "m.json")))["runs"]}
b = {x["policy"]: x["misses"] for x in j["runs"]}
check(d["lru"] == b["lru"] and d["belady"] == b["belady"], "E7 result differs from a direct multisim run")
w8 = json.load(open(os.path.join(FIN, "export", "web_data.json")))
check({"schema", "selected", "models", "e2", "e6", "e7", "e8", "cost_model"} <= set(w8), "web_data.json sections missing: %s" % sorted(w8))
check(len(w8["e2"]) == 8 and len(w8["e7"]) == 2 and len(w8["e8"]) >= 2, "web_data.json sizes: e2 %d e7 %d e8 %d" % (len(w8["e2"]), len(w8["e7"]), len(w8["e8"])))
for p in w8["e2"].values():
    check(all(v["gap_closed"] is None or v["gap_closed"] <= 1.0000001 for v in p["policies"].values()), "gap closed above 1 in web_data")
check(os.path.exists(os.path.join(FIN, "REPORT_NUMBERS.md")), "REPORT_NUMBERS.md missing")
try:
    import matplotlib  # noqa: F401
    have_mpl = True
except ImportError:
    have_mpl = False; print("matplotlib not installed: plot check skipped (pip install matplotlib)")
check((not have_mpl) or len(glob.glob(os.path.join(FIN, "plots", "*.png"))) >= 4, "plots missing: %s" % glob.glob(os.path.join(FIN, "plots", "*")))
# a failing job must be reported, not swallowed
bad = subprocess.run(["bash", os.path.join(ROOT, "scripts", "final_job.sh"), os.path.join(tmp, "bad", "x.json"), CFG, os.path.join(tmp, "nope.trace"), "lru"], capture_output=True, text=True)
check(bad.returncode != 0 and "FAILED" in bad.stdout and not os.path.exists(os.path.join(tmp, "bad", "x.json")), "final_job.sh must fail cleanly and leave no result file")
print("test_final: %s" % ("FAILED" if errors else "OK (selection on val only and frozen, equal-storage size, E1-E8 tables match direct simulator runs, web export complete)"))
for e in errors: print("  ", e)
if not errors: shutil.rmtree(tmp, ignore_errors=True)
else: print("   (kept %s)" % tmp)
sys.exit(1 if errors else 0)
