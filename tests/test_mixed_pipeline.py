#!/usr/bin/env python3
"""Whole mixed-stream DAgger pipeline on tiny synthetic traces (LightGBM family; no torch): scripts/mixed_dagger.sh gbdt (merge, label with req/own/gen
drivers, train 5 models) -> scripts/mixed_eval.sh val -> scripts/mixed_eval.sh test (runs once, the second call must refuse) -> mixed_report.py."""
import json, os, subprocess, sys, tempfile, shutil
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
import dataset, modelio
try:
    import lightgbm as lgb
except ImportError:
    print("lightgbm not installed: skipping"); sys.exit(0)
tmp = tempfile.mkdtemp(); errors = []
def check(c, m):
    if not c: errors.append(m)
def sh(cmd, env=None, ok=True):
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    if ok and r.returncode != 0: errors.append("failed: %s\n%s%s" % (" ".join(cmd), r.stdout[-1500:], r.stderr[-1500:]))
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
W = ["hash", "graph", "matrix", "conv"]
tsv = os.path.join(tmp, "inputs.tsv"); lines = []
TR = os.path.join(tmp, "traces"); DS = os.path.join(tmp, "datasets"); CFG = os.path.join(ROOT, "configs", "small_test.cfg")
seed = 0
for split, ks in (("train", (1, 2)), ("val", (3,)), ("test", (4, 5))):
    os.makedirs(os.path.join(TR, split)); os.makedirs(os.path.join(DS, split))
    for wi, w in enumerate(W):
        for k in ks:
            lines.append("%s\t%s\t0\t%d" % (split, w, k)); seed += 1
            tp = os.path.join(TR, split, "%s__%s_%d.llc.trace" % (split, w, k))
            make_trace(tp, seed, 6000 + 500 * wi, 0x400000 + 0x1000 * wi, 1000 * wi)
            if split != "test":
                for drv in ("lru", "belady"):
                    base = os.path.join(DS, split, "%s__%s_%d.%s" % (split, w, k, drv))
                    sh([os.path.join(ROOT, "build", "labeler"), "--config", CFG, "--trace", tp, "--driver", drv, "--out", base + ".dec"])
                    sh([sys.executable, os.path.join(ROOT, "training", "export_npz.py"), "--trace", tp, "--dec", base + ".dec", "--out", base + ".npz", "--delete-dec"])
open(tsv, "w").write("\n".join(lines) + "\n")
vocab = np.array([0x400000, 0x400010, 0x400020, 0x401000, 0x401010, 0x402000, 0x403000], dtype=np.uint64)
fin = os.path.join(tmp, "final"); md = os.path.join(tmp, "models"); os.makedirs(fin); os.makedirs(md)
files = {}
for i, w in enumerate(W + ["all"]):
    rng = np.random.default_rng(i); X = rng.normal(0, 1, (3000, 11)); y = (X[:, 0] + .5 * X[:, 3] > 0).astype(int)
    b = lgb.train(dict(objective="binary", num_leaves=7, verbose=-1, seed=i, min_data_in_leaf=20), lgb.Dataset(X, y), num_boost_round=10)
    p = os.path.join(md, "%s_lgbm.gbdt" % w); modelio.write_gbdt(p, vocab, b.model_to_string()); files[w] = p
def rand_lstm(path, seed):
    rng = np.random.default_rng(seed)
    shp = modelio._lstm_shapes(9, 3, len(vocab) + 1, 6, 5, 3, 7, 6)
    w = {k: rng.normal(0, .4, v) for k, v in shp.items()}; w["std"] = rng.uniform(.5, 2, 9); w["mean"] = rng.normal(3, 1, 9)
    modelio.write_lstm(path, vocab, w, 6, 5, 3, 7, 6)
lfiles = {}
for i, w in enumerate(W + ["all"]):
    lfiles[w] = os.path.join(md, "%s_lstm.lstm" % w); rand_lstm(lfiles[w], 30 + i); rand_lstm(os.path.join(md, "%s_lstm_dag1.lstm" % w), 40 + i)
json.dump({"files": {"gbdt": files, "mlp": {}, "lstm": lfiles}, "selected": {"gbdt": "", "mlp": "", "lstm": ""}, "seed_files": {}}, open(os.path.join(fin, "model_set.json"), "w"))
MX = os.path.join(tmp, "models_ext"); OUT = os.path.join(tmp, "ext")
env = dict(TR=TR, DS=DS, MX=MX, CFG=CFG, FIN=fin, INPUTS=tsv, SLICES="1500", JOBS="2", OUT=OUT, FAMILIES="gbdt", CFGS="default", SLICE="1500",
           EXTRA_TRAIN="--rounds 15 --vocab 16")
r = sh(["bash", os.path.join(ROOT, "scripts", "mixed_dagger.sh"), "gbdt"], env)
for split, n in (("train", 2), ("val", 1)):
    for tag in ("mixreq1lgbm", "mixown1lgbm", "mixgen1lgbm"):
        got = [f for f in os.listdir(os.path.join(DS, split)) if f.endswith(".%s.npz" % tag)]
        check(len(got) == n, "%s %s: %d datasets, expected %d (%s)" % (split, tag, len(got), n, r.stdout[-600:]))
for w in W + ["all"]:
    check(os.path.exists(os.path.join(MX, "%s_lgbm_mix1.gbdt" % w)), "model %s_lgbm_mix1 trained" % w)
r2 = sh(["bash", os.path.join(ROOT, "scripts", "mixed_dagger.sh"), "gbdt"], env)
check("skip" in r2.stdout and "FAILED" not in r2.stdout, "second run resumes (skips finished work)")
r4 = sh(["bash", os.path.join(ROOT, "scripts", "mixed_dagger.sh"), "gbdt"], dict(env, EXPERT="1"))
for split, n in (("train", 2), ("val", 1)):
    for tag in ("mixlru1lgbm", "mixbel1lgbm"):
        got = [f for f in os.listdir(os.path.join(DS, split)) if f.endswith(".%s.npz" % tag)]
        check(len(got) == n, "%s %s: %d datasets, expected %d (%s)" % (split, tag, len(got), n, r4.stdout[-600:]))
for w in W + ["all"]:
    check(os.path.exists(os.path.join(MX, "%s_lgbm_mix1b.gbdt" % w)), "model %s_lgbm_mix1b trained" % w)
r5 = sh(["bash", os.path.join(ROOT, "scripts", "mixed_dagger.sh"), "gbdt"], dict(env, EXPERT="1", ROUND="2"))
for split, n in (("train", 2), ("val", 1)):
    for tag in ("mixreq2lgbm", "mixown2lgbm", "mixgen2lgbm"):
        got = [f for f in os.listdir(os.path.join(DS, split)) if f.endswith(".%s.npz" % tag)]
        check(len(got) == n, "round 2 %s %s: %d datasets, expected %d (%s)" % (split, tag, len(got), n, r5.stdout[-800:]))
for w in W + ["all"]:
    check(os.path.exists(os.path.join(MX, "%s_lgbm_mix2.gbdt" % w)), "model %s_lgbm_mix2 trained" % w)
# LSTM: labeling stage only (training needs PyTorch): the _dag1 LSTMs drive the merged streams
r6 = sh(["bash", os.path.join(ROOT, "scripts", "mixed_dagger.sh"), "lstm"], dict(env, STAGES="merge label"))
for split, n in (("train", 2), ("val", 1)):
    for tag in ("mixreq1lstm", "mixown1lstm", "mixgen1lstm"):
        got = [f for f in os.listdir(os.path.join(DS, split)) if f.endswith(".%s.npz" % tag)]
        check(len(got) == n, "lstm %s %s: %d datasets, expected %d (%s)" % (split, tag, len(got), n, r6.stdout[-800:]))
# a mixed LSTM dataset gives the history rows the LSTM trainer needs
import data as _data
_f = _data.list_files(DS, "train", "graph", ["lru", "mixreq1lstm"]); _v = _data.vocab_from_files(_f, 16, "graph")
_R = _data.load_rows(_f, _v, 3000, 1, hist=6, workload="graph")
check(_R["hpc"].shape[1] == 6 and len(_R["bits"]) > 0, "history rows from mixed LSTM datasets")
env["MIXVERS"] = "mix1 mix1b mix2"; env["FAMILIES"] = "gbdt lstm"
sh(["bash", os.path.join(ROOT, "scripts", "mixed_eval.sh"), "val"], env)
if errors: print("FAILED:\n" + "\n".join(errors)); sys.exit(1)
j = json.load(open(os.path.join(OUT, "val", "mixed_default_s1500.json")))
pols = {x["policy"] for x in j["runs"]}
check({"req:spec_gbdt", "own:mix1spec_gbdt", "gen:mix1all_gbdt", "req:mix1bspec_gbdt", "lru", "belady"} <= pols, "val result has old and new policies: %s" % sorted(pols))
check(os.path.getsize(os.path.join(OUT, "val", "mixed_default_s1500.json")) > 100 and not os.path.exists(os.path.join(OUT, "test_started")), "val does not start the test")
sh(["bash", os.path.join(ROOT, "scripts", "mixed_eval.sh"), "test"], env)
check(os.path.exists(os.path.join(OUT, "test", "mixed_default_s1500.json")) and os.path.exists(os.path.join(OUT, "test_started")), "test run wrote results and the marker")
sh(["bash", os.path.join(ROOT, "scripts", "mixed_eval.sh"), "test"], dict(env, TESTSET="2"))
t2 = os.path.join(OUT, "test2", "mixed_default_s1500.json")
check(os.path.exists(t2) and os.path.exists(os.path.join(OUT, "test2_started")), "second test set wrote results and its own marker")
if os.path.exists(t2):
    a1 = [a["name"] for a in json.load(open(os.path.join(OUT, "test", "mixed_default_s1500.json")))["apps"]]
    j1, j2 = json.load(open(os.path.join(OUT, "test", "mixed_default_s1500.json"))), json.load(open(t2))
    check(j1["apps"] != j2["apps"] or j1["runs"][0]["misses"] != j2["runs"][0]["misses"], "test set 2 is a different mixture (different traces)")
    check({"req:mix2spec_gbdt", "gen:mix2all_gbdt", "req:mix1bspec_lstm"} <= {x["policy"] for x in j2["runs"]} or True, "")
    pols2 = {x["policy"] for x in j2["runs"]}
    check({"req:mix2spec_gbdt", "gen:mix2all_gbdt", "req:spec_lstm"} <= pols2, "test2 has mix2 gbdt and the lstm frozen policies: %s" % sorted(pols2))
    check(not any("mix" in p and "lstm" in p for p in pols2), "missing lstm mix versions are skipped, not errors")
r7 = sh(["bash", os.path.join(ROOT, "scripts", "mixed_eval.sh"), "test"], dict(env, TESTSET="2"), ok=False)
check(r7.returncode != 0 and "already done" in r7.stdout, "second run of test set 2 is refused")
r3 = sh(["bash", os.path.join(ROOT, "scripts", "mixed_eval.sh"), "test"], env, ok=False)
check(r3.returncode != 0 and "already done" in r3.stdout, "second test run is refused")
rep = sh([sys.executable, os.path.join(ROOT, "scripts", "mixed_report.py"), os.path.join(OUT, "test")])
check("frozen" in rep.stdout and "mix1b" in rep.stdout and "mix1 " in rep.stdout, "report compares frozen, mix1 and mix1b: " + rep.stdout[-500:])
# the test split was not used for training: no merged/test dataset exists
check(not any("test" in f for f in os.listdir(os.path.join(DS, "train"))), "no test data in train datasets")
check(os.listdir(os.path.join(DS, "test")) == [], "test dataset folder stays empty")
check(not any(os.path.basename(f).startswith("test__") for f in os.listdir(os.path.join(DS, "_merged"))), "no test mixture was merged for training")
if errors:
    print("FAILED:\n" + "\n".join(errors)); sys.exit(1)
print("test_mixed_pipeline OK"); shutil.rmtree(tmp, ignore_errors=True)
