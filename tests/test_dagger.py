#!/usr/bin/env python3
"""Pipeline check on a temporary tree with synthetic traces (small_test.cfg): make_datasets.sh -> train (LightGBM; LSTM if PyTorch is
installed) -> dagger_round.sh rounds 1 and 2 -> run_models.sh. Verifies that the DAgger datasets (specialized and generic states) and the
retrained models exist, that round 2 starts from the round-1 models, and that run_models.sh picks up every model (globs) and the mru baseline."""
import json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training")); sys.path.insert(0, os.path.join(ROOT, "tests"))
import dataset
tmp = tempfile.mkdtemp(); errors = []
def check(c, msg):
    if not c: errors.append(msg)
def sh(cmd, env=None):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run(cmd, capture_output=True, text=True, env=e)
    if r.returncode != 0: errors.append("command failed: %s\n%s%s" % (" ".join(cmd), r.stdout[-800:], r.stderr[-1500:]))
    return r
def make_trace(path, seed, n=30000):
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
TR, DS, MD, SIM = (os.path.join(tmp, x) for x in ("traces", "ds", "models", "sim"))
CFG = os.path.join(ROOT, "configs", "small_test.cfg")
for split, seed in (("train", 1), ("train", 2), ("val", 3), ("test", 4)):
    os.makedirs(os.path.join(TR, split), exist_ok=True); make_trace(os.path.join(TR, split, "%s__synth_%d.llc.trace" % (split, seed)), seed)
sh(["bash", os.path.join(ROOT, "scripts", "make_datasets.sh"), TR, CFG, DS], {"JOBS": "2", "MIN_FREE_GB": "0"})
py = sys.executable
for w in ("synth", "all"):
    sh([py, os.path.join(ROOT, "training", "train_lgbm.py"), "--data", DS, "--workload", w, "--out", os.path.join(MD, "%s_lgbm.gbdt" % w), "--rounds", "30", "--leaves", "15"])
env = dict(TR=TR, DS=DS, MD=MD, CFG=CFG, SIM=SIM, WORKLOADS="synth", JOBS="2", EXTRA_TRAIN="--rounds 30 --leaves 15")
for rnd in (1, 2):
    r = sh(["bash", os.path.join(ROOT, "scripts", "dagger_round.sh"), str(rnd), "lgbm"], env)
    for split in ("train", "val"):
        lab = "%s__synth_%d" % (split, 1 if split == "train" else 3)
        for tag in ("dag%dlgbm" % rnd, "dag%dalllgbm" % rnd):
            check(os.path.exists(os.path.join(DS, split, "%s.%s.npz" % (lab, tag))), "round %d: missing dataset %s.%s" % (rnd, lab, tag))
    for w in ("synth", "all"):
        check(os.path.exists(os.path.join(MD, "%s_lgbm_dag%d.gbdt" % (w, rnd))), "round %d: missing model %s_lgbm_dag%d" % (rnd, w, rnd))
    check(not os.path.exists(os.path.join(DS, "test", "test__synth_4.dag%dlgbm.npz" % rnd)), "DAgger touched the test split")
    check("retrain synth (lgbm) on drivers lru,belady" + "".join(",dag%dlgbm" % i for i in range(1, rnd + 1)) in r.stdout, "round %d: wrong driver list for retraining" % rnd)
res = json.load(open(os.path.join(SIM, "val", "val__synth_3.result.json")))
names = [p["name"] for p in res["policies"]]
print("policies in the val run:", ", ".join(names))
for want in ("mru", "gbdt:synth_lgbm", "gbdt:synth_lgbm_dag1", "gbdt:synth_lgbm_dag2", "gbdt:all_lgbm_dag2", "belady"):
    check(want in names, "run_models.sh did not run %s" % want)
out = sh([py, os.path.join(ROOT, "scripts", "summarize.py"), os.path.join(SIM, "val")]).stdout
check("best simple baseline" in out and "gbdt:synth_lgbm_dag2" in out, "summarize.py compact output incomplete")
try:
    import torch  # noqa: F401
    for w in ("synth", "all"):
        sh([py, os.path.join(ROOT, "training", "train_lstm.py"), "--data", DS, "--workload", w, "--out", os.path.join(MD, "%s_lstm.lstm" % w),
            "--epochs", "1", "--hist", "8", "--hidden", "8", "--dc", "8", "--dh", "8", "--max-per-file", "4000"])
    env["EXTRA_TRAIN"] = "--epochs 1 --hist 8 --hidden 8 --dc 8 --dh 8 --max-per-file 4000"; env["RUN_EVAL"] = "0"
    sh(["bash", os.path.join(ROOT, "scripts", "dagger_round.sh"), "1", "lstm"], env)
    check(os.path.exists(os.path.join(MD, "synth_lstm_dag1.lstm")) and os.path.exists(os.path.join(MD, "all_lstm_dag1.lstm")), "LSTM DAgger models missing")
except ImportError:
    print("PyTorch not installed: LSTM DAgger part SKIPPED")
print("test_dagger: %s" % ("FAILED" if errors else "OK (datasets, retraining, round 2 and run_models globs work)"))
for e in errors: print("  ", e)
sys.exit(1 if errors else 0)
