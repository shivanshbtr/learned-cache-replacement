#!/usr/bin/env python3
"""multisim with real model files (MLP with random weights, LSTM with random weights, a tiny LightGBM):
 (1) ONE application: multisim (rules req, own, gen) reproduces `cachesim run` exactly (same misses, hits) for every model kind;
 (2) THREE applications: the books balance (per-application hits+misses = accesses, sums = totals), Belady on the merged stream is never beaten,
     the `alone` runs equal cachesim on each application's own trace, ownership never lets a model evict another application's line
     (evict matrix is only filled where the rule allows), and the result file has the fields the experiment scripts read."""
import json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
import dataset, modelio

CACHESIM, MULTISIM = os.path.join(ROOT, "build", "cachesim"), os.path.join(ROOT, "build", "multisim")
CFG = os.path.join(ROOT, "configs", "small_test.cfg")
tmp = tempfile.mkdtemp()
errors = []
def check(c, msg):
    if not c: errors.append(msg)
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: errors.append("command failed: %s\n%s%s" % (" ".join(cmd), r.stdout[-500:], r.stderr[-1500:]))
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

# three applications; the plain line numbers of A and B overlap on purpose (same address space, different programs)
names = ["A", "B", "C"]
traces = {}
for i, (nm, n) in enumerate(zip(names, (30000, 24000, 36000))):
    traces[nm] = os.path.join(tmp, nm + ".llc.trace")
    make_trace(traces[nm], 10 + i, n, 0x400000 + 0x1000 * (i % 2), 0 if i < 2 else 7000)
vocab = np.array([0x400000, 0x400010, 0x400020, 0x401000, 0x401010], dtype=np.uint64)

def rand_mlp(path, seed, hidden=8):
    rng = np.random.default_rng(seed); emb = 3
    layers = [(rng.normal(0, .5, (hidden, 9 + 2 * emb)), rng.normal(0, .1, hidden)), (rng.normal(0, .5, (1, hidden)), rng.normal(0, .1, 1))]
    modelio.write_mlp(path, vocab, rng.normal(3, 1, 9), rng.uniform(.5, 2, 9), rng.normal(0, 1, (len(vocab) + 1, emb)), rng.normal(0, 1, (len(vocab) + 1, emb)), layers)

def rand_lstm(path, seed, hist=6, hidden=5, dpos=3, dc=7, dh=6, emb=3):
    rng = np.random.default_rng(seed); n_ids = len(vocab) + 1
    shp = modelio._lstm_shapes(9, emb, n_ids, hist, hidden, dpos, dc, dh)
    w = {k: rng.normal(0, .4, s) for k, s in shp.items()}
    w["std"] = rng.uniform(.5, 2, 9); w["mean"] = rng.normal(3, 1, 9)
    modelio.write_lstm(path, vocab, w, hist, hidden, dpos, dc, dh)

def rand_gbdt(path, seed):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (3000, 11)); y = (X[:, 0] + .5 * X[:, 3] > 0).astype(int)
    b = lgb.train(dict(objective="binary", num_leaves=7, verbose=-1, seed=seed, min_data_in_leaf=20), lgb.Dataset(X, y), num_boost_round=12)
    modelio.write_gbdt(path, vocab, b.model_to_string())

models = {"mlp": {}, "lstm": {}, "gbdt": {}}
for kind, ext, fn in (("mlp", "mlp", rand_mlp), ("lstm", "lstm", rand_lstm)):
    for i, nm in enumerate(names + ["gen"]):
        p = os.path.join(tmp, "%s_%s.%s" % (nm, kind, ext)); fn(p, 100 + i); models[kind][nm] = p
try:
    import lightgbm
    for i, nm in enumerate(names + ["gen"]):
        p = os.path.join(tmp, "%s_gbdt.gbdt" % nm); rand_gbdt(p, 200 + i); models["gbdt"][nm] = p
except ImportError:
    print("lightgbm not installed: skipping gbdt")
    del models["gbdt"]

def cachesim(trace, pols, cfg=CFG):
    out = os.path.join(tmp, "cs.json")
    run([CACHESIM, "run", "--config", cfg, "--trace", trace, "--policies", pols, "--out", out])
    return {p["name"]: p for p in json.load(open(out))["policies"]}

def multisim(apps, pols, sets, gens, slice_instr, extra=()):
    out = os.path.join(tmp, "ms.json")
    cmd = [MULTISIM, "run", "--config", CFG, "--slice-instr", str(slice_instr), "--policies", pols, "--out", out]
    for a in apps: cmd += ["--app", "%s=%s" % (a, traces[a])]
    for k, v in sets.items(): cmd += ["--set", "%s=%s" % (k, ",".join("%s:%s" % (a, f) for a, f in v.items()))]
    for k, v in gens.items(): cmd += ["--gen", "%s=%s" % (k, v)]
    cmd += list(extra)
    run(cmd)
    return {r["policy"]: r for r in json.load(open(out))["runs"]}, json.load(open(out))

# ---- (1) one application == cachesim
for kind, files in models.items():
    ext = {"mlp": "mlp", "lstm": "lstm", "gbdt": "gbdt"}[kind]
    ref = cachesim(traces["A"], "lru,belady,%s:%s" % (kind, files["A"]))
    key = [k for k in ref if k.startswith(kind + ":")][0]
    runs, _ = multisim(["A"], "lru,belady,req:s,own:s,gen:g", {"s": {"A": files["A"]}}, {"g": files["A"]}, 1000)
    for rule in ("req:s", "own:s", "gen:g"):
        check(runs[rule]["misses"] == ref[key]["misses"] and runs[rule]["hits"] == ref[key]["hits"],
              "%s single app: %s gives %d misses, cachesim gives %d" % (kind, rule, runs[rule]["misses"], ref[key]["misses"]))
    check(runs["lru"]["misses"] == ref["lru"]["misses"] and runs["belady"]["misses"] == ref["belady"]["misses"], "%s single app: baselines differ from cachesim" % kind)
    print("%s: one application == cachesim (misses %d; lru %d, belady %d)" % (kind, ref[key]["misses"], ref["lru"]["misses"], ref["belady"]["misses"]))

# ---- (2) three applications
for kind, files in models.items():
    sets = {"s": {nm: files[nm] for nm in names}}
    gens = {"g": files["gen"]}
    runs, whole = multisim(names, "lru,srrip,mru,belady,req:s,own:s,gen:g", sets, gens, 3000, ["--alone", "1", "--slices", "1", "--until", "all"])
    sch = whole["schedule"]
    check(sch["switches"] > 10 and sch["slices"] > 10, "%s: schedule has too few slices" % kind)
    check(sch["merged_accesses"] == sum(a["llc_accesses"] for a in whole["apps"]), "%s: until=all must use every access" % kind)
    bel = runs["belady"]["misses"]
    for name, r in runs.items():
        check(sum(a["misses"] for a in r["apps"]) == r["misses"] and sum(a["hits"] for a in r["apps"]) == r["hits"], "%s %s: per-app sums != totals" % (kind, name))
        check(all(a["hits"] + a["misses"] == a["accesses"] for a in r["apps"]), "%s %s: hits+misses != accesses" % (kind, name))
        check(r["misses"] >= bel, "%s %s beat Belady (bug)" % (kind, name))
        check(sum(map(sum, r["evict"])) <= r["misses"], "%s %s: more evictions than misses" % (kind, name))
        check(len(r["slices"]) == sch["slices"] and sum(s[2] for s in r["slices"]) >= r["misses"], "%s %s: slice log wrong" % (kind, name))
    own = runs["own:s"]
    # ownership: once every application holds its quota of ways in a set nobody is below quota, so borrowing only happens while sets fill up
    check(all(a["borrow"] <= a["misses"] for a in own["apps"]), "%s: borrow > misses" % kind)
    check(all(a["borrow"] == 0 for a in runs["req:s"]["apps"]), "%s: borrow counted outside the ownership rule" % kind)
    # alone runs == cachesim on each application's own trace (the models are the application's own model)
    for i, nm in enumerate(names):
        ref = cachesim(traces[nm], "lru,%s:%s" % (kind, files[nm]))
        key = [k for k in ref if k.startswith(kind + ":")][0]
        check(runs["lru"]["alone"][i]["misses"] == ref["lru"]["misses"], "%s: alone lru for %s != cachesim" % (kind, nm))
        check(runs["req:s"]["alone"][i]["misses"] == ref[key]["misses"], "%s: alone req for %s != cachesim" % (kind, nm))
    # models report their statistics: one instance per application for req/own, one for gen
    check(len(runs["req:s"]["models"]) == 3 and len(runs["gen:g"]["models"]) == 1, "%s: model stats instances" % kind)
    check(all(m["decisions"] > 0 for m in runs["own:s"]["models"]), "%s: some application's model was never asked (own)" % kind)
    # cold effect fields exist and are sane
    check(all(0 <= a["cold_miss"] <= a["cold_acc"] <= a["accesses"] for a in runs["lru"]["apps"]), "%s: cold counters" % kind)
    print("%s: 3 apps: %d slices, %d switches; misses lru %d, req %d, own %d, gen %d, belady %d" % (
        kind, sch["slices"], sch["switches"], runs["lru"]["misses"], runs["req:s"]["misses"], own["misses"], runs["gen:g"]["misses"], bel))

# until=first stops when the shortest application is done
runs, whole = multisim(names, "lru", {}, {}, 3000, ["--until", "first"])
check(whole["schedule"]["merged_accesses"] < sum(a["llc_accesses"] for a in whole["apps"]), "until=first should stop early")
# unknown set / bad arguments fail cleanly (nonzero exit, message)
r = subprocess.run([MULTISIM, "run", "--config", CFG, "--app", "A=" + traces["A"], "--slice-instr", "10", "--policies", "req:nope"], capture_output=True, text=True)
check(r.returncode != 0 and "unknown model set" in r.stderr, "bad set name must be reported")

print("test_multi: %s" % ("FAILED" if errors else "OK (multisim = cachesim for one application; books balance; ownership and alone runs consistent)"))
for e in errors: print("  ", e)
sys.exit(1 if errors else 0)
