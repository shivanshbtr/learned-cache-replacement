#!/usr/bin/env python3
"""Sample-window exporter (export/winexport.cpp), no torch needed:
 (1) a window that covers the whole trace reproduces the measured misses of `cachesim run` (lru, drrip, belady, mlp, lstm) and of the labeler on a merged trace
     with the composite drivers req= / own= / gen= (so recording changes nothing);
 (2) a normal window: hits + misses = window length; replaying the recorded events (way, hit) on the recorded initial state of the shown sets is
     consistent (a hit finds its line in that way, a miss never refills a line that is already in the set);
 (3) inspector records: the chosen victim of lru is the LRU way, of belady is in the Belady set, of a learned model is the highest score;
     16 candidates with 9 features each for learned policies; the window never starts inside the warm-up."""
import json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
import dataset, modelio

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

# (1a) whole-trace window == cachesim run ---------------------------------------------------------------------------------
T = tr["hash"]; N = 30000
pols = ["lru", "drrip", "belady", "mlp:" + mf["hash"], "lstm:" + lf["hash"]]
res = os.path.join(tmp, "r.json")
run([B("cachesim"), "run", "--config", CFG, "--trace", T, "--policies", ",".join(pols), "--out", res])
want = [p["misses"] for p in json.load(open(res))["policies"]]
wj = os.path.join(tmp, "w_all.json")
run([B("winexport"), "--config", CFG, "--trace", T, "--policies", ",".join(pols), "--start", "0", "--len", str(N), "--allow-warmup", "1", "--out", wj])
W = json.load(open(wj))
got = [p["measured_misses_to_window_end"] for p in W["policies"]]
check(got == want, "whole-trace window misses %s differ from cachesim %s" % (got, want))

# (1b) merged trace with composite drivers == labeler -----------------------------------------------------------------------
apps = []
for nm in names: apps += ["--app", "%s=%s" % (nm, tr[nm])]
merged, meta = os.path.join(tmp, "m.llc.trace"), os.path.join(tmp, "m.json")
run([B("multisim"), "run" if False else "merge", "--config", CFG] + apps + ["--slice-instr", "300", "--out", merged, "--meta", meta])
drv = {"lru": "lru", "belady": "belady", "req": "req=" + "+".join("mlp:" + mf[n] for n in names), "own": "own=" + "+".join("mlp:" + mf[n] for n in names),
       "gen": "gen=mlp:" + mf["gen"], "genlstm": "gen=lstm:" + lf["gen"]}
mj = os.path.join(tmp, "wm_all.json")
nm_total = json.load(open(meta))["merged_accesses"]
run([B("winexport"), "--config", CFG, "--trace", merged, "--policies", ",".join(drv.values()), "--start", "0", "--len", str(nm_total), "--allow-warmup", "1",
     "--apps", ",".join(names), "--out", mj])
WM = json.load(open(mj))
for (k, d), p in zip(drv.items(), WM["policies"]):
    st = os.path.join(tmp, "st.json")
    run([B("labeler"), "--config", CFG, "--trace", merged, "--driver", d, "--out", os.path.join(tmp, "x.dec"), "--stats", st])
    want_m = json.load(open(st))["measured_misses"]
    check(p["measured_misses_to_window_end"] == want_m, "merged %s: winexport %d, labeler %d" % (k, p["measured_misses_to_window_end"], want_m))

# (2) + (3) normal window ----------------------------------------------------------------------------------------------------
for label, trace, plist, insp in (("single", T, pols, ",".join(pols)), ("merged", merged, list(drv.values()), ",".join([drv["gen"], drv["req"], drv["lru"], drv["belady"]]))):
    wj = os.path.join(tmp, "w_%s.json" % label)
    run([B("winexport"), "--config", CFG, "--trace", trace, "--policies", ",".join(plist), "--start-frac", "0.5", "--len", "2000", "--sets", "12",
         "--inspect", insp, "--max-decisions", "40", "--apps", ",".join(names), "--out", wj])
    W = json.load(open(wj)); L = W["len"]; ways = W["ways"]
    check(L == 2000, "%s: window length %d" % (label, L))
    check(W["start"] >= W["warmup"], "%s: window starts inside the warm-up" % label)
    check(len(W["shown_sets"]) == 12, "%s: shown sets" % label)
    for key in ("set", "line", "pc", "wr"): check(len(W["acc"][key]) == L, "%s: acc.%s length" % (label, key))
    for p in W["policies"]:
        nm = p["name"]
        check(p["window_hits"] + p["window_misses"] == L, "%s %s: hits+misses != len" % (label, nm))
        check(len(p["o"]) == L and len(p["init"]) == 12 * ways, "%s %s: array sizes" % (label, nm))
        # replay on the shown sets
        grid = {s: list(p["init"][i * ways:(i + 1) * ways]) for i, s in enumerate(W["shown_sets"])}
        bad = 0
        for j in range(L):
            s, ln, o = W["acc"]["set"][j], W["acc"]["line"][j], p["o"][j]
            way, hit = o >> 1, o & 1
            if s not in grid: continue
            g = grid[s]
            if hit:
                if g[way] != ln: bad += 1
            else:
                if ln in g: bad += 1
                g[way] = ln
        check(bad == 0, "%s %s: %d inconsistent events in the replay of the shown sets" % (label, nm, bad))
        if "decisions" in p:
            check(len(p["decisions"]) > 0, "%s %s: no decisions recorded" % (label, nm))
            for d in p["decisions"]:
                c = d["cand"]
                check(len(c) == ways, "%s %s: %d candidates" % (label, nm, len(c)))
                if nm == "lru" or nm.startswith("lru"): check(d["v"] == d["lru"], "%s: lru policy did not choose the LRU way" % label)
                if nm == "belady": check(d["v"] in d["bel"], "%s: belady did not choose a Belady way" % label)
                if len(c[0]) > 5:
                    check(len(c[0]) == 16, "%s %s: candidate has %d fields" % (label, nm, len(c[0])))
                    sc = [x[6] for x in c]
                    if label == "single" or nm.startswith("gen="):
                        check(abs(sc[d["v"]] - max(sc)) < 1e-3 * (1 + abs(max(sc))), "%s %s: chosen way is not the top score" % (label, nm))

# (4) continuity: the initial state of a window that starts inside an earlier window equals the replayed state of that earlier window -------------
def window(start, length, sets):
    out = os.path.join(tmp, "c_%d_%d.json" % (start, length))
    run([B("winexport"), "--config", CFG, "--trace", T, "--policies", ",".join(pols), "--start", str(start), "--len", str(length), "--sets", str(sets), "--out", out])
    return json.load(open(out))
A = window(12000, 1500, 400)
ways = A["ways"]
def replayed(pa, upto):
    grid = {s: [(A["lines"][x] if x >= 0 else None) for x in pa["init"][i * ways:(i + 1) * ways]] for i, s in enumerate(A["shown_sets"])}
    for j in range(upto):
        s = A["acc"]["set"][j]; o = pa["o"][j]
        if s in grid and not (o & 1): grid[s][o >> 1] = A["lines"][A["acc"]["line"][j]]
    return grid
for off in range(100, 1400, 130):
    Bw = window(12000 + off, 100, 400)
    for pa, pb in zip(A["policies"], Bw["policies"]):
        grid = replayed(pa, off)
        badc = 0
        for i, s in enumerate(Bw["shown_sets"]):
            if s not in grid: continue
            want = [(Bw["lines"][x] if x >= 0 else None) for x in pb["init"][i * ways:(i + 1) * ways]]
            if want != grid[s]: badc += 1
        check(badc == 0, "continuity at offset %d, %s: %d sets differ" % (off, pa["name"], badc))
print("OK" if not errors else "\n".join(errors))
sys.exit(1 if errors else 0)
