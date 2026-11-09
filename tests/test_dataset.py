#!/usr/bin/env python3
"""Independent check of the labeler + exporter: builds a random LLC trace, labels it with LRU and Belady drivers, and
re-derives every recorded decision in pure Python (separate LRU simulation, brute-force next use)."""
import bisect, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
import dataset

LABELER = os.path.join(ROOT, "build", "labeler")
CFG = os.path.join(ROOT, "configs", "small_test.cfg")   # LLC 128 KB, 16-way, 128 sets
SETS, WAYS = 128, 16
rng = np.random.default_rng(7)
n = 30000
hot = rng.integers(0, 600, n)
scan = 5000 + np.arange(n)
cold = 100000 + rng.integers(0, 20000, n)
pick = rng.integers(0, 3, n)
lines = np.where(pick == 0, hot, np.where(pick == 1, scan % 4000 + 5000, cold)).astype(np.uint64)

tmp = tempfile.mkdtemp()
tpath = os.path.join(tmp, "t.llc.trace")
rec = np.zeros(n, dtype=dataset.TRACE_DTYPE)
rec["addr"] = lines << np.uint64(6); rec["pc"] = 0x400000 + (pick.astype(np.uint64) << np.uint64(4)); rec["icount"] = 3
rec["size"] = 64; rec["flags"] = (rng.integers(0, 4, n) == 0).astype(np.uint8)
hdr = np.zeros(1, dtype=[("magic", "S8"), ("v", "<u4"), ("rs", "<u4"), ("n", "<u8"), ("ic", "<u8"), ("res", "S32")])
hdr["magic"] = b"ACTRACE"; hdr["v"] = 1; hdr["rs"] = 24; hdr["n"] = n; hdr["ic"] = 3 * n
with open(tpath, "wb") as f:
    f.write(hdr.tobytes()); f.write(rec.tobytes())

errors = []
def check(c, msg):
    if not c and len(errors) < 20: errors.append(msg)

# positions of every line, for brute-force next-use
pos = {}
for i, l in enumerate(lines.tolist()): pos.setdefault(l, []).append(i)
def next_use(line, after):
    p = pos[line]; k = bisect.bisect_right(p, after)
    return p[k] if k < len(p) else dataset.NO_NEXT_USE

def python_lru_decisions():
    """Pure-python LRU, 128 sets x 16 ways, empty ways filled lowest first. Yields (i, set, residents[way]=line, victim_way)."""
    tags = [[None] * WAYS for _ in range(SETS)]; stamp = [[0] * WAYS for _ in range(SETS)]; clk = 0; out = []
    for i, l in enumerate(lines.tolist()):
        s = l % SETS; clk += 1
        if l in tags[s]:
            stamp[s][tags[s].index(l)] = clk; continue
        if None in tags[s]:
            w = tags[s].index(None)
        else:
            w = min(range(WAYS), key=lambda x: stamp[s][x]); out.append((i, s, list(tags[s]), w))
        tags[s][w] = l; stamp[s][w] = clk
    return out

for drv in ("lru", "belady"):
    for stride in (1, 3):
        dec = os.path.join(tmp, "x.dec"); npz = os.path.join(tmp, "x.npz")
        r = subprocess.run([LABELER, "--config", CFG, "--trace", tpath, "--driver", drv, "--stride", str(stride), "--out", dec],
                           capture_output=True, text=True)
        check(r.returncode == 0, "labeler failed: " + r.stderr)
        check(len(np.fromfile(dec, dtype="<u4")) > 4000, "too few decisions: test would be vacuous")
        dataset.build_npz(tpath, dec, npz)
        d = dataset.load_npz(npz)
        N = len(d["seq"])
        check(d["next_use"].shape == (N, WAYS), "shape")
        check(d["meta"]["driver"] == drv and d["meta"]["stride"] == stride and d["meta"]["warmup"] == int(0.1 * n), "meta %r" % d["meta"])
        for k in range(N):
            seq, st = int(d["seq"][k]), int(d["set"][k])
            check(st == int(lines[seq]) % SETS, "set mismatch at decision %d" % k)
            nus = []
            for w in range(WAYS):
                last = int(d["last_idx"][k, w]); line = int(lines[last])
                check(line % SETS == st, "resident line in wrong set")
                nu = next_use(line, last); nus.append(nu)
                check(nu == int(d["next_use"][k, w]), "next_use mismatch")
                ins, cnt = int(d["insert_idx"][k, w]), int(d["count"][k, w])
                check(sum(1 for j in pos[line] if ins <= j <= last) == cnt, "count mismatch")
            best = max(nus)
            expect = sum(1 << w for w in range(WAYS) if nus[w] == best)
            check(expect == int(d["label_mask"][k]), "label mask mismatch at decision %d: %x vs %x" % (k, expect, int(d["label_mask"][k])))
            if drv == "belady": check(int(d["label_mask"][k]) >> int(d["driver_victim"][k]) & 1, "belady driver left its own label")
        if drv == "lru":
            ref = python_lru_decisions()
            check(len(ref) % 1 == 0, "")
            sel = ref[::stride]
            check(len(sel) == N, "decision count %d vs python LRU %d" % (N, len(sel)))
            for k, (i, s, res, vw) in enumerate(sel[:N]):
                check(i == int(d["seq"][k]) and vw == int(d["driver_victim"][k]), "LRU decision %d differs from python LRU" % k)
                got = [int(lines[int(d["last_idx"][k, w])]) for w in range(WAYS)]
                check(got == res, "resident lines differ from python LRU at decision %d" % k)
        base = dataset.offline_baselines(d)
        check(0.0 <= base["lru (oldest last access)"] <= 1.0, "baseline range")
print("test_dataset: %s" % ("FAILED" if errors else "OK (labels, state, counts, next_use re-derived independently; strides 1 and 3; drivers lru and belady)"))
for e in errors: print("  ", e)
sys.exit(1 if errors else 0)
