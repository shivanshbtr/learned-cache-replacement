#!/usr/bin/env python3
"""Summarize all *.result.json under a folder.
Default: one compact block per trace: policy, LLC misses, share of the LRU-to-Belady gap closed, and % fewer misses than the BEST SIMPLE baseline
(lru, fifo, random, srrip, brrip, drrip, bip, mru; positive = fewer misses). Models (names containing ':') are marked with '*'.
--wide: the old one-row-per-trace table (policies as columns).   --csv FILE: write the data (long format; wide format with --wide).
usage: summarize.py <folder> [--wide] [--csv out.csv]"""
import csv, glob, json, os, sys
argv = sys.argv[1:]
wide = "--wide" in argv
csv_out = argv[argv.index("--csv") + 1] if "--csv" in argv else None
args = [x for i, x in enumerate(argv) if not x.startswith("--") and not (i and argv[i - 1] == "--csv")]
root = args[0] if args else "results/traces"
runs = []
for f in sorted(glob.glob(os.path.join(root, "**", "*.result.json"), recursive=True)):
    r = json.load(open(f)); runs.append((os.path.basename(f).replace(".result.json", ""), r))
if not runs: print("no result files found under", root); sys.exit(0)
SIMPLE = ["lru", "fifo", "random", "srrip", "brrip", "drrip", "bip", "mru"]
order = SIMPLE
names = []
for _, r in runs:
    for p in r["policies"]:
        if p["name"] not in names: names.append(p["name"])

def gap(p, lru, bel):
    if lru and bel and lru["misses"] > bel["misses"]:
        return 100.0 * (lru["misses"] - p["misses"]) / (lru["misses"] - bel["misses"])
    return None

if wide:
    cols = [n for n in order if n in names] + sorted(n for n in names if n not in order and n != "belady") + ["belady"]
    cols = [c for c in cols if c in names]
    rows = []
    for label, r in runs:
        p = {x["name"]: x for x in r["policies"]}
        lru, bel = p.get("lru"), p.get("belady")
        row = [label, r["trace"]["llc_accesses"]]
        for c in cols:
            if c not in p: row.append("-"); continue
            cell = str(p[c]["misses"])
            g = gap(p[c], lru, bel)
            if c not in ("lru", "belady") and g is not None: cell += " (%.0f%%)" % g
            row.append(cell)
        rows.append(row)
    hdr = ["label", "LLC acc"] + cols
    w = [max(len(str(x[i])) for x in rows + [hdr]) for i in range(len(hdr))]
    print("misses per policy; (x%) = share of the LRU-to-Belady gap closed")
    for r in [hdr] + rows: print("  ".join(str(c).rjust(w[i]) if i else str(c).ljust(w[i]) for i, c in enumerate(r)))
    if csv_out:
        with open(csv_out, "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(hdr); wr.writerows(rows)
    sys.exit(0)

long_rows = []
for label, r in runs:
    P = r["policies"]; p = {x["name"]: x for x in P}
    lru, bel = p.get("lru"), p.get("belady")
    simple = [x for x in P if x["name"] in SIMPLE]
    best = min(simple, key=lambda x: x["misses"]) if simple else None
    print("\n== %s   (LLC accesses %d)   best simple baseline: %s" % (label, r["trace"]["llc_accesses"], ("%s %d" % (best["name"], best["misses"])) if best else "-"))
    pol = [x for x in P if x["name"] in SIMPLE] + [x for x in P if x["name"] not in SIMPLE and x["name"] != "belady"] + ([bel] if bel else [])
    nw = max(len(x["name"]) for x in pol) + 2
    print("   %-*s %10s %10s %14s" % (nw, "policy", "misses", "gap closed", "vs best simple"))
    for x in pol:
        g = gap(x, lru, bel); is_model = ":" in x["name"]
        vs = (100.0 * (best["misses"] - x["misses"]) / best["misses"]) if best and best["misses"] else None
        gs = "-" if g is None or x["name"] in ("lru", "belady") else "%.0f%%" % g
        vss = "-" if vs is None or x["name"] == "belady" else "%+.1f%%" % vs
        print("   %-*s %10d %10s %14s" % (nw, ("* " if is_model else "") + x["name"], x["misses"], gs, vss))
        long_rows.append([label, r["trace"]["llc_accesses"], x["name"], x["misses"], "" if g is None else round(g, 2), "" if vs is None else round(vs, 2)])
print("\ngap closed = (LRU - policy) / (LRU - Belady);  vs best simple = % fewer misses than the best of lru/fifo/random/srrip/brrip/drrip/bip/mru (+ is better);  * = learned model")
if csv_out:
    with open(csv_out, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["label", "llc_accesses", "policy", "misses", "gap_closed_pct", "vs_best_simple_pct"]); wr.writerows(long_rows)
