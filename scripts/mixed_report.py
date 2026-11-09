#!/usr/bin/env python3
"""Table of the mixed-stream DAgger evaluation: for every result file in a folder, total misses and change versus LRU / versus the best
simple baseline for every policy, per-application MPKI, and old (frozen) vs new (mix) models side by side.
usage: python3 scripts/mixed_report.py results/ext/val"""
import glob, json, os, sys

SIMPLE = ["lru", "srrip", "drrip", "bip", "mru"]
d = sys.argv[1]
for f in sorted(glob.glob(os.path.join(d, "mixed_*.json"))):
    j = json.load(open(f)); R = {r["policy"]: r for r in j["runs"]}
    lru = R["lru"]["misses"]; bel = R["belady"]["misses"] if "belady" in R else None
    best = min((R[n]["misses"], n) for n in SIMPLE if n in R)
    print("\n== %s   (LLC %d KB, %d-way, slice %d instr)   best simple baseline: %s %d misses (%+.1f%% vs LRU)" % (
        os.path.basename(f), j["llc_size"] // 1024, j["llc_ways"], j["schedule"]["slice_instr"], best[1], best[0], 100.0 * (lru - best[0]) / lru))
    print("%-22s %10s %9s %9s %8s | %s" % ("policy", "misses", "vs LRU", "vs best", "gap%", "MPKI per application: " + " ".join(a["name"] for a in j["apps"])))
    for n, r in R.items():
        gap = ("%.1f" % (100.0 * (lru - r["misses"]) / (lru - bel))) if bel is not None and lru > bel else "-"
        print("%-22s %10d %+8.1f%% %+8.1f%% %8s | %s" % (n, r["misses"], 100.0 * (lru - r["misses"]) / lru, 100.0 * (best[0] - r["misses"]) / best[0], gap,
                                                    " ".join("%.2f" % a["mpki"] for a in r["apps"])))
    print("(positive = fewer misses than LRU / than the best simple baseline; gap% = share of the LRU-to-Belady gap closed)")
    import re
    found = sorted({(m.group(2), m.group(4)) for n in R for m in [re.match(r"(req|own|gen):(mix\w*?)(spec|all)_(\w+)$", n)] if m})
    for v, fam in found:
        for rule, old, new in (("req", "req:spec_%s", "req:%sspec_%s"), ("own", "own:spec_%s", "own:%sspec_%s"), ("gen", "gen:all_%s", "gen:%sall_%s")):
            o, n_ = old % fam, new % (v, fam)
            if o in R and n_ in R:
                print("  %s %-4s frozen %d -> %s %d misses (%+.1f%%)" % (fam, rule, R[o]["misses"], v, R[n_]["misses"], 100.0 * (R[o]["misses"] - R[n_]["misses"]) / R[o]["misses"]))
