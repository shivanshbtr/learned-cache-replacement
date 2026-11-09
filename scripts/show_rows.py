#!/usr/bin/env python3
"""Print a few eviction decisions of a dataset file in readable form (what the models are trained on).
usage: python3 scripts/show_rows.py results/datasets/train/<file>.npz [--n 3] [--at 1000]     (--at: first decision index to show)"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
import dataset, features
ap = argparse.ArgumentParser(); ap.add_argument("file"); ap.add_argument("--n", type=int, default=2); ap.add_argument("--at", type=int, default=2000)
a = ap.parse_args()
d = dataset.load_npz(a.file); m = d["meta"]
print("file: %s\n  trace: %s\n  driver (who evicted while recording): %s   ways: %d   decisions stored: %d   trace accesses: %d%s" % (
    os.path.basename(a.file), os.path.basename(m["trace"]), m["driver"], m["ways"], len(d["seq"]), len(d["trace_line"]),
    ("\n  merged applications: %s" % m["apps"]) if "apps" in m else ""))
sel = features.select(d, True, True)
sel = sel[sel >= a.at][:a.n] if (sel >= a.at).any() else sel[:a.n]
vocab = features.build_vocab([d], 64)   # illustration only: pc_id = rank of the PC among the most frequent PCs of THIS file (training uses a pooled vocabulary)
num, pcid, curid, bits = features.build(d, vocab, sel)
for k, i in enumerate(sel):
    seq = int(d["seq"][i]); pc = int(d["trace_pc"][seq])
    print("\n=== decision #%d: at access %d, set %d. Access brings in line %#x (PC %#x, %s). All %d ways are full; pick a victim." % (
        i, seq, int(d["set"][i]), int(d["trace_line"][seq]) & ((1 << 48) - 1), pc & ((1 << 56) - 1), "store" if d["trace_write"][seq] else "load", m["ways"]))
    if "apps" in m: print("    requested by application: %s" % m["apps"][pc >> 56])
    print("way | since_last  age  count reused last_gap rank  last_wr | pc_id | line (app)        | Belady victim? | driver evicted")
    for w in range(m["ways"]):
        li = int(d["last_idx"][i, w]); line = int(d["trace_line"][li]); f = num[k, w]
        print("%3d | %9.2f %6.2f %5.2f %5d %8.2f %5.2f %5d | %5d | %#12x (%s) | %-14s | %s" % (
            w, f[0], f[1], f[2], f[3], f[4], f[5], f[7], pcid[k, w], line & ((1 << 48) - 1), (m["apps"][line >> 48] if "apps" in m else "-"),
            "YES" if bits[k, w] else "", "<== evicted" if w == int(d["driver_victim"][i]) else ""))
    print("  current access: pc_id=%d  is_store=%d     (features are log2-scaled: since_last 10 means ~1000 accesses ago)" % (curid[k], int(num[k, 0, 8])))
