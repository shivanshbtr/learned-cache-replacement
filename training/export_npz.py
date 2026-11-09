#!/usr/bin/env python3
"""Pack an LLC trace and the labeler's .dec file into one .npz dataset file and print offline baselines.
usage: python3 training/export_npz.py --trace X.llc.trace --dec X.lru.dec --out X.lru.npz [--delete-dec]"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True); ap.add_argument("--dec", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--delete-dec", action="store_true"); ap.add_argument("--meta-json", default=None, help="merged traces: JSON from multisim merge (adds apps, slice_instr to the dataset meta)")
a = ap.parse_args()
import json
dataset.build_npz(a.trace, a.dec, a.out, json.load(open(a.meta_json)) if a.meta_json else None)
d = dataset.load_npz(a.out)           # re-load to prove the file is readable
print("wrote %s (%.1f MB): %d decisions" % (a.out, os.path.getsize(a.out) / 1e6, len(d["seq"])))
for k, v in dataset.offline_baselines(d).items():
    print("   %-28s %.3f" % (k, v))
if a.delete_dec:
    os.remove(a.dec)
