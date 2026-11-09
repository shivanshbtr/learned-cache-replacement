#!/usr/bin/env python3
"""Offline comparison: top-1 agreement with the Belady victim set (higher = better) for models and simple rules.
usage: python3 training/eval_offline.py --data results/datasets --split val|test --models models/a.mlp models/b.gbdt [--drivers lru]
Rows = dataset files; columns = random, LRU, MRU, FIFO rules and every model. Only measured, informative decisions count.
(The real metric is misses in the simulator; this is the fast screening step.)"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data, dataset, features, modelio

def scores_for(path, d, sel):
    if path.endswith(".mlp"):
        m = modelio.read_mlp(path)
        num, pcid, curid, bits = features.build(d, m["vocab"], sel)
        return modelio.predict_mlp(m, num, pcid, curid), bits
    if path.endswith(".lstm"):
        m = modelio.read_lstm(path)
        num, pcid, curid, bits = features.build(d, m["vocab"], sel)
        hpc, hnum = features.build_hist(d, m["vocab"], sel, m["hist"])
        return modelio.predict_lstm(m, num, pcid, curid, hpc, hnum), bits
    import lightgbm as lgb
    vocab, text = modelio.read_gbdt(path)
    num, pcid, curid, bits = features.build(d, vocab, sel)
    X = features.gbdt_matrix(num, pcid, curid)
    return lgb.Booster(model_str=text).predict(X, raw_score=True).reshape(len(bits), -1), bits

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/datasets"); ap.add_argument("--split", default="val"); ap.add_argument("--workload", default="all")
    ap.add_argument("--models", nargs="*", default=[]); ap.add_argument("--drivers", default="lru"); ap.add_argument("--max-rows", type=int, default=200000)
    a = ap.parse_args()
    files = data.list_files(a.data, a.split, a.workload, a.drivers.split(","))
    cols = ["random", "LRU", "MRU", "FIFO"] + [os.path.basename(m) for m in a.models]
    rows = []
    for f in files:
        d = dataset.load_npz(f)
        sel = features.select(d, True, True, a.max_rows, 0)
        if len(sel) == 0: continue
        last = d["last_idx"][sel].astype(np.int64); ins = d["insert_idx"][sel].astype(np.int64)
        bits = dataset.mask_to_bits(d["label_mask"][sel], d["meta"]["ways"]); r = np.arange(len(sel))
        vals = [float((bits.sum(1) / bits.shape[1]).mean()), float(bits[r, last.argmin(1)].mean()), float(bits[r, last.argmax(1)].mean()), float(bits[r, ins.argmin(1)].mean())]
        for m in a.models:
            sc, b = scores_for(m, d, sel); vals.append(data.top1(sc, b))
        rows.append((os.path.basename(f).replace(".npz", ""), len(sel), vals))
    if not rows: sys.exit("no files")
    w = max(len(r[0]) for r in rows)
    print("%-*s %8s " % (w, "file", "decisions") + " ".join("%12s" % c[:12] for c in cols))
    for name, n, vals in rows:
        print("%-*s %8d " % (w, name, n) + " ".join("%12.3f" % v for v in vals))
    print("%-*s %8s " % (w, "MEAN", "") + " ".join("%12.3f" % np.mean([r[2][i] for r in rows]) for i in range(len(cols))))
