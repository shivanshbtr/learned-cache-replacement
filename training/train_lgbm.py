#!/usr/bin/env python3
"""Train a LightGBM eviction model (candidate-level binary classifier: is this way a Belady victim?).
usage: python3 training/train_lgbm.py --data results/datasets --workload hash --out models/hash_lgbm.gbdt
       [--drivers lru,belady] [--leaves 31] [--rounds 300] [--vocab 64] [--max-per-file 60000] [--seed 1]
The eviction decision = the way with the highest score. Validation = files under <data>/val (same workload)."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightgbm as lgb
import data, features, modelio

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="results/datasets"); ap.add_argument("--workload", default="all"); ap.add_argument("--out", required=True)
ap.add_argument("--drivers", default="lru,belady"); ap.add_argument("--leaves", type=int, default=31); ap.add_argument("--rounds", type=int, default=300)
ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--vocab", type=int, default=64)
ap.add_argument("--max-per-file", type=int, default=60000); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()
drivers = a.drivers.split(",")
tr = data.list_files(a.data, "train", a.workload, drivers); va = data.list_files(a.data, "val", a.workload, drivers)
if not tr or not va: raise SystemExit("no train/val files for workload %s under %s" % (a.workload, a.data))
t0 = time.time()
vocab = data.vocab_from_files(tr, a.vocab, a.workload)
T = data.load_rows(tr, vocab, a.max_per_file, a.seed, workload=a.workload); V = data.load_rows(va, vocab, a.max_per_file, a.seed + 1000, workload=a.workload)
Xt, Xv = features.gbdt_matrix(T["num"], T["pcid"], T["curid"]), features.gbdt_matrix(V["num"], V["pcid"], V["curid"])
yt, yv = T["bits"].reshape(-1).astype(np.float32), V["bits"].reshape(-1).astype(np.float32)
for R in (T, V):
    for k in ("num", "pcid", "curid"): R.pop(k)   # the matrices above hold everything the booster needs (frees gigabytes for the generic models)
print("train: %d files, %d decisions; val: %d files, %d decisions; vocab %d PCs" % (len(tr), len(T["bits"]), len(va), len(V["bits"]), len(vocab)))
params = dict(objective="binary", learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=50, seed=a.seed, verbose=-1,
              deterministic=True, force_row_wise=True, num_threads=a.threads)
booster = lgb.train(params, lgb.Dataset(Xt, yt), num_boost_round=a.rounds, valid_sets=[lgb.Dataset(Xv, yv)],
                    callbacks=[lgb.early_stopping(25, verbose=False)])
best = booster.best_iteration or a.rounds
W = T["bits"].shape[1]
sv = booster.predict(Xv, num_iteration=best, raw_score=True).reshape(-1, W)
st = booster.predict(Xt, num_iteration=best, raw_score=True).reshape(-1, W)
res = dict(kind="gbdt", workload=a.workload, drivers=drivers, leaves=a.leaves, trees=best, vocab=len(vocab), seed=a.seed,
           train_files=[os.path.basename(f) for f in tr], val_files=[os.path.basename(f) for f in va],
           train_top1=data.top1(st, T["bits"]), val_top1=data.top1(sv, V["bits"]), seconds=round(time.time() - t0, 1))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
modelio.write_gbdt(a.out, vocab, booster.model_to_string(num_iteration=best))
res["bytes"] = os.path.getsize(a.out)
json.dump(res, open(a.out + ".json", "w"), indent=1)
print("saved %s (%d trees, %.1f KB): train top-1 %.3f, val top-1 %.3f (informative measured decisions)" % (a.out, best, res["bytes"] / 1024, res["train_top1"], res["val_top1"]))
