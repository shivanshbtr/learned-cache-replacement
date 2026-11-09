#!/usr/bin/env python3
"""Train the MLP eviction model (PyTorch): a per-candidate scorer with PC embeddings, trained with a set-level softmax
cross-entropy against the Belady victim SET (tied victims: loss = -log of the total probability of the tied set).
usage: python3 training/train_mlp.py --data results/datasets --workload hash --out models/hash_mlp.mlp
       [--hidden 32,32] [--emb 4] [--epochs 30] [--vocab 64] [--max-per-file 60000] [--seed 1]"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn
import data, features, modelio

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="results/datasets"); ap.add_argument("--workload", default="all"); ap.add_argument("--out", required=True)
ap.add_argument("--drivers", default="lru,belady"); ap.add_argument("--hidden", default="32,32"); ap.add_argument("--emb", type=int, default=4)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--batch", type=int, default=512); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--patience", type=int, default=6); ap.add_argument("--vocab", type=int, default=64)
ap.add_argument("--max-per-file", type=int, default=60000); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
if a.threads: torch.set_num_threads(a.threads)
drivers = a.drivers.split(",")
tr = data.list_files(a.data, "train", a.workload, drivers); va = data.list_files(a.data, "val", a.workload, drivers)
if not tr or not va: raise SystemExit("no train/val files for workload %s under %s" % (a.workload, a.data))
t0 = time.time()
vocab = data.vocab_from_files(tr, a.vocab, a.workload)
T = data.load_rows(tr, vocab, a.max_per_file, a.seed, workload=a.workload); V = data.load_rows(va, vocab, a.max_per_file, a.seed + 1000, workload=a.workload)
N, W, F = T["num"].shape
mean, std = data.mean_std(T["num"])
print("train: %d files, %d decisions; val: %d files, %d decisions; vocab %d PCs; %d ways" % (len(tr), N, len(va), len(V["bits"]), len(vocab), W))

class Net(nn.Module):
    def __init__(s, n_ids, emb, hidden):
        super().__init__()
        s.el, s.ec = nn.Embedding(n_ids, emb), nn.Embedding(n_ids, emb)
        dims = [F + 2 * emb] + hidden + [1]
        s.lin = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
    def forward(s, num, pcid, cur):
        Wn = num.shape[1]
        h = torch.cat([num, s.el(pcid), s.ec(cur)[:, None, :].expand(-1, Wn, -1)], -1)
        for i, l in enumerate(s.lin):
            h = l(h)
            if i < len(s.lin) - 1: h = torch.relu(h)
        return h.squeeze(-1)

def tens(R):   # normalizes R["num"] in place (raw values are not needed afterwards)
    return (torch.from_numpy(data.normalize(R["num"], mean, std, inplace=True)), torch.from_numpy(R["pcid"].astype(np.int64)),
            torch.from_numpy(R["curid"].astype(np.int64)), torch.from_numpy(R["bits"]))
tT, tV = tens(T), tens(V)
net = Net(len(vocab) + 1, a.emb, [int(x) for x in a.hidden.split(",")])
opt = torch.optim.Adam(net.parameters(), lr=a.lr)

def loss_fn(sc, bits):
    return (torch.logsumexp(sc, 1) - torch.logsumexp(sc.masked_fill(~bits, float("-inf")), 1)).mean()
def evaluate(t):
    net.eval()
    with torch.no_grad():
        sc = torch.cat([net(t[0][i:i + 8192], t[1][i:i + 8192], t[2][i:i + 8192]) for i in range(0, len(t[0]), 8192)])
    net.train()
    return float(loss_fn(sc, t[3])), float(t[3][torch.arange(len(sc)), sc.argmax(1)].float().mean())

best, best_state, bad = -1.0, None, 0
for ep in range(a.epochs):
    perm = torch.randperm(N)
    tot = 0.0
    for i in range(0, N, a.batch):
        b = perm[i:i + a.batch]
        opt.zero_grad()
        l = loss_fn(net(tT[0][b], tT[1][b], tT[2][b]), tT[3][b])
        l.backward(); opt.step(); tot += float(l) * len(b)
    vl, v1 = evaluate(tV)
    print("epoch %2d  train loss %.4f  val loss %.4f  val top-1 %.4f" % (ep + 1, tot / N, vl, v1), flush=True)
    if v1 > best + 1e-4: best, bad, best_state = v1, 0, {k: x.clone() for k, x in net.state_dict().items()}
    else:
        bad += 1
        if bad >= a.patience: break
net.load_state_dict(best_state)
_, tr1 = evaluate(tT); vl, v1 = evaluate(tV)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
sd = {k: x.numpy() for k, x in net.state_dict().items()}
layers = [(sd["lin.%d.weight" % i], sd["lin.%d.bias" % i]) for i in range(len(net.lin))]
modelio.write_mlp(a.out, vocab, mean, std, sd["el.weight"], sd["ec.weight"], layers)
m = modelio.read_mlp(a.out)
res = dict(kind="mlp", workload=a.workload, drivers=drivers, hidden=a.hidden, emb=a.emb, vocab=len(vocab), seed=a.seed, params=modelio.mlp_params(m),
           train_files=[os.path.basename(f) for f in tr], val_files=[os.path.basename(f) for f in va], train_top1=tr1, val_top1=v1,
           bytes=os.path.getsize(a.out), seconds=round(time.time() - t0, 1))
json.dump(res, open(a.out + ".json", "w"), indent=1)
print("saved %s (%d params, %.1f KB): train top-1 %.3f, val top-1 %.3f" % (a.out, res["params"], res["bytes"] / 1024, tr1, v1))
