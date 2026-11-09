#!/usr/bin/env python3
"""Train the small LSTM + attention eviction model (PARROT-inspired, PyTorch).
Design (see docs/models.md): an LSTM runs over the last H LLC accesses (ending with the current one); every candidate line of the set
attends over the H hidden states (general attention + learned positional embeddings); a dense layer gives one score per line.
Loss = set-level softmax cross-entropy against the Belady victim SET (as the MLP) + `--aux` x MSE of an auxiliary head on the
log reuse distance (training only, not exported).
usage: python3 training/train_lstm.py --data results/datasets --workload hash --out models/hash_lstm.lstm
       [--hist 16] [--hidden 32] [--emb 4] [--dpos 4] [--dc 32] [--dh 32] [--aux 0.1] [--epochs 20] [--vocab 64] [--max-per-file 40000] [--seed 1]"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn
import data, features, modelio

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="results/datasets"); ap.add_argument("--workload", default="all"); ap.add_argument("--out", required=True)
ap.add_argument("--drivers", default="lru,belady")
ap.add_argument("--hist", type=int, default=16); ap.add_argument("--hidden", type=int, default=32); ap.add_argument("--emb", type=int, default=4)
ap.add_argument("--dpos", type=int, default=4); ap.add_argument("--dc", type=int, default=32); ap.add_argument("--dh", type=int, default=32)
ap.add_argument("--aux", type=float, default=0.1)
ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--patience", type=int, default=5); ap.add_argument("--vocab", type=int, default=64)
ap.add_argument("--max-per-file", type=int, default=40000); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
if a.threads: torch.set_num_threads(a.threads)
drivers = a.drivers.split(",")
tr = data.list_files(a.data, "train", a.workload, drivers); va = data.list_files(a.data, "val", a.workload, drivers)
if not tr or not va: raise SystemExit("no train/val files for workload %s under %s" % (a.workload, a.data))
t0 = time.time()
H = a.hist
vocab = data.vocab_from_files(tr, a.vocab, a.workload)
T = data.load_rows(tr, vocab, a.max_per_file, a.seed, hist=H, workload=a.workload); V = data.load_rows(va, vocab, a.max_per_file, a.seed + 1000, hist=H, workload=a.workload)
N, W, F = T["num"].shape
mean, std = data.mean_std(T["num"])
print("train: %d files, %d decisions; val: %d files, %d decisions; vocab %d PCs; %d ways; history %d" % (len(tr), N, len(va), len(V["bits"]), len(vocab), W, H))
K = a.hidden + a.dpos


class Net(nn.Module):
    def __init__(s, n_ids):
        super().__init__()
        s.emb = nn.Embedding(n_ids, a.emb)
        s.lstm = nn.LSTM(a.emb + features.HIST_FEATS, a.hidden, batch_first=True)
        s.pos = nn.Parameter(torch.randn(H, a.dpos) * 0.1)
        s.a1 = nn.Linear(F + 2 * a.emb, a.dc); s.q = nn.Linear(a.dc, K)
        s.bl = nn.Linear(a.dc + a.hidden, a.dh); s.v = nn.Linear(a.dh, 1); s.auxh = nn.Linear(a.dh, 1)

    def forward(s, num, pcid, cur, hpc, hnum):
        B, Wn = num.shape[0], num.shape[1]
        Hs, _ = s.lstm(torch.cat([s.emb(hpc), hnum], -1))                                   # [B,H,hidden]
        c = torch.relu(s.a1(torch.cat([num, s.emb(pcid), s.emb(cur)[:, None, :].expand(-1, Wn, -1)], -1)))   # [B,W,dc]
        u = s.q(c)                                                                          # [B,W,K]
        keys = torch.cat([Hs, s.pos[None].expand(B, -1, -1)], -1)                           # [B,H,K]
        att = torch.softmax(torch.einsum("bwk,bhk->bwh", u, keys) * np.float32(1.0 / np.sqrt(K)), -1)
        g = torch.einsum("bwh,bhd->bwd", att, Hs)
        z = torch.relu(s.bl(torch.cat([c, g], -1)))
        return s.v(z).squeeze(-1), s.auxh(z).squeeze(-1)


def tens(R, inplace=False):   # inplace: overwrite R["num"] with the normalized values (the training set does not need the raw ones)
    return dict(num=torch.from_numpy(data.normalize(R["num"], mean, std, inplace=inplace)), pcid=torch.from_numpy(R["pcid"].astype(np.int64)),
                cur=torch.from_numpy(R["curid"].astype(np.int64)), bits=torch.from_numpy(R["bits"]), hpc=torch.from_numpy(R["hpc"]),
                hnum=torch.from_numpy(R["hnum"]), rd=torch.from_numpy(R["rd"]))


def batch(t, idx):
    return t["num"][idx], t["pcid"][idx], t["cur"][idx], t["hpc"][idx].long(), t["hnum"][idx]


tT, tV = tens(T, True), tens(V)   # V["num"] stays raw: the export self-check needs it
net = Net(len(vocab) + 1)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)


def ce(sc, bits):
    return (torch.logsumexp(sc, 1) - torch.logsumexp(sc.masked_fill(~bits, float("-inf")), 1)).mean()


def evaluate(t):
    net.eval()
    n = len(t["bits"]); sc = []
    with torch.no_grad():
        for i in range(0, n, 2048):
            sc.append(net(*batch(t, torch.arange(i, min(n, i + 2048))))[0])
    sc = torch.cat(sc)
    net.train()
    return float(ce(sc, t["bits"])), float(t["bits"][torch.arange(n), sc.argmax(1)].float().mean())


best, best_state, bad = -1.0, None, 0
for ep in range(a.epochs):
    perm = torch.randperm(N)
    tot = 0.0
    for i in range(0, N, a.batch):
        b = perm[i:i + a.batch]
        opt.zero_grad()
        sc, aux = net(*batch(tT, b))
        l = ce(sc, tT["bits"][b])
        if a.aux > 0: l = l + a.aux * ((aux - tT["rd"][b]) ** 2).mean()
        l.backward(); opt.step(); tot += float(l) * len(b)
    vl, v1 = evaluate(tV)
    print("epoch %2d  train loss %.4f  val loss %.4f  val top-1 %.4f  (%.0fs)" % (ep + 1, tot / N, vl, v1, time.time() - t0), flush=True)
    if v1 > best + 1e-4: best, bad, best_state = v1, 0, {k: x.clone() for k, x in net.state_dict().items()}
    else:
        bad += 1
        if bad >= a.patience: break
net.load_state_dict(best_state)
_, tr1 = evaluate(tT); vl, v1 = evaluate(tV)
sd = {k: x.numpy() for k, x in net.state_dict().items()}
w = dict(mean=mean, std=std, emb=sd["emb.weight"], w_ih=sd["lstm.weight_ih_l0"], w_hh=sd["lstm.weight_hh_l0"], b=sd["lstm.bias_ih_l0"] + sd["lstm.bias_hh_l0"],
         pos=sd["pos"], a1w=sd["a1.weight"], a1b=sd["a1.bias"], qw=sd["q.weight"], qb=sd["q.bias"], bw=sd["bl.weight"], bb=sd["bl.bias"],
         vw=sd["v.weight"].reshape(-1), vb=sd["v.bias"].reshape(-1))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
modelio.write_lstm(a.out, vocab, w, H, a.hidden, a.dpos, a.dc, a.dh)
m = modelio.read_lstm(a.out)
# self-check: the exported numpy model must give the same scores as the PyTorch model (guards the export and the C++ inference path)
sel = np.arange(min(len(V["bits"]), 2000))
with torch.no_grad():
    net.eval(); ts = net(*batch(tV, torch.from_numpy(sel)))[0].numpy()
ns = modelio.predict_lstm(m, V["num"][sel], V["pcid"][sel], V["curid"][sel], V["hpc"][sel], V["hnum"][sel])
diff = float(np.abs(ts - ns).max())
print("export self-check: max |torch - numpy| score difference %.2e" % diff)
if diff > 1e-3: raise SystemExit("EXPORT MISMATCH between PyTorch and numpy forward pass (%.3e)" % diff)
res = dict(kind="lstm", workload=a.workload, drivers=drivers, hist=H, hidden=a.hidden, emb=a.emb, dpos=a.dpos, dc=a.dc, dh=a.dh, aux=a.aux, vocab=len(vocab),
           seed=a.seed, params=modelio.lstm_params(m), train_files=[os.path.basename(f) for f in tr], val_files=[os.path.basename(f) for f in va],
           train_top1=tr1, val_top1=v1, bytes=os.path.getsize(a.out), seconds=round(time.time() - t0, 1))
json.dump(res, open(a.out + ".json", "w"), indent=1)
print("saved %s (%d params, %.1f KB): train top-1 %.3f, val top-1 %.3f" % (a.out, res["params"], res["bytes"] / 1024, tr1, v1))
