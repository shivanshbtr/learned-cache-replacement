#!/usr/bin/env python3
"""Planning helper for the final experiments (Batch E+F). Subcommands:

  select      choose the final model variants from the VALIDATION results only and freeze them in <final>/model_set.json
  policies    print the comma-separated cachesim policy list for one test workload and one group
  multi-args  print the multisim --set/--gen arguments for one model family
  apps        print the four applications (name=trace path) used for E7/E8 (first test input of each workload in configs/inputs.tsv)
  drivers     print the training drivers of the chosen DAgger round (for the equal-storage generic model and the seed runs)
  bighidden   print the hidden width of the generic MLP whose file size matches the sum of the four specialized MLP files

Model file naming (what train_all.sh, dagger_round.sh and the scripts of this batch write):
  <models>/<w>_{mlp,lgbm,lstm}<sfx>.{mlp,gbdt,lstm}      specialized (w = hash graph matrix conv)
  <models>/all_...                                       generic, same architecture
  <models>/allbig_mlp<sfx>.mlp                           generic with the total size of the four specialized models (E3 equal storage)
  <models_seeds>/<name>_s2.mlp, _s3.mlp                  extra training seeds of the MLPs
sfx is '' (Belady + LRU data only), '_dag1', '_dag2', ...
"""
import argparse, glob, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "training"))
WORKLOADS = ["hash", "graph", "matrix", "conv"]
FAM = {"mlp": ("mlp", "mlp"), "gbdt": ("lgbm", "gbdt"), "lstm": ("lstm", "lstm")}   # kind -> (file stem, extension)
BASE = ["lru", "fifo", "random", "srrip", "drrip", "bip", "mru"]


def fname(md, w, kind, sfx, seed=None):
    stem, ext = FAM[kind]
    return os.path.join(md, "%s_%s%s%s.%s" % (w, stem, sfx, "_s%d" % seed if seed else "", ext))


def policy_name(path):
    kind = {"mlp": "mlp", "gbdt": "gbdt", "lstm": "lstm"}[path.rsplit(".", 1)[1]]
    return "%s:%s" % (kind, os.path.basename(path).rsplit(".", 1)[0])


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def gap(p, lru, bel):
    return (lru - p) / (lru - bel) if lru > bel else None


def val_gaps(sim_val):
    """{policy name: {workload: gap closed}} and {workload: {policy: misses}} from the validation result files."""
    gaps, misses = {}, {}
    for f in sorted(glob.glob(os.path.join(sim_val, "*.result.json"))):
        label = os.path.basename(f)[:-len(".result.json")]
        w = label.split("__", 1)[1].split("_", 1)[0]
        r = json.load(open(f)); p = {x["name"]: x["misses"] for x in r["policies"]}
        if "lru" not in p or "belady" not in p: continue
        misses[w] = p
        for n, m in p.items():
            g = gap(m, p["lru"], p["belady"])
            if g is not None: gaps.setdefault(n, {})[w] = g
    return gaps, misses


def cmd_select(a):
    out = os.path.join(a.final, "model_set.json")
    if os.path.exists(out) and os.path.exists(os.path.join(a.final, "test_started")) and not a.force:
        print("model set is frozen (the test split has been started): %s" % out); return
    gaps, misses = val_gaps(a.val)
    if len(misses) < 4: sys.exit("need validation results for all four workloads in %s (run scripts/run_models.sh val first)" % a.val)
    selected, table = {}, {}
    for kind, (stem, ext) in FAM.items():
        variants = set()
        for f in glob.glob(os.path.join(a.models, "all_%s*.%s" % (stem, ext))):
            sfx = os.path.basename(f)[len("all_%s" % stem):-len(".%s" % ext)]
            if sfx == "" or (sfx.startswith("_dag") and sfx[4:].isdigit()): variants.add(sfx)
        best = None
        for sfx in sorted(variants, key=lambda s: (int(s[4:]) if s else 0)):
            vals, ok = [], True
            for w in WORKLOADS:
                for who in (w, "all"):
                    n = policy_name("%s/%s_%s%s.%s" % (a.models, who, stem, sfx, ext))
                    g = gaps.get(n, {}).get(w)
                    if g is None: ok = False
                    else: vals.append(g)
            if not ok:
                print("  %-5s variant %-6s incomplete on val, skipped" % (kind, sfx or "base")); continue
            score = sum(vals) / len(vals)
            table.setdefault(kind, {})[sfx] = score
            print("  %-5s variant %-6s mean gap closed on val (4 workloads x specialized+generic): %.4f" % (kind, sfx or "base", score))
            if best is None or score > best[1] + 1e-3: best = (sfx, score)
        if best is None: sys.exit("no complete variant of %s on val" % kind)
        selected[kind] = best[0]
        print("=> %s: %s" % (kind, best[0] or "base"))
    files, seedfiles, hashes = {}, {}, {}
    for kind, (stem, ext) in FAM.items():
        files[kind] = {w: fname(a.models, w, kind, selected[kind]) for w in WORKLOADS}
        files[kind]["all"] = "%s/all_%s%s.%s" % (a.models, stem, selected[kind], ext)
    big = "%s/allbig_mlp%s.mlp" % (a.models, selected["mlp"])
    if os.path.exists(big): files["mlp"]["allbig"] = big
    else: print("note: %s not found (train it with scripts/train_final_extras.sh); E3 equal-storage will be skipped" % big)
    for who in WORKLOADS + ["all", "allbig"]:
        lst = []
        for s in (2, 3):
            stem = "%s_mlp%s_s%d.mlp" % (who, selected["mlp"], s)
            if os.path.exists(os.path.join(a.seeds, stem)): lst.append(os.path.join(a.seeds, stem))
        if lst: seedfiles[who] = lst
    for k in files.values():
        for p in k.values():
            if not os.path.exists(p): sys.exit("model file missing: %s" % p)
            hashes[p] = sha(p)
    for l in seedfiles.values():
        for p in l: hashes[p] = sha(p)
    valinfo = {}
    for kind in FAM:
        for who in WORKLOADS + ["all"]:
            n = policy_name(files[kind][who]) if who in files[kind] else None
            valinfo["%s/%s" % (kind, who)] = {w: round(gaps.get(n, {}).get(w, float("nan")), 4) for w in WORKLOADS}
    os.makedirs(a.final, exist_ok=True)
    json.dump({"version": 1, "selected": selected, "val_mean_gap": table, "files": files, "seed_files": seedfiles, "sha256": hashes,
               "val_gap_closed": valinfo, "rule": "per model family, the variant with the highest mean gap closed on the VALIDATION traces "
               "(4 workloads, specialized and generic models together; ties within 0.001 go to the earlier round); test results never used"},
              open(out, "w"), indent=1)
    print("frozen: %s" % out)


def load_set(path):
    return json.load(open(path))


def cmd_policies(a):
    s = load_set(a.set); w = a.workload
    def pol(kind, who): return policy_name(s["files"][kind][who]) if who in s["files"][kind] else None
    def path(kind, who): return s["files"][kind][who]
    def spec(kind, p): return "%s:%s" % (kind, p)
    if a.group == "base": l = BASE + ["belady"]
    elif a.group == "core": l = ["lru", "srrip", "drrip", "bip", "mru", "belady", spec("mlp", path("mlp", w)), spec("mlp", path("mlp", "all"))]
    elif a.group == "mlp":
        l = ["lru", "belady", spec("mlp", path("mlp", w)), spec("mlp", path("mlp", "all"))]
        if "allbig" in s["files"]["mlp"]: l.append(spec("mlp", path("mlp", "allbig")))
        l += [spec("mlp", path("mlp", x)) for x in WORKLOADS if x != w]   # E4 cross-workload
        for who in (w, "all", "allbig"):
            l += [spec("mlp", p) for p in s["seed_files"].get(who, [])]    # extra training seeds (E3 mean and std)
    elif a.group in ("gbdt", "lstm"): l = ["lru", "belady", spec(a.group, path(a.group, w)), spec(a.group, path(a.group, "all"))]
    else: sys.exit("unknown group")
    print(",".join(l))


def cmd_multi_args(a):
    s = load_set(a.set); kind = a.family
    parts = ["--set", "spec_%s=%s" % (kind, ",".join("%s:%s" % (w, s["files"][kind][w]) for w in WORKLOADS)), "--gen", "all_%s=%s" % (kind, s["files"][kind]["all"])]
    if kind == "mlp" and "allbig" in s["files"]["mlp"]: parts += ["--gen", "allbig_mlp=%s" % s["files"]["mlp"]["allbig"]]
    print(" ".join(parts))


def primary_traces(tsv, tr):
    """First test input of every workload in inputs.tsv (the one listed first), as {workload: trace path}."""
    out = {}
    for line in open(tsv):
        line = line.strip()
        if not line or line.startswith("#"): continue
        f = line.split()
        if f[0] != "test" or f[1] in out: continue
        out[f[1]] = os.path.join(tr, "test", "test__%s%s.llc.trace" % (f[1], "".join("_" + x for x in f[3:])))
    return out


def cmd_apps(a):
    p = primary_traces(a.inputs, a.traces)
    print(" ".join("--app %s=%s" % (w, p[w]) for w in WORKLOADS if w in p))


def rounds(sfx): return int(sfx[4:]) if sfx.startswith("_dag") else 0


def cmd_drivers(a):
    sfx = load_set(a.set)["selected"]["mlp"]; R = rounds(sfx)
    if a.who == "spec": d = ["lru", "belady"] + ["dag%dmlp" % r for r in range(1, R + 1)]
    else: d = ["lru", "belady"] + sum([["dag%dmlp" % r, "dag%dallmlp" % r] for r in range(1, R + 1)], [])
    print(",".join(d))


def cmd_variant(a):
    print(load_set(a.set)["selected"]["mlp"])


def cmd_bighidden(a):
    import tempfile, numpy as np, modelio
    sfx = a.sfx
    spec = [modelio.read_mlp(fname(a.models, w, "mlp", sfx)) for w in WORKLOADS]
    total = sum(os.path.getsize(fname(a.models, w, "mlp", sfx)) for w in WORKLOADS)
    gen = modelio.read_mlp("%s/all_mlp%s.mlp" % (a.models, sfx))
    dims = gen["dims"]
    emb, n_ids, n_num = gen["emb_last"].shape[1], gen["emb_last"].shape[0], len(gen["mean"])
    n_hidden = len(dims) - 2
    best = None
    with tempfile.TemporaryDirectory() as td:
        for h in range(8, 513):
            ds = [n_num + 2 * emb] + [h] * n_hidden + [1]
            layers = [(np.zeros((ds[i + 1], ds[i]), np.float32), np.zeros(ds[i + 1], np.float32)) for i in range(len(ds) - 1)]
            p = os.path.join(td, "m.mlp")
            modelio.write_mlp(p, gen["vocab"], np.zeros(n_num), np.ones(n_num), np.zeros((n_ids, emb)), np.zeros((n_ids, emb)), layers)
            d = abs(os.path.getsize(p) - total)
            if best is None or d < best[0]: best = (d, h, os.path.getsize(p))
    print(",".join([str(best[1])] * n_hidden))
    print("# target %d bytes (sum of the four specialized files), chosen file size %d bytes" % (total, best[2]), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--final", default=os.path.join(ROOT, "results", "final")); p.add_argument("--models", default=os.path.join(ROOT, "models"))
        p.add_argument("--seeds", default=os.path.join(ROOT, "models_seeds")); p.add_argument("--val", default=os.path.join(ROOT, "results", "sim", "val"))
        p.add_argument("--set", default=None)
    p = sub.add_parser("select"); common(p); p.add_argument("--force", action="store_true"); p.set_defaults(f=cmd_select)
    p = sub.add_parser("policies"); common(p); p.add_argument("--workload", required=True); p.add_argument("--group", required=True); p.set_defaults(f=cmd_policies)
    p = sub.add_parser("multi-args"); common(p); p.add_argument("--family", required=True); p.set_defaults(f=cmd_multi_args)
    p = sub.add_parser("apps"); common(p); p.add_argument("--inputs", default=os.path.join(ROOT, "configs", "inputs.tsv"))
    p.add_argument("--traces", default=os.path.join(ROOT, "results", "traces")); p.set_defaults(f=cmd_apps)
    p = sub.add_parser("drivers"); common(p); p.add_argument("--who", choices=["spec", "all"], required=True); p.set_defaults(f=cmd_drivers)
    p = sub.add_parser("variant"); common(p); p.set_defaults(f=cmd_variant)
    p = sub.add_parser("bighidden"); common(p); p.add_argument("--sfx", required=True); p.set_defaults(f=cmd_bighidden)
    a = ap.parse_args()
    if a.set is None: a.set = os.path.join(a.final, "model_set.json")
    a.f(a)


if __name__ == "__main__":
    main()
