#!/usr/bin/env python3
"""Planning helper for mixed-stream DAgger (docs/mixed_dagger.md). Subcommands:
  mixtures    one line per mixture of the TRAIN or VAL split:  NAME --app hash=<trace> --app graph=<trace> ...   (k-th input of every workload, k = 0,1,..)
  testmix     --app arguments of the n-th TEST input of every workload (n = 2: the second test mixture, used only by the evaluation script)
  driver      composite labeler driver:  req=mlp:a+mlp:b+..  /  own=..  /  gen=mlp:file   (models: the frozen set (+ --sfx) or --models DIR --ver V)
  drivers     comma list of the solo training drivers (lru,belady,dag1..) of the base variant, without the mixed ones
  multi-args  multisim --set/--gen arguments for the frozen models and every mixed-DAgger version that exists (--versions mix1,mix1b,mix2)
  pols        the matching multisim policy names (req:/own:/gen:)
Families: mlp, gbdt, lstm. Mixture rule: mixture k of the TRAIN split uses the k-th train input of each workload (a workload with fewer inputs repeats its
last one); the VAL split has one mixture. Applications are always in the order hash graph matrix conv."""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLOADS = ["hash", "graph", "matrix", "conv"]
FAM = {"mlp": "mlp", "gbdt": "lgbm", "lstm": "lstm"}     # family -> tag stem used in dataset driver tags and model names
EXT = {"mlp": "mlp", "gbdt": "gbdt", "lstm": "lstm"}
PRE = {"mlp": "mlp", "gbdt": "gbdt", "lstm": "lstm"}     # makePolicy prefix


def split_inputs(tsv, tr, split):
    out = {w: [] for w in WORKLOADS}
    for line in open(tsv):
        line = line.strip()
        if not line or line.startswith("#"): continue
        f = line.split()
        if f[0] == split and f[1] in out:
            out[f[1]].append(os.path.join(tr, split, "%s__%s%s.llc.trace" % (split, f[1], "".join("_" + x for x in f[3:]))))
    return out


def with_sfx(p, sfx):
    if not sfx: return p
    r, e = os.path.splitext(p)
    return r + sfx + e


def files_of(a):
    """model files per workload and 'all': from the frozen set (+ optional suffix) or from --models DIR --ver V."""
    if getattr(a, "ver", None):
        stem = FAM[a.family]
        return {w: os.path.join(a.models, "%s_%s_%s.%s" % (w, stem, a.ver, EXT[a.family])) for w in WORKLOADS + ["all"]}
    s = json.load(open(a.set))
    return {w: with_sfx(p, getattr(a, "sfx", "")) for w, p in s["files"][a.family].items()}


def cmd_mixtures(a):
    if a.split == "test": sys.exit("the test split is not used for training mixtures")
    inp = split_inputs(a.inputs, a.traces, a.split)
    if any(not v for v in inp.values()): sys.exit("a workload has no %s input" % a.split)
    for k in range(max(len(v) for v in inp.values())):
        print("mix%d%s %s" % (k, a.suffix, " ".join("--app %s=%s" % (w, inp[w][min(k, len(inp[w]) - 1)]) for w in WORKLOADS)))


def cmd_testmix(a):
    inp = split_inputs(a.inputs, a.traces, "test")
    if any(len(v) < a.n for v in inp.values()): sys.exit("every workload needs %d test inputs" % a.n)
    print(" ".join("--app %s=%s" % (w, inp[w][a.n - 1]) for w in WORKLOADS))


def cmd_driver(a):
    F = files_of(a); pre = PRE[a.family]
    if a.mode == "gen": print("gen=%s:%s" % (pre, F["all"]))
    else: print("%s=%s" % (a.mode, "+".join("%s:%s" % (pre, F[w]) for w in WORKLOADS)))


def cmd_drivers(a):
    sfx = a.sfx if a.sfx else json.load(open(a.set))["selected"].get(a.family, "")
    R = int(sfx[4:]) if sfx.startswith("_dag") else 0
    t = FAM[a.family]
    if a.who == "spec": d = ["lru", "belady"] + ["dag%d%s" % (r, t) for r in range(1, R + 1)]
    else: d = ["lru", "belady"] + sum([["dag%d%s" % (r, t), "dag%dall%s" % (r, t)] for r in range(1, R + 1)], [])
    print(",".join(d))


def found_versions(a):
    out = []
    for v in a.versions.split(","):
        new = {w: os.path.join(a.models, "%s_%s_%s.%s" % (w, FAM[a.family], v, EXT[a.family])) for w in WORKLOADS + ["all"]}
        if all(os.path.exists(p) for p in new.values()): out.append((v, new))
        else: print("note: no complete %s %s model set in %s: skipped" % (a.family, v, a.models), file=sys.stderr)
    return out


def cmd_multi_args(a):
    """multisim --set/--gen arguments: the frozen models (spec_<f>, all_<f>; --frozen-sfx for the LSTM extension _dag1) and every mixed-DAgger version (<v>spec_<f>, <v>all_<f>)."""
    s = json.load(open(a.set)); F = {w: with_sfx(p, a.frozen_sfx) for w, p in s["files"][a.family].items()}
    out = ["--set spec_%s=%s --gen all_%s=%s" % (a.family, ",".join("%s:%s" % (w, F[w]) for w in WORKLOADS), a.family, F["all"])]
    for v, new in found_versions(a):
        out.append("--set %sspec_%s=%s --gen %sall_%s=%s" % (v, a.family, ",".join("%s:%s" % (w, new[w]) for w in WORKLOADS), v, a.family, new["all"]))
    print(" ".join(out))


def cmd_pols(a):
    p = ["req:spec_%s" % a.family, "own:spec_%s" % a.family, "gen:all_%s" % a.family]
    for v, _ in found_versions(a):
        p += ["req:%sspec_%s" % (v, a.family), "own:%sspec_%s" % (v, a.family), "gen:%sall_%s" % (v, a.family)]
    print(",".join(p))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    SET = os.path.join(ROOT, "results", "final", "model_set.json")
    INP = os.path.join(ROOT, "configs", "inputs.tsv"); TRD = os.path.join(ROOT, "results", "traces"); MD = os.path.join(ROOT, "models_ext")
    p = sub.add_parser("mixtures"); p.add_argument("--split", required=True); p.add_argument("--suffix", default="")
    p.add_argument("--inputs", default=INP); p.add_argument("--traces", default=TRD); p.set_defaults(f=cmd_mixtures)
    p = sub.add_parser("testmix"); p.add_argument("--n", type=int, default=2); p.add_argument("--inputs", default=INP); p.add_argument("--traces", default=TRD); p.set_defaults(f=cmd_testmix)
    p = sub.add_parser("driver"); p.add_argument("--set", default=SET); p.add_argument("--family", choices=list(FAM), required=True); p.add_argument("--mode", choices=["req", "own", "gen"], required=True)
    p.add_argument("--sfx", default=""); p.add_argument("--ver", default=""); p.add_argument("--models", default=MD); p.set_defaults(f=cmd_driver)
    p = sub.add_parser("drivers"); p.add_argument("--set", default=SET); p.add_argument("--family", choices=list(FAM), required=True); p.add_argument("--who", choices=["spec", "all"], required=True)
    p.add_argument("--sfx", default=""); p.set_defaults(f=cmd_drivers)
    for n, f in (("multi-args", cmd_multi_args), ("pols", cmd_pols)):
        p = sub.add_parser(n); p.add_argument("--set", default=SET); p.add_argument("--family", choices=list(FAM), required=True)
        p.add_argument("--models", default=MD); p.add_argument("--versions", default="mix1"); p.add_argument("--frozen-sfx", default=""); p.set_defaults(f=f)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
