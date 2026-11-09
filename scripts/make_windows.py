#!/usr/bin/env python3
"""Make the sample windows for the web simulation (no new experiment: it replays existing TEST traces with the frozen models; nothing is selected).
  python3 scripts/make_windows.py [--len 4000] [--out results/web/windows]
Writes   <out>/single_<workload>.json   (4 files: lru, drrip, belady, MLP/GBDT/LSTM specialized and generic; inspector for the two MLPs)
         <out>/shared_default.json      (the E8 test mixture, merged round-robin, 2 MB: lru, drrip, belady; gen/req/own rules with the frozen MLP and the mixed-stream
                                        versions mix1, mix1b, mix2 of MLP, GBDT, LSTM that exist in models_ext; inspector for gen and req of mix1b MLP)
Window = 4000 accesses starting at the middle of the trace (never in the warm-up). Every policy is replayed from the start of the trace, so it takes
roughly 15 to 40 minutes (the LSTM runs are the slow part). Env/arguments override the default paths."""
import argparse, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import final_plan, mixed_plan
W = ["hash", "graph", "matrix", "conv"]

ap = argparse.ArgumentParser()
ap.add_argument("--len", type=int, default=4000)
ap.add_argument("--out", default=os.path.join(ROOT, "results/web/windows"))
ap.add_argument("--cfg", default=os.path.join(ROOT, "configs/default.cfg"))
ap.add_argument("--set", default=os.path.join(ROOT, "results/final/model_set.json"))
ap.add_argument("--inputs", default=os.path.join(ROOT, "configs/inputs.tsv"))
ap.add_argument("--traces", default=os.path.join(ROOT, "results/traces"))
ap.add_argument("--models", default=os.path.join(ROOT, "models_ext"))
ap.add_argument("--bin", default=os.path.join(ROOT, "build"))
ap.add_argument("--ver", default="mix1b", help="mixed-stream version used in the shared window")
ap.add_argument("--slice", type=int, default=10000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
S = json.load(open(a.set))["files"]
prim = final_plan.primary_traces(a.inputs, a.traces)
for w in W:
    if not os.path.exists(prim[w]): sys.exit("missing trace " + prim[w])

def run(cmd):
    t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: sys.exit("FAILED: %s\n%s%s" % (" ".join(cmd), r.stdout[-500:], r.stderr[-1500:]))
    print(r.stdout.strip().splitlines()[0], "(%.0f s)" % (time.time() - t0), flush=True)

# 1. single applications: each application with its OWN trace; specialized model of that workload and the generic model ("all")
for w in W:
    pols, labs = ["lru", "drrip", "belady"], ["LRU", "DRRIP", "Belady (oracle)"]
    for fam, nm, pre in (("mlp", "MLP", "mlp"), ("gbdt", "GBDT", "gbdt"), ("lstm", "LSTM", "lstm")):
        pols += ["%s:%s" % (pre, S[fam][w]), "%s:%s" % (pre, S[fam]["all"])]
        labs += ["%s specialized" % nm, "%s generic" % nm]
    insp = [p for p in pols if p.startswith("mlp:")]
    run([os.path.join(a.bin, "winexport"), "--config", a.cfg, "--trace", prim[w], "--policies", ",".join(pols), "--labels", ",".join(labs),
         "--start-frac", "0.5", "--len", str(a.len), "--sets", "32", "--inspect", ",".join(insp), "--max-decisions", "150", "--apps", w,
         "--out", os.path.join(a.out, "single_%s.json" % w)])

# 2. shared LLC: the E8 test mixture merged like multisim (slice 10,000 instructions, until the first application finishes)
apps = []
for w in W: apps += ["--app", "%s=%s" % (w, prim[w])]
merged = os.path.join(a.out, "_merged.llc.trace")
run([os.path.join(a.bin, "multisim"), "merge", "--config", a.cfg] + apps + ["--slice-instr", str(a.slice), "--until", "first", "--out", merged])
def drv(fam, mode, ver=""):
    ns = argparse.Namespace(set=a.set, family=fam, mode=mode, sfx="", ver=ver, models=a.models)
    import io, contextlib
    b = io.StringIO()
    with contextlib.redirect_stdout(b): mixed_plan.cmd_driver(ns)
    return b.getvalue().strip()
# policies of the shared window: the three simple ones, then rule x family x version. Frozen = the frozen single-application models (MLP only).
# Models that do not exist in --models are skipped with a note, so a missing file never blocks the other windows.
pols = ["lru", "drrip", "belady"]; labs = ["LRU", "DRRIP", "Belady (oracle)"]
def add(label, fam, mode, ver=""):
    if ver and not all(os.path.exists(os.path.join(a.models, "%s_%s_%s.%s" % (w, mixed_plan.FAM[fam], ver, mixed_plan.EXT[fam]))) for w in W + ["all"]):
        print("note: no %s %s model set in %s: skipped %s" % (fam, ver, a.models, label), flush=True); return
    pols.append(drv(fam, mode, ver)); labs.append(label)
add("gen: frozen MLP", "mlp", "gen")
for v in ("mix1", "mix1b", "mix2"): add("gen: %s MLP" % v, "mlp", "gen", v)
add("gen: mix1b GBDT", "gbdt", "gen", "mix1b"); add("gen: mix1b LSTM", "lstm", "gen", "mix1b")
add("req: frozen MLP", "mlp", "req"); add("req: mix1b MLP", "mlp", "req", "mix1b"); add("req: mix1b LSTM", "lstm", "req", "mix1b")
add("own: frozen MLP", "mlp", "own"); add("own: mix1b MLP", "mlp", "own", "mix1b")
insp = [p for p, l in zip(pols, labs) if l in ("gen: mix1b MLP", "req: mix1b MLP")]
run([os.path.join(a.bin, "winexport"), "--config", a.cfg, "--trace", merged, "--policies", ",".join(pols), "--labels", ",".join(labs),
     "--start-frac", "0.5", "--len", str(a.len), "--sets", "32", "--inspect", ",".join(insp), "--max-decisions", "150",
     "--apps", ",".join(W), "--out", os.path.join(a.out, "shared_default.json")])
os.remove(merged)
print("done:", a.out)
