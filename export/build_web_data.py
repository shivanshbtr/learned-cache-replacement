#!/usr/bin/env python3
"""Build the data folder of the static web simulation (web_simulation/data/*.js). No new experiment: it only re-packs existing result files.
  python3 export/build_web_data.py --final DIR_with_final_results --ext DIR_with_ext_results [--windows results/web/windows] [--out web_simulation/data]
Inputs
  --final    folder of final_results.zip: export/web_data.json, tables/*.csv, model_set.json
  --ext      folder of ext_results.zip:   results/ext/{val,test,test2}/mixed_*.json, results/ext/lstm_test/test/*.result.json
  --windows  folder written by scripts/make_windows.py (single_<workload>.json, shared_default.json); optional, the page shows a hint when missing
Outputs (plain .js files that set one global each, so index.html also works when opened from a folder, where browsers block fetch() of .json):
  main.js  -> window.AC_MAIN   (final results: E2 to E8, overhead, cost model, tables)
  ext.js   -> window.AC_EXT    (mixed-stream extension: val / test / test2 x 2 MB / 8 MB, LSTM-DAgger test)
  win_<name>.js -> window.AC_WIN[name]   (sample windows)
Every number is copied or computed from the inputs by the formulas written here (gap closed = (LRU - m) / (LRU - Belady); 'vs best simple' = (best - m) / best).
"""
import argparse, csv, glob, json, os, re, sys

SIMPLE = ["lru", "srrip", "drrip", "bip", "mru"]
POL = re.compile(r"^(req|own|gen):(mix1b|mix1|mix2)?(spec|all)_(mlp|gbdt|lstm)$")


def read_csv(p):
    with open(p, newline="") as f: return [r for r in csv.reader(f)]


def jwrite(path, var, obj, key=None):
    body = json.dumps(obj, separators=(",", ":"))
    with open(path, "w") as f:
        if key is None: f.write("window.%s=%s;\n" % (var, body))
        else: f.write("window.%s=window.%s||{};window.%s[%s]=%s;\n" % (var, var, var, json.dumps(key), body))


def build_main(final, out):
    wd = json.load(open(os.path.join(final, "export", "web_data.json")))
    tabs = {}
    for p in sorted(glob.glob(os.path.join(final, "tables", "*.csv"))): tabs[os.path.basename(p)[:-4]] = read_csv(p)
    ms = json.load(open(os.path.join(final, "model_set.json")))
    wd["tables"] = tabs
    wd["val_mean_gap"] = ms.get("val_mean_gap", {})
    jwrite(os.path.join(out, "main.js"), "AC_MAIN", wd)
    return wd


def ext_set(path):
    j = json.load(open(path)); R = {r["policy"]: r for r in j["runs"]}
    lru, bel = R["lru"]["misses"], R["belady"]["misses"]
    best_name = min(SIMPLE, key=lambda k: R[k]["misses"]); best = R[best_name]["misses"]
    runs = []
    for name, r in R.items():
        m = POL.match(name)
        e = {"name": name, "misses": r["misses"], "mpki": r["mpki"], "app_mpki": [round(a["mpki"], 3) for a in r["apps"]],
             "app_misses": [a["misses"] for a in r["apps"]],
             "gap": (lru - r["misses"]) / (lru - bel) if lru != bel else None, "vs_best": (best - r["misses"]) / best, "vs_lru": (lru - r["misses"]) / lru}
        if m: e.update(rule=m.group(1), version=m.group(2) or "frozen", scope="spec" if m.group(3) == "spec" else "gen", family=m.group(4))
        else: e.update(rule=None, version=None, scope=None, family=None)
        runs.append(e)
    return {"llc_size": j["llc_size"], "ways": j["llc_ways"], "apps": [a["name"] for a in j["apps"]], "app_llc_accesses": [a["llc_accesses"] for a in j["apps"]],
            "schedule": j["schedule"], "config_hash": j["config_hash"], "best_simple": best_name, "runs": runs}


def build_ext(ext, out):
    E = {"sets": {}, "note": "gap = share of the LRU-to-Belady gap closed; vs_best = fewer misses than the best simple policy of that run (negative = more). "
                              "One mixture per set, one training seed: differences of a few points are within noise."}
    base = os.path.join(ext, "results", "ext")
    for s in ("val", "test", "test2"):
        E["sets"][s] = {}
        for tag, fn in (("default", "mixed_default_s10000.json"), ("llc8m", "mixed_llc8m_16w_s10000.json")):
            p = os.path.join(base, s, fn)
            if os.path.exists(p): E["sets"][s][tag] = ext_set(p)
    # LSTM DAgger test (8 test traces, 2 MB): mean gap closed, specialized = mean over the 4 workloads of the mean of that workload's 2 traces
    per = {}
    for f in sorted(glob.glob(os.path.join(base, "lstm_test", "test", "*.result.json"))):
        j = json.load(open(f)); wl = os.path.basename(f).split("__")[1].split("_")[0]
        for p in j["policies"]: per.setdefault(p["name"], {}).setdefault(wl, []).append(p["gap_closed"])
    def mean(x): return sum(x) / len(x)
    spec = {"MLP _dag1": "mlp:%s_mlp_dag1", "LSTM base": "lstm:%s_lstm", "LSTM _dag1": "lstm:%s_lstm_dag1"}
    gen = {"MLP _dag1": "mlp:all_mlp_dag1", "LSTM base": "lstm:all_lstm", "LSTM _dag1": "lstm:all_lstm_dag1"}
    L = {"per_workload": {}, "specialized_mean": {}, "generic_mean": {}, "traces": 8}
    wls = ["hash", "graph", "matrix", "conv"]
    for k, pat in spec.items():
        pw = {w: mean(per[pat % w][w]) for w in wls if pat % w in per}
        L["per_workload"][k] = pw; L["specialized_mean"][k] = mean(list(pw.values()))
    for k, nm in gen.items():
        v = [x for w in per[nm].values() for x in w]; L["generic_mean"][k] = mean(v)
    for k in ("srrip", "drrip", "bip", "mru"):
        L.setdefault("simple_mean", {})[k] = mean([x for w in per[k].values() for x in w])
    E["lstm_dagger_test"] = L
    jwrite(os.path.join(out, "ext.js"), "AC_EXT", E)
    return E


def build_windows(win, out):
    n = 0
    for p in sorted(glob.glob(os.path.join(win, "*.json"))):
        name = os.path.basename(p)[:-5]
        j = json.load(open(p)); j["trace"] = os.path.basename(j.get("trace", ""))   # no local paths in the shipped data
        jwrite(os.path.join(out, "win_%s.js" % name), "AC_WIN", j, key=name); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True); ap.add_argument("--ext", required=True)
    ap.add_argument("--windows", default=""); ap.add_argument("--out", default="web_simulation/data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wd = build_main(a.final, a.out); E = build_ext(a.ext, a.out)
    nw = build_windows(a.windows, a.out) if a.windows and os.path.isdir(a.windows) else 0
    if not nw: print("note: no sample windows found (run scripts/make_windows.py, then build again); the page shows a hint instead of the animation")
    print("main.js: %d e2 traces, %d e7 slice lengths, %d e8 configs; ext.js: sets %s; windows: %d" % (
        len(wd["e2"]), len(wd["e7"]), len(wd["e8"]), {s: list(v) for s, v in E["sets"].items()}, nw))


if __name__ == "__main__":
    main()
