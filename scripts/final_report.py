#!/usr/bin/env python3
"""Aggregate the final (test split) results into tables, CSV files, plots and the JSON for the web simulation.
usage: python3 scripts/final_report.py [--final results/final] [--cost configs/cost_model.json] [--no-plots]
Reads  <final>/model_set.json, e2/ (E1-E4), e5/, multi/ (E7, E8), e6/ ;  writes  <final>/tables/*.csv|md, plots/*.png, export/web_data.json, REPORT_NUMBERS.md
Definitions: gap closed = (LRU - policy) / (LRU - Belady) on measured LLC misses; 'vs best simple' = % fewer misses than the best of
lru fifo random srrip brrip drrip bip mru on the same trace (+ is better). Model roles come from model_set.json (frozen before the test run)."""
import argparse, csv, glob, json, math, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from final_plan import policy_name, WORKLOADS, FAM

SIMPLE = ["lru", "fifo", "random", "srrip", "brrip", "drrip", "bip", "mru"]
KINDS = ["mlp", "gbdt", "lstm"]
MD_OUT = []   # (section title, markdown) collected for REPORT_NUMBERS.md


def gap(m, lru, bel):
    return (lru - m) / (lru - bel) if lru > bel else None


def pct(x, d=1):
    return "-" if x is None else "%.*f%%" % (d, 100 * x)


def md_table(hdr, rows):
    w = [max(len(str(r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([line(hdr), "| " + " | ".join("-" * w[i] for i in range(len(hdr))) + " |"] + [line(r) for r in rows])


def write_csv(path, hdr, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(hdr); wr.writerows(rows)


def section(tabdir, name, title, hdr, rows, note=""):
    write_csv(os.path.join(tabdir, name + ".csv"), hdr, rows)
    md = md_table(hdr, rows) + ("\n\n" + note if note else "")
    open(os.path.join(tabdir, name + ".md"), "w").write("### %s\n\n%s\n" % (title, md))
    MD_OUT.append((title, md))
    print("\n== %s\n%s" % (title, md))


def meanstd(v):
    return (sum(v) / len(v), statistics.stdev(v) if len(v) > 1 else 0.0)


# ------------------------------------------------------------------ loading
def load_runs(folder):
    """{label: {"w", "pol": {name: record}, "acc", "hash"}} merged over the group files <label>.<group>.result.json."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(folder, "*.result.json"))):
        name = os.path.basename(f)[:-len(".result.json")]
        label = name.rsplit(".", 1)[0]
        r = json.load(open(f))
        d = runs.setdefault(label, {"w": label.split("__", 1)[1].split("_", 1)[0], "pol": {}, "acc": r["trace"]["llc_accesses"], "hash": r["config"]["hash"]})
        for p in r["policies"]:
            if p["name"] in d["pol"] and d["pol"][p["name"]]["misses"] != p["misses"]:
                print("WARNING: %s: policy %s has different miss counts in different group files (%d vs %d)" % (label, p["name"], d["pol"][p["name"]]["misses"], p["misses"]))
            d["pol"].setdefault(p["name"], p)
    return runs


def roles_of(ms):
    """names of the model policies per role: spec[kind][w], gen[kind], big, seeds[who]."""
    F = ms["files"]
    spec = {k: {w: policy_name(F[k][w]) for w in WORKLOADS} for k in KINDS}
    gen = {k: policy_name(F[k]["all"]) for k in KINDS}
    big = policy_name(F["mlp"]["allbig"]) if "allbig" in F["mlp"] else None
    seeds = {who: [policy_name(p) for p in lst] for who, lst in ms.get("seed_files", {}).items()}
    return spec, gen, big, seeds


def stats_of(rec, lru, bel, best):
    m = rec["misses"]
    return m, gap(m, lru, bel), (100.0 * (best - m) / best if best else None)


# ------------------------------------------------------------------ E1 / E2
def e2(runs, ms, tabdir):
    spec, gen, big, seeds = roles_of(ms)
    rows, long_rows, per_w = [], [], {}
    for label in sorted(runs, key=lambda l: (WORKLOADS.index(runs[l]["w"]) if runs[l]["w"] in WORKLOADS else 9, l)):
        d = runs[label]; w = d["w"]; P = d["pol"]
        lru, bel = P["lru"]["misses"], P["belady"]["misses"]
        simple = {n: P[n]["misses"] for n in SIMPLE if n in P}
        bname = min(simple, key=simple.get); best = simple[bname]
        cols = [("lru", "lru"), ("best simple (%s)" % bname, bname)] + [(n, n) for n in SIMPLE if n in P and n not in ("lru", bname)]
        for k in KINDS:
            cols += [("%s specialized" % k, spec[k][w]), ("%s generic" % k, gen[k])]
        if big: cols.append(("mlp generic-big", big))
        cols.append(("belady", "belady"))
        first = True
        for role, pn in cols:
            if pn not in P: continue
            m, g, vs = stats_of(P[pn], lru, bel, best)
            long_rows.append([label, w, role, pn, m, "" if g is None else round(100 * g, 2), "" if vs is None else round(vs, 2)])
            rows.append([label if first else "", role, m, pct(g) if role not in ("lru", "belady") else "-", ("%+.1f%%" % vs) if (vs is not None and role != "belady") else "-"])
            first = False
            per_w.setdefault(w, {}).setdefault(role.split(" (")[0], []).append(g)
    write_csv(os.path.join(tabdir, "e2_test_long.csv"), ["trace", "workload", "role", "policy", "misses", "gap_closed_pct", "vs_best_simple_pct"], long_rows)
    section(tabdir, "e2_test", "E1/E2: test split, LLC misses, gap closed, gain over the best simple baseline", ["trace", "policy", "misses", "gap closed", "vs best simple"], rows,
            "Gap closed = (LRU - policy) / (LRU - Belady). 'best simple' = fewest misses among lru, fifo, random, srrip, drrip, bip, mru on that trace.")
    hdr = ["workload", "best simple", "mlp specialized", "mlp generic", "gbdt specialized", "gbdt generic", "lstm specialized", "lstm generic"]
    srows = []
    for w in WORKLOADS:
        if w not in per_w: continue
        # the best simple policy can differ per trace, so recompute its gap per trace
        bs = []
        for label, d in runs.items():
            if d["w"] != w: continue
            P = d["pol"]; sm = min(P[n]["misses"] for n in SIMPLE if n in P)
            bs.append(gap(sm, P["lru"]["misses"], P["belady"]["misses"]))
        avg = lambda v: [x for x in v if x is not None]
        def cell(role):
            v = avg(per_w[w].get(role, []))
            return pct(sum(v) / len(v)) if v else "-"
        srows.append([w, pct(sum(avg(bs)) / len(avg(bs))) if avg(bs) else "-"] + [cell(r) for r in hdr[2:]])
    section(tabdir, "e2_summary", "E2 summary: mean gap closed over the test inputs of each workload", hdr, srows)
    return per_w


# ------------------------------------------------------------------ E3
def e3(runs, ms, tabdir):
    spec, gen, big, seeds = roles_of(ms)
    rows, long_rows, adv = [], [], {}
    for label in sorted(runs):
        d = runs[label]; w = d["w"]; P = d["pol"]
        if w not in WORKLOADS or "mlp:" not in "".join(P): continue
        lru, bel = P["lru"]["misses"], P["belady"]["misses"]
        if lru <= bel: continue
        def arm(main, who):
            names = [main] + seeds.get(who, [])
            g = [gap(P[n]["misses"], lru, bel) for n in names if n in P]
            return g
        A = {"specialized": arm(spec["mlp"][w], w), "generic (equal size)": arm(gen["mlp"], "all")}
        if big: A["generic-big (equal total storage)"] = arm(big, "allbig")
        base = None
        for name, g in A.items():
            if not g: continue
            m, s = meanstd(g)
            if name == "specialized": base = m
            rows.append([label if name == "specialized" else "", name, len(g), "%.1f%% +- %.1f" % (100 * m, 100 * s), ("%+.1f pts" % (100 * (base - m))) if (base is not None and name != "specialized") else "-"])
            long_rows.append([label, w, name, len(g), round(100 * m, 3), round(100 * s, 3)])
            adv.setdefault(w, {}).setdefault(name, []).append(m)
        for k in ("gbdt", "lstm"):
            sg, gg = gap(P[spec[k][w]]["misses"], lru, bel) if spec[k][w] in P else None, gap(P[gen[k]]["misses"], lru, bel) if gen[k] in P else None
            if sg is None or gg is None: continue
            rows.append(["", "%s specialized vs generic" % k, 1, "%.1f%% vs %.1f%%" % (100 * sg, 100 * gg), "%+.1f pts" % (100 * (sg - gg))])
            long_rows.append([label, w, k + " specialized", 1, round(100 * sg, 3), 0]); long_rows.append([label, w, k + " generic", 1, round(100 * gg, 3), 0])
    write_csv(os.path.join(tabdir, "e3_long.csv"), ["trace", "workload", "arm", "n_seeds", "gap_closed_mean_pct", "gap_closed_std_pct"], long_rows)
    section(tabdir, "e3", "E3: specialized vs generic (equal per-model size, equal total storage); gap closed, mean +- std over training seeds", ["trace", "arm", "seeds", "gap closed", "specialized - this arm"], rows,
            "Training seeds change only the final training run on identical labeled data (the DAgger data were produced by the seed-1 models). A positive last column means specialization wins by that many points of gap.")
    return adv


# ------------------------------------------------------------------ E4
def e4(runs, ms, tabdir):
    spec, gen, big, seeds = roles_of(ms)
    cells = {}
    for label, d in runs.items():
        w = d["w"]; P = d["pol"]
        if w not in WORKLOADS or P["lru"]["misses"] <= P["belady"]["misses"]: continue
        for x in WORKLOADS + ["generic"]:
            n = gen["mlp"] if x == "generic" else spec["mlp"][x]
            if n in P: cells.setdefault((x, w), []).append(gap(P[n]["misses"], P["lru"]["misses"], P["belady"]["misses"]))
    rows = []
    for x in WORKLOADS + ["generic"]:
        rows.append([("model of " + x) if x != "generic" else "generic model"] + [pct(sum(cells[(x, w)]) / len(cells[(x, w)])) if (x, w) in cells else "-" for w in WORKLOADS])
    section(tabdir, "e4", "E4: cross-workload matrix (MLP of the row, run on the test traces of the column): mean gap closed", ["", *["on " + w for w in WORKLOADS]], rows,
            "Diagonal = the model on its own workload. Negative = worse than LRU. A model never trained on a workload still has to run on it when applications are mixed up.")
    return cells


# ------------------------------------------------------------------ E5
def e5(final, ms, tabdir, e2runs):
    spec, gen, big, seeds = roles_of(ms)
    rows = []
    dirs = sorted(d for d in glob.glob(os.path.join(final, "e5", "*")) if os.path.isdir(d))
    sets = [("default (2 MB, 16-way)", e2runs)] + [(os.path.basename(d), load_runs(d)) for d in dirs]
    for name, runs in sets:
        for w in WORKLOADS:
            vals = {"best simple": [], "spec": [], "gen": [], "lru": [], "belady": []}
            hashes = set()
            for label, d in runs.items():
                if d["w"] != w: continue
                P = d["pol"]; hashes.add(d["hash"])
                if P["lru"]["misses"] <= P["belady"]["misses"]: continue
                lru, bel = P["lru"]["misses"], P["belady"]["misses"]
                vals["best simple"].append(gap(min(P[n]["misses"] for n in SIMPLE if n in P), lru, bel))
                if spec["mlp"][w] in P: vals["spec"].append(gap(P[spec["mlp"][w]]["misses"], lru, bel))
                if gen["mlp"] in P: vals["gen"].append(gap(P[gen["mlp"]]["misses"], lru, bel))
                vals["lru"].append(lru); vals["belady"].append(bel)
            if not vals["lru"] and not vals["spec"]:
                rows.append([name, w, "no headroom (LRU = Belady)", "-", "-", "-"]); continue
            f = lambda v: pct(sum(v) / len(v)) if v else "-"
            rows.append([name, w, f(vals["best simple"]), f(vals["spec"]), f(vals["gen"]), ",".join(sorted(hashes))])
    if len(sets) > 1:
        section(tabdir, "e5", "E5: sensitivity to the LLC configuration (models trained at 2 MB / 16-way, NOT retrained): mean gap closed on the test traces",
                ["LLC config", "workload", "best simple", "mlp specialized", "mlp generic", "config hash"], rows,
                "The L1/L2 filters are unchanged, so the same LLC traces are replayed on a different LLC. Rows without headroom have LRU = Belady on all test traces of that workload.")


# ------------------------------------------------------------------ E6
def e6(final, ms, tabdir, cost):
    spec, gen, big, seeds = roles_of(ms)
    st = {}   # policy -> list of model stats
    for f in sorted(glob.glob(os.path.join(final, "e6", "*.result.json"))):
        r = json.load(open(f))
        for p in r["policies"]:
            if "model" in p: st.setdefault(p["name"], []).append(p["model"])
    if not st: print("E6: no overhead results found (run scripts/final_overhead.sh)"); return {}
    med = lambda v: statistics.median(v)
    rows, out = [], {}
    for k in KINDS:
        names = [("specialized %s (%s)" % (k, w), spec[k][w]) for w in WORKLOADS] + [("generic %s" % k, gen[k])]
        if k == "mlp" and big: names.append(("generic-big mlp", big))
        for role, n in names:
            if n not in st: continue
            s = st[n]
            m = {"params": s[0]["params"], "bytes": s[0]["model_bytes"], "ops": med([x["ops_per_decision"] for x in s]), "lat_mean_ns": med([x["lat_mean_ns"] for x in s]),
                 "lat_p50_ns": med([x["lat_p50_ns"] for x in s]), "lat_p99_ns": med([x["lat_p99_ns"] for x in s]), "decisions": sum(x["decisions"] for x in s), "kind": k}
            out[n] = dict(m, role=role)
            rows.append([role, m["params"], m["bytes"], "%.0f" % m["ops"], "%.1f" % (m["lat_mean_ns"] / 1000), "%.1f" % (m["lat_p50_ns"] / 1000), "%.1f" % (m["lat_p99_ns"] / 1000)])
    section(tabdir, "e6_models", "E6: model cost (software inference in the simulator; latency in microseconds, median over the runs)",
            ["model", "params", "bytes", "ops/decision", "latency mean us", "p50 us", "p99 us"], rows,
            "ops/decision counts multiply-adds as 2 ops (trees: node visits). A real LLC replacement decision has to finish in about 100 ns, so these software numbers say how much faster a dedicated accelerator would have to be.")
    tot = []
    for k in KINDS:
        sp = [out[spec[k][w]]["bytes"] for w in WORKLOADS if spec[k][w] in out]
        if len(sp) == 4 and gen[k] in out:
            tot.append([k, sum(sp), out[gen[k]]["bytes"], out[big]["bytes"] if (k == "mlp" and big in out) else "-", "%.1fx" % (sum(sp) / out[gen[k]]["bytes"])])
    section(tabdir, "e6_storage", "E6: total storage of four specialized models vs the generic model(s), bytes", ["family", "4 specialized", "generic", "generic-big", "4 specialized / generic"], tot)
    act = []
    for k in KINDS:
        for w in WORKLOADS:
            n = spec[k][w]
            if n not in out: continue
            for tf in cost["t_fixed_us"]:
                for bw in cost["bandwidth_gbps"]:
                    act.append([k, w, out[n]["bytes"], tf, bw, round(tf + out[n]["bytes"] / (bw * 1e3), 3), tf])
    write_csv(os.path.join(tabdir, "e6_activation.csv"), ["family", "workload", "model_bytes", "t_fixed_us", "bandwidth_GBps", "on_demand_us", "resident_us"], act)
    arows = []
    for k in KINDS:
        sizes = [out[spec[k][w]]["bytes"] for w in WORKLOADS if spec[k][w] in out]
        if not sizes: continue
        b = sum(sizes) / len(sizes)
        for bw in cost["bandwidth_gbps"]:
            arows.append([k, int(b), bw] + ["%.2f" % (tf + b / (bw * 1e3)) for tf in cost["t_fixed_us"]])
    section(tabdir, "e6_activation_summary", "E6: activation time T = T_fixed + bytes / bandwidth (microseconds, mean model size of the family)",
            ["family", "mean bytes", "bandwidth GB/s"] + ["T_fixed=%gus" % t for t in cost["t_fixed_us"]], arows,
            "Resident mode (all models already in the model store) costs only T_fixed. This is a cost MODEL, not a measurement.")
    return out


# ------------------------------------------------------------------ multi-application runs
def load_multi(final, prefix):
    """{key (e.g. 'default_s10000' or 's123'): {"sched", "apps", "runs": {policy: run}, "cfg"}} merged over group files."""
    out = {}
    for f in sorted(glob.glob(os.path.join(final, "multi", prefix + "*.json"))):
        name = os.path.basename(f)[:-5]
        key = name.rsplit(".", 1)[0][len(prefix):]
        r = json.load(open(f))
        d = out.setdefault(key, {"sched": r["schedule"], "apps": [a["name"] for a in r["apps"]], "runs": {}, "llc": "%dKB/%d-way" % (r["llc_size"] // 1024, r["llc_ways"]), "hash": r["config_hash"]})
        for run in r["runs"]:
            d["runs"].setdefault(run["policy"], run)
    return out


def best_simple(runs):
    c = {n: runs[n]["misses"] for n in SIMPLE if n in runs}
    return min(c, key=c.get) if c else None


def e7(final, ms, tabdir, cost, e6s):
    D = load_multi(final, "e7_")
    if not D: print("E7: no results found"); return {}
    spec, gen, big, seeds = roles_of(ms)
    keys = sorted(D, key=lambda k: D[k]["sched"]["slice_instr"])
    apps = D[keys[0]]["apps"]
    polnames = [n for n in ["lru", "srrip", "drrip", "bip", "mru"] if n in D[keys[0]]["runs"]] + sorted(n for n in D[keys[0]]["runs"] if ":" in n) + ["belady"]
    rows = []
    for k in keys:
        d = D[k]; R = d["runs"]; bs = best_simple(R)
        for n in polnames:
            if n not in R: continue
            m = R[n]["misses"]
            rows.append([d["sched"]["slice_instr"] if n == polnames[0] else "", d["sched"]["switches"] if n == polnames[0] else "", n, m,
                         "%+.1f%%" % (100.0 * (R["lru"]["misses"] - m) / R["lru"]["misses"]) if n != "lru" else "-",
                         ("%+.1f%%" % (100.0 * (R[bs]["misses"] - m) / R[bs]["misses"])) if bs and n not in ("belady",) and n != bs else "-"])
    section(tabdir, "e7_misses", "E7: switching between the %d applications (round robin), total LLC misses by time-slice length" % len(apps),
            ["slice (instructions)", "switches", "policy", "misses", "vs LRU", "vs best simple"], rows,
            "The LLC contents persist across switches (lines of the other applications are evicted, not flushed), so every resume starts partly cold. req:spec = the active application's own model; gen:* = one generic model for all.")
    # interference and cold-cache effect
    irows = []
    for k in keys:
        d = D[k]; R = d["runs"]
        for n in ["lru", next((x for x in R if x.startswith("req:spec_mlp")), None)]:
            if not n or n not in R or "alone" not in R[n]: continue
            for i, a in enumerate(apps):
                sh, al = R[n]["apps"][i], R[n]["alone"][i]
                cold = sh["cold_miss"] / sh["cold_acc"] if sh["cold_acc"] else float("nan")
                tot = sh["misses"] / sh["accesses"] if sh["accesses"] else float("nan")
                irows.append([d["sched"]["slice_instr"] if (n == "lru" and i == 0) else "", n if i == 0 else "", a, "%.2f" % al["mpki"], "%.2f" % sh["mpki"],
                              "%.2fx" % (sh["mpki"] / al["mpki"]) if al["mpki"] else "-", "%.2f" % cold if cold == cold else "-", "%.2f" % tot if tot == tot else "-"])
    section(tabdir, "e7_interference", "E7: cost of switching per application: MPKI alone vs MPKI when time-sliced, and miss rate right after a switch-in",
            ["slice (instr)", "policy", "application", "MPKI alone", "MPKI sliced", "ratio", "miss rate first %d accesses after switch-in" % D[keys[0]]["sched"]["cold_k"], "miss rate overall"], irows)
    # cost model
    main = cost["main"]
    def act_ns(nbytes, mode, p):
        return p["t_fixed_us"] * 1e3 + (nbytes / (p["bandwidth_gbps"]) if mode == "on_demand" else 0.0)   # bytes / (GB/s) = ns
    crow, be = [], []
    for k in keys:
        d = D[k]; R = d["runs"]; bs = best_simple(R)
        for n in sorted(x for x in R if ":" in x):
            r = R[n]
            saved = R[bs]["misses"] - r["misses"]
            dec = sum(m.get("decisions", 0) for m in r["models"])
            byts = [m.get("model_bytes", 0) for m in r["models"]]
            sl = r.get("slices", [])
            prev, sw_apps = None, []
            for s in sl:
                if prev is not None and s[0] != prev: sw_apps.append(s[0])
                prev = s[0]
            for mode in ("resident", "on_demand"):
                for L in cost["hw_latency_ns"]:
                    p = dict(main, t_fixed_us=main["t_fixed_us"])
                    act = 0.0
                    if n.startswith("req:") or n.startswith("own:"):
                        act = sum(act_ns(byts[a], mode, p) for a in sw_apps)
                    stall = dec * max(0.0, L - main["t_miss_ns"])
                    net = saved * main["t_miss_ns"] - act - stall
                    crow.append([d["sched"]["slice_instr"], n, mode, L, saved, dec, round(act / 1e6, 3), round(stall / 1e6, 3), round(saved * main["t_miss_ns"] / 1e6, 3), round(net / 1e6, 3)])
    write_csv(os.path.join(tabdir, "e7_cost_model.csv"), ["slice_instr", "policy", "mode", "decision_latency_ns", "misses_saved_vs_best_simple", "decisions", "activation_ms", "decision_stall_ms", "miss_time_saved_ms", "net_ms"], crow)
    # break-even slice length (analytic): per switch cost c, saved time per instruction sigma at the longest slice -> S* = c / (sigma - stall per instr)
    last = D[keys[-1]]; R = last["runs"]; bs = best_simple(R)
    for n in sorted(x for x in R if x.startswith("req:")):
        r = R[n]; icount = r["measured_icount"] or 1
        saved_t = (R[bs]["misses"] - r["misses"]) * main["t_miss_ns"]
        byts = [m.get("model_bytes", 0) for m in r["models"]]
        c_res = act_ns(sum(byts) / len(byts), "resident", main); c_od = act_ns(sum(byts) / len(byts), "on_demand", main)
        for L in cost["hw_latency_ns"]:
            stall = sum(m.get("decisions", 0) for m in r["models"]) * max(0.0, L - main["t_miss_ns"])
            sigma = (saved_t - stall) / icount
            be.append([n, L, "%.1f" % (saved_t / 1e6), "%.1f" % (stall / 1e6)] + [("%.0f instr (%.2f ms)" % (c / sigma, c / sigma * cost["ns_per_instr"] / 1e6)) if sigma > 0 else "never (net loss)" for c in (c_res, c_od)])
    section(tabdir, "e7_breakeven", "E7: break-even slice length (specialized models): slices shorter than this lose more to activation than they save in misses (cost model, main parameters)",
            ["policy", "decision latency ns", "miss time saved vs best simple (ms)", "decision stall (ms)", "resident: slice >=", "on demand: slice >="], be,
            "Cost model parameters (configs/cost_model.json): miss penalty %g ns, T_fixed %g us, bandwidth %g GB/s, %g ns per instruction (assumed 3 GHz, IPC 1). A decision stalls the cache only by the part of its latency above the miss penalty. These are assumptions, swept in e7_cost_model.csv." %
            (main["t_miss_ns"], main["t_fixed_us"], main["bandwidth_gbps"], cost["ns_per_instr"]))
    return {k: D[k] for k in keys}


def jain(v):
    return (sum(v) ** 2) / (len(v) * sum(x * x for x in v)) if v and sum(x * x for x in v) > 0 else float("nan")


def e8(final, ms, tabdir):
    D = load_multi(final, "e8_")
    if not D: print("E8: no results found"); return {}
    out = {}
    for key in sorted(D):
        d = D[key]; R = d["runs"]; apps = d["apps"]; bs = best_simple(R)
        pols = [n for n in ["lru", "srrip", "drrip", "bip", "mru"] if n in R] + sorted(n for n in R if ":" in n) + (["belady"] if "belady" in R else [])
        rows = []
        for n in pols:
            r = R[n]
            sh = [a["mpki"] for a in r["apps"]]
            al = [a["mpki"] for a in r["alone"]] if "alone" in r else None
            npv = [(al[i] / sh[i]) if sh[i] > 0 else 1.0 for i in range(len(apps))] if al else None
            ev = r["evict"]; tot = sum(map(sum, ev)); cross = sum(ev[i][j] for i in range(len(ev)) for j in range(len(ev)) if i != j)
            rows.append([n, r["misses"], "%.2f" % r["mpki"], ("%+.1f%%" % (100.0 * (R["lru"]["misses"] - r["misses"]) / R["lru"]["misses"])) if n != "lru" else "-"] +
                        ["%.2f" % x for x in sh] + [("%.2f" % max(sh[i] / al[i] for i in range(len(sh)) if al[i] > 0)) if al else "-", ("%.3f" % jain(npv)) if npv else "-",
                        ("%.1f%%" % (100.0 * cross / tot)) if tot else "-"])
            out.setdefault(key, {})[n] = {"misses": r["misses"], "mpki": r["mpki"], "app_mpki": sh, "alone_mpki": al, "jain": jain(npv) if npv else None,
                                         "cross_evict_share": cross / tot if tot else None, "borrow": [a["borrow"] for a in r["apps"]], "evict": ev}
        section(tabdir, "e8_" + key, "E8: %d applications sharing the LLC (%s, slice %d instr, %d slices, until first finished), %s" % (len(apps), d["llc"], d["sched"]["slice_instr"], d["sched"]["slices"], key),
                ["policy", "misses", "MPKI", "vs LRU"] + ["MPKI " + a for a in apps] + ["max slowdown", "Jain fairness", "cross-app evictions"], rows,
                "max slowdown = largest MPKI(shared) / MPKI(alone, same policy); Jain fairness over normalized progress MPKI(alone)/MPKI(shared) (1 = no interference); cross-app evictions = share of evictions that removed another application's line. req = requester's model, own = ownership (quota ways/apps, a model only evicts its own application's lines), gen = one generic model.")
    return out


# ------------------------------------------------------------------ web export
def export(final, ms, e2runs, e6s, e7d, e8d, cost):
    spec, gen, big, seeds = roles_of(ms)
    out = {"schema": 1, "note": "Numbers computed on complete test traces by the C++ simulator; timelines are per-slice aggregates.",
           "selected": ms["selected"], "cost_model": cost, "models": {}, "e2": {}, "e6": e6s}
    for k in KINDS:
        for w in WORKLOADS + ["all"]:
            p = ms["files"][k][w]; n = policy_name(p)
            out["models"]["%s/%s" % (k, w)] = {"policy": n, "bytes": os.path.getsize(p), "sha256": ms["sha256"][p]}
    for label, d in e2runs.items():
        P = d["pol"]
        if "lru" not in P or "belady" not in P: continue
        out["e2"][label] = {"workload": d["w"], "llc_accesses": d["acc"], "config_hash": d["hash"],
                            "policies": {n: {"misses": p["misses"], "mpki": p["mpki"], "gap_closed": gap(p["misses"], P["lru"]["misses"], P["belady"]["misses"])} for n, p in P.items()}}
    out["e7"] = {k: {"slice_instr": d["sched"]["slice_instr"], "switches": d["sched"]["switches"], "apps": d["apps"],
                     "runs": {n: {"misses": r["misses"], "mpki": r["mpki"], "apps": [{"app": a["app"], "misses": a["misses"], "mpki": a["mpki"]} for a in r["apps"]],
                                  "slices": r.get("slices") if (n in ("lru", "belady") or n.startswith("req:spec_mlp") or n.startswith("gen:all_mlp")) else None}
                              for n, r in d["runs"].items()}} for k, d in e7d.items()}
    out["e8"] = e8d
    os.makedirs(os.path.join(final, "export"), exist_ok=True)
    json.dump(out, open(os.path.join(final, "export", "web_data.json"), "w"))
    print("\nwrote %s (%d KB)" % (os.path.join(final, "export", "web_data.json"), os.path.getsize(os.path.join(final, "export", "web_data.json")) // 1024))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=os.path.join(ROOT, "results", "final")); ap.add_argument("--cost", default=os.path.join(ROOT, "configs", "cost_model.json"))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    ms = json.load(open(os.path.join(a.final, "model_set.json")))
    cost = json.load(open(a.cost))
    tabdir = os.path.join(a.final, "tables")
    os.makedirs(tabdir, exist_ok=True)
    print("model set (frozen on val): %s" % ms["selected"])
    runs = load_runs(os.path.join(a.final, "e2"))
    if not runs: sys.exit("no E2 results in %s/e2 (run scripts/final_e2.sh)" % a.final)
    e2(runs, ms, tabdir); e3(runs, ms, tabdir); e4(runs, ms, tabdir); e5(a.final, ms, tabdir, runs)
    e6s = e6(a.final, ms, tabdir, cost)
    e7d = e7(a.final, ms, tabdir, cost, e6s); e8d = e8(a.final, ms, tabdir)
    export(a.final, ms, runs, e6s, e7d, e8d, cost)
    with open(os.path.join(a.final, "REPORT_NUMBERS.md"), "w") as f:
        f.write("# Final results (test split)\n\nModel set frozen on validation results: %s\n\n" % json.dumps(ms["selected"]))
        for t, md in MD_OUT: f.write("### %s\n\n%s\n\n" % (t, md))
    if not a.no_plots:
        try:
            import final_plots
            final_plots.make(a.final, runs, ms, e7d, e8d)
        except ImportError as e:
            print("plots skipped (%s); pip install matplotlib to get them" % e)


if __name__ == "__main__":
    main()
