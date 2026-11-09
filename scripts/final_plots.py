"""Plots for the final results (called by final_report.py). Writes PNG files to <final>/plots/. Only numbers already in the result files are drawn.
Colors: one fixed color per role in every figure (baseline grey, specialized blue, generic orange, Belady black)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from final_plan import policy_name, WORKLOADS

SIMPLE = ["lru", "fifo", "random", "srrip", "brrip", "drrip", "bip", "mru"]
C = {"simple": "#8c8c8c", "spec": "#2a6fb0", "gen": "#e07b1a", "big": "#7a4fb5", "gbdt": "#2e8b57", "lstm": "#b5446e", "belady": "#111111", "lru": "#c9c9c9"}


def _gap(m, lru, bel):
    return 100.0 * (lru - m) / (lru - bel) if lru > bel else None


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("plot:", path)


def e2_plot(final, runs, ms):
    F = ms["files"]
    roles = [("best simple", None, C["simple"]), ("mlp specialized", "spec", C["spec"]), ("mlp generic", "gen", C["gen"]),
             ("gbdt specialized", "gbdt", C["gbdt"]), ("lstm specialized", "lstm", C["lstm"])]
    labels = sorted(runs, key=lambda l: (WORKLOADS.index(runs[l]["w"]) if runs[l]["w"] in WORKLOADS else 9, l))
    labels = [l for l in labels if runs[l]["pol"]["lru"]["misses"] > runs[l]["pol"]["belady"]["misses"]]
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(labels)), 4.2))
    width = 0.8 / len(roles)
    for j, (name, key, col) in enumerate(roles):
        ys = []
        for l in labels:
            P, w = runs[l]["pol"], runs[l]["w"]
            lru, bel = P["lru"]["misses"], P["belady"]["misses"]
            if key is None:
                m = min(P[n]["misses"] for n in SIMPLE if n in P)
            else:
                pn = policy_name(F["mlp"][w] if key == "spec" else F["mlp"]["all"]) if key in ("spec", "gen") else policy_name(F[key][w])
                m = P[pn]["misses"] if pn in P else None
            ys.append(_gap(m, lru, bel) if m is not None else 0)
        ax.bar([i + j * width for i in range(len(labels))], ys, width, label=name, color=col)
    ax.axhline(100, color=C["belady"], lw=1, ls="--")
    ax.text(len(labels) - 0.4, 101, "Belady = 100%", ha="right", va="bottom", fontsize=8)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(labels))])
    ax.set_xticklabels([l.replace("test__", "") for l in labels], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("gap closed (%)  (LRU = 0, Belady = 100)")
    ax.set_title("Test split: share of the LRU-to-Belady gap closed")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    _save(fig, os.path.join(final, "plots", "e2_gap_closed.png"))


def e3_plot(final, runs, ms):
    F = ms["files"]; big = policy_name(F["mlp"]["allbig"]) if "allbig" in F["mlp"] else None
    arms = [("specialized", lambda w: policy_name(F["mlp"][w]), C["spec"]), ("generic (equal size)", lambda w: policy_name(F["mlp"]["all"]), C["gen"])]
    if big:
        arms.append(("generic (equal total storage)", lambda w: big, C["big"]))
    vals = {a[0]: [] for a in arms}
    ws = []
    for w in WORKLOADS:
        gs = {a[0]: [] for a in arms}
        for l, d in runs.items():
            if d["w"] != w:
                continue
            P = d["pol"]
            lru, bel = P["lru"]["misses"], P["belady"]["misses"]
            for name, fn, _ in arms:
                if fn(w) in P and lru > bel:
                    gs[name].append(_gap(P[fn(w)]["misses"], lru, bel))
        if all(gs[a[0]] for a in arms):
            ws.append(w)
            for a in arms:
                vals[a[0]].append(sum(gs[a[0]]) / len(gs[a[0]]))
    if not ws:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    width = 0.8 / len(arms)
    for j, (name, _, col) in enumerate(arms):
        ax.bar([i + j * width for i in range(len(ws))], vals[name], width, label=name, color=col)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(ws))]); ax.set_xticklabels(ws)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_ylabel("gap closed (%), mean over test inputs"); ax.set_title("E3: specialized vs generic MLP"); ax.legend(fontsize=8)
    _save(fig, os.path.join(final, "plots", "e3_specialized_vs_generic.png"))


def e7_plot(final, e7d):
    if not e7d:
        return
    keys = sorted(e7d, key=lambda k: e7d[k]["sched"]["slice_instr"])
    x = [e7d[k]["sched"]["slice_instr"] for k in keys]
    R0 = e7d[keys[0]]["runs"]
    series = [("lru", C["lru"], "LRU"), ("belady", C["belady"], "Belady")]
    for n in R0:
        if n.startswith("req:spec_mlp"):
            series.append((n, C["spec"], "per-application MLPs"))
        elif n.startswith("gen:all_mlp") and "big" not in n:
            series.append((n, C["gen"], "generic MLP"))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n, col, lab in series:
        ys = [e7d[k]["runs"][n]["misses"] for k in keys if n in e7d[k]["runs"]]
        if len(ys) == len(x):
            ax.plot(x, ys, marker="o", color=col, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("time slice (instructions)"); ax.set_ylabel("LLC misses (all applications)")
    ax.set_title("E7: round-robin switching, misses by slice length"); ax.legend(fontsize=8)
    _save(fig, os.path.join(final, "plots", "e7_switching.png"))


def e8_plot(final, e8d):
    for key, R in (e8d or {}).items():
        pols = [n for n in R if n in ("lru", "srrip", "drrip", "bip", "mru", "belady") or ":" in n]
        if not pols:
            continue
        apps = len(R[pols[0]]["app_mpki"])
        fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(pols) + 3), 4))
        width = 0.8 / apps
        for a in range(apps):
            ax.bar([i + a * width for i in range(len(pols))], [R[n]["app_mpki"][a] for n in pols], width, label="application %d" % (a + 1))
        ax.set_xticks([i + 0.4 - width / 2 for i in range(len(pols))]); ax.set_xticklabels(pols, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("MPKI per application"); ax.set_title("E8: shared LLC (%s)" % key); ax.legend(fontsize=8)
        _save(fig, os.path.join(final, "plots", "e8_%s.png" % key))


def make(final, runs, ms, e7d, e8d):
    e2_plot(final, runs, ms)
    e3_plot(final, runs, ms)
    e7_plot(final, e7d)
    e8_plot(final, e8d)
