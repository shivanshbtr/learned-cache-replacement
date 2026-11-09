#!/usr/bin/env bash
# THE final evaluation (Batch E+F). Freezes the model set from the VALIDATION results, then runs the TEST split once and builds tables/plots/JSON.
# Stages (env STAGES, default "select e2 multi e6 report"): select = freeze model set; e2 = E1-E5; multi = E7+E8; e6 = overhead (quiet machine!); report.
# Resumable: finished result files are skipped. Once the test split has been started, the model set is frozen and cannot be re-selected (rm results/final/test_started to override).
# usage: scripts/run_final.sh        env: JOBS (default 4)  JOBS_MULTI (default 3)  see final_e2.sh / final_multi.sh / final_overhead.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIN="${FIN:-$ROOT/results/final}"; export FIN
mkdir -p "$FIN"
for b in cachesim multisim; do [ -x "$ROOT/build/$b" ] || { echo "Run 'make all' first."; exit 1; }; done
for s in ${STAGES:-select e2 multi e6 report}; do
  case "$s" in
    select) python3 "$ROOT/scripts/final_plan.py" select --final "$FIN" ${SEL_ARGS:-} || exit 1
            [ -f "$FIN/test_started" ] || date -u +%FT%TZ > "$FIN/test_started"
            cp "$FIN/model_set.json" "$FIN/model_set.frozen.json";;
    e2)     bash "$ROOT/scripts/final_e2.sh";;
    multi)  JOBS="${JOBS_MULTI:-3}" bash "$ROOT/scripts/final_multi.sh";;
    e6)     bash "$ROOT/scripts/final_overhead.sh";;
    report) python3 "$ROOT/scripts/final_report.py" --final "$FIN";;
    *) echo "unknown stage $s"; exit 1;;
  esac
done
