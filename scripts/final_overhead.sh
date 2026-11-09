#!/usr/bin/env bash
# E6: decision latency, ops per decision, parameters and bytes of every final model. RUN THIS ON A QUIET MACHINE (nothing else running):
# the jobs run one after the other and use only the first E6_ACC accesses of the first test trace of each workload (latency needs
# decisions, not the whole trace; miss counts from this run are NOT reported).
# usage: scripts/final_overhead.sh      env: TR FIN CFG INPUTS E6_ACC (default 600000)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TR="${TR:-$ROOT/results/traces}"; FIN="${FIN:-$ROOT/results/final}"; CFG="${CFG:-$ROOT/configs/default.cfg}"; INPUTS="${INPUTS:-$ROOT/configs/inputs.tsv}"
SET="$FIN/model_set.json"; PLAN="python3 $ROOT/scripts/final_plan.py"
[ -f "$SET" ] || { echo "no $SET"; exit 1; }
load=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)
awk -v l="$load" 'BEGIN{ if (l+0 > 1.5) print "WARNING: load average " l " is high; latency numbers will be noisy. Stop other jobs and rerun with FORCE_E6=1." }'
for pair in $($PLAN apps --set "$SET" --inputs "$INPUTS" --traces "$TR" | sed 's/--app //g'); do
  w=${pair%%=*}; f=${pair#*=}; label="$(basename "$f" .llc.trace)"
  pol="lru"
  for g in mlp gbdt lstm; do pol="$pol,$($PLAN policies --set "$SET" --workload "$w" --group "$g" | cut -d, -f3-4)"; done
  big="$($PLAN policies --set "$SET" --workload "$w" --group mlp | cut -d, -f5)"; case "$big" in mlp:*allbig*) pol="$pol,$big";; esac
  out="$FIN/e6/$label.result.json"; [ "${FORCE_E6:-0}" = 1 ] && rm -f "$out"
  bash "$ROOT/scripts/final_job.sh" "$out" "$CFG" "$f" "$pol" --max-accesses "${E6_ACC:-600000}"
done
