#!/usr/bin/env bash
# Run baselines AND the trained models in the simulator on every trace of a split (the real metric: LLC misses).
# usage: scripts/run_models.sh [split=val|test|train] [traces] [models_dir] [outroot]
#   defaults: val  results/traces  models  results/sim
# env: JOBS (default 3)   CONFIG (default configs/default.cfg)   SUMMARY_ARGS (e.g. --wide)
# Policies per trace: lru fifo random srrip drrip bip mru belady + EVERY model file of that workload and of the generic model:
#   <models_dir>/<w>_*.{mlp,gbdt,lstm} and <models_dir>/all_*.{mlp,gbdt,lstm}   (so DAgger variants such as hash_mlp_dag1.mlp appear automatically)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPLIT="${1:-val}"; TR="${2:-$ROOT/results/traces}"; MD="${3:-$ROOT/models}"; OUT="${4:-$ROOT/results/sim}"; CFG="${CONFIG:-$ROOT/configs/default.cfg}"
mkdir -p "$OUT/$SPLIT"
export ROOT CFG MD OUT SPLIT
ls "$TR/$SPLIT"/*.llc.trace 2>/dev/null | xargs -P "${JOBS:-3}" -I{} bash -c '
  shopt -s nullglob
  f="{}"; label=$(basename "$f" .llc.trace); w=${label#*__}; w=${w%%_*}
  pol="lru,fifo,random,srrip,drrip,bip,mru"
  for m in "$MD/${w}"_*.mlp "$MD/${w}"_*.gbdt "$MD/${w}"_*.lstm "$MD"/all_*.mlp "$MD"/all_*.gbdt "$MD"/all_*.lstm; do
    case "$m" in *.mlp) pol="$pol,mlp:$m";; *.gbdt) pol="$pol,gbdt:$m";; *.lstm) pol="$pol,lstm:$m";; esac
  done
  "$ROOT/build/cachesim" run --config "$CFG" --trace "$f" --policies "$pol,belady" --out "$OUT/$SPLIT/$label.result.json" > "$OUT/$SPLIT/$label.txt" 2>&1 && echo "done $label" || echo "FAILED $label (see $OUT/$SPLIT/$label.txt)"
'
python3 "$ROOT/scripts/summarize.py" "$OUT/$SPLIT" ${SUMMARY_ARGS:-}
