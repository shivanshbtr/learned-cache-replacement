#!/usr/bin/env bash
# E1 to E5 on the TEST split: every test trace is run with the baselines and the frozen final models (results/final/model_set.json).
#   groups per trace: base (baselines), mlp (own + generic + equal-storage generic + cross-workload models + extra seeds), gbdt, lstm
#   E5 (config sensitivity): the same traces with other LLC sizes/associativity, baselines + specialized and generic MLP (no retraining)
# usage: scripts/final_e2.sh        env: JOBS (default 4) TR FIN CFG SKIP_E5=1 E5_CFGS="llc1m_16w llc4m_16w llc2m_8w" INCLUDE_LSTM=0
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TR="${TR:-$ROOT/results/traces}"; FIN="${FIN:-$ROOT/results/final}"; CFG="${CFG:-$ROOT/configs/default.cfg}"
SET="$FIN/model_set.json"; PLAN="python3 $ROOT/scripts/final_plan.py"
[ -f "$SET" ] || { echo "no $SET: run scripts/run_final.sh (or final_plan.py select) first"; exit 1; }
JL="$(mktemp)"
GROUPS_E2="base mlp gbdt"; [ "${INCLUDE_LSTM:-1}" = 1 ] && GROUPS_E2="$GROUPS_E2 lstm"
for f in "$TR"/test/*.llc.trace; do
  [ -f "$f" ] || continue
  label="$(basename "$f" .llc.trace)"; w=${label#*__}; w=${w%%_*}
  for g in $GROUPS_E2; do
    echo "$FIN/e2/$label.$g.result.json $CFG $f $($PLAN policies --set "$SET" --workload "$w" --group "$g")" >> "$JL"
  done
  if [ "${SKIP_E5:-0}" != 1 ]; then
    for c in ${E5_CFGS:-llc1m_16w llc4m_16w llc2m_8w}; do
      echo "$FIN/e5/$c/$label.core.result.json $ROOT/configs/$c.cfg $f $($PLAN policies --set "$SET" --workload "$w" --group core)" >> "$JL"
    done
  fi
done
echo "E1-E5 (test split): $(wc -l < "$JL") jobs, ${JOBS:-4} in parallel"
xargs -P "${JOBS:-4}" -L 1 bash "$ROOT/scripts/final_job.sh" < "$JL"
rm -f "$JL"
