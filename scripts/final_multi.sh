#!/usr/bin/env bash
# E7 (switching A->B->C->D->A... with several time-slice lengths) and E8 (four applications sharing the LLC, arbitration rules) on the TEST split.
# Applications: the first test input of each workload in configs/inputs.tsv. Models: the frozen set. Runs stop when the first application has finished.
# usage: scripts/final_multi.sh      env: JOBS (default 3) TR FIN CFG INPUTS  E7_DIV="1000 100 10 4" (slice = shortest trace / each number)
#        E8_SLICES="10000" (instructions)  E8_CFGS="default llc8m_16w"  INCLUDE_LSTM_MULTI=0
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TR="${TR:-$ROOT/results/traces}"; FIN="${FIN:-$ROOT/results/final}"; CFG="${CFG:-$ROOT/configs/default.cfg}"; INPUTS="${INPUTS:-$ROOT/configs/inputs.tsv}"
SET="$FIN/model_set.json"; PLAN="python3 $ROOT/scripts/final_plan.py"
[ -f "$SET" ] || { echo "no $SET: run scripts/run_final.sh first"; exit 1; }
APPS="$($PLAN apps --set "$SET" --inputs "$INPUTS" --traces "$TR")"
for t in $(echo "$APPS" | tr ' ' '\n' | grep '=' | sed 's/.*=//'); do [ -f "$t" ] || { echo "missing trace $t"; exit 1; }; done
IMIN="$("$ROOT/build/multisim" info $(for t in $(echo "$APPS" | tr ' ' '\n' | grep '=' | sed 's/.*=//'); do echo "--trace $t"; done) | sed 's/.*total_icount=//' | sort -n | head -1)"
echo "applications: $APPS"; echo "shortest trace: $IMIN instructions"
MLP="$($PLAN multi-args --set "$SET" --family mlp)"; GB="$($PLAN multi-args --set "$SET" --family gbdt)"; LS="$($PLAN multi-args --set "$SET" --family lstm)"
BIG=""; case "$MLP" in *allbig_mlp*) BIG=",gen:allbig_mlp";; esac
BASEP="lru,srrip,drrip,bip,mru,belady"
JL="$(mktemp)"
add() {  # out cfg slice policies setargs extra
  # no trailing blank: xargs -L 1 would join a line that ends in a space with the next line
  echo "$1 --config $2 --slice-instr $3 --policies $4 $5 $APPS --until first --alone 1${6:+ $6}" >> "$JL"
}
for d in ${E7_DIV:-1000 100 10 4}; do
  S=$(( IMIN / d )); [ "$S" -ge 1 ] || S=1
  add "$FIN/multi/e7_s$S.base.json" "$CFG" "$S" "$BASEP" "" "--slices 1"
  add "$FIN/multi/e7_s$S.mlp.json" "$CFG" "$S" "req:spec_mlp,gen:all_mlp$BIG" "$MLP" "--slices 1"
  add "$FIN/multi/e7_s$S.gbdt.json" "$CFG" "$S" "req:spec_gbdt,gen:all_gbdt" "$GB" "--slices 1"
  [ "${INCLUDE_LSTM_MULTI:-0}" = 1 ] && add "$FIN/multi/e7_s$S.lstm.json" "$CFG" "$S" "req:spec_lstm,gen:all_lstm" "$LS" "--slices 1"
done
for c in ${E8_CFGS:-default llc8m_16w}; do
  cf="$ROOT/configs/$c.cfg"; [ "$c" = default ] && cf="$CFG"
  for S in ${E8_SLICES:-10000}; do
    add "$FIN/multi/e8_${c}_s$S.base.json" "$cf" "$S" "$BASEP" "" ""
    add "$FIN/multi/e8_${c}_s$S.mlp.json" "$cf" "$S" "req:spec_mlp,own:spec_mlp,gen:all_mlp$BIG" "$MLP" ""
    add "$FIN/multi/e8_${c}_s$S.gbdt.json" "$cf" "$S" "req:spec_gbdt,own:spec_gbdt,gen:all_gbdt" "$GB" ""
    [ "${INCLUDE_LSTM_MULTI:-0}" = 1 ] && add "$FIN/multi/e8_${c}_s$S.lstm.json" "$cf" "$S" "req:spec_lstm,own:spec_lstm,gen:all_lstm" "$LS" ""
  done
done
echo "E7/E8: $(wc -l < "$JL") jobs, ${JOBS:-3} in parallel"
xargs -P "${JOBS:-3}" -L 1 bash "$ROOT/scripts/final_multi_job.sh" < "$JL"
rm -f "$JL"
