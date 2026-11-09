#!/usr/bin/env bash
# Extension: the LSTM trained with one DAgger round (_dag1) on the TEST split, ONCE, next to the base LSTM and the frozen MLP.
# Chosen on val only (mean gap closed of the specialized LSTMs: base vs _dag1, see results/sim/val). The frozen model set is NOT changed.
# usage: scripts/ext_lstm_test.sh        env: JOBS (3) TR CFG MD FIN OUT
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TR="${TR:-$ROOT/results/traces}"; CFG="${CFG:-$ROOT/configs/default.cfg}"; MD="${MD:-$ROOT/models}"; FIN="${FIN:-$ROOT/results/final}"; OUT="${OUT:-$ROOT/results/ext/lstm_test}"
[ -e "$OUT/test_started" ] && { echo "already done ($OUT/test_started); results are in $OUT/test. Not running again."; exit 1; }
for w in hash graph matrix conv all; do [ -f "$MD/${w}_lstm_dag1.lstm" ] || { echo "missing $MD/${w}_lstm_dag1.lstm"; exit 1; }; done
mkdir -p "$OUT/test"; date -u +%FT%TZ > "$OUT/test_started"
sha256sum "$MD"/*_lstm_dag1.lstm >> "$OUT/test_started"
export ROOT CFG MD FIN OUT
ls "$TR"/test/*.llc.trace | xargs -P "${JOBS:-3}" -I{} bash -c '
  f="{}"; label=$(basename "$f" .llc.trace); w=${label#*__}; w=${w%%_*}
  core="$(python3 "$ROOT/scripts/final_plan.py" policies --set "$FIN/model_set.json" --workload "$w" --group core)"
  pol="$core,lstm:$MD/${w}_lstm.lstm,lstm:$MD/${w}_lstm_dag1.lstm,lstm:$MD/all_lstm.lstm,lstm:$MD/all_lstm_dag1.lstm"
  "$ROOT/build/cachesim" run --config "$CFG" --trace "$f" --policies "$pol" --out "$OUT/test/$label.result.json" > "$OUT/test/$label.txt" 2>&1 && echo "done $label" || echo "FAILED $label"
'
python3 "$ROOT/scripts/summarize.py" "$OUT/test"
