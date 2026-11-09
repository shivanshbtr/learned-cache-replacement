#!/usr/bin/env bash
# Evaluate the mixed-stream DAgger models (extension) on a shared LLC, next to the frozen models and the baselines.
#   scripts/mixed_eval.sh val     one mixture of the VAL inputs                    -> results/ext/val/       (look, choose; may be repeated)
#   scripts/mixed_eval.sh test    TESTSET=1 (default): the E8 test mixture (first test input of each workload) -> results/ext/test/   (marker test_started)
#                                 TESTSET=2: the SECOND test mixture (second test input of each workload)    -> results/ext/test2/  (marker test2_started)
#   Each test set can be run ONCE (the script refuses a second run). Plan: all new candidates (LightGBM, LSTM, mix2) go to test set 2 together with the MLP versions,
#   so one table shows every version on inputs that no earlier choice has seen.
# env: MIXVERS="mix1 mix1b mix2" (versions that do not exist for a family are skipped)  FAMILIES="mlp gbdt lstm"  CFGS="default llc8m_16w"  SLICE=10000
#      INPUTS TR CFG FIN MX OUT       The frozen LSTM in the tables is the extension model _dag1 (the frozen set holds the base LSTM, which is weaker).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPLIT="${1:?usage: mixed_eval.sh val|test}"
TR="${TR:-$ROOT/results/traces}"; CFG="${CFG:-$ROOT/configs/default.cfg}"; FIN="${FIN:-$ROOT/results/final}"; INPUTS="${INPUTS:-$ROOT/configs/inputs.tsv}"; MX="${MX:-$ROOT/models_ext}"; OUT="${OUT:-$ROOT/results/ext}"
SET="$FIN/model_set.json"; SLICE="${SLICE:-10000}"; TESTSET="${TESTSET:-1}"; MIXVERS="${MIXVERS:-mix1}"; VERS="$(echo $MIXVERS | tr ' ' ',')"
PLAN="python3 $ROOT/scripts/mixed_plan.py"
[ -f "$SET" ] || { echo "no $SET"; exit 1; }
DIR="$SPLIT"; [ "$SPLIT" = test ] && [ "$TESTSET" = 2 ] && DIR=test2
if [ "$SPLIT" = test ]; then
  MARK="$OUT/test_started"; [ "$TESTSET" = 2 ] && MARK="$OUT/test2_started"
  [ -e "$MARK" ] && { echo "this extension test run was already done ($MARK). Results are in $OUT/$DIR. Not running again."; exit 1; }
  if [ "$TESTSET" = 2 ]; then APPS="$($PLAN testmix --n 2 --inputs "$INPUTS" --traces "$TR")" || exit 1
  else APPS="$(python3 "$ROOT/scripts/final_plan.py" apps --set "$SET" --inputs "$INPUTS" --traces "$TR")"; fi
  mkdir -p "$OUT/$DIR"; date -u +%FT%TZ > "$MARK"
  python3 - "$MX" "$MARK" <<'PY'
import hashlib, glob, sys
with open(sys.argv[2], "a") as f:
    for p in sorted(glob.glob(sys.argv[1] + "/*_mix[12]*.*")):
        if not p.endswith((".json", ".log")): f.write("%s %s\n" % (hashlib.sha256(open(p, "rb").read()).hexdigest()[:16], p))
PY
elif [ "$SPLIT" = val ]; then
  APPS="$($PLAN mixtures --split val --inputs "$INPUTS" --traces "$TR" | head -1 | cut -d' ' -f2-)"
else echo "split must be val or test"; exit 1; fi
mkdir -p "$OUT/$DIR"
BASEP="lru,srrip,drrip,bip,mru,belady"
for c in ${CFGS:-default llc8m_16w}; do
  cf="$ROOT/configs/$c.cfg"; [ "$c" = default ] && cf="$CFG"
  POL="$BASEP"; ARGS=""
  for f in ${FAMILIES:-mlp}; do
    FS=""; [ "$f" = lstm ] && FS="--frozen-sfx _dag1"
    ARGS="$ARGS $($PLAN multi-args --set "$SET" --family "$f" --models "$MX" --versions "$VERS" $FS)" || exit 1
    POL="$POL,$($PLAN pols --set "$SET" --family "$f" --models "$MX" --versions "$VERS")"
  done
  # shellcheck disable=SC2086
  "$ROOT/build/multisim" run --config "$cf" --slice-instr "$SLICE" --policies "$POL" $ARGS $APPS --until first --alone 1 --out "$OUT/$DIR/mixed_${c}_s$SLICE.json" | tee "$OUT/$DIR/mixed_${c}_s$SLICE.txt"
done
python3 "$ROOT/scripts/mixed_report.py" "$OUT/$DIR"
