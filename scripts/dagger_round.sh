#!/usr/bin/env bash
# One DAgger round: let the CURRENT model drive the cache on the train and val traces (never test), label the states it visits with
# Belady, add them to the training data, retrain, and compare in the simulator.
# usage: scripts/dagger_round.sh ROUND KIND        ROUND = 1, 2, ...   KIND = mlp | lgbm | lstm
#   Round 1 starts from models/<w>_<kind> and models/all_<kind>; round n starts from the *_dag<n-1> models.
#   Two kinds of visited states are collected per trace: driven by the specialized model of that trace's workload (tag dag<R><kind>) and
#   driven by the generic model (tag dag<R>all<kind>). Specialized models retrain on lru,belady + all specialized dag tags;
#   the generic model on lru,belady + all dag tags.
#   New models: models/<w>_<kind>_dag<R>.<ext> and models/all_<kind>_dag<R>.<ext>.  Then run_models.sh val is run (RUN_EVAL=0 to skip).
# env (all optional): TR (traces) DS (datasets) MD (models) CFG (config) SIM (sim results) WORKLOADS (default "hash graph matrix conv")
#   JOBS (parallel labeling jobs, default 3)  STRIDE (default 2)  EXTRA_TRAIN (extra args for the training script)  RUN_EVAL (default 1)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
R="${1:?usage: dagger_round.sh ROUND KIND}"; KIND="${2:?usage: dagger_round.sh ROUND KIND (mlp|lgbm|lstm)}"
TR="${TR:-$ROOT/results/traces}"; DS="${DS:-$ROOT/results/datasets}"; MD="${MD:-$ROOT/models}"; CFG="${CFG:-$ROOT/configs/default.cfg}"; SIM="${SIM:-$ROOT/results/sim}"
case "$KIND" in
  mlp)  EXT=mlp;  DRV=mlp;  TRAIN=train_mlp.py;;
  lgbm) EXT=gbdt; DRV=gbdt; TRAIN=train_lgbm.py;;
  lstm) EXT=lstm; DRV=lstm; TRAIN=train_lstm.py;;
  *) echo "KIND must be mlp, lgbm or lstm"; exit 1;;
esac
WL="${WORKLOADS:-hash graph matrix conv}"
PREV=$((R - 1))
base() { if [ "$PREV" -eq 0 ]; then echo "$MD/$1_$KIND.$EXT"; else echo "$MD/$1_${KIND}_dag$PREV.$EXT"; fi; }
[ -x "$ROOT/build/labeler" ] || { echo "Run 'make all' first."; exit 1; }
LIST="$(mktemp)"
for f in "$TR"/train/*.llc.trace "$TR"/val/*.llc.trace; do
  [ -f "$f" ] || continue
  split="$(basename "$(dirname "$f")")"; label="$(basename "$f" .llc.trace)"; w=${label#*__}; w=${w%%_*}
  case " $WL " in *" $w "*) ;; *) continue;; esac
  echo "$f $split $label dag$R$KIND $(base "$w")" >> "$LIST"
  echo "$f $split $label dag${R}all$KIND $(base all)" >> "$LIST"
done
echo "DAgger round $R ($KIND): $(wc -l < "$LIST") labeling jobs, ${JOBS:-3} in parallel"
export ROOT CFG DS DRV STRIDE="${STRIDE:-2}"
xargs -P "${JOBS:-3}" -L 1 bash -c '
  f=$0; split=$1; label=$2; tag=$3; model=$4
  [ -f "$model" ] || { echo "skip $label.$tag (no model $model)"; exit 0; }
  d="$DS/$split"; mkdir -p "$d"; npz="$d/$label.$tag.npz"
  [ -f "$npz" ] && { echo "skip $label.$tag (done)"; exit 0; }
  "$ROOT/build/labeler" --config "$CFG" --trace "$f" --driver "$DRV:$model" --stride "$STRIDE" --out "$d/$label.$tag.dec" --stats "$d/$label.$tag.labeler.json" > "$d/$label.$tag.log" 2>&1 &&
  python3 "$ROOT/training/export_npz.py" --trace "$f" --dec "$d/$label.$tag.dec" --out "$npz" --delete-dec >> "$d/$label.$tag.log" 2>&1 &&
  echo "done  $label.$tag" || echo "FAILED $label.$tag (see $d/$label.$tag.log)"
' < "$LIST"
rm -f "$LIST"
SPEC="lru,belady"; ALL="lru,belady"
for r in $(seq 1 "$R"); do SPEC="$SPEC,dag$r$KIND"; ALL="$ALL,dag$r$KIND,dag${r}all$KIND"; done
mkdir -p "$MD"
for w in $WL all; do
  drivers="$SPEC"; [ "$w" = all ] && drivers="$ALL"
  echo "=== retrain $w ($KIND) on drivers $drivers" | tee -a "$MD/dagger.log"
  python3 "$ROOT/training/$TRAIN" --data "$DS" --workload "$w" --drivers "$drivers" --out "$MD/${w}_${KIND}_dag$R.$EXT" ${EXTRA_TRAIN:-} 2>&1 | tee -a "$MD/dagger.log" | tail -2
done
if [ "${RUN_EVAL:-1}" = 1 ]; then
  CONFIG="$CFG" "$ROOT/scripts/run_models.sh" val "$TR" "$MD" "$SIM"
fi
