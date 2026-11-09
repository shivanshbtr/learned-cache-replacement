#!/usr/bin/env bash
# Extra models for E3, trained on the SAME labeled data as the final MLPs (no new DAgger rounds):
#   allbig_mlp<sfx>  generic MLP whose file size equals the sum of the four specialized MLP files (equal total storage)
#   extra training seeds (2 and 3) of every final MLP (specialized, generic, big generic), into models_seeds/  -> mean and std over 3 seeds in E3
# usage: scripts/train_final_extras.sh       env: JOBS (default 3)  THREADS (default 2)  DS MD SEEDMD FIN
# Needs results/final/model_set.json (python3 scripts/final_plan.py select first: it freezes nothing, it only reads validation results).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DS="${DS:-$ROOT/results/datasets}"; MD="${MD:-$ROOT/models}"; SEEDMD="${SEEDMD:-$ROOT/models_seeds}"; FIN="${FIN:-$ROOT/results/final}"
PLAN="python3 $ROOT/scripts/final_plan.py"; SET="$FIN/model_set.json"
[ -f "$SET" ] || { echo "run: python3 scripts/final_plan.py select"; exit 1; }
SFX="$($PLAN variant --set "$SET")"; DSPEC="$($PLAN drivers --set "$SET" --who spec)"; DALL="$($PLAN drivers --set "$SET" --who all)"
H="$($PLAN bighidden --set "$SET" --models "$MD" --sfx "$SFX")"
echo "MLP variant '$SFX'; spec drivers $DSPEC; generic drivers $DALL; big generic hidden $H"
mkdir -p "$SEEDMD"
JL="$(mktemp)"
TH="--threads ${THREADS:-2}"
[ -f "$MD/allbig_mlp$SFX.mlp" ] || echo "allbig 1 $DALL $H $MD/allbig_mlp$SFX.mlp" >> "$JL"
for s in 2 3; do
  for w in hash graph matrix conv; do [ -f "$SEEDMD/${w}_mlp${SFX}_s$s.mlp" ] || echo "$w $s $DSPEC 32,32 $SEEDMD/${w}_mlp${SFX}_s$s.mlp" >> "$JL"; done
  [ -f "$SEEDMD/all_mlp${SFX}_s$s.mlp" ] || echo "all $s $DALL 32,32 $SEEDMD/all_mlp${SFX}_s$s.mlp" >> "$JL"
  [ -f "$SEEDMD/allbig_mlp${SFX}_s$s.mlp" ] || echo "all $s $DALL $H $SEEDMD/allbig_mlp${SFX}_s$s.mlp" >> "$JL"
done
echo "$(wc -l < "$JL") training jobs, ${JOBS:-3} in parallel"
export ROOT DS TH
xargs -P "${JOBS:-3}" -L 1 bash -c '
  w=$0; seed=$1; drv=$2; hid=$3; out=$4
  [ "$w" = allbig ] && w=all
  python3 "$ROOT/training/train_mlp.py" --data "$DS" --workload "$w" --drivers "$drv" --hidden "$hid" --seed "$seed" $TH --out "$out" > "$out.log" 2>&1 &&
  { echo "done   $(basename "$out")"; tail -1 "$out.log"; } || echo "FAILED $(basename "$out") (see $out.log)"
' < "$JL"
rm -f "$JL"
echo "Next: python3 scripts/final_plan.py select   (adds the new files to the model set), then scripts/run_final.sh"
