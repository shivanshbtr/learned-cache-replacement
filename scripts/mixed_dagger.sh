#!/usr/bin/env bash
# Mixed-stream DAgger (extension, docs/mixed_dagger.md). Models drive the cache on MERGED multi-application streams (hash+graph+matrix+conv, round-robin by
# instructions, the E8 setup but built from TRAIN and VAL inputs only, never test), the visited states are labeled with Belady on the merged stream, and new
# models are trained on solo data + these merged states.
#   labeling drivers:  req=  (each requester's own model chooses among all lines)   own=  (ownership rule)   gen=  (generic model)   lru   belady (EXPERT=1)
#   dataset tags:      mix<mode><round><kind>   e.g. mixreq1mlp mixown1lgbm mixgen2lstm mixlru1mlp mixbel1mlp     files: <DS>/<split>/<split>__mix<k>s<slice>.<tag>.npz
#   new models:        <MX>/<w>_<kind>_<version>.<ext> and <MX>/all_<kind>_<version>.<ext>      (MX = models_ext; the main models/ folder is untouched)
# usage: scripts/mixed_dagger.sh KIND      KIND = mlp | gbdt | lstm         Resumable: finished mixtures, datasets and models are skipped.
#   ROUND=1 (default): the frozen models drive (LSTM: the _dag1 models, BASE_SFX).  EXPERT=1 also records the merged streams with LRU and Belady driving
#           (tags mixlru1<kind>, mixbel1<kind>) and trains version mix1b (without EXPERT: version mix1).
#   ROUND=2: the models of version PREV (default mix1b) drive; new datasets mixreq2/mixown2/mixgen2 (+ the round-1 expert data); trains version mix2.
# env: ROUND (1|2)  EXPERT (0/1)  PREV (mix1b)  BASE_SFX (lstm: _dag1)  TR DS MX CFG FIN INPUTS  JOBS (labeling jobs in parallel, default 2)
#      SLICES (instructions per time slice, default "10000")  STRIDE (2)  EXTRA_TRAIN (extra args for the training script)  STAGES="merge label train"
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KIND="${1:?usage: mixed_dagger.sh mlp|gbdt|lstm}"
TR="${TR:-$ROOT/results/traces}"; DS="${DS:-$ROOT/results/datasets}"; MX="${MX:-$ROOT/models_ext}"; CFG="${CFG:-$ROOT/configs/default.cfg}"; FIN="${FIN:-$ROOT/results/final}"; INPUTS="${INPUTS:-$ROOT/configs/inputs.tsv}"
SET="$FIN/model_set.json"; PLAN="python3 $ROOT/scripts/mixed_plan.py"; STAGES="${STAGES:-merge label train}"; ROUND="${ROUND:-1}"; PREV="${PREV:-mix1b}"
case "$KIND" in mlp) TAG=mlp; EXT=mlp; TRAIN=train_mlp.py; BSFX="${BASE_SFX:-}";; gbdt) TAG=lgbm; EXT=gbdt; TRAIN=train_lgbm.py; BSFX="${BASE_SFX:-}";;
  lstm) TAG=lstm; EXT=lstm; TRAIN=train_lstm.py; BSFX="${BASE_SFX-_dag1}";; *) echo "KIND must be mlp, gbdt or lstm"; exit 1;; esac
[ "$ROUND" = 1 ] || [ "$ROUND" = 2 ] || { echo "ROUND must be 1 or 2"; exit 1; }
[ -f "$SET" ] || { echo "no $SET (the model set must be frozen first)"; exit 1; }
[ -x "$ROOT/build/multisim" ] && [ -x "$ROOT/build/labeler" ] || { echo "Run 'make all' first."; exit 1; }
# who drives in this round
if [ "$ROUND" = 1 ]; then DARGS="--family $KIND ${BSFX:+--sfx $BSFX}"; else DARGS="--family $KIND --ver $PREV --models $MX"; fi
MER="$DS/_merged"; mkdir -p "$MER" "$MX"
export ROOT CFG DS MER STRIDE="${STRIDE:-2}"

if [[ " $STAGES " == *" merge "* ]]; then
  : > "$MER/jobs.txt"
  for split in train val; do
    for S in ${SLICES:-10000}; do
      $PLAN mixtures --split "$split" --suffix "s$S" --inputs "$INPUTS" --traces "$TR" | while read -r name rest; do echo "$MER/${split}__$name $S $CFG $rest" >> "$MER/jobs.txt"; done
    done
  done
  echo "merge: $(wc -l < "$MER/jobs.txt") mixtures"
  while read -r base S cfg rest; do
    [ -s "$base.meta.json" ] && continue
    # shellcheck disable=SC2086
    "$ROOT/build/multisim" merge --config "$cfg" $rest --slice-instr "$S" --until first --out "$base.llc.trace" --meta "$base.meta.json" > "$base.merge.log" 2>&1 \
      && echo "merged $(basename "$base"): $(cat "$base.merge.log")" || { echo "FAILED merge $(basename "$base") (see $base.merge.log)"; rm -f "$base.meta.json"; }
  done < "$MER/jobs.txt"
fi

if [[ " $STAGES " == *" label "* ]]; then
  LIST="$(mktemp)"
  MODES="req own gen"; [ "$ROUND" = 1 ] && [ "${EXPERT:-0}" = 1 ] && MODES="$MODES lru bel"
  for mode in $MODES; do
    case "$mode" in lru) DRV=lru;; bel) DRV=belady;; *) DRV="$($PLAN driver $DARGS --set "$SET" --mode "$mode")" || exit 1;; esac
    t="mix${mode}${ROUND}${TAG}"; [ "$mode" = bel ] && t="mixbel${ROUND}${TAG}"; [ "$mode" = lru ] && t="mixlru${ROUND}${TAG}"
    for meta in "$MER"/*.meta.json; do
      [ -f "$meta" ] || continue
      base="${meta%.meta.json}"; lab="$(basename "$base")"; split="${lab%%__*}"
      echo "$base $split $lab $t $DRV" >> "$LIST"
    done
  done
  echo "label: $(wc -l < "$LIST") jobs, ${JOBS:-2} in parallel"
  xargs -P "${JOBS:-2}" -L 1 bash -c '
    base=$0; split=$1; lab=$2; tag=$3; drv=$4
    d="$DS/$split"; mkdir -p "$d"; npz="$d/$lab.$tag.npz"
    [ -f "$npz" ] && { echo "skip $lab.$tag (done)"; exit 0; }
    "$ROOT/build/labeler" --config "$CFG" --trace "$base.llc.trace" --driver "$drv" --stride "$STRIDE" --out "$d/$lab.$tag.dec" --stats "$d/$lab.$tag.labeler.json" > "$d/$lab.$tag.log" 2>&1 &&
    python3 "$ROOT/training/export_npz.py" --trace "$base.llc.trace" --dec "$d/$lab.$tag.dec" --out "$npz" --meta-json "$base.meta.json" --delete-dec >> "$d/$lab.$tag.log" 2>&1 &&
    echo "done  $lab.$tag" || { echo "FAILED $lab.$tag (see $d/$lab.$tag.log)"; rm -f "$d/$lab.$tag.dec"; }
  ' < "$LIST"
  rm -f "$LIST"
  echo "(merged traces kept in $MER; delete them with: rm -rf $MER  when all KINDs and rounds are labeled)"
fi

if [[ " $STAGES " == *" train "* ]]; then
  # round-1 mixed tags that every later version also uses
  R1="mixreq1${TAG},mixown1${TAG}"; VER=mix1
  if [ "$ROUND" = 1 ]; then
    [ "${EXPERT:-0}" = 1 ] && { R1="$R1,mixlru1${TAG},mixbel1${TAG}"; VER=mix1b; }
    MIXT="$R1"; MIXG="$R1,mixgen1${TAG}"
  else
    VER=mix2; MIXT="$R1,mixlru1${TAG},mixbel1${TAG},mixreq2${TAG},mixown2${TAG}"; MIXG="$R1,mixlru1${TAG},mixbel1${TAG},mixgen1${TAG},mixreq2${TAG},mixown2${TAG},mixgen2${TAG}"
  fi
  for w in hash graph matrix conv all; do
    drivers="$($PLAN drivers --set "$SET" --family "$KIND" --who spec --sfx "$BSFX"),$MIXT"
    [ "$w" = all ] && drivers="$($PLAN drivers --set "$SET" --family "$KIND" --who all --sfx "$BSFX"),$MIXG"
    out="$MX/${w}_${TAG}_${VER}.$EXT"
    [ -f "$out" ] && { echo "skip $out (done)"; continue; }
    echo "=== train $w ($KIND, $VER) on drivers $drivers" | tee -a "$MX/mixed_dagger.log"
    python3 "$ROOT/training/$TRAIN" --data "$DS" --workload "$w" --drivers "$drivers" --out "$out" ${EXTRA_TRAIN:-} 2>&1 | tee -a "$MX/mixed_dagger.log" | tail -2
  done
fi
