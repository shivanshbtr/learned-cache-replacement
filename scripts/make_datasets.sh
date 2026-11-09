#!/usr/bin/env bash
# Belady-label every LLC trace under <traceroot> and pack each into a .npz dataset file.
# usage: scripts/make_datasets.sh [traceroot] [config] [outroot]
#   defaults: results/traces  configs/default.cfg  results/datasets
# env: DRIVERS (default "lru,belady": which policy drives the cache states that get labeled)
#      STRIDE  (default 2: keep every 2nd decision)   JOBS (default 4)   KEEP_DEC=1 keeps the raw .dec files
# Output: <outroot>/<split>/<label>.<driver>.npz   (resumable: existing .npz files are skipped)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TR="${1:-$ROOT/results/traces}"; CFG="${2:-$ROOT/configs/default.cfg}"; OUT="${3:-$ROOT/results/datasets}"
[ -x "$ROOT/build/labeler" ] || { echo "Run 'make all' first."; exit 1; }
python3 -c "import numpy" 2>/dev/null || { echo "numpy missing: pip install numpy"; exit 1; }
MIN_FREE_GB="${MIN_FREE_GB:-20}" "$ROOT/scripts/check_disk.sh" "$OUT" 2>/dev/null || { mkdir -p "$OUT"; "$ROOT/scripts/check_disk.sh" "$OUT" || exit 1; }
LIST="$(mktemp)"
for f in "$TR"/*/*.llc.trace; do
  [ -f "$f" ] || continue
  split="$(basename "$(dirname "$f")")"; label="$(basename "$f" .llc.trace)"
  IFS=',' read -ra DL <<< "${DRIVERS:-lru,belady}"
  for drv in "${DL[@]}"; do echo "$f $split $label $drv" >> "$LIST"; done
done
echo "$(wc -l < "$LIST") labeling jobs, ${JOBS:-4} in parallel"
export ROOT CFG OUT STRIDE="${STRIDE:-2}" KEEP_DEC="${KEEP_DEC:-0}"
xargs -P "${JOBS:-4}" -L 1 bash -c '
  f=$0; split=$1; label=$2; drv=$3
  d="$OUT/$split"; mkdir -p "$d"; npz="$d/$label.$drv.npz"
  [ -f "$npz" ] && { echo "skip $label.$drv (done)"; exit 0; }
  "$ROOT/build/labeler" --config "$CFG" --trace "$f" --driver "$drv" --stride "$STRIDE" --out "$d/$label.$drv.dec" --stats "$d/$label.$drv.labeler.json" > "$d/$label.$drv.log" 2>&1 &&
  python3 "$ROOT/training/export_npz.py" --trace "$f" --dec "$d/$label.$drv.dec" --out "$npz" $( [ "$KEEP_DEC" = 1 ] || echo --delete-dec ) >> "$d/$label.$drv.log" 2>&1 &&
  echo "done  $label.$drv" || echo "FAILED $label.$drv (see $d/$label.$drv.log)"
' < "$LIST"
rm -f "$LIST"
python3 "$ROOT/training/dataset.py" $(find "$OUT" -name '*.npz' | sort | head -60) 2>&1 | head -400
