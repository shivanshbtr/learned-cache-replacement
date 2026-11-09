#!/usr/bin/env bash
# One multi-application job: multisim run -> <out>.  Resumable and atomic like final_job.sh.
# usage: scripts/final_multi_job.sh OUT multisim-args...      (the --out argument is added here)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$1"; shift
[ -s "$OUT" ] && { echo "skip   $(basename "$OUT") (done)"; exit 0; }
mkdir -p "$(dirname "$OUT")"
if "$ROOT/build/multisim" run "$@" --out "$OUT.part" > "${OUT%.json}.txt" 2>&1; then
  mv "$OUT.part" "$OUT"; echo "done   $(basename "$OUT")"
else
  echo "FAILED $(basename "$OUT") (see ${OUT%.json}.txt)"; exit 1
fi
