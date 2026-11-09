#!/usr/bin/env bash
# One single-application job: cachesim run -> <out>.  Resumable (skips an existing result), atomic (writes <out>.part first).
# usage: scripts/final_job.sh OUT CONFIG TRACE POLICIES [extra cachesim args...]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$1"; CFG="$2"; TRACE="$3"; POL="$4"; shift 4
[ -s "$OUT" ] && { echo "skip   $(basename "$OUT") (done)"; exit 0; }
mkdir -p "$(dirname "$OUT")"
if "$ROOT/build/cachesim" run --config "$CFG" --trace "$TRACE" --policies "$POL" --out "$OUT.part" "$@" > "${OUT%.json}.txt" 2>&1; then
  mv "$OUT.part" "$OUT"; echo "done   $(basename "$OUT")"
else
  echo "FAILED $(basename "$OUT") (see ${OUT%.json}.txt)"; exit 1
fi
