#!/usr/bin/env bash
# Build a clean, small copy of the project for GitHub. Does NOT touch the working folder.
# usage: bash scripts/package_repo.sh [SRC] [DST]
#   SRC default: the folder that holds this script's parent (the project)   DST default: ~/appcache_repo
# Copies: code, configs, tests, docs, scripts, web simulation, frozen models, SMALL result tables, run logs.
# Leaves out: build/, results/traces, results/datasets (use scripts/pack_data.sh for the traces).
set -euo pipefail
SRC="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
DST="${2:-$HOME/appcache_repo}"
[ -d "$SRC/simulator" ] || { echo "ERROR: $SRC does not look like the project (no simulator/)"; exit 1; }
if [ -e "$DST" ] && [ -n "$(ls -A "$DST" 2>/dev/null)" ]; then echo "ERROR: $DST exists and is not empty. Choose another folder or remove it yourself."; exit 1; fi
mkdir -p "$DST"
echo "copy from $SRC to $DST"
for d in applications belady configs docs export scripts simulator tests trace_collection training web_simulation models models_ext models_seeds report; do
  if [ -d "$SRC/$d" ]; then mkdir -p "$DST/$d"; cp -a "$SRC/$d/." "$DST/$d/"; else echo "  (skipped, not found: $d)"; fi
done
for f in Makefile README.md requirements.txt .gitignore; do [ -f "$SRC/$f" ] && cp -a "$SRC/$f" "$DST/$f"; done
# small results only
for r in final ext web sim; do
  if [ -d "$SRC/results/$r" ]; then mkdir -p "$DST/results/$r"; cp -a "$SRC/results/$r/." "$DST/results/$r/"; else echo "  (skipped, not found: results/$r)"; fi
done
# run logs from the project root
mkdir -p "$DST/logs"; for l in "$SRC"/*.log; do [ -f "$l" ] && cp -a "$l" "$DST/logs/"; done
# never copy compiled files or caches
find "$DST" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$DST" \( -name '*.o' -o -name '*.pyc' -o -name '*.trace' -o -name '*.zip' \) -type f -delete
echo "--- size of the copy:"; du -sh "$DST"
echo "--- biggest files (GitHub refuses files over 100 MB):"; find "$DST" -type f -printf '%s %p\n' | sort -rn | head -5 | awk '{printf "%.1f MB  %s\n",$1/1048576,$2}'
BIG=$(find "$DST" -type f -size +50M | wc -l); [ "$BIG" -eq 0 ] && echo "OK: no file over 50 MB" || { echo "WARNING: $BIG file(s) over 50 MB, do not push them"; exit 2; }
echo "--- files: $(find "$DST" -type f | wc -l)"
