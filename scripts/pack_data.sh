#!/usr/bin/env bash
# Pack the BIG data for separate storage (Drive, GitHub Release, college server). Does not change anything in the project.
# usage: bash scripts/pack_data.sh [PROJECT_DIR] [OUT_DIR]     default OUT_DIR: ~/appcache_data
# Output: traces.tar.gz  (results/traces, the raw data: Valgrind has to be re-run to get these)
#         CHECKSUMS.sha256 (checksum of the tarball and of every trace file)
#         datasets_manifest.txt (name and size of every file in results/datasets; they can be re-made from the traces)
set -euo pipefail
P="${1:-$(cd "$(dirname "$0")/.." && pwd)}"; OUT="${2:-$HOME/appcache_data}"
[ -d "$P/results/traces" ] || { echo "ERROR: no $P/results/traces"; exit 1; }
mkdir -p "$OUT"
echo "1/3 file checksums of the traces (about 1 GB)"
( cd "$P" && find results/traces -type f | sort | xargs sha256sum ) > "$OUT/CHECKSUMS.sha256"
echo "2/3 traces.tar.gz"
tar -C "$P" -czf "$OUT/traces.tar.gz" results/traces
( cd "$OUT" && sha256sum traces.tar.gz >> CHECKSUMS.sha256 )
echo "3/3 datasets manifest (names and sizes only)"
( cd "$P" && find results/datasets -type f -printf '%s\t%p\n' | sort -k2 ) > "$OUT/datasets_manifest.txt" 2>/dev/null || true
ls -la "$OUT"; du -sh "$OUT"
echo "Verify later with:  cd <folder with the data> && sha256sum -c CHECKSUMS.sha256   (after extracting traces.tar.gz)"
