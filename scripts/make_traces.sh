#!/usr/bin/env bash
# Trace every input of the train / val / test split (configs/inputs.tsv) in parallel. Resumable: finished jobs are skipped.
# usage: scripts/make_traces.sh [outroot] [config] [inputs.tsv]
#   defaults: results/traces  configs/default.cfg  configs/inputs.tsv
# env:   JOBS (parallel jobs, default 4)   MIN_FREE_GB (default 20)   ONLY (comma list of workloads, e.g. hash,graph)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/results/traces}"; CFG="${2:-$ROOT/configs/default.cfg}"; TSV="${3:-$ROOT/configs/inputs.tsv}"
JOBS="${JOBS:-4}"
mkdir -p "$OUT"
"$ROOT/scripts/check_disk.sh" "$OUT" || exit 1
[ -x "$ROOT/build/cachesim" ] || { echo "Run 'make all workloads' first."; exit 1; }
LIST="$(mktemp)"
# columns: split  workload  max_records  args...   (lines starting with # are comments)
grep -v '^\s*#' "$TSV" | grep -v '^\s*$' | awk -v only="${ONLY:-}" 'BEGIN{n=split(only,a,","); for(i=1;i<=n;i++) ok[a[i]]=1} only=="" || ($2 in ok)' > "$LIST"
echo "$(wc -l < "$LIST") jobs, $JOBS in parallel, output in $OUT"
start=$(date +%s)
# xargs -L1 splits each line on blanks into: split workload max_records args...
xargs -P "$JOBS" -L 1 bash "$ROOT/scripts/trace_job.sh" "$OUT" "$CFG" < "$LIST"
rc=$?
rm -f "$LIST"
echo "finished in $(( $(date +%s) - start )) s (xargs status $rc)"
python3 "$ROOT/scripts/summarize.py" "$OUT"
