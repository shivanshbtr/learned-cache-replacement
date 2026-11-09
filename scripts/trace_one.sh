#!/usr/bin/env bash
# Trace ONE workload run: valgrind lackey -> lackey2trace -> L1/L2 filter -> <outdir>/<label>.llc.trace
# The workload runs under `env -i` (empty environment) with ASLR off, so stack addresses do not depend on your shell.
# usage: scripts/trace_one.sh <workload> <config> <outdir> [workload args...]
# env:   MAX_RECORDS (default 0 = no cap)   MIN_FREE_GB (default 20)   LABEL (default: workload + args)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ $# -ge 3 ] || { sed -n '2,7p' "$0"; exit 2; }
W="$1"; CFG="$2"; OUT="$3"; shift 3
mkdir -p "$OUT"
"$ROOT/scripts/check_disk.sh" "$OUT" || exit 1
VG="$(command -v valgrind)" || { echo "valgrind not found (sudo apt install valgrind)"; exit 1; }
[ -x "$ROOT/build/cachesim" ] && [ -x "$ROOT/build/lackey2trace" ] && [ -x "$ROOT/build/apps/$W" ] || { echo "Run 'make all workloads' first."; exit 1; }

ADDR="$(nm "$ROOT/build/apps/$W" | awk '$3=="roi_marker"{print $1}')"
[ -n "$ADDR" ] || { echo "roi_marker symbol not found in build/apps/$W"; exit 1; }
LABEL="${LABEL:-$W$(printf '_%s' "$@")}"
LLC="$OUT/$LABEL.llc.trace"
echo "== $LABEL  (roi_marker at 0x$ADDR)"

# lackey writes its trace to fd 3 (its log fd); the program's own stdout/stderr go away.
env -i PATH=/usr/bin:/bin setarch "$(uname -m)" -R "$VG" --tool=lackey --trace-mem=yes --log-fd=3 "$ROOT/build/apps/$W" "$@" \
    3>&1 >/dev/null 2>"$OUT/$LABEL.valgrind.err" \
  | "$ROOT/build/lackey2trace" --roi-addr "0x$ADDR" --out - --max-records "${MAX_RECORDS:-0}" 2>"$OUT/$LABEL.lackey2trace.log" \
  | "$ROOT/build/cachesim" filter --config "$CFG" --in - --out "$LLC" --stats "$OUT/$LABEL.filter.json" 2>"$OUT/$LABEL.filter.log"
st=("${PIPESTATUS[@]}")
cat "$OUT/$LABEL.lackey2trace.log" "$OUT/$LABEL.filter.log"
# st[0] (valgrind) may be non-zero when lackey2trace stops reading at ROI end; only the other two matter.
if [ "${st[1]}" != 0 ] || [ "${st[2]}" != 0 ]; then echo "pipeline failed (lackey2trace=${st[1]} filter=${st[2]}); see $OUT/$LABEL.*.log"; exit 1; fi
