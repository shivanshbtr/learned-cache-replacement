#!/usr/bin/env bash
# One trace job for make_traces.sh: trace + baseline-policy comparison; writes <label>.done when finished.
# usage: scripts/trace_job.sh <outroot> <config> <split> <workload> <max_records> [workload args...]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTROOT="$1"; CFG="$2"; SPLIT="$3"; W="$4"; MAXR="$5"; shift 5
DIR="$OUTROOT/$SPLIT"
mkdir -p "$DIR"
LABEL="${SPLIT}__${W}$(printf '_%s' "$@")"
if [ -f "$DIR/$LABEL.done" ]; then echo "skip $LABEL (done)"; exit 0; fi
rm -f "$DIR/$LABEL".*
t0=$(date +%s)
{
  LABEL="$LABEL" MAX_RECORDS="$MAXR" bash "$ROOT/scripts/trace_one.sh" "$W" "$CFG" "$DIR" "$@" &&
  "$ROOT/build/cachesim" run --config "$CFG" --trace "$DIR/$LABEL.llc.trace" \
      --policies lru,fifo,random,srrip,brrip,drrip,bip,belady --out "$DIR/$LABEL.result.json"
} >"$DIR/$LABEL.job.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  echo "{\"split\":\"$SPLIT\",\"workload\":\"$W\",\"args\":\"$*\",\"max_records\":$MAXR,\"seconds\":$(( $(date +%s) - t0 ))}" > "$DIR/$LABEL.done"
  echo "done  $LABEL ($(( $(date +%s) - t0 )) s)"
else
  echo "FAILED $LABEL (see $DIR/$LABEL.job.log)"
fi
exit $rc
