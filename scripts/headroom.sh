#!/usr/bin/env bash
# Headroom check: trace one workload (scripts/trace_one.sh), then compare the policies at the LLC.
# usage: scripts/headroom.sh <workload> <config> <outdir> [workload args...]
#   <workload> is a program in build/apps/ (matrix graph conv hash seq), built by `make workloads`.
# env: POLICIES (default lru,fifo,random,srrip,drrip,bip,belady)   MAX_RECORDS (default 0 = no cap)   MIN_FREE_GB (default 20)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ $# -ge 3 ] || { sed -n '2,6p' "$0"; exit 2; }
W="$1"; CFG="$2"; OUT="$3"; shift 3
bash "$ROOT/scripts/trace_one.sh" "$W" "$CFG" "$OUT" "$@" || exit 1
LABEL="${LABEL:-$W$(printf '_%s' "$@")}"
"$ROOT/build/cachesim" run --config "$CFG" --trace "$OUT/$LABEL.llc.trace" --policies "${POLICIES:-lru,fifo,random,srrip,drrip,bip,belady}" --out "$OUT/$LABEL.result.json"
