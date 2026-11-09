#!/usr/bin/env bash
# Reproducibility check: trace the same small run twice and compare the md5 of the two LLC traces.
# usage: scripts/repro_check.sh [config]     (default configs/default.cfg; uses a small hash run, about 1 minute)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="${1:-$ROOT/configs/default.cfg}"
T="$(mktemp -d)"
ARGS=(16 20000 60000 1 1)
for i in 1 2; do
  MIN_FREE_GB="${MIN_FREE_GB:-20}" bash "$ROOT/scripts/trace_one.sh" hash "$CFG" "$T/run$i" "${ARGS[@]}" >"$T/log$i.txt" 2>&1 || { cat "$T/log$i.txt"; echo "REPRO CHECK: pipeline failed"; exit 1; }
done
a=$(md5sum < "$T"/run1/*.llc.trace); b=$(md5sum < "$T"/run2/*.llc.trace)
n=$(stat -c %s "$T"/run1/*.llc.trace)
rm -rf "$T"
if [ "$a" = "$b" ]; then echo "REPRO CHECK: OK (two runs gave identical LLC traces, $n bytes, md5 ${a%% *})"; else echo "REPRO CHECK: FAILED (traces differ: $a vs $b)"; exit 1; fi
