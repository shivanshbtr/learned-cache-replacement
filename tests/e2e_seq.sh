#!/usr/bin/env bash
# End-to-end check on a program with a known answer: valgrind -> lackey2trace -> filter -> policies.
# seq reads 16384 distinct lines x 3 passes. With the small test config (LLC 128 KB) LRU/FIFO/SRRIP must miss every
# measured access, and Belady must hit about 2048 lines per pass after the first.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
command -v valgrind >/dev/null || { echo "e2e_seq: SKIPPED (valgrind not installed)"; exit 0; }
OUT="$(mktemp -d)"
MIN_FREE_GB=1 bash scripts/headroom.sh seq configs/small_test.cfg "$OUT" 1048576 3 >"$OUT/log.txt" 2>&1 || { cat "$OUT/log.txt"; echo "e2e_seq FAILED: pipeline"; exit 1; }
python3 - "$OUT"/seq_1048576_3.result.json <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
t=r["trace"]; p={x["name"]:x for x in r["policies"]}
errs=[]
if t["llc_accesses"]!=16384*3: errs.append("llc_accesses %d != %d"%(t["llc_accesses"],16384*3))
if t["unique_lines"]!=16384: errs.append("unique_lines %d"%t["unique_lines"])
m=t["measured_accesses"]
for n in ("lru","fifo","srrip"):
    if p[n]["misses"]!=m: errs.append("%s misses %d != measured %d"%(n,p[n]["misses"],m))
if not (p["belady"]["hits"]>=0.05*m): errs.append("belady hits too low: %d of %d"%(p["belady"]["hits"],m))
if not (p["belady"]["misses"]<p["lru"]["misses"]): errs.append("belady did not beat lru")
if errs:
    print("e2e_seq FAILED:"); [print(" ",e) for e in errs]; sys.exit(1)
print("e2e_seq OK: %d LLC accesses, LRU misses %d/%d, Belady hits %d"%(t["llc_accesses"],p["lru"]["misses"],m,p["belady"]["hits"]))
PY
rc=$?
rm -rf "$OUT"
exit $rc
