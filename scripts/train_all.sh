#!/usr/bin/env bash
# Train LightGBM and MLP for every workload (specialized) and for all workloads together (generic).
# usage: scripts/train_all.sh [datasets] [models_dir]     defaults: results/datasets  models
# env: WORKLOADS (default "hash graph matrix conv all")   DRIVERS (default lru,belady)   EXTRA_LGBM / EXTRA_MLP = extra args
# Output: <models_dir>/<w>_lgbm.gbdt  <w>_mlp.mlp  (+ .json with validation scores), log in <models_dir>/train.log
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DS="${1:-$ROOT/results/datasets}"; MD="${2:-$ROOT/models}"
mkdir -p "$MD"
python3 -c "import numpy, lightgbm, torch" 2>/dev/null || { echo "missing python packages: pip install numpy lightgbm; pip install torch --index-url https://download.pytorch.org/whl/cpu"; exit 1; }
for w in ${WORKLOADS:-hash graph matrix conv all}; do
  echo "=== $w: LightGBM" | tee -a "$MD/train.log"
  python3 "$ROOT/training/train_lgbm.py" --data "$DS" --workload "$w" --drivers "${DRIVERS:-lru,belady}" --out "$MD/${w}_lgbm.gbdt" ${EXTRA_LGBM:-} 2>&1 | tee -a "$MD/train.log" | tail -2
  echo "=== $w: MLP" | tee -a "$MD/train.log"
  python3 "$ROOT/training/train_mlp.py" --data "$DS" --workload "$w" --drivers "${DRIVERS:-lru,belady}" --out "$MD/${w}_mlp.mlp" ${EXTRA_MLP:-} 2>&1 | tee -a "$MD/train.log" | tail -2
done
echo "--- offline comparison on validation inputs (top-1 agreement with Belady; higher is better) ---"
for w in ${WORKLOADS:-hash graph matrix conv all}; do
  [ "$w" = all ] && continue
  python3 "$ROOT/training/eval_offline.py" --data "$DS" --split val --workload "$w" --drivers lru \
     --models "$MD/${w}_lgbm.gbdt" "$MD/${w}_mlp.mlp" "$MD/all_lgbm.gbdt" "$MD/all_mlp.mlp" 2>/dev/null | tee -a "$MD/offline_val.txt"
done
