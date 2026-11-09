#!/usr/bin/env bash
# Train the LSTM + attention model for every workload (specialized) and for all workloads together (generic).
# usage: scripts/train_lstm_all.sh [datasets] [models_dir]     defaults: results/datasets  models
# env: WORKLOADS (default "hash graph matrix conv all")   DRIVERS (default lru,belady)   EXTRA_LSTM = extra train_lstm.py args (e.g. "--epochs 10 --hist 8")
# Output: <models_dir>/<w>_lstm.lstm (+ .json with validation scores), log in <models_dir>/train_lstm.log
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DS="${1:-$ROOT/results/datasets}"; MD="${2:-$ROOT/models}"
mkdir -p "$MD"
python3 -c "import numpy, torch" 2>/dev/null || { echo "missing python packages: pip install numpy; pip install torch --index-url https://download.pytorch.org/whl/cpu"; exit 1; }
for w in ${WORKLOADS:-hash graph matrix conv all}; do
  echo "=== $w: LSTM+attention" | tee -a "$MD/train_lstm.log"
  python3 "$ROOT/training/train_lstm.py" --data "$DS" --workload "$w" --drivers "${DRIVERS:-lru,belady}" --out "$MD/${w}_lstm.lstm" ${EXTRA_LSTM:-} 2>&1 | tee -a "$MD/train_lstm.log" | tail -3
done
echo "--- offline comparison on validation inputs (top-1 agreement with Belady; higher is better) ---"
for w in ${WORKLOADS:-hash graph matrix conv all}; do
  [ "$w" = all ] && continue
  python3 "$ROOT/training/eval_offline.py" --data "$DS" --split val --workload "$w" --drivers lru \
     --models "$MD/${w}_lstm.lstm" "$MD/${w}_mlp.mlp" "$MD/${w}_lgbm.gbdt" "$MD/all_lstm.lstm" 2>/dev/null | tee -a "$MD/offline_val_lstm.txt"
done
