#!/usr/bin/env bash
# One resumable chain for the whole mixed-stream extension (everything after MLP round 1, which is already done):
#   1 LightGBM round 1 (expert data + req/own/gen)   2 LSTM round 1 (the _dag1 LSTMs drive)   3 MLP round 2   4 LightGBM round 2   5 LSTM round 2
#   then the VAL comparison of all versions. Each step skips what is already finished, so after a sleep or power-off just run the same command again.
# usage: nohup bash scripts/mixed_all.sh > mixed_all.log 2>&1 &            env: JOBS (2)  EXTRA_MLP EXTRA_GBDT EXTRA_LSTM (extra training args)  STEPS="1 2 3 4 5"
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT" || exit 1
export JOBS="${JOBS:-2}"
step() { local n="$1"; shift; [[ " ${STEPS:-1 2 3 4 5} " == *" $n "* ]] || return 0; echo "=========== step $n: $* ($(date +%H:%M))"; "$@" || echo "!!!!!!!!!! step $n FAILED (see above); continuing with the next step"; }
export EXTRA_TRAIN="${EXTRA_GBDT:---threads 4}"
step 1 env EXPERT=1 bash scripts/mixed_dagger.sh gbdt
export EXTRA_TRAIN="${EXTRA_LSTM:---epochs 12 --threads 4}"
step 2 env EXPERT=1 bash scripts/mixed_dagger.sh lstm
export EXTRA_TRAIN="${EXTRA_MLP:---threads 4}"
step 3 env ROUND=2 bash scripts/mixed_dagger.sh mlp
export EXTRA_TRAIN="${EXTRA_GBDT:---threads 4}"
step 4 env ROUND=2 bash scripts/mixed_dagger.sh gbdt
export EXTRA_TRAIN="${EXTRA_LSTM:---epochs 12 --threads 4}"
step 5 env ROUND=2 bash scripts/mixed_dagger.sh lstm
echo "=========== val comparison ($(date +%H:%M))"
MIXVERS="mix1 mix1b mix2" FAMILIES="mlp gbdt lstm" bash scripts/mixed_eval.sh val
echo "=========== all done ($(date +%H:%M))"
