# Application-Carried Learned Cache Replacement

Learning Cache Replacement from Belady: DAgger Training from Single Applications to Shared Caches.
B.Tech project, School of Computer Science and Engineering, KIIT Deemed to be University, Bhubaneswar (2026-2027).
Guide: Dr. Bindu Agarwalla. Team: Shivansh Batra, Priyam Kumar, Shristi Mishra, Jatin Aryan, Shreya Singh.

Simulation only: no hardware was built or modified. Small models (MLP, LightGBM, LSTM with attention) imitate Belady's optimal
replacement policy on the last-level cache (LLC), are trained with DAgger, and are tested on held-out program inputs for single
applications, for time-sliced switching and for four applications sharing one LLC.

## Main results (all on held-out test inputs, 2 MB 16-way LLC; details in the report)
- MLP closes 46% to 47% (hash) and 36% (graph) of the LRU-to-Belady gap; best simple policy 35% and 23%. Matrix and conv: 100%.
- One generic model is as good as four specialized models and needs 3.3 to 4.6 times less storage.
- Models trained on one application fail on other applications and in a shared cache.
- Mixed-stream DAgger on the shared cache: generic models have 5% to 9% fewer misses than the best simple policy at 2 MB and up to 16% fewer at 8 MB (one test mixture, one seed; the ranking of versions is not stable).
- Software decision time (7 to 146 microseconds) is far above a hardware budget of about 100 ns.

## What is in this repository
| Path | Content |
|---|---|
| `applications/` | the 4 C workloads (hash, graph, matrix, conv) with region-of-interest markers |
| `trace_collection/` | Valgrind lackey trace converter |
| `simulator/` | cache simulator (L1/L2 filter, LLC, LRU/FIFO/Random/MRU/SRRIP/BRRIP/DRRIP/BIP/Belady, learned policies) and the multi-application engine |
| `belady/` | Belady labeler (training data) |
| `training/` | features, dataset, MLP / LightGBM / LSTM training, model export |
| `export/`, `web_simulation/` | exporter and the static web simulation (open `web_simulation/index.html`, works offline) |
| `configs/`, `scripts/`, `tests/`, `docs/` | configuration, pipeline scripts, tests, design notes |
| `models/`, `models_ext/`, `models_seeds/` | the trained models (about 12 MB) |
| `results/final`, `results/ext`, `results/web`, `results/sim` | small result tables, plots and web data |
| `logs/` | logs of the long runs |
| `report/` | the project report (PDF) |

## Not in the repository (large data)
The raw traces (about 1.1 GB) and the datasets (about 26 GB) are kept outside git, on Hugging Face: https://huggingface.co/datasets/Shivanshfrm9thdim/learned-cache-replacement-data . See `docs/DATA.md` for the download commands, the checksum checks and how to rebuild them.

## Quick start
```
sudo apt update && sudo apt install -y build-essential valgrind binutils python3 python3-venv unzip
python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt
make all workloads && make test      # builds the simulator and runs all tests
```
To look at the results without running anything: open `web_simulation/index.html` in a browser.

## Setup (Ubuntu 22.04 / WSL2)
```
sudo apt update && sudo apt install -y build-essential valgrind binutils python3 unzip
make all workloads      # builds build/cachesim, build/lackey2trace, build/apps/*
make test               # unit tests + end-to-end test (needs valgrind)
```

## Headroom check (one workload)
```
scripts/headroom.sh <workload> <config> <outdir> [workload args...]
scripts/headroom.sh hash configs/default.cfg results/headroom 17 100000 300000 1
```
Output: table of LRU, FIFO, Random, SRRIP, Belady (misses, hit rate, MPKI, gap closed) plus
`<outdir>/<label>.result.json`, `.filter.json`, `.llc.trace`, and logs.
Workload arguments are listed at the top of each file in `applications/`.
`MAX_RECORDS=N` caps the raw trace (pilot runs), `POLICIES=lru,belady` limits policies, `MIN_FREE_GB` sets the disk guard.

## Manual steps (what the script does)
```
setarch -R valgrind --tool=lackey --trace-mem=yes --log-fd=3 build/apps/hash 17 100000 300000 1 3>&1 >/dev/null 2>/dev/null \
 | build/lackey2trace --roi-addr 0x<roi_marker from nm> --out - \
 | build/cachesim filter --config configs/default.cfg --in - --out hash.llc.trace
build/cachesim run --config configs/default.cfg --trace hash.llc.trace --out hash.result.json
```

## Batch B: traces for the whole input split, then datasets
```
make all workloads && make test                 # build and test (also needs python3 + numpy)
bash scripts/repro_check.sh                     # two traced runs must give identical md5
bash scripts/make_traces.sh                     # all inputs in configs/inputs.tsv -> results/traces/{train,val,test}/ (JOBS=4)
bash scripts/make_datasets.sh                   # Belady labels -> results/datasets/{train,val,test}/*.npz
```
Both scripts are resumable. Formats: `docs/dataset_format.md`. Smoke versions: `configs/inputs_smoke.tsv` with `configs/small_test.cfg`.

## Batch C: train models and run them in the simulator
```
python3 -m venv ~/venv && source ~/venv/bin/activate          # once; run the activate line in every new terminal
pip install numpy lightgbm && pip install torch --index-url https://download.pytorch.org/whl/cpu
bash scripts/train_all.sh                                     # LightGBM + MLP per workload and generic -> models/
bash scripts/run_models.sh val                                # simulator misses: baselines vs models (val inputs)
```
Details: `docs/models.md`.

## Batch D: LSTM + attention, MRU baseline, DAgger
```
bash scripts/train_lstm_all.sh                # LSTM + attention per workload and generic -> models/<w>_lstm.lstm
bash scripts/dagger_round.sh 1 lgbm           # DAgger round 1 (also: mlp, lstm): relabel visited states, retrain, compare on val
bash scripts/run_models.sh val                # baselines (now with mru) + every model in models/; compact summary
```
Tests: `make test` also runs `tests/test_lstm.py` (C++ LSTM = numpy model, decision by decision) and `tests/test_dagger.py` (pipeline on synthetic traces).

## Batch E+F: final evaluation (test split, run ONCE), switching, shared LLC
```
bash scripts/train_final_extras.sh          # equal-total-storage generic MLP + training seeds 2 and 3 (needs PyTorch, about 20 to 30 min)
bash scripts/run_final.sh                   # stages: select (freeze model set on val), e2 (E1-E5), multi (E7, E8), e6 (latency, quiet machine), report
```
Outputs: `results/final/tables/*.csv|md`, `plots/*.png`, `export/web_data.json`, `REPORT_NUMBERS.md`. Stages can be chosen: `STAGES="e6 report" bash scripts/run_final.sh`.
Multi-application engine and cost model: `docs/multi_app.md`. Tests: `tests/test_multi.*`, `tests/test_final.py`.

## Extension: mixed-stream DAgger and LSTM DAgger
`scripts/mixed_dagger.sh mlp` (merge, label with the shared-cache driver, train `models_ext/*_mix1.*`), `scripts/mixed_eval.sh val|test`; see docs/mixed_dagger.md.
Tests: `python3 tests/test_mixed.py`, `python3 tests/test_mixed_pipeline.py`.
