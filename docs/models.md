# Models and the simulator hook

## Task
At every eviction decision (miss in a full set) the model scores the 16 resident lines; the highest score is evicted.
Target = Belady's victim SET. Features are past-only (see `training/features.py`, mirrored in `simulator/learned.h`):
time since last access, age, access count, re-use flag, last inter-access gap, recency rank, relative recency, last/current store flags,
PC id of the line's last access and PC id of the current access (vocabulary of the 64 most frequent training PCs, 0 = unknown).

## Models
| Model | Script | Training | Size knobs |
|---|---|---|---|
| LightGBM (GBDT) | `training/train_lgbm.py` | candidate-level binary classifier (is this way a Belady victim), early stopping on val inputs | `--leaves`, `--rounds` |
| MLP | `training/train_mlp.py` | per-candidate scorer with two PC embeddings, set-level softmax cross-entropy against the tied victim set (loss = -log P(tied set)) | `--hidden`, `--emb` |
| LSTM + attention (PARROT-inspired, small) | `training/train_lstm.py` | LSTM over the last H LLC accesses; each candidate attends over the H hidden states; set-level softmax cross-entropy + optional auxiliary MSE on log reuse distance | `--hist`, `--hidden`, `--emb`, `--dpos`, `--dc`, `--dh`, `--aux` |

Only decisions where not every way ties (`dataset.informative`) are used for training; early stopping and validation use the val inputs only.
Specialized model = trained on one workload's train inputs (`--workload hash`); generic model = all workloads (`--workload all`).

## C++ inference (no ONNX Runtime)
`mlp:<file.mlp>`, `gbdt:<file.gbdt>` and `lstm:<file.lstm>` are policy names for `cachesim run --policies` and `labeler --driver`.
Formats: `training/modelio.py`. LightGBM models are written in LightGBM's own text format (numerical splits only) and walked by hand in C++.
`statsJson()` reports parameters (GBDT: nodes), model bytes, ops per decision (MLP: multiply-adds x2 per candidate x ways; GBDT: measured node comparisons), decision count, mean / p50 / p99 decision latency in ns.
Latency is measured around `victim()` only (feature building + inference) with `steady_clock`, in the simulator on the evaluation machine: it is a software cost, not hardware latency.

## LSTM + attention (Batch D)
Reference: PARROT (Liu et al., ICML 2020, arXiv 2006.16239): LSTM over embedded (address, PC) per access, past H hidden states, each cache line attends over them
(general attention + positional embeddings), dense layer + softmax over lines, ranking loss on reuse distance plus auxiliary loss on log reuse distance, DAgger.
**Our small version, differences to PARROT:** no per-address embeddings (numeric features of the candidate + PC-id embeddings instead; address
history enters only as line-address deltas), default H=16 / LSTM 32 / embeddings 4 (PARROT: H=80, 128, 64), loss = softmax cross-entropy over the tied Belady victim
set (not NDCG), auxiliary reuse-distance head used in training only (`--aux`, not exported), DAgger limited to round 1 by default.
Step input (S = emb+5): `[emb[pc id]; sign(delta line); log2(1+|delta|)/20; is_write; same_set_as_current; valid]`; padding steps (before access 0) use pc id 0 and zeros.
Candidate: `x = [normalized 9 features; emb[last pc]; emb[cur pc]]`, `c = relu(A1 x)`, query `u = Wq c`, keys `[h_i ; pos_i]`, `a = softmax(u.k/sqrt(K))`, `g = sum a_i h_i`, `score = v . relu(B [c; g])`.
The history is the last H LLC accesses ending with the current one, LSTM started from zero state (stateless per decision, so a decision needs H LSTM steps: this is why it is the most expensive model; ops per decision are reported).
`tests/test_lstm.py`: random-weight models (3 shapes) and, if PyTorch is installed, a model trained by `train_lstm.py` are run by the C++ policy; every victim must equal the numpy argmax (`modelio.predict_lstm`). `train_lstm.py` also checks torch == numpy scores before it writes the result.

## DAgger (round 1 by default)
`scripts/dagger_round.sh ROUND KIND` (KIND = mlp | lgbm | lstm): the current model drives the cache on every train and val trace (never test), the labeler labels the visited states with Belady
(files `<trace>.dag<R><kind>.npz` for states driven by the workload's specialized model, `<trace>.dag<R>all<kind>.npz` for states driven by the generic model), the model is retrained on lru + belady + all dag datasets
(`models/<w>_<kind>_dag<R>.*`, `models/all_<kind>_dag<R>.*`) and `run_models.sh val` compares. Tested end-to-end on synthetic traces by `tests/test_dagger.py`.

## Model inventory (every model trained in the project)
File name: `<workload>_<kind><variant>.<ext>`, with a `.json` file next to each model that holds its settings.
- **workload:** `hash`, `graph`, `matrix`, `conv` = specialized (trained on that workload's train inputs); `all` = generic (all workloads); `allbig` = generic MLP whose hidden width was chosen so that its file size matches the four specialized MLPs together (experiment E3).
- **kind / extension:** `mlp` (`.mlp`), `lgbm` (`.gbdt`), `lstm` (`.lstm`).
- **variant:** none = trained on the LRU-driven and Belady-driven datasets; `_dag1`, `_dag2` = after DAgger round 1 or 2 on single applications; `_s2`, `_s3` = the `_dag1` recipe with training seeds 2 and 3 (the plain `_dag1` is seed 1); `_mix1`, `_mix1b`, `_mix2` = mixed-stream DAgger on a shared cache (extension, see `docs/mixed_dagger.md`).

| Folder | Models | Files |
|---|---|---|
| `models/` | for each of hash, graph, matrix, conv and all: MLP base, `_dag1`, `_dag2`; LightGBM base, `_dag1`; LSTM base, `_dag1`; plus `allbig_mlp_dag1` | 36 |
| `models_seeds/` | MLP `_dag1` with seeds 2 and 3 for hash, graph, matrix, conv, all and allbig (training-noise check for E3) | 12 |
| `models_ext/` | mixed-stream models for hash, graph, matrix, conv and all: MLP `_mix1`, `_mix1b`, `_mix2`; LightGBM `_mix1b`, `_mix2`; LSTM `_mix1b`, `_mix2` | 35 |

Single-application DAgger round 2 exists for the MLP only. The mixed-stream LightGBM and LSTM models exist as `_mix1b` and `_mix2` only.

**Official model set** (`results/final/model_set.json`, chosen on validation results only and frozen before the test run): MLP `_dag1`, LightGBM `_dag1`, LSTM base.
Mean gap closed on the four validation traces (specialized and generic models together): MLP 65.3% base, 72.6% `_dag1`, 72.7% `_dag2`; LightGBM 54.8% base, 71.2% `_dag1`; LSTM 67.6% base.
The MLP `_dag2` is ahead by less than 0.001, so the tie rule (the earlier round wins) selects `_dag1`.
The LSTM `_dag1` was trained after the freeze, judged on validation and run once on test as an extension (decision log items 32 and 33); the frozen set was not changed.

**Size of the models in the official set** (parameters / bytes; LightGBM parameters = tree nodes; source: `results/final/tables/e6_models.md`):

| Model | hash | graph | matrix | conv | generic (`all`) |
|---|---|---|---|---|---|
| MLP | 1,739 / 7,044 | 2,203 / 9,364 | 1,731 / 7,004 | 1,739 / 7,044 | 2,203 / 9,364 |
| LightGBM | 18,300 / 1,058,844 | 13,237 / 769,533 | 3,050 / 179,814 | 2,623 / 154,851 | 7,869 / 471,637 |
| LSTM | 9,363 / 37,540 | 9,595 / 38,932 | 9,359 / 37,516 | 9,363 / 37,540 | 9,595 / 38,932 |

The equal-storage generic MLP `allbig` has 7,495 parameters / 30,532 bytes, against 30,456 bytes for the four specialized MLPs together.

## Checks
`tests/test_models.py`: trains both models on a synthetic dataset, lets the C++ policy drive the cache (`labeler --driver gbdt:...`),
re-evaluates the Python model on each recorded state, and requires the same victim for every decision (0 mismatches). C++ unit test parses a hand-made LightGBM model.

## Pipeline
```
bash scripts/train_all.sh            # models/<workload>_{lgbm.gbdt,mlp.mlp} and all_*, offline top-1 table on val inputs
bash scripts/train_lstm_all.sh       # LSTM + attention per workload and generic (needs PyTorch)
bash scripts/dagger_round.sh 1 lgbm  # DAgger round 1 for a model kind (mlp | lgbm | lstm), retrains and re-evaluates on val
bash scripts/run_models.sh val       # simulator misses of baselines (incl. mru) + EVERY model file found in models/   (results/sim/val)
bash scripts/run_models.sh test      # same on held-out test inputs
```
Report rule: choose the best model on **val** results, report **test** results once.
