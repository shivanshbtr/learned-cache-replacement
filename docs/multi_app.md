# Multi-application LLC: switching (E7) and shared cache (E8)

Code: `simulator/multi.h` (engine), `simulator/multisim.cpp` (command line), tests `tests/test_multi.cpp`, `tests/test_multi.py`.

## Model of the system (every point is a simulation choice)
- Each application has its own LLC trace (already filtered by L1/L2). A **schedule** merges them round-robin: an application runs until it has
  consumed `--slice-instr` instructions (at least one record), then the next unfinished application runs. `--until first` stops when the first
  application finishes (every application is then active in the whole run); `--until all` plays every trace to its end.
- The schedule does not depend on hits and misses (there is no timing model), so Belady can be computed on the merged stream (`mergedNextUse`).
- Every line has an owner: `tag = line | app << 48`. The **set** comes from the plain line address, so applications never hit on each other's
  lines but compete for the same sets. Plain line addresses must be below 2^48.
- Time is the global access counter. Per-line metadata (LRU stamps, features such as "time since last access") counts accesses of all applications.
- Per-application model instances see every hit and fill of the merged stream. An instance of application X sees the PC of an access only if X
  issued it; PCs of other applications are shown as 0 (unknown code).
- Warm-up: the first `warmup_frac` of the merged stream is excluded from the statistics, as for a single application.

## Arbitration rules (who picks the victim when application R misses in a full set)
| Token | Rule |
|---|---|
| `lru`, `srrip`, `belady`, ... | one policy for the whole cache (baselines) |
| `req:SET` | R's own model scores every line of the set and may evict any line, also another application's |
| `own:SET` | ownership: quota = ways / applications. If R owns at least `quota` lines in the set, R's model evicts one of ITS OWN lines. Otherwise the application with the most lines in the set gives up one of ITS OWN lines, chosen by ITS model (counted as `borrow`). A model never chooses among another application's lines. Scores are computed from the whole set; only allowed ways can win (`victimAllowed`). |
| `gen:NAME` | one generic model for the whole cache |
Trust discussion: `own` bounds how much a badly trained or adversarial model can hurt the others, because it can only ever evict its own lines.

## Metrics
Per application: accesses, hits, misses, MPKI (shared) and MPKI alone (`--alone 1` runs each application alone with the same policy);
cold counters (misses in the first `--cold-k` accesses after each switch-in); evict matrix [requester][owner of the evicted line];
max slowdown = max MPKI(shared)/MPKI(alone); Jain fairness over MPKI(alone)/MPKI(shared); share of cross-application evictions; `--slices 1` adds a per-slice (app, accesses, misses) timeline.

## Activation cost model (E7, configs/cost_model.json: assumptions, not measurements)
`T_activate = T_fixed + bytes / bandwidth` (on-demand) or `T_fixed` (resident). A decision stalls the cache only by the part of its latency above the miss
penalty. Net benefit = misses saved vs the best simple policy x miss penalty - activation time at every switch-in - decision stalls. `final_report.py` prints the
sweep (`e7_cost_model.csv`) and the break-even slice length (`e7_breakeven`) for the main parameters.

## Usage
`multisim run --config C --app NAME=LLC.trace ... --slice-instr N [--until first|all] --policies lru,belady,req:S,own:S,gen:G --set S=app:file,... --gen G=file [--alone 1] [--slices 1] [--out R.json]`
E7/E8 on the test split: `scripts/final_multi.sh` (called by `scripts/run_final.sh`).
