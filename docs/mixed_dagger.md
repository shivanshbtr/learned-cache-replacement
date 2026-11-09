# Mixed-stream DAgger (extension after the frozen E1-E8 results)

## Why
E7/E8 showed that models trained on one application alone lose to DRRIP/BIP on a shared LLC. Reason: in training, every state came from a cache that only
one application used; on a shared LLC the model sees sets full of other applications' lines and unknown PCs. DAgger fixes distribution shift by labeling the states the
learned policy itself visits. Here the "environment" is the shared cache.

## What is done
1. **Merge** (`multisim merge`): the TRAIN inputs (mixture k = k-th train input of hash, graph, matrix, conv) and the VAL inputs (one mixture) are interleaved
   exactly like E8 (round robin, `--slice-instr` instructions per turn, stop when the first application ends). The result is ONE LLC trace; application id a is stored in
   line bits 48+ (tag differs, set does not) and PC bits 56+. A `.meta.json` lists the application names.
2. **Label** (`labeler --driver req=A+B+C+D | own=... | gen=M`, `simulator/composite.h`): the frozen models drive the shared cache with the same rule as in `multisim`
   (req: requester's model chooses among all lines; own: ownership quota; gen: generic model). At every eviction the Belady victim set on the merged stream is recorded.
   Equality with `multisim` is a test (`tests/test_mixed.py`: identical measured misses for lru, req, own, gen, belady at two slice lengths; a mutation that removes PC masking fails it).
3. **Train** (`training/data.py: mix_view`): the model of workload W is trained on solo data (lru, belady, dag1) PLUS the merged datasets, where it only sees decisions
   requested by its own application, and the PC of every other application's access is shown as 0 (as in the simulator). The generic model sees all decisions and all PCs.
   Vocabulary never spends a slot on PC 0. New files: `models_ext/<w>_<kind>_mix1.*` and `all_<kind>_mix1.*`; `models/` and the frozen set are untouched.
4. **Evaluate** (`scripts/mixed_eval.sh`): val mixture first (can be repeated, used to look and decide), then the E8 test mixture ONCE (a marker file with the SHA-256 of the new models
   is written before the run, a second run refuses). Policies side by side: baselines, frozen req/own/gen, mixed-DAgger req/own/gen.

## Variant mix1b: expert-driven shared-cache data added (`EXPERT=1`)
mix1 recorded the merged streams only with LEARNED drivers (req, own, gen). In the usual DAgger recipe the expert (here Belady, plus LRU as the trivial policy used for the
single-application data) first drives the NEW environment, then the learner's own states are added. mix1 skipped that step. mix1b adds two datasets per mixture:
`mixlru1<kind>` (LRU drives the merged stream) and `mixbel1<kind>` (Belady drives it), labeled with Belady like all others. Models: `<w>_<kind>_mix1b.<ext>`.
Choice between mix1 and mix1b is made on the val mixture only; the better one gets the single test run.

## All three model families, and round 2
- **LightGBM and LSTM** use the same merged datasets and drivers: `gbdt=...`, `lstm=...` composite drivers. For the LSTM the drivers of round 1 are the extension models `_dag1`
  (`BASE_SFX=_dag1`, the better LSTM; the frozen set holds the base LSTM). The LSTM history ring is global over the merged stream and uses PLAIN line addresses (the simulator strips the
  application bits, and so does `data.mix_view` for the history features); a test checks that labeler and multisim give identical misses for LSTM drivers.
- **Round 2** (`ROUND=2`): the round-1 models (version `PREV`, default mix1b) drive the shared cache; new datasets `mixreq2/mixown2/mixgen2<kind>`; version `mix2` is trained on solo data + all round-1 data + round-2 data.
- **Chain:** `scripts/mixed_all.sh` runs gbdt round 1, lstm round 1, mlp round 2, gbdt round 2, lstm round 2, then the val comparison of all versions (resumable).
- **Test protocol:** the E8 test mixture was already used once for the MLP versions mix1 and mix1b (results in the log). All new candidates (gbdt/lstm mix1b, all mix2) are run ONCE on a SECOND test mixture
  (`TESTSET=2`: the second test input of every workload), together with the MLP versions so one table shows everything on inputs that no earlier choice has seen. Nothing is chosen on test.

## Integrity rules
- The test split is not used for training: mixtures come from `train` and `val` inputs only (`mixed_plan.py mixtures` refuses `test`).
- Main E1-E8 results do not change. The extension is reported in its own section; the frozen set stays the official model set.
- Held-out claim: val mixture has inputs the training mixtures never contained. Test mixture inputs are the E8 applications, never seen in any round.

## Limits
- Two rounds only: round 1 (`mix1`, `mix1b`) and round 2 (`mix2`). Further rounds were not run.
- Slice length in training = the one used in E8 (10000 instructions) unless SLICES lists more; other slice lengths are generalization.
- Four applications always in the order hash, graph, matrix, conv; each application is present exactly once. A different mix of applications is not tested.
- The LSTM history uses line deltas, which are not defined across applications; on merged streams it uses plain line addresses with the application bits removed, as in the simulator.
- Only the PC-level information is hidden between applications; sets and time are shared.

## Commands
```
make all
scripts/mixed_dagger.sh mlp          # merge + label + train (hours, resumable)
scripts/mixed_eval.sh val            # look at val
EXPERT=1 scripts/mixed_dagger.sh mlp # adds LRU/Belady-driven merged data, trains mix1b
MIXVERS="mix1 mix1b" scripts/mixed_eval.sh val
MIXVERS="mix1 mix1b" scripts/mixed_eval.sh test   # once (or only the chosen version)
```
