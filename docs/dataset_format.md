# Dataset format (labeler output)

## What one record is
One **eviction decision** = a miss in a set whose ways are all valid. The labeler replays an LLC trace with a *driver* policy
(`lru`, `belady`, later a learned policy). The driver decides what is really evicted; the labeler also computes Belady's victim set.

`.dec` file (written by `build/labeler`): 64-byte header, then `N` records of `rec_words` little-endian uint32.

Header (struct `DecHeader`): magic `ACDEC`, version 1, ways, rec_words (= 4 + 4*ways), stride, sets, line_bytes, num_records (u64), trace_len (u64), warmup (u64), driver name (8 chars).

Record: `[seq, set, label_mask, driver_victim]` then for each way `w`: `[last_idx, insert_idx, count, next_use]`.

| Field | Meaning | Model input? |
|---|---|---|
| seq | index in the LLC trace of the missing access | yes (current access: line and PC from the trace) |
| set | set index | yes |
| label_mask | bit w = way w holds a Belady victim (several bits when lines are never used again) | **TARGET** |
| driver_victim | way the driver evicted | no (analysis) |
| last_idx | index of the latest access to the resident line (its line address and PC are `trace[last_idx]`) | yes |
| insert_idx | index of the access that brought the line in (age = seq - insert_idx) | yes |
| count | accesses to the line since insertion, fill included | yes |
| next_use | index of the next access to the line (0xFFFFFFFF = never) | **NO: label information only** |

## Hand-worked example (matches the unit test)
1 set, 2 ways, trace A B C A B, driver LRU:
* i2 (C misses): ways hold A(last 0, next 3) and B(last 1, next 4). Belady evicts B: mask 0b10. LRU evicts A (driver_victim 0).
* i3 (A misses): ways hold C(last 2, never) and B(last 1, next 4). Belady evicts C: mask 0b01. LRU evicts B.
* i4 (B misses): ways hold C(never) and A(never): tie, mask 0b11. LRU evicts C.

## `.npz` file (made by `training/export_npz.py`, loaded by `training/dataset.py`)
One trace plus its decisions: `seq set label_mask driver_victim` [N]; `last_idx insert_idx count next_use` [N, ways];
`trace_line trace_pc trace_write trace_icount` [T]; `meta` (JSON: config facts, driver, stride, warmup).
History for sequence models (LSTM): the last H accesses before `seq` are `trace_line[seq-H:seq]` and `trace_pc[seq-H:seq]`.

## Rules for training code
* Use `measured(d)` (seq >= warmup) for evaluation. Drop `~informative(d)` rows for training (all ways tied = no signal).
* Never feed `next_use` or `label_mask` into a model. Normalisation statistics from training files only.
* Split by input instance (`results/datasets/{train,val,test}/`), never by random rows.
* Offline baselines (`python3 training/dataset.py file.npz`): top-1 agreement of random, LRU, MRU, FIFO and the driver with the Belady victim set.
