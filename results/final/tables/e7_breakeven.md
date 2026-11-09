### E7: break-even slice length (specialized models): slices shorter than this lose more to activation than they save in misses (cost model, main parameters)

| policy        | decision latency ns | miss time saved vs best simple (ms) | decision stall (ms) | resident: slice >= | on demand: slice >= |
| ------------- | ------------------- | ----------------------------------- | ------------------- | ------------------ | ------------------- |
| req:spec_gbdt |                  20 |                               -17.4 |                 0.0 |   never (net loss) |    never (net loss) |
| req:spec_gbdt |                 100 |                               -17.4 |                35.6 |   never (net loss) |    never (net loss) |
| req:spec_gbdt |                1000 |                               -17.4 |              1637.5 |   never (net loss) |    never (net loss) |
| req:spec_mlp  |                  20 |                               -13.5 |                 0.0 |   never (net loss) |    never (net loss) |
| req:spec_mlp  |                 100 |                               -13.5 |                34.0 |   never (net loss) |    never (net loss) |
| req:spec_mlp  |                1000 |                               -13.5 |              1565.3 |   never (net loss) |    never (net loss) |

Cost model parameters (configs/cost_model.json): miss penalty 80 ns, T_fixed 10 us, bandwidth 10 GB/s, 0.33 ns per instruction (assumed 3 GHz, IPC 1). A decision stalls the cache only by the part of its latency above the miss penalty. These are assumptions, swept in e7_cost_model.csv.
