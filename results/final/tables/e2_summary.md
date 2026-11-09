### E2 summary: mean gap closed over the test inputs of each workload

| workload | best simple | mlp specialized | mlp generic | gbdt specialized | gbdt generic | lstm specialized | lstm generic |
| -------- | ----------- | --------------- | ----------- | ---------------- | ------------ | ---------------- | ------------ |
| hash     |       35.3% |           46.3% |       47.1% |            41.0% |        40.7% |            39.2% |        36.6% |
| graph    |       22.7% |           35.8% |       36.2% |            34.9% |        34.2% |            32.9% |        33.0% |
| matrix   |       96.4% |          100.0% |      100.0% |           100.0% |       100.0% |           100.0% |       100.0% |
| conv     |       53.5% |          100.0% |      100.0% |            99.5% |        99.8% |           100.0% |        86.4% |
