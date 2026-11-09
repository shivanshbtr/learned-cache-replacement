### E6: model cost (software inference in the simulator; latency in microseconds, median over the runs)

| model                     | params |   bytes | ops/decision | latency mean us | p50 us | p99 us |
| ------------------------- | ------ | ------- | ------------ | --------------- | ------ | ------ |
| specialized mlp (hash)    |   1739 |    7044 |        52528 |             8.5 |    8.4 |   14.6 |
| specialized mlp (graph)   |   2203 |    9364 |        52528 |            13.8 |   12.1 |   39.0 |
| specialized mlp (matrix)  |   1731 |    7004 |        52528 |            10.7 |   10.4 |   18.5 |
| specialized mlp (conv)    |   1739 |    7044 |        52528 |            11.3 |   10.8 |   21.9 |
| generic mlp               |   2203 |    9364 |        52528 |            12.1 |   11.1 |   26.4 |
| generic-big mlp           |   7495 |   30532 |       220528 |            50.3 |   46.3 |  134.0 |
| specialized gbdt (hash)   |  18300 | 1058844 |        16762 |           146.9 |  146.0 |  441.8 |
| specialized gbdt (graph)  |  13237 |  769533 |        16921 |           135.5 |  125.0 |  354.4 |
| specialized gbdt (matrix) |   3050 |  179814 |         5472 |            13.8 |   13.1 |   28.3 |
| specialized gbdt (conv)   |   2623 |  154851 |         3161 |             7.8 |    7.4 |   16.8 |
| generic gbdt              |   7869 |  471637 |        12889 |            59.5 |   53.9 |  130.7 |
| specialized lstm (hash)   |   9363 |   37540 |       325632 |           125.8 |  111.1 |  364.8 |
| specialized lstm (graph)  |   9595 |   38932 |       325632 |           120.5 |  114.0 |  308.3 |
| specialized lstm (matrix) |   9359 |   37516 |       325632 |           173.4 |  134.2 |  604.0 |
| specialized lstm (conv)   |   9363 |   37540 |       325632 |           117.5 |  103.7 |  300.6 |
| generic lstm              |   9595 |   38932 |       325632 |           116.1 |  100.7 |  365.7 |

ops/decision counts multiply-adds as 2 ops (trees: node visits). A real LLC replacement decision has to finish in about 100 ns, so these software numbers say how much faster a dedicated accelerator would have to be.
