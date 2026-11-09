# Final results (test split)

Model set frozen on validation results: {"mlp": "_dag1", "gbdt": "_dag1", "lstm": ""}

### E1/E2: test split, LLC misses, gap closed, gain over the best simple baseline

| trace                            |              policy |  misses | gap closed | vs best simple |
| -------------------------------- | ------------------- | ------- | ---------- | -------------- |
| test__hash_18_100000_1500000_0_5 |                 lru |  883189 |          - |         -24.2% |
|                                  |   best simple (bip) |  710872 |      35.5% |          +0.0% |
|                                  |                fifo | 1012321 |     -26.6% |         -42.4% |
|                                  |              random | 1008836 |     -25.9% |         -41.9% |
|                                  |               srrip |  784766 |      20.3% |         -10.4% |
|                                  |               drrip |  715736 |      34.5% |          -0.7% |
|                                  |                 mru | 1276610 |     -81.0% |         -79.6% |
|                                  |     mlp specialized |  665140 |      44.9% |          +6.4% |
|                                  |         mlp generic |  662413 |      45.5% |          +6.8% |
|                                  |    gbdt specialized |  702589 |      37.2% |          +1.2% |
|                                  |        gbdt generic |  704091 |      36.9% |          +1.0% |
|                                  |    lstm specialized |  696203 |      38.5% |          +2.1% |
|                                  |        lstm generic |  708379 |      36.0% |          +0.4% |
|                                  |     mlp generic-big |  661625 |      45.6% |          +6.9% |
|                                  |              belady |  397661 |          - |              - |
| test__hash_18_100000_1500000_1_5 |                 lru |  559732 |          - |         -23.8% |
|                                  |   best simple (bip) |  452148 |      35.0% |          +0.0% |
|                                  |                fifo |  675581 |     -37.7% |         -49.4% |
|                                  |              random |  674744 |     -37.5% |         -49.2% |
|                                  |               srrip |  499473 |      19.6% |         -10.5% |
|                                  |               drrip |  454856 |      34.2% |          -0.6% |
|                                  |                 mru |  990326 |    -140.2% |        -119.0% |
|                                  |     mlp specialized |  413392 |      47.7% |          +8.6% |
|                                  |         mlp generic |  409946 |      48.8% |          +9.3% |
|                                  |    gbdt specialized |  422361 |      44.7% |          +6.6% |
|                                  |        gbdt generic |  422993 |      44.5% |          +6.4% |
|                                  |    lstm specialized |  437146 |      39.9% |          +3.3% |
|                                  |        lstm generic |  445396 |      37.2% |          +1.5% |
|                                  |     mlp generic-big |  410118 |      48.7% |          +9.3% |
|                                  |              belady |  252654 |          - |              - |
| test__graph_100000_8_0_5_4       |                 lru |  432038 |          - |         -15.7% |
|                                  |   best simple (bip) |  373485 |      28.9% |          +0.0% |
|                                  |                fifo |  571871 |     -69.0% |         -53.1% |
|                                  |              random |  564478 |     -65.3% |         -51.1% |
|                                  |               srrip |  414579 |       8.6% |         -11.0% |
|                                  |               drrip |  383421 |      24.0% |          -2.7% |
|                                  |                 mru | 1342526 |    -449.0% |        -259.5% |
|                                  |     mlp specialized |  346864 |      42.0% |          +7.1% |
|                                  |         mlp generic |  346792 |      42.0% |          +7.1% |
|                                  |    gbdt specialized |  350045 |      40.4% |          +6.3% |
|                                  |        gbdt generic |  351385 |      39.8% |          +5.9% |
|                                  |    lstm specialized |  354134 |      38.4% |          +5.2% |
|                                  |        lstm generic |  354215 |      38.4% |          +5.2% |
|                                  |     mlp generic-big |  346351 |      42.3% |          +7.3% |
|                                  |              belady |  229240 |          - |              - |
| test__graph_100000_8_1_5_4       |                 lru |  338197 |          - |          -8.4% |
|                                  |   best simple (bip) |  311945 |      16.5% |          +0.0% |
|                                  |                fifo |  422896 |     -53.2% |         -35.6% |
|                                  |              random |  432961 |     -59.5% |         -38.8% |
|                                  |               srrip |  340826 |      -1.7% |          -9.3% |
|                                  |               drrip |  332626 |       3.5% |          -6.6% |
|                                  |                 mru |  844848 |    -318.1% |        -170.8% |
|                                  |     mlp specialized |  291099 |      29.6% |          +6.7% |
|                                  |         mlp generic |  289774 |      30.4% |          +7.1% |
|                                  |    gbdt specialized |  291466 |      29.3% |          +6.6% |
|                                  |        gbdt generic |  292606 |      28.6% |          +6.2% |
|                                  |    lstm specialized |  294640 |      27.3% |          +5.5% |
|                                  |        lstm generic |  294056 |      27.7% |          +5.7% |
|                                  |     mlp generic-big |  289596 |      30.5% |          +7.2% |
|                                  |              belady |  178921 |          - |              - |
| test__matrix_700_0_1             |                 lru | 2255616 |          - |         -97.9% |
|                                  | best simple (drrip) | 1139586 |      95.7% |          +0.0% |
|                                  |                fifo | 2255616 |       0.0% |         -97.9% |
|                                  |              random | 1728323 |      45.2% |         -51.7% |
|                                  |               srrip | 2255616 |       0.0% |         -97.9% |
|                                  |                 bip | 1149867 |      94.8% |          -0.9% |
|                                  |                 mru | 1176312 |      92.6% |          -3.2% |
|                                  |     mlp specialized | 1089727 |     100.0% |          +4.4% |
|                                  |         mlp generic | 1089727 |     100.0% |          +4.4% |
|                                  |    gbdt specialized | 1089762 |     100.0% |          +4.4% |
|                                  |        gbdt generic | 1089742 |     100.0% |          +4.4% |
|                                  |    lstm specialized | 1089766 |     100.0% |          +4.4% |
|                                  |        lstm generic | 1089728 |     100.0% |          +4.4% |
|                                  |     mlp generic-big | 1089727 |     100.0% |          +4.4% |
|                                  |              belady | 1089727 |          - |              - |
| test__matrix_800_0_1             |                 lru | 2255031 |          - |         -60.9% |
|                                  | best simple (drrip) | 1401123 |      97.0% |          +0.0% |
|                                  |                fifo | 2255031 |       0.0% |         -60.9% |
|                                  |              random | 2007843 |      28.1% |         -43.3% |
|                                  |               srrip | 2255031 |       0.0% |         -60.9% |
|                                  |                 bip | 1427471 |      94.0% |          -1.9% |
|                                  |                 mru | 1429927 |      93.7% |          -2.1% |
|                                  |     mlp specialized | 1374912 |     100.0% |          +1.9% |
|                                  |         mlp generic | 1374912 |     100.0% |          +1.9% |
|                                  |    gbdt specialized | 1374921 |     100.0% |          +1.9% |
|                                  |        gbdt generic | 1374918 |     100.0% |          +1.9% |
|                                  |    lstm specialized | 1374941 |     100.0% |          +1.9% |
|                                  |        lstm generic | 1374941 |     100.0% |          +1.9% |
|                                  |     mlp generic-big | 1374912 |     100.0% |          +1.9% |
|                                  |              belady | 1374912 |          - |              - |
| test__conv_720_5_3_1             |                 lru |  174488 |          - |         -22.4% |
|                                  |   best simple (bip) |  142582 |      51.6% |          +0.0% |
|                                  |                fifo |  174488 |       0.0% |         -22.4% |
|                                  |              random |  162686 |      19.1% |         -14.1% |
|                                  |               srrip |  174488 |       0.0% |         -22.4% |
|                                  |               drrip |  144592 |      48.4% |          -1.4% |
|                                  |                 mru |  158161 |      26.4% |         -10.9% |
|                                  |     mlp specialized |  112698 |     100.0% |         +21.0% |
|                                  |         mlp generic |  112692 |     100.0% |         +21.0% |
|                                  |    gbdt specialized |  113048 |      99.4% |         +20.7% |
|                                  |        gbdt generic |  112869 |      99.7% |         +20.8% |
|                                  |    lstm specialized |  112699 |     100.0% |         +21.0% |
|                                  |        lstm generic |  125451 |      79.4% |         +12.0% |
|                                  |     mlp generic-big |  112694 |     100.0% |         +21.0% |
|                                  |              belady |  112690 |          - |              - |
| test__conv_832_3_6_1             |                 lru |  466707 |          - |         -22.4% |
|                                  | best simple (drrip) |  381162 |      55.4% |          +0.0% |
|                                  |                fifo |  466707 |       0.0% |         -22.4% |
|                                  |              random |  449370 |      11.2% |         -17.9% |
|                                  |               srrip |  466707 |       0.0% |         -22.4% |
|                                  |                 bip |  384002 |      53.6% |          -0.7% |
|                                  |                 mru |  451641 |       9.8% |         -18.5% |
|                                  |     mlp specialized |  312340 |     100.0% |         +18.1% |
|                                  |         mlp generic |  312336 |     100.0% |         +18.1% |
|                                  |    gbdt specialized |  313107 |      99.5% |         +17.9% |
|                                  |        gbdt generic |  312592 |      99.8% |         +18.0% |
|                                  |    lstm specialized |  312350 |     100.0% |         +18.1% |
|                                  |        lstm generic |  322590 |      93.4% |         +15.4% |
|                                  |     mlp generic-big |  312336 |     100.0% |         +18.1% |
|                                  |              belady |  312335 |          - |              - |

Gap closed = (LRU - policy) / (LRU - Belady). 'best simple' = fewest misses among lru, fifo, random, srrip, drrip, bip, mru on that trace.

### E2 summary: mean gap closed over the test inputs of each workload

| workload | best simple | mlp specialized | mlp generic | gbdt specialized | gbdt generic | lstm specialized | lstm generic |
| -------- | ----------- | --------------- | ----------- | ---------------- | ------------ | ---------------- | ------------ |
| hash     |       35.3% |           46.3% |       47.1% |            41.0% |        40.7% |            39.2% |        36.6% |
| graph    |       22.7% |           35.8% |       36.2% |            34.9% |        34.2% |            32.9% |        33.0% |
| matrix   |       96.4% |          100.0% |      100.0% |           100.0% |       100.0% |           100.0% |       100.0% |
| conv     |       53.5% |          100.0% |      100.0% |            99.5% |        99.8% |           100.0% |        86.4% |

### E3: specialized vs generic (equal per-model size, equal total storage); gap closed, mean +- std over training seeds

| trace                            |                               arm | seeds |       gap closed | specialized - this arm |
| -------------------------------- | --------------------------------- | ----- | ---------------- | ---------------------- |
| test__conv_720_5_3_1             |                       specialized |     3 |    100.0% +- 0.0 |                      - |
|                                  |              generic (equal size) |     3 |    100.0% +- 0.0 |               -0.0 pts |
|                                  | generic-big (equal total storage) |     3 |    100.0% +- 0.0 |               +0.0 pts |
|                                  |       gbdt specialized vs generic |     1 |   99.4% vs 99.7% |               -0.3 pts |
|                                  |       lstm specialized vs generic |     1 |  100.0% vs 79.4% |              +20.6 pts |
| test__conv_832_3_6_1             |                       specialized |     3 |    100.0% +- 0.0 |                      - |
|                                  |              generic (equal size) |     3 |    100.0% +- 0.0 |               -0.0 pts |
|                                  | generic-big (equal total storage) |     3 |    100.0% +- 0.0 |               +0.0 pts |
|                                  |       gbdt specialized vs generic |     1 |   99.5% vs 99.8% |               -0.3 pts |
|                                  |       lstm specialized vs generic |     1 |  100.0% vs 93.4% |               +6.6 pts |
| test__graph_100000_8_0_5_4       |                       specialized |     3 |     41.8% +- 0.2 |                      - |
|                                  |              generic (equal size) |     3 |     42.1% +- 0.0 |               -0.2 pts |
|                                  | generic-big (equal total storage) |     3 |     42.2% +- 0.0 |               -0.4 pts |
|                                  |       gbdt specialized vs generic |     1 |   40.4% vs 39.8% |               +0.7 pts |
|                                  |       lstm specialized vs generic |     1 |   38.4% vs 38.4% |               +0.0 pts |
| test__graph_100000_8_1_5_4       |                       specialized |     3 |     29.3% +- 0.3 |                      - |
|                                  |              generic (equal size) |     3 |     30.1% +- 0.3 |               -0.8 pts |
|                                  | generic-big (equal total storage) |     3 |     30.2% +- 0.3 |               -0.9 pts |
|                                  |       gbdt specialized vs generic |     1 |   29.3% vs 28.6% |               +0.7 pts |
|                                  |       lstm specialized vs generic |     1 |   27.3% vs 27.7% |               -0.4 pts |
| test__hash_18_100000_1500000_0_5 |                       specialized |     3 |     44.7% +- 0.2 |                      - |
|                                  |              generic (equal size) |     3 |     45.4% +- 0.6 |               -0.7 pts |
|                                  | generic-big (equal total storage) |     3 |     45.1% +- 0.5 |               -0.4 pts |
|                                  |       gbdt specialized vs generic |     1 |   37.2% vs 36.9% |               +0.3 pts |
|                                  |       lstm specialized vs generic |     1 |   38.5% vs 36.0% |               +2.5 pts |
| test__hash_18_100000_1500000_1_5 |                       specialized |     3 |     47.7% +- 0.2 |                      - |
|                                  |              generic (equal size) |     3 |     48.6% +- 0.5 |               -0.9 pts |
|                                  | generic-big (equal total storage) |     3 |     48.3% +- 0.4 |               -0.6 pts |
|                                  |       gbdt specialized vs generic |     1 |   44.7% vs 44.5% |               +0.2 pts |
|                                  |       lstm specialized vs generic |     1 |   39.9% vs 37.2% |               +2.7 pts |
| test__matrix_700_0_1             |                       specialized |     3 |    100.0% +- 0.0 |                      - |
|                                  |              generic (equal size) |     3 |    100.0% +- 0.0 |               -0.0 pts |
|                                  | generic-big (equal total storage) |     3 |    100.0% +- 0.0 |               -0.0 pts |
|                                  |       gbdt specialized vs generic |     1 | 100.0% vs 100.0% |               -0.0 pts |
|                                  |       lstm specialized vs generic |     1 | 100.0% vs 100.0% |               -0.0 pts |
| test__matrix_800_0_1             |                       specialized |     3 |    100.0% +- 0.0 |                      - |
|                                  |              generic (equal size) |     3 |    100.0% +- 0.0 |               -0.0 pts |
|                                  | generic-big (equal total storage) |     3 |    100.0% +- 0.0 |               -0.0 pts |
|                                  |       gbdt specialized vs generic |     1 | 100.0% vs 100.0% |               -0.0 pts |
|                                  |       lstm specialized vs generic |     1 | 100.0% vs 100.0% |               +0.0 pts |

Training seeds change only the final training run on identical labeled data (the DAgger data were produced by the seed-1 models). A positive last column means specialization wins by that many points of gap.

### E4: cross-workload matrix (MLP of the row, run on the test traces of the column): mean gap closed

|                 | on hash | on graph | on matrix | on conv |
| --------------- | ------- | -------- | --------- | ------- |
| model of hash   |   46.3% |    15.4% |     91.9% |    0.0% |
| model of graph  | -143.9% |    35.8% |      0.0% |   99.6% |
| model of matrix | -110.8% |  -374.7% |    100.0% |  100.0% |
| model of conv   | -110.4% |  -425.3% |     93.7% |  100.0% |
| generic model   |   47.1% |    36.2% |    100.0% |  100.0% |

Diagonal = the model on its own workload. Negative = worse than LRU. A model never trained on a workload still has to run on it when applications are mixed up.

### E5: sensitivity to the LLC configuration (models trained at 2 MB / 16-way, NOT retrained): mean gap closed on the test traces

| LLC config             | workload | best simple | mlp specialized | mlp generic |      config hash |
| ---------------------- | -------- | ----------- | --------------- | ----------- | ---------------- |
| default (2 MB, 16-way) |     hash |       35.3% |           46.3% |       47.1% | aad10428c2aa096c |
| default (2 MB, 16-way) |    graph |       22.7% |           35.8% |       36.2% | aad10428c2aa096c |
| default (2 MB, 16-way) |   matrix |       96.4% |          100.0% |      100.0% | aad10428c2aa096c |
| default (2 MB, 16-way) |     conv |       53.5% |          100.0% |      100.0% | aad10428c2aa096c |
| llc1m_16w              |     hash |       37.1% |           50.6% |       50.5% | a4b69918085ec0b5 |
| llc1m_16w              |    graph |       43.7% |           62.7% |       62.8% | a4b69918085ec0b5 |
| llc1m_16w              |   matrix |       95.4% |          100.0% |       99.9% | a4b69918085ec0b5 |
| llc1m_16w              |     conv |       56.8% |          100.0% |      100.0% | a4b69918085ec0b5 |
| llc2m_8w               |     hash |       33.3% |           47.3% |       48.3% | c7249e0b85304ea7 |
| llc2m_8w               |    graph |       20.4% |           37.3% |       37.6% | c7249e0b85304ea7 |
| llc2m_8w               |   matrix |       93.7% |          100.0% |       99.5% | c7249e0b85304ea7 |
| llc2m_8w               |     conv |       55.0% |          100.0% |      100.0% | c7249e0b85304ea7 |
| llc4m_16w              |     hash |       17.1% |           24.0% |       26.2% | 1165faf386505d17 |
| llc4m_16w              |    graph |       38.8% |           69.4% |       69.8% | 1165faf386505d17 |
| llc4m_16w              |   matrix |       97.7% |          100.0% |       99.7% | 1165faf386505d17 |
| llc4m_16w              |     conv |       69.5% |          100.0% |      100.0% | 1165faf386505d17 |

The L1/L2 filters are unchanged, so the same LLC traces are replayed on a different LLC. Rows without headroom have LRU = Belady on all test traces of that workload.

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

### E6: total storage of four specialized models vs the generic model(s), bytes

| family | 4 specialized | generic | generic-big | 4 specialized / generic |
| ------ | ------------- | ------- | ----------- | ----------------------- |
| mlp    |         30456 |    9364 |       30532 |                    3.3x |
| gbdt   |       2163042 |  471637 |           - |                    4.6x |
| lstm   |        151528 |   38932 |           - |                    3.9x |

### E6: activation time T = T_fixed + bytes / bandwidth (microseconds, mean model size of the family)

| family | mean bytes | bandwidth GB/s | T_fixed=1us | T_fixed=10us | T_fixed=100us |
| ------ | ---------- | -------------- | ----------- | ------------ | ------------- |
| mlp    |       7614 |              1 |        8.61 |        17.61 |        107.61 |
| mlp    |       7614 |             10 |        1.76 |        10.76 |        100.76 |
| mlp    |       7614 |            100 |        1.08 |        10.08 |        100.08 |
| gbdt   |     540760 |              1 |      541.76 |       550.76 |        640.76 |
| gbdt   |     540760 |             10 |       55.08 |        64.08 |        154.08 |
| gbdt   |     540760 |            100 |        6.41 |        15.41 |        105.41 |
| lstm   |      37882 |              1 |       38.88 |        47.88 |        137.88 |
| lstm   |      37882 |             10 |        4.79 |        13.79 |        103.79 |
| lstm   |      37882 |            100 |        1.38 |        10.38 |        100.38 |

Resident mode (all models already in the model store) costs only T_fixed. This is a cost MODEL, not a measurement.

### E7: switching between the 4 applications (round robin), total LLC misses by time-slice length

| slice (instructions) | switches |         policy |  misses |  vs LRU | vs best simple |
| -------------------- | -------- | -------------- | ------- | ------- | -------------- |
| 40022                |     3997 |            lru | 2481450 |       - |         -23.2% |
|                      |          |          srrip | 2281082 |   +8.1% |         -13.2% |
|                      |          |          drrip | 2014940 |  +18.8% |              - |
|                      |          |            bip | 2089918 |  +15.8% |          -3.7% |
|                      |          |            mru | 3968773 |  -59.9% |         -97.0% |
|                      |          |   gen:all_gbdt | 2638225 |   -6.3% |         -30.9% |
|                      |          |    gen:all_mlp | 2477056 |   +0.2% |         -22.9% |
|                      |          | gen:allbig_mlp | 2524663 |   -1.7% |         -25.3% |
|                      |          |  req:spec_gbdt | 2840480 |  -14.5% |         -41.0% |
|                      |          |   req:spec_mlp | 2907637 |  -17.2% |         -44.3% |
|                      |          |         belady | 1581395 |  +36.3% |              - |
| 400222               |      397 |            lru | 2612880 |       - |         -34.9% |
|                      |          |          srrip | 2286079 |  +12.5% |         -18.0% |
|                      |          |          drrip | 1937004 |  +25.9% |              - |
|                      |          |            bip | 1982805 |  +24.1% |          -2.4% |
|                      |          |            mru | 3948324 |  -51.1% |        -103.8% |
|                      |          |   gen:all_gbdt | 2093791 |  +19.9% |          -8.1% |
|                      |          |    gen:all_mlp | 2372899 |   +9.2% |         -22.5% |
|                      |          | gen:allbig_mlp | 2417541 |   +7.5% |         -24.8% |
|                      |          |  req:spec_gbdt | 2347457 |  +10.2% |         -21.2% |
|                      |          |   req:spec_mlp | 2408863 |   +7.8% |         -24.4% |
|                      |          |         belady | 1554165 |  +40.5% |              - |
| 4002223              |       37 |            lru | 1689412 |       - |              - |
|                      |          |          srrip | 1702117 |   -0.8% |          -0.8% |
|                      |          |          drrip | 1704841 |   -0.9% |          -0.9% |
|                      |          |            bip | 1937866 |  -14.7% |         -14.7% |
|                      |          |            mru | 3861529 | -128.6% |        -128.6% |
|                      |          |   gen:all_gbdt | 1871312 |  -10.8% |         -10.8% |
|                      |          |    gen:all_mlp | 2315238 |  -37.0% |         -37.0% |
|                      |          | gen:allbig_mlp | 2317655 |  -37.2% |         -37.2% |
|                      |          |  req:spec_gbdt | 1755591 |   -3.9% |          -3.9% |
|                      |          |   req:spec_mlp | 1871141 |  -10.8% |         -10.8% |
|                      |          |         belady | 1331865 |  +21.2% |              - |
| 10005559             |       13 |            lru | 1442700 |       - |          -1.5% |
|                      |          |          srrip | 1421696 |   +1.5% |              - |
|                      |          |          drrip | 1429141 |   +0.9% |          -0.5% |
|                      |          |            bip | 1971158 |  -36.6% |         -38.6% |
|                      |          |            mru | 3792799 | -162.9% |        -166.8% |
|                      |          |   gen:all_gbdt | 1757559 |  -21.8% |         -23.6% |
|                      |          |    gen:all_mlp | 2079846 |  -44.2% |         -46.3% |
|                      |          | gen:allbig_mlp | 2181603 |  -51.2% |         -53.5% |
|                      |          |  req:spec_gbdt | 1639791 |  -13.7% |         -15.3% |
|                      |          |   req:spec_mlp | 1591036 |  -10.3% |         -11.9% |
|                      |          |         belady |  955735 |  +33.8% |              - |

The LLC contents persist across switches (lines of the other applications are evicted, not flushed), so every resume starts partly cold. req:spec = the active application's own model; gen:* = one generic model for all.

### E7: cost of switching per application: MPKI alone vs MPKI when time-sliced, and miss rate right after a switch-in

| slice (instr) |       policy | application | MPKI alone | MPKI sliced | ratio | miss rate first 2000 accesses after switch-in | miss rate overall |
| ------------- | ------------ | ----------- | ---------- | ----------- | ----- | --------------------------------------------- | ----------------- |
| 40022         |          lru |        hash |      10.28 |       26.86 | 2.61x |                                          0.74 |              0.74 |
|               |              |       graph |      12.12 |       22.20 | 1.83x |                                          0.32 |              0.31 |
|               |              |      matrix |      17.88 |       17.88 | 1.00x |                                          1.00 |              1.00 |
|               |              |        conv |       1.44 |        1.44 | 1.00x |                                          1.00 |              1.00 |
|               | req:spec_mlp |        hash |       7.59 |       33.11 | 4.36x |                                          0.91 |              0.91 |
|               |              |       graph |       9.73 |       27.85 | 2.86x |                                          0.39 |              0.39 |
|               |              |      matrix |       8.64 |       17.88 | 2.07x |                                          1.00 |              1.00 |
|               |              |        conv |       0.96 |        1.44 | 1.49x |                                          1.00 |              1.00 |
| 400222        |          lru |        hash |      10.28 |       27.74 | 2.70x |                                          0.82 |              0.76 |
|               |              |       graph |      12.12 |       25.59 | 2.11x |                                          0.45 |              0.36 |
|               |              |      matrix |      17.88 |       17.88 | 1.00x |                                          1.00 |              1.00 |
|               |              |        conv |       1.44 |        1.44 | 1.00x |                                          1.00 |              1.00 |
|               | req:spec_mlp |        hash |       7.59 |       31.21 | 4.11x |                                          0.92 |              0.86 |
|               |              |       graph |       9.73 |       16.45 | 1.69x |                                          0.25 |              0.23 |
|               |              |      matrix |       8.64 |       17.88 | 2.07x |                                          1.00 |              1.00 |
|               |              |        conv |       0.96 |        1.44 | 1.49x |                                          1.00 |              1.00 |
| 4002223       |          lru |        hash |      10.28 |       14.18 | 1.38x |                                          1.00 |              0.39 |
|               |              |       graph |      12.12 |       15.75 | 1.30x |                                          1.00 |              0.22 |
|               |              |      matrix |      17.88 |       17.88 | 1.00x |                                          1.00 |              1.00 |
|               |              |        conv |       1.44 |        1.44 | 1.00x |                                          1.00 |              1.00 |
|               | req:spec_mlp |        hash |       7.59 |       23.66 | 3.12x |                                          1.00 |              0.65 |
|               |              |       graph |       9.73 |       11.47 | 1.18x |                                          0.21 |              0.16 |
|               |              |      matrix |       8.64 |       17.87 | 2.07x |                                          1.00 |              1.00 |
|               |              |        conv |       0.96 |        1.44 | 1.49x |                                          1.00 |              1.00 |
| 10005559      |          lru |        hash |      10.28 |       11.92 | 1.16x |                                          1.00 |              0.33 |
|               |              |       graph |      12.12 |       13.32 | 1.10x |                                          1.00 |              0.19 |
|               |              |      matrix |      17.88 |       17.90 | 1.00x |                                          1.00 |              1.00 |
|               |              |        conv |       1.44 |        1.44 | 1.00x |                                          1.00 |              1.00 |
|               | req:spec_mlp |        hash |       7.59 |       19.45 | 2.56x |                                          0.80 |              0.54 |
|               |              |       graph |       9.73 |       11.27 | 1.16x |                                          0.07 |              0.16 |
|               |              |      matrix |       8.64 |       17.90 | 2.07x |                                          1.00 |              1.00 |
|               |              |        conv |       0.96 |        1.44 | 1.50x |                                          1.00 |              1.00 |

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

### E8: 4 applications sharing the LLC (2048KB/16-way, slice 10000 instr, 15990 slices, until first finished), default_s10000

| policy         |  misses |  MPKI | vs LRU | MPKI hash | MPKI graph | MPKI matrix | MPKI conv | max slowdown | Jain fairness | cross-app evictions |
| -------------- | ------- | ----- | ------ | --------- | ---------- | ----------- | --------- | ------------ | ------------- | ------------------- |
| lru            | 3229206 | 17.81 |      - |     28.99 |      24.64 |       17.88 |      1.44 |         2.82 |         0.855 |               67.5% |
| srrip          | 2978663 | 16.43 |  +7.8% |     26.15 |      20.49 |       17.88 |      1.44 |         2.85 |         0.871 |               66.8% |
| drrip          | 2633063 | 14.52 | +18.5% |     22.96 |      18.85 |       15.41 |      1.44 |         2.75 |         0.930 |               74.0% |
| bip            | 2717589 | 14.99 | +15.8% |     24.13 |      20.65 |       15.09 |      1.44 |         2.91 |         0.915 |               74.5% |
| mru            | 4499277 | 24.81 | -39.3% |     31.64 |      61.87 |       15.60 |      1.44 |         1.74 |         0.947 |               70.7% |
| gen:all_gbdt   | 3012205 | 16.61 |  +6.7% |     34.11 |      25.79 |       11.55 |      1.44 |         4.39 |         0.852 |               75.9% |
| gen:all_mlp    | 3024787 | 16.68 |  +6.3% |     36.30 |      22.78 |       12.16 |      1.44 |         4.82 |         0.861 |               76.9% |
| gen:allbig_mlp | 2936191 | 16.19 |  +9.1% |     36.17 |      21.33 |       11.70 |      1.44 |         4.80 |         0.862 |               73.3% |
| own:spec_gbdt  | 3462478 | 19.09 |  -7.2% |     27.41 |      32.72 |       17.88 |      1.44 |         3.54 |         0.883 |                0.0% |
| own:spec_mlp   | 3041762 | 16.77 |  +5.8% |     26.56 |      25.43 |       16.03 |      1.44 |         3.50 |         0.911 |                0.0% |
| req:spec_gbdt  | 3732374 | 20.58 | -15.6% |     29.79 |      37.88 |       17.88 |      1.44 |         3.86 |         0.855 |               92.7% |
| req:spec_mlp   | 3738607 | 20.62 | -15.8% |     33.14 |      34.69 |       17.88 |      1.44 |         4.37 |         0.850 |               83.7% |
| belady         | 2157901 | 11.90 | +33.2% |     12.50 |      13.13 |       16.91 |      1.44 |         2.70 |         0.958 |               67.2% |

max slowdown = largest MPKI(shared) / MPKI(alone, same policy); Jain fairness over normalized progress MPKI(alone)/MPKI(shared) (1 = no interference); cross-app evictions = share of evictions that removed another application's line. req = requester's model, own = ownership (quota ways/apps, a model only evicts its own application's lines), gen = one generic model.

### E8: 4 applications sharing the LLC (8192KB/16-way, slice 10000 instr, 15990 slices, until first finished), llc8m_16w_s10000

| policy         |  misses |  MPKI | vs LRU | MPKI hash | MPKI graph | MPKI matrix | MPKI conv | max slowdown | Jain fairness | cross-app evictions |
| -------------- | ------- | ----- | ------ | --------- | ---------- | ----------- | --------- | ------------ | ------------- | ------------------- |
| lru            | 1347085 |  7.43 |      - |      9.40 |      10.89 |        8.05 |      1.44 |       159.57 |         0.313 |               65.0% |
| srrip          | 1012781 |  5.58 | +24.8% |      7.43 |      10.76 |        4.34 |      1.44 |        86.05 |         0.325 |               65.7% |
| drrip          |  798702 |  4.40 | +40.7% |      6.71 |       9.16 |        2.47 |      1.44 |        48.90 |         0.343 |               70.7% |
| bip            |  760486 |  4.19 | +43.5% |      6.72 |      10.05 |        1.46 |      1.44 |        31.12 |         0.354 |               67.3% |
| mru            | 2232970 | 12.31 | -65.8% |     16.74 |      28.17 |        8.10 |      1.44 |       160.49 |         0.271 |               77.5% |
| gen:all_gbdt   |  978855 |  5.40 | +27.3% |     13.58 |       6.48 |        2.89 |      1.44 |        62.85 |         0.341 |               69.5% |
| gen:all_mlp    |  942251 |  5.20 | +30.1% |     17.78 |       6.76 |        0.07 |      1.44 |        82.30 |         0.545 |               51.8% |
| gen:allbig_mlp |  898060 |  4.95 | +33.3% |     14.74 |       7.75 |        0.49 |      1.44 |        68.22 |         0.412 |               62.5% |
| own:spec_gbdt  | 1582621 |  8.73 | -17.5% |      9.66 |      10.57 |       11.45 |      1.44 |       227.00 |         0.311 |                1.7% |
| own:spec_mlp   | 1295852 |  7.15 |  +3.8% |      8.66 |      10.30 |        8.00 |      1.44 |       158.52 |         0.317 |                2.1% |
| req:spec_gbdt  | 1407352 |  7.76 |  -4.5% |     11.53 |      14.64 |        5.89 |      1.44 |       116.85 |         0.302 |               93.4% |
| req:spec_mlp   | 1173824 |  6.47 | +12.9% |     17.37 |       8.14 |        2.89 |      1.44 |        80.44 |         0.325 |               58.7% |
| belady         |  464480 |  2.56 | +65.5% |      4.41 |       6.37 |        0.30 |      1.44 |        20.40 |         0.516 |               67.3% |

max slowdown = largest MPKI(shared) / MPKI(alone, same policy); Jain fairness over normalized progress MPKI(alone)/MPKI(shared) (1 = no interference); cross-app evictions = share of evictions that removed another application's line. req = requester's model, own = ownership (quota ways/apps, a model only evicts its own application's lines), gen = one generic model.

