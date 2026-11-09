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
