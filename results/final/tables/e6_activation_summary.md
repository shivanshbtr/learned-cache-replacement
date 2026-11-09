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
