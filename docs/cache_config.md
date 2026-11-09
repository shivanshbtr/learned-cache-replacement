# Cache config (key=value, `#` comments, sizes accept K/M/G = powers of 1024)
| Key | Default | Meaning |
|---|---|---|
| line_bytes | 64 | power of two, 8..128 |
| l1_size, l1_ways | 32K, 8 | L1D (LRU filter) |
| l2_size, l2_ways | 256K, 8 | L2 (LRU filter) |
| llc_size, llc_ways | 2M, 16 | LLC (policy under test), 2048 sets by default |
| warmup_frac | 0.1 | first fraction of LLC accesses only warm the cache and are not counted |
| seed | 1 | Random policy seed |
| srrip_bits | 2 | SRRIP RRPV width, 1..4 |
Sets must be a power of two; size must be a multiple of ways*line_bytes. Unknown keys are errors.
Every result JSON carries a config hash (FNV-1a of the canonical settings).
`small_test.cfg` (8K/32K/128K) exists only so small programs overflow the LLC in tests. It is not the project config.


Policies available in `cachesim run --policies`: `lru fifo random srrip brrip drrip bip belady`. BIP/BRRIP/DRRIP use fixed constants (1/32 bimodal probability, 10-bit PSEL) and the RRPV width `srrip_bits`.
