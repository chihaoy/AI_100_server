
### Block A: T=128 sorted layout, capacity sweep (E1/E2 mirror)

| Case | Precision | Chunk, ms (pooled) | Rounds | µs per token | MXFP6 / FP16 (same graph) | Output vs MXFP6 anchor | rel L2 vs FP16 ref | Counts exact |
|---|---|---:|---|---:|---:|---|---:|---|
| cc_c128 | FP16 | 7.284 | 7.295 / 7.258 / 7.287 | 56.9 | 1.000 | – | 0.00e+00 | True |
| cc_c128 | MXFP6 | 5.460 | 5.481 / 5.410 / 5.495 | 42.7 | 0.750 | bit-exact | 1.13e-02 | True |
| cc_c128 | MXFP6 no flag | 10.845 | 10.817 / 10.841 / 10.872 | 84.7 | 1.489 | bit-exact | 1.13e-02 | True |
| cc_c64 | MXFP6 | 5.531 | 5.581 / 5.559 / 5.469 | 43.2 | – | bit-exact | 1.13e-02 | True |
| cc_c32 | MXFP6 | 5.540 | 5.591 / 5.524 / 5.494 | 43.3 | – | bit-exact | 1.13e-02 | True |
| cc_c16 | MXFP6 | 5.361 | 5.384 / 5.303 / 5.393 | 41.9 | – | bit-exact | 1.13e-02 | True |
| cc_c8 | MXFP6 | 5.465 | 5.434 / 5.463 / 5.513 | 42.7 | – | bit-exact | 1.13e-02 | True |
| cc_c4 | MXFP6 | 5.338 | 5.395 / 5.253 / 5.354 | 41.7 | – | bit-exact | 1.13e-02 | True |
| cc_c2 | FP16 | 7.266 | 7.271 / 7.243 / 7.291 | 56.8 | 1.000 | – | 0.00e+00 | True |
| cc_c2 | MXFP6 | 5.348 | 5.389 / 5.317 / 5.392 | 41.8 | 0.736 | bit-exact | 1.13e-02 | True |
| cc_c2 | MXFP6 no flag | 8.563 | 8.547 / 8.561 / 8.582 | 66.9 | 1.178 | bit-exact | 1.13e-02 | True |
| hot96_cold2 | MXFP6 | 5.342 | 5.396 / 5.308 / 5.336 | 41.7 | – | bit-exact | 1.13e-02 | True |
| hot92_cold2 | MXFP6 | 5.313 | 5.323 / 5.257 / 5.349 | 41.5 | – | bit-exact | 1.13e-02 | True |

### Block B: T=128 placement (E4/E5 mirror)

| Case | Precision | Chunk, ms (pooled) | Rounds | µs per token | MXFP6 / FP16 (same graph) | Output vs MXFP6 anchor | rel L2 vs FP16 ref | Counts exact |
|---|---|---:|---|---:|---:|---|---:|---|
| c2tree_sorted | FP16 | 4.953 | 4.987 / 4.958 / 4.895 | 38.7 | 1.000 | – | 0.00e+00 | True |
| c2tree_sorted | MXFP6 | 3.276 | 3.207 / 3.323 / 3.285 | 25.6 | 0.661 | bit-exact | 1.13e-02 | True |
| c2tree_sorted | MXFP6 no flag | 5.644 | 5.614 / 5.604 / 5.702 | 44.1 | 1.139 | bit-exact | 1.13e-02 | True |
| c2tree_native | FP16 | 4.212 | 4.203 / 4.165 / 4.272 | 32.9 | 1.000 | – | 1.06e-05 | True |
| c2tree_native | MXFP6 | 2.777 | 2.869 / 2.783 / 2.724 | 21.7 | 0.659 | rel 1.0e-05 | 1.13e-02 | True |
| c2tree_bal | FP16 | 3.946 | 4.045 / 3.932 / 3.859 | 30.8 | 1.000 | – | 1.18e-05 | True |
| c2tree_bal | MXFP6 | 2.825 | 2.894 / 2.766 / 2.769 | 22.1 | 0.716 | rel 1.2e-05 | 1.13e-02 | True |
| e4native_128_128 | MXFP6 | 5.129 | 5.099 / 5.169 / 5.113 | 40.1 | – | rel 1.0e-05 | 1.13e-02 | True |
| e4native_92_34 | MXFP6 | 5.053 | 5.048 / 5.032 / 5.098 | 39.5 | – | rel 1.0e-05 | 1.13e-02 | True |

### Block D: chunk size × capacity (E6/E7 mirror)

| Case | Precision | Chunk, ms (pooled) | Rounds | µs per token | MXFP6 / FP16 (same graph) | Output vs MXFP6 anchor | rel L2 vs FP16 ref | Counts exact |
|---|---|---:|---|---:|---:|---|---:|---|
| T128_lpt_128_128 | MXFP6 | 2.871 | 2.954 / 2.898 / 2.820 | 22.4 | – | rel 7.0e-05 | 9.85e-02 | True |
| T128_hc_128_128 | FP16 | 3.928 | 3.978 / 3.894 / 3.924 | 30.7 | 1.000 | – | 7.15e-05 | True |
| T128_hc_128_128 | MXFP6 | 2.848 | 2.797 / 2.854 / 2.967 | 22.2 | 0.725 | bit-exact | 9.85e-02 | True |
| T128_hc_92_2 | FP16 | 3.896 | 3.894 / 3.858 / 3.925 | 30.4 | 1.000 | – | 7.15e-05 | True |
| T128_hc_92_2 | MXFP6 | 2.596 | 2.612 / 2.599 / 2.589 | 20.3 | 0.666 | bit-exact | 9.85e-02 | True |
| T256_lpt_256_256 | MXFP6 | 5.729 | 5.835 / 5.603 / 5.758 | 22.4 | – | rel 6.8e-05 | 9.69e-02 | True |
| T256_hc_256_256 | FP16 | 6.767 | 6.834 / 6.652 / 6.791 | 26.4 | 1.000 | – | 0.00e+00 | True |
| T256_hc_256_256 | MXFP6 | 5.914 | 5.913 / 5.936 / 5.888 | 23.1 | 0.874 | bit-exact | 9.69e-02 | True |
| T256_hc_256_4 | MXFP6 | 5.574 | 5.535 / 5.546 / 5.642 | 21.8 | – | bit-exact | 9.69e-02 | True |
| T256_hc_158_256 | MXFP6 | 5.650 | 5.588 / 5.720 / 5.650 | 22.1 | – | bit-exact | 9.69e-02 | True |
| T256_hc_158_32 | MXFP6 | 5.227 | 5.223 / 5.238 / 5.209 | 20.4 | – | bit-exact | 9.69e-02 | True |
| T256_hc_158_4 | FP16 | 6.509 | 6.510 / 6.574 / 6.478 | 25.4 | 1.000 | – | 0.00e+00 | True |
| T256_hc_158_4 | MXFP6 | 5.197 | 5.250 / 5.213 / 5.146 | 20.3 | 0.798 | bit-exact | 9.69e-02 | True |
| T512_lpt_512_512 | MXFP6 | 11.571 | 11.698 / 11.400 / 11.616 | 22.6 | – | rel 4.4e-05 | 3.70e-02 | True |
| T512_hc_512_512 | FP16 | 12.312 | 12.293 / 12.383 / 12.279 | 24.0 | 1.000 | – | 0.00e+00 | True |
| T512_hc_512_512 | MXFP6 | 11.347 | 11.444 / 11.337 / 11.275 | 22.2 | 0.922 | bit-exact | 3.70e-02 | True |
| T512_hc_512_8 | MXFP6 | 10.558 | 10.671 / 10.462 / 10.609 | 20.6 | – | bit-exact | 3.70e-02 | True |
| T512_hc_296_512 | MXFP6 | 10.729 | 10.689 / 10.765 / 10.759 | 21.0 | – | bit-exact | 3.70e-02 | True |
| T512_hc_296_32 | FP16 | 11.285 | 11.346 / 11.283 / 11.221 | 22.0 | 1.000 | – | 0.00e+00 | True |
| T512_hc_296_32 | MXFP6 | 9.924 | 9.913 / 9.880 / 9.978 | 19.4 | 0.879 | bit-exact | 3.70e-02 | True |
| T512_hc_296_8 | MXFP6 | 10.031 | 10.074 / 9.990 / 10.049 | 19.6 | – | bit-exact | 3.70e-02 | True |

### Block C: all-active plans (E3 mirror), FP16 flag vs MXFP6 flag, medians ms

| Plan | uniform | mild_a | skew_a | skew_b | skew_c |
|---|---:|---:|---:|---:|---:|
| aggregate_c32_32 | 4.936 → 2.977 (0.60, L2 9.4e-02) | 4.891 → 2.967 (0.61, L2 9.8e-02) | 4.905 → 2.964 (0.60, L2 9.9e-02) | 4.888 → 2.943 (0.60, L2 8.3e-02) | 4.840 → 2.984 (0.62, L2 8.8e-02) |
| aggregate_c32_4 |  | 4.911 → 2.999 (0.61, L2 9.8e-02) | 4.931 → 3.042 (0.62, L2 9.9e-02) |  |  |
| aggregate_c8_8 | 4.964 → 2.988 (0.60, L2 9.4e-02) |  |  |  |  |
| identity_c32_32 | 4.903 → 2.982 (0.61, L2 9.4e-02) | 4.923 → 2.969 (0.60, L2 9.8e-02) | 4.917 → 3.008 (0.61, L2 9.9e-02) | 4.915 → 2.980 (0.61, L2 8.3e-02) | 4.925 → 3.006 (0.61, L2 8.8e-02) |
| identity_c8_8 | 4.954 → 3.076 (0.62, L2 9.4e-02) |  |  |  |  |
| sorted_a_c32_2 |  |  | 4.969 → 3.018 (0.61, L2 9.9e-02) |  |  |
| sorted_a_c32_32 | 4.903 → 3.010 (0.61, L2 9.4e-02) | 4.919 → 2.990 (0.61, L2 9.8e-02) | 4.923 → 3.033 (0.62, L2 9.9e-02) | 4.890 → 3.011 (0.62, L2 8.3e-02) | 4.913 → 2.975 (0.61, L2 8.8e-02) |
| sorted_a_c32_4 |  | 5.040 → 3.069 (0.61, L2 9.8e-02) | 5.035 → 3.037 (0.60, L2 9.9e-02) |  |  |
| sorted_b_c32_32 | 4.963 → 3.021 (0.61, L2 9.4e-02) | 4.927 → 3.005 (0.61, L2 9.8e-02) | 4.932 → 3.020 (0.61, L2 9.9e-02) | 4.962 → 3.000 (0.60, L2 8.3e-02) | 4.922 → 2.982 (0.61, L2 8.8e-02) |
| sorted_c_c32_32 | 4.954 → 2.978 (0.60, L2 9.4e-02) | 4.911 → 2.960 (0.60, L2 9.8e-02) | 4.965 → 3.014 (0.61, L2 9.9e-02) | 4.932 → 3.006 (0.61, L2 8.3e-02) | 4.970 → 3.020 (0.61, L2 8.8e-02) |
| stripe_c8_8 | 4.968 → 3.004 (0.60, L2 9.4e-02) |  |  |  |  |

**Policy means over the five workloads (frozen selections from all_active/RESULTS.md):**

| Policy | FP16, ms | MXFP6, ms | MXFP6 / FP16 |
|---|---:|---:|---:|
| one static plan | 4.892 | 2.967 | 0.606 |
| fixed layout + capacity oracle | 4.907 | 2.991 | 0.610 |
| fixed shape + placement oracle | 4.918 | 2.991 | 0.608 |
| joint oracle (frozen) | 4.956 | 3.015 | 0.608 |
| no regrouping, 32/32 | 4.917 | 2.989 | 0.608 |
| best tested plan per workload (fp16) | 4.885 | | [4.903(sorted_a_c32_32) 4.891(aggregate_c32_32) 4.905(aggregate_c32_32) 4.888(aggregate_c32_32) 4.840(aggregate_c32_32)] |
| best tested plan per workload (mx) | 2.964 | | [2.977(aggregate_c32_32) 2.960(sorted_c_c32_32) 2.964(aggregate_c32_32) 2.943(aggregate_c32_32) 2.975(sorted_a_c32_32)] |

MXFP6 plan spread per workload: uniform: 2.977–3.076 ms; mild_a: 2.960–3.069 ms; skew_a: 2.964–3.042 ms; skew_b: 2.943–3.011 ms; skew_c: 2.975–3.020 ms
