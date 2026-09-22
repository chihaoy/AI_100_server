# All-active oracle measurements

Generated from validated raw outputs and timing samples. Policy selections are frozen from the screen; confirmation does not select new winners. All means weight the five routing workloads equally.

## Fresh confirmation

| Workload | One static plan, ms | Fixed layout + capacity oracle, ms | Joint oracle, ms |
|---|---:|---:|---:|
| uniform | 5.4992 | 5.4722 | 5.5176 |
| mild_a | 5.5114 | 5.4402 | 5.4573 |
| skew_a | 5.4854 | 5.4157 | 5.4046 |
| skew_b | 5.4613 | 5.4613 | 5.4822 |
| skew_c | 5.4242 | 5.4242 | 5.4242 |
| **Mean** | 5.4763 | 5.4427 | 5.4572 |

| Workload | Fixed shape / static placement, ms | Fixed shape / placement oracle, ms |
|---|---:|---:|
| uniform | 5.4992 | 5.5320 |
| mild_a | 5.5114 | 5.5610 |
| skew_a | 5.4854 | 5.4811 |
| skew_b | 5.4613 | 5.4822 |
| skew_c | 5.4242 | 5.4242 |

| Comparison | Saving, ms | Latency reduction | Speedup | Paired round wins |
|---|---:|---:|---:|---:|
| Joint oracle vs One static plan | 0.0191 | 0.35% | 1.0035× | 4/5 |
| Joint oracle vs Fixed layout / capacity oracle | -0.0145 | -0.27% | 0.9973× | 2/5 |
| Fixed shape / placement oracle vs Fixed shape / static placement | -0.0198 | -0.36% | 0.9964× | 1/5 |

## Frozen selections

| Policy | Workload | Plan |
|---|---|---|
| One static plan | uniform | `aggregate_c32_32` |
| One static plan | mild_a | `aggregate_c32_32` |
| One static plan | skew_a | `aggregate_c32_32` |
| One static plan | skew_b | `aggregate_c32_32` |
| One static plan | skew_c | `aggregate_c32_32` |
| Fixed shape / static placement | uniform | `aggregate_c32_32` |
| Fixed shape / static placement | mild_a | `aggregate_c32_32` |
| Fixed shape / static placement | skew_a | `aggregate_c32_32` |
| Fixed shape / static placement | skew_b | `aggregate_c32_32` |
| Fixed shape / static placement | skew_c | `aggregate_c32_32` |
| Fixed layout / capacity oracle | uniform | `aggregate_c8_8` |
| Fixed layout / capacity oracle | mild_a | `aggregate_c32_4` |
| Fixed layout / capacity oracle | skew_a | `aggregate_c32_4` |
| Fixed layout / capacity oracle | skew_b | `aggregate_c32_32` |
| Fixed layout / capacity oracle | skew_c | `aggregate_c32_32` |
| Fixed shape / placement oracle | uniform | `sorted_c_c32_32` |
| Fixed shape / placement oracle | mild_a | `sorted_c_c32_32` |
| Fixed shape / placement oracle | skew_a | `sorted_a_c32_32` |
| Fixed shape / placement oracle | skew_b | `sorted_b_c32_32` |
| Fixed shape / placement oracle | skew_c | `aggregate_c32_32` |
| Joint oracle | uniform | `stripe_c8_8` |
| Joint oracle | mild_a | `sorted_a_c32_4` |
| Joint oracle | skew_a | `sorted_a_c32_2` |
| Joint oracle | skew_b | `sorted_b_c32_32` |
| Joint oracle | skew_c | `aggregate_c32_32` |

## Per-pair timing spread

| Plan | Workload | Median, ms | P10–P90, ms | Round medians, ms |
|---|---|---:|---:|---|
| `aggregate_c32_32` | mild_a | 5.5114 | 5.3649–5.6712 | 5.5292, 5.5664, 5.4316, 5.5229, 5.4786 |
| `aggregate_c32_32` | skew_a | 5.4854 | 5.3816–5.5978 | 5.4934, 5.4772, 5.4492, 5.4811, 5.4886 |
| `aggregate_c32_32` | skew_b | 5.4613 | 5.3685–5.5604 | 5.4183, 5.4960, 5.4376, 5.4553, 5.4898 |
| `aggregate_c32_32` | skew_c | 5.4242 | 5.3005–5.5710 | 5.5012, 5.4105, 5.3605, 5.3412, 5.4896 |
| `aggregate_c32_32` | uniform | 5.4992 | 5.3899–5.6571 | 5.5706, 5.4300, 5.4879, 5.4507, 5.5991 |
| `aggregate_c32_4` | mild_a | 5.4402 | 5.3290–5.5980 | 5.4078, 5.4294, 5.4656, 5.4117, 5.4757 |
| `aggregate_c32_4` | skew_a | 5.4157 | 5.2934–5.5607 | 5.3527, 5.4710, 5.3618, 5.4244, 5.5075 |
| `aggregate_c8_8` | uniform | 5.4722 | 5.3428–5.6877 | 5.3945, 5.5140, 5.4350, 5.5094, 5.5331 |
| `sorted_a_c32_2` | skew_a | 5.4046 | 5.2979–5.5382 | 5.4296, 5.3911, 5.4099, 5.4187, 5.3768 |
| `sorted_a_c32_32` | skew_a | 5.4811 | 5.3491–5.6517 | 5.5228, 5.4414, 5.4626, 5.4176, 5.5684 |
| `sorted_a_c32_4` | mild_a | 5.4573 | 5.3079–5.6241 | 5.4796, 5.4593, 5.4894, 5.3646, 5.4336 |
| `sorted_b_c32_32` | skew_b | 5.4822 | 5.3893–5.6310 | 5.4953, 5.4637, 5.4695, 5.5274, 5.4709 |
| `sorted_c_c32_32` | mild_a | 5.5610 | 5.4282–5.7055 | 5.6690, 5.5211, 5.5854, 5.5262, 5.4793 |
| `sorted_c_c32_32` | uniform | 5.5320 | 5.3834–5.6686 | 5.5208, 5.5497, 5.4257, 5.6264, 5.5193 |
| `stripe_c8_8` | uniform | 5.5176 | 5.3618–5.6588 | 5.4264, 5.4766, 5.5524, 5.5515, 5.5865 |

## Separate instrumented profiles

Each row uses every metric from the same median-device-duration sample of three validated captures. Group 0/1 identify source-graph groups; the compiler can change their compute order. The gap is between the groups in actual chronological order. Phase windows are not additive costs.

| Plan / input | Device, ms | Actual HMX order | Group 0 HMX span, ms | Gap, ms | Group 1 HMX span, ms | Combine, ms | P2P, MiB | Projection DDR copies, MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `aggregate_c32_32` / skew_b | 5.0138 | group0_then_group1 | 1.6796 | 0.3663 | 1.8139 | 0.8518 | 79.523 | 1131.000 |
| `identity_c32_32` / uniform | 4.9266 | group0_then_group1 | 1.6548 | 0.3720 | 1.7827 | 0.8272 | 79.523 | 1131.000 |
| `identity_c32_32` / skew_b | 4.9289 | group0_then_group1 | 1.6415 | 0.3371 | 1.7807 | 0.8735 | 79.523 | 1131.000 |
| `sorted_b_c32_32` / skew_b | 4.9379 | group0_then_group1 | 1.6645 | 0.3485 | 1.8133 | 0.8126 | 79.523 | 1131.000 |
| `sorted_b_c32_2` / skew_b | 4.9218 | group0_then_group1 | 1.6713 | 0.3935 | 1.8959 | 0.8104 | 42.950 | 1068.000 |
| `sorted_b_reverse_c2_32` / skew_b | 4.9434 | group1_then_group0 | 1.6879 | 0.1714 | 1.6239 | 0.8691 | 42.950 | 1068.000 |

Projection DDR bytes are observed copy-descriptor sizes attributed to the six projections, not memory-controller counters or an exact classification of weight versus activation traffic. Every source graph retains the same 1152 MiB of trained weights.

| Plan / input | Group 0/1 HMX compute, max-core µs | Group 0/1 HMX wait, max-core ms | Group 0/1 HMX events | Static constants, MiB |
|---|---:|---:|---:|---:|
| `aggregate_c32_32` / skew_b | 74.166 / 89.479 | 1.626 / 1.754 | 384 / 384 | 1152.278 |
| `identity_c32_32` / uniform | 93.229 / 81.354 | 1.616 / 1.727 | 384 / 384 | 1152.278 |
| `identity_c32_32` / skew_b | 70.312 / 67.292 | 1.582 / 1.732 | 384 / 384 | 1152.278 |
| `sorted_b_c32_32` / skew_b | 79.323 / 71.875 | 1.614 / 1.751 | 384 / 384 | 1152.278 |
| `sorted_b_c32_2` / skew_b | 90.938 / 46.146 | 1.597 / 1.851 | 384 / 384 | 1152.278 |
| `sorted_b_reverse_c2_32` / skew_b | 55.312 / 83.906 | 1.638 / 1.577 | 384 / 384 | 1152.278 |

Compute and wait maxima may come from different cores; they are not additive critical-path costs. HMX waits use `opSyncDurUs` and are clipped to the observed projection interval. They include dependencies and scheduling, not only DDR transfers.

## Validation

- screen: 18,000 timed invocations, 180 saved outputs; maximum relative L2 versus FP32 reference 0.0812%.
- confirm: 7,500 timed invocations, 75 saved outputs; maximum relative L2 versus FP32 reference 0.0791%.
- 18 profiling outputs match their same-graph uninstrumented references bit for bit.
- Counts remain exact, every expert is active, and no routing assignments overflow.
- P10/P90 are invocation spread, not confidence intervals. Only the final output of each timing round is saved.
