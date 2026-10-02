# MLSys draft: RankTier

Draft submission in the MLSys 2025 template (`mlsys2025.sty`, blind review mode). "RankTier" is a placeholder name,
set once by the `\sys` macro in `main.tex`.

## Build

```
python3 make_figures.py .. figures     # needs numpy + matplotlib; reads ../realcase/routing* and ../full_model/e36/
tectonic -X compile main.tex           # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`\usepackage[T1]{fontenc}` is needed under tectonic (XeTeX), which otherwise falls back to Latin Modern because
`times.sty` has no TU fonts; under pdfLaTeX it is harmless. The figure PDFs embed TrueType (Type 42) fonts from
matplotlib, not Type 1.

The repository whitelist tracks only `*.md`, `*.py`, `*.sh` and a few others, so `main.tex`, `references.bib`, the
style files, `figures/overview.tex` and the PDFs stay untracked unless added with `git add -f`.

## Where the numbers come from

Section numbers refer to `../README.md`. Experiments E38–E42 follow `../../../../../RANKTIER_EXPERIMENT_PLAN.md` (the
repository root's plan); their outputs live under `../full_model/e38`–`e41`.

| Paper | Source |
|---|---|
| Table 3, §6.3 capacity attribution A–E (default and tuned tiles), §6.5 sort cost | E38: `../full_model/e38/e38_summary.txt` (`../scripts/e38_summary.py`), sessions `e38b_T*_S*.txt` (`../scripts/run_e38b.sh`, queue `run_e38c.sh`), two-layer tile sweep `e38a_two_layer_T*.txt` (`run_e38a.sh`); graphs `build_full_tiered.py --order identity`, `build_full_stack.py --regroup cardmajor` |
| §3.1, §6.3 and App. F tile predication (rows computed per lane, weight reads of empty experts, C at 512 KiB) | E43: `../README.md` 3.39; `../full_model/e39/e43_tile_predication.txt` (`../scripts/tile_predication.py` on the layer-1 work tables and each program's routing counts), `../full_model/e39/e43_C512_profile.txt` (`../scripts/run_e38p.sh 512 e38C:512`) |
| §6.3 tile explanation, Table 8 (App. F) | E38 profiles: `../full_model/e39/e38p_T512_profiles.txt`, block rows `e38p_T512_fix_profiles.txt` (`../scripts/run_e38p.sh`); weight bytes (bank-gather reads, or static weight reads without spills) from the corrected `stage_profile.py` in `../full_model/e39/stage_profiles_T512.txt`; operation durations from the layer-1 work tables `../full_model/kvtest/prof_e38_512_*/analysis/sample0/work.csv` |
| §6.4 representation vs. lowering (after-GEMM time, cross-SoC bytes) | E42: `../full_model/e39/e42_T512_combine_profiles.txt` (`run_e38p.sh` on the ladder's combine variants `trunc2_512_e42R0..3`), full-model latencies from E36 |
| §6.6 and Table 9 (App. G) fixed-size blocks | E39: `../full_model/e39/e39_T512_P{1,2,3}.txt`, `e39_T{256,128}_P1dg.txt`, table by `../scripts/e39_summary.py`; graphs `build_full_tiered.py --blocks B [--block-budget] [--direct-gather]` via `build_e39_graphs.sh` (`E39_DG=1`), the `--block-pad invalid` graphs and RankTier's `--direct-gather` graph by direct `build_full_tiered.py` calls (`../README.md` §5), budgets `e39_block_budget.py`; profiles `e38p_T512_fix_profiles.txt` |
| §6.7 consecutive chunks, overflow detection and recovery | E40: `../full_model/e40/e40c_T*.txt` (`run_e40c.sh`), `e40a.txt` (two-layer, `run_e40a.sh`), `e40b.txt` (full model, `run_e40b.sh`); host `../scripts/moe_qwen3_multichunk_host.cpp`, plans `e40_plan.py`; graphs `build_full_tiered.py --capacity-inputs`, `truncate_layers.py --keep-counts`; 1024-token reference `../full_model/ref1024/` (`heldout_T1024_ids.npy`) |
| §6.8 held-out routing, per-workload and leave-one-out calibration | E41: `../full_model/e41/e41_analysis.txt` (`../scripts/e41_analysis.py`, which maps the capture's card-major lanes back to experts), chunks `e41_make_chunks.py`, device capture `run_e41.sh`; the 30 evaluation chunks that hold calibration documents of the deployed capacities: README 3.37 |
| Earlier 2×2 ablation (now §6.3, last paragraph: tiers under the export's combine) | E37: `../full_model/e37/e37b_full_model.txt`, `e37_summary.txt`, `e37_ablation.json`, `../scripts/run_e37b.sh`, `e37_summary.py`; two-layer check `run_e37a.sh`, `e37a_two_layer.txt`; padding-only graphs from `../scripts/build_full_tiered.py --combine dense` |
| Table 2, Figure 3 (App. E), Table 11 (App. K), §6.2 | E36: `../ladder/e36_full_ladder.txt`, `../full_model/e36/e36_ladder.json`, `../scripts/run_e36.sh`, `e36_summary.py` |
| Figure 1, Appendix Table 6 | `make_figures.py` on the routing captures `../realcase/routing*` (E21, E24, E28, E29) |
| Workload statistics, fixed-layout overflow (§3.1, App. B) | E24, 3.22; largest per-expert loads from E28/E29, 3.25–3.26 |
| Tensor-unit time 44/310/921 µs (§3.1) | E18, 3.16 |
| Hot stage 2.35 of 9.24 ms at T=512 (§3.1) | E32, 3.29 (instrumented, run-time sort program) |
| Down-projection slicing, 121.6 / 36 / 24 MiB (§3.2, §5.1) | 1 and the summary at the top (E1) |
| Dense accumulator behavior (§3.2) | 3.1–3.2, E6 (3.6), E13 (3.12) |
| Table 1, combine share 51/64/68% | E8 profile table, 3.8 |
| Gather probe 2.796 ms; other formulations; partitions (§3.3) | E20 (3.18), E22 (3.20), E23 (3.21) |
| Core mapping, lanes per SoC (§3.3) | E19, 3.17 |
| Sequential stages, stage windows, Figure 4, three vs four tiers | E34, 3.30 |
| Head-parallel attention (§5.1) | E17, 3.15 |
| Dense-path rewrites (§5.2) | E10, 3.10 |
| Token-owned combine, lowering pitfalls (§4.2, §5.2) | E11, E15, E16 (3.11, 3.13, 3.14) |
| Elementwise final sum (§5.2, §6.4) | E30, E31 (3.27, 3.28) |
| Rank sort, sort-first, first-GEMM wait (§4.1, §5.3) | E22, E26, E27 (3.20, 3.23, 3.24) |
| Tile sizes, Table 7 (App. D) | E27, E29 (3.24, 3.26) |
| Tier sizing rule, rows vs stages (§4.1, §5.3) | E34 and `../scripts/tier_budget.py` |
| Split first tier (+6%), no depth-first (+1–2%), Table 10 (App. I) | E35: `../tiers/e35_idle_cores.txt`, `../scripts/run_e35.sh` |
| Per-layer first-tier capacity, co-location | E29 (3.26), E30 (3.27) |
| Power and slow SoCs (§6.10) | E31, 3.28; clocks and board power of RankTier and the padded programs: E38 telemetry, 3.34 (`../full_model/e38/e38_summary.txt`) |
| Appendix C | E14, 3.12 |
| §7 and Table 4 (what transfers): stage weight floor 0.6 ms MXFP6 / 1.6 ms FP16; a 16-lane stage at the floor with 92 rows, 0.15–0.4 ms above it with 158, tensor-unit bound with 296 | 3.1 (0.1 ms per FP16 expert), E8 (3.8, 37 µs per MXFP6 expert), E13 (3.12), E18 (3.16) |
| §7: reduction series split into rewrites 1.60–1.69× and token-owned + elementwise 1.22–1.48×; production vs naive T/T 1.85× / 1.11–1.13× | E36 ladder (steps 2→3, 3→5, 1 vs 2) |
| §7: four or five tiers (48–58% of the two-tier rows, not stated in the paper) | E34, 3.30 (`../scripts/tier_budget.py`) |

## Open before submission

- Consecutive chunks are tested on the full model, two at T=128/256/512 and four at T=128 (also four on two-layer cuts);
  decode is not (the MoE rewrites fix T).
- The overflow fallback is implemented as capacity profiles in one program and tested with a forced overflow; the deployed
  overflow rate is measured only on the held-out chunks of §6.8.
- A second MoE model for the routing analysis (plan 4b) and the second-platform study (plan 5) are not done.
- The name, the author block and the related-work coverage need the authors' pass.
