# MLSys draft: RankTier

Draft submission in the MLSys 2025 template (`mlsys2025.sty`, blind review mode). "RankTier" is a placeholder name,
set once by the `\sys` macro in `main.tex`.

The draft follows the framing of `../../../../../RANKTIER_PAPER_REFRAMING_AND_MOTIVATION.md` (the repository root):
runtime-varying MoE work inside a statically partitioned graph, with rank-bound capacities and ownership-preserving
aggregation as two parts of one design. The motivation has three panels: executed padding (Figure 1), the capacity ×
aggregation interaction (Figure 2) and equivalent arithmetic with different placement (Figure 3).

## Layout

`main.tex` holds the preamble, the macros and the title block, and `\input`s one file per `\section` in order: the
abstract and sections 1 `introduction`, 2 `background`, 3 `motivation`, 4 `design`, 5 `realization`, 6 `evaluation`,
7 `transfer` and 8 `conclusion` from `sections/`, and appendices A `flags`, B `realization`, C `calibration`,
D `one_layer`, E `tiles`, F `ladder`, G `attribution_profiles`, H `phases`, I `custom`, J `blocks`, K `rejected`,
L `stages` and M `accuracy` from `appendix/`. Each file starts with `% !TEX root = ../main.tex`, so editors build
`main.tex` from it. The "What transfers" table of §7 (`tab:transfer`) is defined in `sections/evaluation.tex`: a
`table*` is set no earlier than the next page, which from there is the page where §7 starts. The body must end on
page 10.

## Build

```
../scripts/paper_numbers.py ../full_model paper_numbers.json   # joint fit of E46-E55 (needs numpy)
python3 make_figures.py .. figures     # needs numpy + matplotlib; reads paper_numbers.json, ../realcase/routing*, ../full_model/e48, e52
tectonic -X compile main.tex           # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`\usepackage[T1]{fontenc}` is needed under tectonic (XeTeX), which otherwise falls back to Latin Modern because
`times.sty` has no TU fonts; under pdfLaTeX it is harmless. The figure PDFs embed TrueType (Type 42) fonts from
matplotlib, not Type 1.

The repository whitelist tracks only `*.md`, `*.py`, `*.sh` and a few others, so `main.tex`, `sections/*.tex`,
`appendix/*.tex`, `references.bib`, the style files, `figures/overview.tex`, `paper_numbers.json` and the PDFs stay
untracked unless added with `git add -f`. Add a new section file the same way: `main.tex` is tracked, and a commit without
the file breaks the build.

## Where the numbers come from

Section numbers refer to `../README.md`. Experiments E38–E42 follow `../../../../../RANKTIER_EXPERIMENT_PLAN.md`, E44–E51
`../../../../../RANKTIER_REDUCTION_FOLLOWUP_PLAN.md` (§11) and E52–E55 `../../../../../RANKTIER_E47_E51_ASSESSMENT_AND_NEXT_STEPS.md`
(§8). Every full-model latency in the body comes from one program + session fit of log latency over all timed passes of
E46–E55, in which every session also times ladder step 5 (`s5os`) as an anchor: `../scripts/paper_numbers.py` writes
`paper_numbers.json`, which `make_figures.py` reads. Program keys: `prod` production, `s2` naive T/T, `s2os` + head-group
sum, `s3sos` + scans, `s3os` + dense repairs, `s4os` token-owned, `s5os` + elementwise sum, `Dos`/`Eos` two/three tiers,
`Aos`–`Cos` attribution, `cdf`/`edf` repaired dense combine at T / with tiers, `ls`/`lg` local top-k with SoC slices /
global rows, `ctopk`/`ctopkg` top-k from all SoCs, `tcadd`/`tcsl` lane-indexed on the lane axis / SoC slices, `tcc`
TokenCombine.

| Paper | Source |
|---|---|
| Table 1, Figure 5 (App. F), §6.2, abstract and introduction speedups | joint fit (`paper_numbers.json`); sessions `../full_model/e47/` (ladder, attribution), `e55/` (production, `../scripts/run_e55.sh`) |
| Figure 2, §3.2, §6.3 capacity × aggregation | E50: `../full_model/e50/` (`../scripts/run_e50.sh`; dense-with-tiers graphs `build_full_tiered.py --combine densefix`) |
| Table 2, §6.3 attribution A–E at tuned tiles, §6.5 two vs three tiers | E47 sessions (`../scripts/run_e47.sh`), joint fit |
| Table 3 latencies, Figure 3 left, §3.3, §6.4 | E48 (`run_e48.sh`; `build_full_stack.py --combine tokencentric_slices`, `build_full_dyncard.py --combine topk|topk_gather`) and E52 (`run_e52.sh`; `build_full_dyncard.py --combine local_global|local_slices`), joint fit |
| Table 3 MiB column, Figure 3 right | instrumented two-layer profiles: `../full_model/e48/e48_T512_profiles.txt`, `../full_model/e52/e52_T512_profiles.txt` (`run_e38p.sh`) |
| App. H (Table 9), §6.4 local reduction | `../scripts/combine_breakdown.py` on `../full_model/kvtest/prof_e38_512_p2{D,tcadd}_def/analysis/sample0/work.csv`; assessment §8.2 |
| §6.10, App. I (TokenCombine) | `../scripts/customop/tokencombine/` (E49 logs in `../full_model/e49/`), full model E51/E54 (`run_e51.sh`, `run_e54.sh`; `../full_model/e54/e46_e54_summary.txt`); assessment §8.3 |
| §5 (ii), App. B.2 head-group sum (31% of a layer; 17/67/130 ms) | E44 `../full_model/e44_layers.txt` (`../scripts/e44_layers.py`), E45/E46 (`run_e45.sh`, `run_e46.sh`), joint fit `s2` vs `s2os` |
| §6.5 tier spans, gaps and GEMM time | E44 `../full_model/e44_layers.txt` (before the attention fix) |
| §6.5 stage windows, three vs four tiers, Figure 6 (App. L), one session at T=256 | E34, 3.30 (`../tiers/e34a_two_layer.txt`, `../tiers/e34c_full_T256.txt`) |
| App. G (Table 8) and §4.6 tile predication | E38 profiles `../full_model/e39/e38p_T512_profiles.txt`, `e38p_T512_fix_profiles.txt`; E43 `../full_model/e39/e43_tile_predication.txt`, `e43_C512_profile.txt` |
| §6.6 and Table 10 (App. J) fixed-size blocks | E39: `../full_model/e39/e39_T512_P{1,2,3}.txt`, `e39_T{256,128}_P1dg.txt`, `../scripts/e39_summary.py` |
| §4.4 and §6.7 consecutive chunks, overflow detection and recovery | E40: `../full_model/e40/e40c_T*.txt`, `e40a.txt`, `e40b.txt` |
| §6.8 held-out routing, per-workload and leave-one-out calibration, headroom | E41: `../full_model/e41/e41_analysis.txt` (`../scripts/e41_analysis.py`); the 30 evaluation chunks with calibration documents: README 3.37 |
| Table 12 (App. M), §6.9 | joint fit (`l2`, `next_token_matches` in `paper_numbers.json`); FP32 references `../full_model/ref`, `ref256`, `ref512` |
| Table 5 (App. A) compile times | compile lines of `../full_model/e4[6-9]/`, `e5[0-5]/*run_T*.txt`; TokenCombine `e51_aot.txt`, `e54_aot.txt` |
| Figure 1, Table 6 (App. C) | `make_figures.py` on the routing captures `../realcase/routing*` (E21, E24, E28, E29) |
| Workload statistics, fixed-layout overflow (§3.1, App. C) | E24, 3.22; largest per-expert loads from E28/E29, 3.25–3.26 |
| Tensor-unit time 44/310/921 µs (§3.1) | E18, 3.16 |
| Down-projection slicing, 121.6 / 36 / 24 MiB (§5, App. B.1) | 1 and the summary at the top (E1) |
| Dense accumulator behavior (App. B.1) | 3.1–3.2, E6 (3.6), E13 (3.12) |
| Gather probe 2.796 ms (§5 (iii)), partitions (App. B.4, K) | E20 (3.18), E22 (3.20), E23 (3.21) |
| Core mapping, lanes per SoC (§5, App. B.1) | E19, 3.17 |
| Head-parallel attention (§5, App. B.2) | E17, 3.15 |
| Dense-path rewrites (App. B.3) | E10, 3.10 |
| Elementwise final sum (§4.5, App. B.3) | E30, E31 (3.27, 3.28) |
| Rank sort, sort-first, first-GEMM wait (§4.2, App. B.4) | E22, E26, E27 (3.20, 3.23, 3.24) |
| Tile sizes, Table 7 (App. E) | E27, E29 (3.24, 3.26) |
| Tier sizing rule, rows of three/four/five tiers (§4.3) | E34 and `../scripts/tier_budget.py` |
| Split first tier, no depth-first, Table 11 (App. K) | E35: `../tiers/e35_idle_cores.txt`, `../scripts/run_e35.sh` |
| Power and slow SoCs (§6.11) | E31, 3.28; E38 telemetry, 3.34 |
| Appendix D | E14, 3.12 |
| §7 stage weight floor 0.6 ms MXFP6; a 16-lane stage at the floor with 92 rows, tensor-unit bound with 296 | 3.1, E8 (3.8), E13 (3.12), E18 (3.16) |

## Open before submission

- Tier-structure selection: the deployed 8/8/16 structure was chosen by comparison; at T=256 two tiers are 2.5% faster.
  A selection method is being designed and is described as open work in §4.3 and §6.5.
- Consecutive chunks are tested on the full model, two at T=128/256/512 and four at T=128; decode is not (the MoE rewrites
  fix T). The block, multi-chunk, tile-size and stage measurements predate the attention fix.
- The overflow fallback is implemented as capacity profiles in one program and tested with a forced overflow; the deployed
  overflow rate is measured only on the evaluation chunks of §6.8.
- TokenCombine is measured at T=512 only, without tiers.
- A second MoE model (plan 4b) and a second platform (plan 5) are not done.
- The name, the author block and the related-work coverage need the authors' pass.
