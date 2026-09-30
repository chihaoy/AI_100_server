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

Section numbers refer to `../README.md`; E35, E36 and E37 are not yet written up there.

| Paper | Source |
|---|---|
| Table 4, §6.3 ablation (naive T/T = step 2; padding-only cells) | E37: `../full_model/e37/e37b_full_model.txt`, `e37_summary.txt`, `e37_ablation.json`, `../scripts/run_e37b.sh`, `e37_summary.py`; two-layer check `run_e37a.sh`, `e37a_two_layer.txt`; padding-only graphs from `../scripts/build_full_tiered.py --combine dense` |
| Table 3, Figure 3, Table 6, capacity usage (§6.2, 6.5, 6.6) | E36: `../ladder/e36_full_ladder.txt`, `../full_model/e36/e36_ladder.json`, `../scripts/run_e36.sh`, `e36_summary.py` |
| Figure 1, Appendix Table 8 | `make_figures.py` on the routing captures `../realcase/routing*` (E21, E24, E28, E29) |
| Workload statistics, fixed-layout overflow (§3.1, App. B) | E24, 3.22; largest per-expert loads from E28/E29, 3.25–3.26 |
| Tensor-unit time 44/310/921 µs (§3.1) | E18, 3.16 |
| Hot stage 2.35 of 9.24 ms at T=512 (§3.1) | E32, 3.29 (instrumented, run-time sort program) |
| Down-projection slicing, 121.6 / 36 / 24 MiB (§3.2, §4.1) | 1 and the summary at the top (E1) |
| Dense accumulator behavior (§3.2) | 3.1–3.2, E6 (3.6), E13 (3.12) |
| Table 1, combine share 51/64/68% | E8 profile table, 3.8 |
| Gather probe 2.796 ms; other formulations; partitions (§3.3) | E20 (3.18), E22 (3.20), E23 (3.21) |
| Core mapping, lanes per SoC (§3.3) | E19, 3.17 |
| Sequential stages, stage windows, Figure 4, three vs four tiers | E34, 3.30 |
| Head-parallel attention (§4.1) | E17, 3.15 |
| Dense-path rewrites (§4.2) | E10, 3.10 |
| Token-owned combine, lowering pitfalls, one-layer numbers (§4.2, §6.3) | E11, E15, E16 (3.11, 3.13, 3.14) |
| Elementwise final sum, MoE exchange 6 MiB / 3% (§4.2, §6.3) | E30, E31 (3.27, 3.28) |
| Rank sort, sort-first, first-GEMM wait (§4.3) | E22, E26, E27 (3.20, 3.23, 3.24) |
| Tile sizes, Table 5 | E27, E29 (3.24, 3.26) |
| Tier sizing rule, rows vs stages (§4.4) | E34 and `../scripts/tier_budget.py` |
| Split first tier (+6%), no depth-first (+1–2%) | E35: `../tiers/e35_idle_cores.txt`, `../scripts/run_e35.sh` |
| Per-layer first-tier capacity, co-location | E29 (3.26), E30 (3.27) |
| Power and slow SoCs (§6.7) | E31, 3.28 |
| Appendix C | E14, 3.12 |

## Open before submission

- Only the first prefill chunk is tested; multi-chunk and decode correctness of the expert-parallel programs are not.
- End-to-end timing uses one prompt per chunk length; tiers 2 and 3 use calibrated capacities with no overflow
  fallback implemented.
- The name, the author block and the related-work coverage need the authors' pass.
