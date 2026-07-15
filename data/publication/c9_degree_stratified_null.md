# Canonical F5a: degree-matched pathway anchor null

**Status:** Current secondary analysis on the canonical log2(x+1), measured-only, bounded-h2 landscapes.

## What this test is

For each of eight terms discovered on this cohort and then fixed before the measured-only/log2 method transfer, this analysis asks whether member anchors have a larger pathway-level location of `-slope` than nonmember anchors with comparable regulatory-network degree. A larger `-slope` means higher mean moderated-|t| in the measured hop-1 shell than in the measured hop-2 shell.

The fixed/canonical anchor scope is the same robust ranking used for the canonical fixed-term GSEA: valid anchors with at least 20 measured hop-1 neighbors. The May all-valid-anchor scope is retained as a sensitivity. The fixed/canonical match variable is the persisted full-INDRA degree used by the production per-anchor degree-preserving permutation; matching on measured hop-1 shell size is another sensitivity. The actual May window is retained: control/member degree ratio in [0.8, 1.25]. One matched control is sampled per member, with replacement, in each of 9,999 deterministic Monte Carlo replicates. The fixed/canonical endpoint is the term mean `-slope`; the median is a sensitivity. Raw one-sided empirical p-values are multiplied by eight and compared with the nominal 0.05 reference, separately within each contrast, anchor scope, degree metric, and reported statistic.

> **Selection integrity:** these eight terms were derived from discovery on the same cohort and were not prospectively preregistered. The eightfold threshold handles only the arithmetic multiplicity across the fixed list during this method-transfer check; it is not a post-selection FWER guarantee or selective-inference correction.

> This is not an analysis of `slope_pvalue`, and it is not GSEA. It does not produce or validate an NES. It is a same-data conditional sensitivity asking whether the simple degree-matched reference reproduces the term-level location shift in the already-computed anchor scores.

## Fixed/canonical result: matching on full-INDRA degree

### C9 vs Sporadic

| Term | n matched | observed mean | matched-null mean (95% interval) | empirical p | eightfold p | meets threshold |
|---|---:|---:|---:|---:|---:|:---:|
| mRNA Splicing | 60 | +0.1550 | -0.0015 [-0.0384, +0.0361] | 1.0e-04 | 8.0e-04 | yes |
| Processing Capped Pre-mRNA | 74 | +0.1545 | -0.0044 [-0.0368, +0.0290] | 1.0e-04 | 8.0e-04 | yes |
| mRNA splicing, via spliceosome | 55 | +0.1514 | +0.0000 [-0.0403, +0.0404] | 1.0e-04 | 8.0e-04 | yes |
| chromosome | 185 | +0.0999 | -0.0069 [-0.0264, +0.0127] | 1.0e-04 | 8.0e-04 | yes |
| chromatin | 106 | +0.1080 | -0.0002 [-0.0269, +0.0271] | 1.0e-04 | 8.0e-04 | yes |
| nucleocytoplasmic transport | 37 | +0.1240 | +0.0043 [-0.0424, +0.0526] | 1.0e-04 | 8.0e-04 | yes |
| nuclear pore | 10 | +0.2020 | +0.0087 [-0.0876, +0.1083] | 2.0e-04 | 0.0016 | yes |
| Vpr-mediated nuclear import | 8 | +0.1080 | +0.0110 [-0.0778, +0.1068] | 0.0231 | 0.1848 | no |

**Pattern:** 7/8 terms meet the fixed/canonical mean-score threshold; 7/8 also meet the median-sensitivity threshold.

### C9 vs Control

| Term | n matched | observed mean | matched-null mean (95% interval) | empirical p | eightfold p | meets threshold |
|---|---:|---:|---:|---:|---:|:---:|
| mRNA Splicing | 60 | +0.0497 | -0.0483 [-0.0779, -0.0176] | 1.0e-04 | 8.0e-04 | yes |
| Processing Capped Pre-mRNA | 74 | +0.0503 | -0.0500 [-0.0766, -0.0230] | 1.0e-04 | 8.0e-04 | yes |
| mRNA splicing, via spliceosome | 55 | +0.0513 | -0.0491 [-0.0805, -0.0168] | 1.0e-04 | 8.0e-04 | yes |
| chromosome | 185 | +0.0489 | -0.0534 [-0.0685, -0.0384] | 1.0e-04 | 8.0e-04 | yes |
| chromatin | 106 | +0.0549 | -0.0480 [-0.0688, -0.0268] | 1.0e-04 | 8.0e-04 | yes |
| nucleocytoplasmic transport | 37 | +0.0417 | -0.0416 [-0.0779, -0.0046] | 1.0e-04 | 8.0e-04 | yes |
| nuclear pore | 10 | +0.0750 | -0.0424 [-0.1164, +0.0358] | 0.0025 | 0.0200 | yes |
| Vpr-mediated nuclear import | 8 | +0.0348 | -0.0343 [-0.1059, +0.0448] | 0.0397 | 0.3176 | no |

**Pattern:** 7/8 terms meet the fixed/canonical mean-score threshold; 7/8 also meet the median-sensitivity threshold.

### Sporadic vs Control

| Term | n matched | observed mean | matched-null mean (95% interval) | empirical p | eightfold p | meets threshold |
|---|---:|---:|---:|---:|---:|:---:|
| mRNA Splicing | 60 | +0.0351 | +0.0355 [+0.0114, +0.0590] | 0.5164 | 1.0000 | no |
| Processing Capped Pre-mRNA | 74 | +0.0354 | +0.0362 [+0.0140, +0.0589] | 0.5231 | 1.0000 | no |
| mRNA splicing, via spliceosome | 55 | +0.0270 | +0.0368 [+0.0108, +0.0633] | 0.7658 | 1.0000 | no |
| chromosome | 185 | +0.0298 | +0.0329 [+0.0201, +0.0460] | 0.6805 | 1.0000 | no |
| chromatin | 106 | +0.0334 | +0.0323 [+0.0157, +0.0491] | 0.4471 | 1.0000 | no |
| nucleocytoplasmic transport | 37 | +0.0338 | +0.0349 [+0.0061, +0.0643] | 0.5230 | 1.0000 | no |
| nuclear pore | 10 | +0.1011 | +0.0394 [-0.0221, +0.1043] | 0.0297 | 0.2376 | no |
| Vpr-mediated nuclear import | 8 | +0.0991 | +0.0360 [-0.0264, +0.1028] | 0.0302 | 0.2416 | no |

**Pattern:** 0/8 terms meet the fixed/canonical mean-score threshold; 0/8 also meet the median-sensitivity threshold.

**Key nuance:** Vpr-mediated nuclear import is the sole fixed/canonical-scope C9 non-threshold term (only 8 eligible term members). Its one-sided mean p-values are 0.0231 for C9-vs-sporadic and 0.0397 for C9-vs-control, but neither meets the eightfold threshold. The fixed/canonical-scope result is therefore 7/8, not the May artifact's all-eight claim. The May all-valid scope still returns 8/8 in both C9 contrasts, but that scope cannot substitute for the robust ranking used by the canonical fixed-term GSEA.

## Scope and degree sensitivities

| Contrast | sensitivity | mean threshold count | median threshold count | minimum match coverage |
|---|---|---:|---:|---:|
| C9 vs Sporadic | fixed/canonical scope; measured hop-1 degree | 7/8 | 7/8 | 100.0% |
| C9 vs Sporadic | May all-valid scope; full-INDRA degree | 8/8 | 8/8 | 100.0% |
| C9 vs Sporadic | all-valid scope; measured hop-1 degree | 8/8 | 8/8 | 100.0% |
| C9 vs Control | fixed/canonical scope; measured hop-1 degree | 7/8 | 7/8 | 100.0% |
| C9 vs Control | May all-valid scope; full-INDRA degree | 8/8 | 8/8 | 100.0% |
| C9 vs Control | all-valid scope; measured hop-1 degree | 8/8 | 8/8 | 100.0% |
| Sporadic vs Control | fixed/canonical scope; measured hop-1 degree | 0/8 | 0/8 | 100.0% |
| Sporadic vs Control | May all-valid scope; full-INDRA degree | 0/8 | 2/8 | 100.0% |
| Sporadic vs Control | all-valid scope; measured hop-1 degree | 2/8 | 2/8 | 100.0% |

### Exact May-method reproduction

The historical one-draw procedure (all valid anchors, full-INDRA degree, one match sampled per member, duplicate controls discarded, one-sided Mann-Whitney U) is retained only as a reproducibility sensitivity. After multiplying its raw p-values by eight, its threshold counts are:

- C9 vs Sporadic: 8/8 terms.
- C9 vs Control: 8/8 terms.
- Sporadic vs Control: 1/8 terms.

Because that result depends on one random matched set and silently reduces its size when controls repeat, it is not the fixed/canonical endpoint.


## Interpretation boundary

In this same-data conditional sensitivity on the fixed/canonical scope, the mean `-slope` exceeds the full-INDRA degree-matched reference at the eightfold threshold for 7/8 C9-vs-sporadic terms and 7/8 C9-vs-control terms; the corresponding sporadic-vs-control count is 0/8. The 7/7/0 pattern weighs against a simple degree-location explanation for those seven terms, but does not rule it out, provide selective-inference control, or show that the GSEA result is network-independent.

This same-data control is conditional on the discovery-derived term list, frozen INDRA term membership, INDRA degree snapshot in the distance sidecar, and the same INDRA regulatory graph that generated the slopes. Its 7/7/0 pattern weighs against a simple degree-location explanation for seven terms in each C9 contrast, but does not rule that explanation out. It cannot establish network independence, biological causality, external replication, individual-anchor significance, or post-selection familywise error control. Term overlap also means the eight rows are not independent.

Machine-readable values and full input hashes are in `data/publication/c9_degree_stratified_null.json`; frozen term members are in `data/publication/c9_degree_stratified_null_terms.json`.

## Reproduce

```bash
uv run --no-sync python scripts/run_c9_degree_stratified_null.py
```

The default is offline and consumes the frozen term snapshot. `--refresh-terms` deliberately replaces that snapshot from CoGEx and should be treated as a corpus update, not an ordinary rerun.
