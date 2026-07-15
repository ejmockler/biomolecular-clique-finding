# Canonical size-matched HGNC-set null

**Status:** canonical auxiliary analysis regenerated on the production log2(x+1) scale.

## Design

For each contrast, `RotationTestEngine` fits `log2(intensity+1) ~ condition + Sex` and supplies the production empirical-Bayes moderated *t*. UniProt rows are mapped to HGNC IDs; the sole duplicated HGNC measurement is aggregated by maximum *t*² so every random-set unit is one gene. For each of the eight discovery-derived fixed terms, the observed mean *t*² is compared with 10,000 uniform same-size HGNC sets sampled without replacement from that contrast's finite moderated-*t* background.

One-tailed empirical p = `(1 + count(null >= observed)) / 10,001`. The family readout is Bonferroni-8: raw p < 0.00625.

## Input and fit accounting

- Source matrix: 3,264 feature rows (3,263 human UniProt rows plus the iRT standard); 3,262 mapped rows collapse to 3,261 HGNC genes; 2 unmapped.
- Metadata-matched samples: 423; primary arms: C9 = 25, Sporadic = 294, Control = 91.
- C9 vs Sporadic: n=319; HGNC background=3,261; EB d0=4.9847; background mean/median *t*²=1.955/1.030.
- C9 vs Control: n=116; HGNC background=3,261; EB d0=5.5097; background mean/median *t*²=1.323/0.611.
- Sporadic vs Control: n=385; HGNC background=3,261; EB d0=4.9219; background mean/median *t*²=0.993/0.436.

## Results

Each cell is `observed mean t² (null median / null 95th); raw empirical p; Bonferroni-8 pass`.

| Cluster | Term | n | C9 vs Sporadic | C9 vs Control | Sporadic vs Control |
|---|---|---:|---|---|---|
| Splicing | mRNA Splicing | 137 | 3.307 (1.945 / 2.299); p=0.00010; ✓ | 2.167 (1.314 / 1.570); p=0.00010; ✓ | 0.983 (0.987 / 1.198); p=0.51345; — |
| Splicing | Processing of Capped Intron-Containing Pre-mRNA | 183 | 3.540 (1.948 / 2.251); p=0.00010; ✓ | 2.418 (1.321 / 1.538); p=0.00010; ✓ | 0.962 (0.991 / 1.170); p=0.60894; — |
| Splicing | mRNA splicing, via spliceosome | 118 | 3.251 (1.950 / 2.330); p=0.00010; ✓ | 2.106 (1.319 / 1.596); p=0.00020; ✓ | 0.954 (0.990 / 1.219); p=0.60344; — |
| Chromatin | chromosome | 321 | 3.094 (1.956 / 2.175); p=0.00010; ✓ | 2.311 (1.321 / 1.479); p=0.00010; ✓ | 0.936 (0.991 / 1.127); p=0.76432; — |
| Chromatin | chromatin | 174 | 3.125 (1.952 / 2.256); p=0.00010; ✓ | 2.185 (1.320 / 1.544); p=0.00010; ✓ | 1.030 (0.991 / 1.174); p=0.35926; — |
| Transport | nucleocytoplasmic transport | 81 | 3.638 (1.941 / 2.410); p=0.00010; ✓ | 2.344 (1.311 / 1.658); p=0.00010; ✓ | 0.953 (0.989 / 1.272); p=0.59094; — |
| Transport | nuclear pore | 40 | 4.404 (1.931 / 2.615); p=0.00010; ✓ | 3.010 (1.303 / 1.804); p=0.00010; ✓ | 1.239 (0.975 / 1.403); p=0.14659; — |
| Transport | Vpr-mediated nuclear import of PICs | 26 | 5.097 (1.926 / 2.816); p=0.00010; ✓ | 3.832 (1.304 / 1.929); p=0.00010; ✓ | 0.949 (0.965 / 1.498); p=0.52405; — |

## Readout

The Bonferroni-8 pass pattern is **8/8/0** for C9-vs-Sporadic / C9-vs-Control / Sporadic-vs-Control.

The terms were discovered on this same cohort and then fixed for method transfer. The eightfold threshold handles arithmetic multiplicity across the reported rerun tests; it is not a post-selection FWER or selective-inference guarantee.

This graph-independent analysis controls pathway size and the heavy-tailed moderated-*t*² background within each contrast. It does not make the overlapping terms independent, remove the contrast sample-size imbalance, or license causal, mechanistic, individual-protein, or external-cohort claims.

## Reproduction

```bash
uv run --no-sync python scripts/run_size_matched_null.py
```

The default run is offline: the full term libraries, measured intersections, and UniProt-to-HGNC map are frozen in `data/publication/c9_size_matched_null_inputs.json`. Refreshing those inputs is a separate, explicit network operation via `--refresh-frozen-inputs`.
