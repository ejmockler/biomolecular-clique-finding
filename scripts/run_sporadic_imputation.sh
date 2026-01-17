#!/bin/bash
# Sporadic ALS Proteomics Imputation Pipeline
#
# Dataset: 357 samples
#   - 170 Sporadic ALS (CASE): No C9orf72 expansion, no known mutations
#   - 187 Healthy Controls (CTRL)
#
# Excluded: 221 samples (C9orf72 carriers, familial ALS, other mutations)

set -e

cd /Users/noot/Documents/biomolecular-clique-finding

echo "=== Sporadic ALS Proteomics Imputation ==="
echo "Date: $(date)"
echo ""

.venv/bin/python3 -m cliquefinder impute \
    --input aals_cohort1-6_counts_merged.csv \
    --output output/proteomics/sporadic \
    --metadata data/sporadic_als_filtered_metadata.csv \
    --phenotype-filter CASE CTRL \
    --phenotype-col phenotype \
    --method mad-z \
    --threshold 7.5 \
    --upper-only \
    --mode within_group \
    --group-cols phenotype \
    --impute-strategy soft-clip \
    --no-residual-pass \
    --no-global-cap \
    --log-transform \
    --classify-sex \
    --sex-labels-col Sex \
    --sex-within-phenotype

echo ""
echo "=== Imputation Complete ==="
echo "Output: output/proteomics/sporadic.*"
