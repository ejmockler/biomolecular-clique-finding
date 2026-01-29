#!/bin/bash
#
# AnswerALS Sporadic ALS Proteomics Imputation Example
# =====================================================
#
# This is an AnswerALS-specific example demonstrating the cliquefinder CLI
# for sporadic ALS proteomics imputation.
#
# Dataset: 357 samples from AnswerALS Cohort 1-6
#   - 170 Sporadic ALS (CASE): No C9orf72 expansion, no known mutations
#   - 187 Healthy Controls (CTRL)
#   - Excluded: 221 samples (C9orf72 carriers, familial ALS, other mutations)
#
# IMPORTANT: This script contains absolute paths that need customization.
# Update the following paths before running:
#   - PROJECT_DIR: Path to biomolecular-clique-finding repository
#   - Input data paths (aals_cohort1-6_counts_merged.csv)
#   - Metadata paths (data/sporadic_als_filtered_metadata.csv)
#
# Usage:
#   bash examples/als/run_sporadic_imputation.sh
#

set -e

# ============================================================================
# CONFIGURATION - CUSTOMIZE THESE PATHS
# ============================================================================

# TODO: Update this to your project directory
PROJECT_DIR="/Users/noot/Documents/biomolecular-clique-finding"

cd "$PROJECT_DIR"

echo "=== Sporadic ALS Proteomics Imputation ==="
echo "Date: $(date)"
echo ""

# ============================================================================
# IMPUTATION PIPELINE
# ============================================================================

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
