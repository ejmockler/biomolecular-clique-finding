# Metadata Propagation Implementation

## Problem Statement

The cross-modal integration pipeline was failing to perform stratified expression filtering on RNA-seq data because the RNA dataset lacked required metadata columns (`phenotype`, `Sex_predicted`). This metadata existed in the proteomics dataset but wasn't being transferred to the RNA samples.

### Error Message
```
WARNING: Missing stratification columns in RNA metadata. Skipping expression filter.
```

## Solution Overview

Implemented metadata propagation from proteomics to RNA samples by matching participants across modalities using the participant ID extraction pattern `r'(NEU[A-Z0-9]+)'`.

### Key Changes

**File Modified:** `/Users/noot/Documents/biomolecular-clique-finding/src/cliquefinder/cli/analyze.py`

**Location:** Lines 271-502 (replacing original lines 271-372)

## Implementation Details

### 1. New Helper Function: `propagate_metadata_to_rna()`

Located inside the RNA dataset loading section (lines 279-364), this function:

**Inputs:**
- `proteomics_sample_ids`: List of sample IDs from proteomics matrix
- `proteomics_metadata`: DataFrame with proteomics sample metadata
- `rna_sample_ids`: List of sample IDs from RNA matrix
- `participant_pattern`: Regex pattern for extracting participant IDs (default: `r'(NEU[A-Z0-9]+)'`)

**Process:**
1. Extract participant IDs from proteomics samples using regex pattern
2. Build mapping: `participant_id -> proteomics_sample_id` (keeps first match per participant)
3. For each RNA sample:
   - Extract participant ID
   - Find matching proteomics sample
   - If matched:
     - Extract `phenotype` from RNA sample ID prefix (CASE/CTRL)
     - Copy `Sex_predicted` or `Sex` from proteomics metadata
     - Track match statistics
   - If unmatched:
     - Infer `phenotype` from sample ID prefix
     - Set `Sex` to `pd.NA`

**Output:**
- DataFrame indexed by RNA sample IDs with columns:
  - `phenotype`: CASE or CTRL (from sample ID prefix)
  - `Sex`: Copied from proteomics metadata if available
  - `participant_id`: Extracted participant ID (for provenance)
  - `matched_proteomics_sample`: Source proteomics sample ID (for traceability)

### 2. Execution Flow

```
1. Load proteomics data with metadata
2. Load RNA data (with matrix if --skip-expression-filter not set)
3. **[NEW]** Propagate metadata from proteomics to RNA via participant ID matching
4. **[OPTIONAL]** Merge external metadata if --metadata provided (override/enrich)
5. Perform stratified expression filtering (now works with propagated metadata!)
6. Cross-modal ID mapping and sample alignment
7. Continue with clique discovery...
```

### 3. Sample ID Formats

**Proteomics:**
- Format: `CASE_NEUAA295HHE-9014-P_D3`
- Participant ID: `NEUAA295HHE` (extracted via regex)
- Phenotype: `CASE` (from prefix)

**RNA:**
- Format: `CASE-NEUAA295HHE-5310-T_P010`
- Participant ID: `NEUAA295HHE` (extracted via regex)
- Phenotype: `CASE` (from prefix)

**Matching Logic:**
- Both samples have participant `NEUAA295HHE`
- Match established: RNA sample inherits Sex from proteomics

## Matching Statistics

The implementation logs detailed statistics:

```python
logger.info(
    f"Metadata propagation: {n_matched}/{len(rna_sample_ids)} RNA samples matched to proteomics "
    f"({100*n_matched/len(rna_sample_ids):.1f}%)"
)

logger.info(
    f"RNA metadata summary: {n_with_phenotype} samples with phenotype, "
    f"{n_with_sex} samples with Sex"
)
```

**Expected Results:**
- Matched participants: ~381/416 (based on your description)
- All RNA samples: ~578
- Match rate: ~66% (381/578)
- All RNA samples get `phenotype` (from sample ID prefix)
- Matched RNA samples get `Sex` (from proteomics metadata)

## Compatibility with Existing Code

### 1. External Metadata Override (Lines 385-502)

The external metadata merging (if `--metadata` provided) now:
- Runs AFTER metadata propagation
- Uses `combine_first()` to override propagated values with external metadata where available
- Preserves all propagated values for samples without external metadata

### 2. Expression Filtering (Lines 504-566)

No changes required! The expression filter now:
- Finds `phenotype` column in RNA metadata (propagated from proteomics)
- Finds `Sex` column in RNA metadata (propagated from proteomics)
- Performs stratified filtering as designed

### 3. Sample Alignment (Lines 571-597)

Uses existing `SampleAlignedCrossModalMapper` - no changes needed.

## Graceful Degradation

The implementation handles edge cases:

1. **Unmatched RNA samples:**
   - Get `phenotype` inferred from sample ID prefix
   - Get `Sex = pd.NA`
   - Not excluded from analysis

2. **Missing Sex in proteomics:**
   - Checks both `Sex_predicted` and `Sex` columns
   - Falls back to `pd.NA` if neither exists

3. **Invalid participant IDs:**
   - Regex returns None
   - Sample treated as unmatched

4. **No proteomics data:**
   - Function not called if `matrix` is None
   - Pipeline continues with external metadata only

## Testing Recommendations

### Unit Test Example

```python
def test_metadata_propagation():
    # Setup
    prot_sample_ids = ['CASE_NEUAA295HHE-9014-P_D3', 'CTRL_W14C179-7929-P_B2']
    prot_metadata = pd.DataFrame({
        'Sex_predicted': ['Female', 'Male'],
        'Age': [45, 52]
    }, index=prot_sample_ids)

    rna_sample_ids = ['CASE-NEUAA295HHE-5310-T_P010', 'CASE-NEUBB123XYZ-1234-T_P001']

    # Execute
    result = propagate_metadata_to_rna(prot_sample_ids, prot_metadata, rna_sample_ids)

    # Assert
    assert result.loc['CASE-NEUAA295HHE-5310-T_P010', 'phenotype'] == 'CASE'
    assert result.loc['CASE-NEUAA295HHE-5310-T_P010', 'Sex'] == 'Female'
    assert result.loc['CASE-NEUBB123XYZ-1234-T_P001', 'phenotype'] == 'CASE'
    assert pd.isna(result.loc['CASE-NEUBB123XYZ-1234-T_P001', 'Sex'])
```

### Integration Test

Run the full pipeline with actual data:

```bash
cliquefinder analyze \
  --input proteomics_3264x436.csv \
  --rna-filter aals_cohort1-6_counts_merged.csv \
  --output results/test_metadata_propagation \
  --min-cpm 1.0 \
  --min-prevalence 0.10 \
  --stratify-by phenotype Sex \
  --discover \
  --workers 4
```

**Expected Log Output:**
```
Loading RNA filter: aals_cohort1-6_counts_merged.csv
RNA data: 60664 genes x 578 samples, type=ensembl_gene
Propagating metadata from proteomics to RNA samples...
Metadata propagation: 381/578 RNA samples matched to proteomics (65.9%)
RNA metadata summary: 578 samples with phenotype, 381 samples with Sex
RNA matrix metadata columns: ['phenotype', 'Sex', 'participant_id', 'matched_proteomics_sample']
Applying stratified expression filter...
Expression filter: 45000/60664 genes passed (74.2%)
```

## Benefits

1. **Enables stratified expression filtering:** RNA samples now have required metadata
2. **Preserves provenance:** Tracks which proteomics sample provided each metadata value
3. **Flexible matching:** Uses robust regex pattern that works across format variations
4. **Clear logging:** Reports exact match statistics for QC
5. **Graceful handling:** Unmatched samples still usable with inferred phenotype
6. **Compatible:** Works seamlessly with existing external metadata pipeline

## Future Enhancements

1. **Alternative participant ID patterns:**
   - Make pattern configurable via CLI argument
   - Support multiple patterns for different cohorts

2. **Metadata confidence scores:**
   - Track propagation method (matched vs inferred)
   - Add quality flags for metadata provenance

3. **Validation checks:**
   - Compare inferred phenotype with propagated phenotype
   - Flag mismatches for manual review

4. **Metadata visualization:**
   - Generate QC plots showing match rates by cohort
   - Highlight samples with incomplete metadata
