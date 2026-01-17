# Genetic Subtype Contrast Analysis

## Overview

The differential CLI now supports flexible genetic subtype analyses through the `--genetic-contrast` flag. This allows comparing familial ALS cases (e.g., C9orf72 carriers) against sporadic ALS cases.

## New Features

### 1. CLI Arguments

- `--mode {clique,protein}`: Analysis mode (default: clique)
  - `clique`: Standard clique-level analysis (backwards compatible)
  - `protein`: Protein-level differential analysis only

- `--genetic-contrast MUTATION`: Genetic subtype contrast
  - Example: `--genetic-contrast C9orf72`
  - Automatically creates binary phenotype: mutation carriers vs sporadic ALS
  - Filters to CASE samples only (excludes healthy controls)
  - Known mutations: C9orf72, SOD1, TARDBP, FUS, SETX, Multiple, Other

### 2. Sample Selection Logic

The `derive_genetic_phenotype()` function implements:

```python
# C9orf72 familial ALS carriers
c9_mask = (metadata['phenotype'] == 'CASE') &
          (metadata['ClinReport_Mutations_Details'] == 'C9orf72')

# Sporadic ALS (no known mutation)
known_mutations = ['C9orf72', 'SOD1', 'Multiple', 'Other', 'SETX', 'TARDBP (TDP43)', 'FUS']
sals_mask = (metadata['phenotype'] == 'CASE') &
            (~metadata['ClinReport_Mutations_Details'].isin(known_mutations) |
             metadata['ClinReport_Mutations_Details'].isna())
```

## Usage Examples

### Example 1: C9orf72 vs Sporadic ALS (Clique-Level)

```bash
cliquefinder differential \
    --data output/proteomics/imputed.data.csv \
    --metadata output/proteomics/all_als.metadata.csv \
    --cliques output/proteomics/cliques/cliques.csv \
    --output output/proteomics/differential_c9 \
    --genetic-contrast C9orf72 \
    --subject-col subject_id \
    --mode clique \
    --fdr-threshold 0.05
```

**Output:**
- Creates contrast: `C9ORF72_vs_SPORADIC`
- Reports sample counts for each group
- Warns if underpowered (n < 30)
- Saves `genetic_contrast: C9orf72` in parameters JSON

### Example 2: SOD1 vs Sporadic ALS (Protein-Level)

```bash
cliquefinder differential \
    --data output/proteomics/imputed.data.csv \
    --metadata output/proteomics/all_als.metadata.csv \
    --cliques output/proteomics/cliques/cliques.csv \
    --output output/proteomics/differential_sod1_proteins \
    --genetic-contrast SOD1 \
    --subject-col subject_id \
    --mode protein \
    --fdr-threshold 0.05
```

**Output:**
- Skips clique loading (protein-only mode)
- Runs MSstats-style protein-level differential analysis
- Faster for exploratory analysis or when cliques not needed

### Example 3: Standard CASE vs CTRL (Backwards Compatible)

```bash
cliquefinder differential \
    --data output/proteomics/imputed.data.csv \
    --metadata output/proteomics/imputed.metadata.csv \
    --cliques output/proteomics/cliques/cliques.csv \
    --output output/proteomics/differential \
    --condition-col phenotype \
    --subject-col subject_id \
    --contrast CASE_vs_CTRL CASE CTRL
```

**Backwards compatible** - no changes to existing workflows.

### Example 4: Permutation Testing with Genetic Contrast

```bash
cliquefinder differential \
    --data output/proteomics/imputed.data.csv \
    --metadata output/proteomics/all_als.metadata.csv \
    --cliques output/proteomics/cliques/cliques.csv \
    --output output/proteomics/differential_c9_perm \
    --genetic-contrast C9orf72 \
    --subject-col subject_id \
    --permutation-test \
    --n-permutations 1000 \
    --gpu
```

**Uses competitive permutation testing** instead of standard FDR correction.

## Sample Size Considerations

The implementation includes automatic warnings for underpowered analyses:

- **WARNING** if either group has n < 30: "Statistical power may be limited"
- **WARNING** if either group has n < 10: "Results should be interpreted with caution"

### Expected Sample Sizes (ALS proteomics dataset)

Based on `output/proteomics/all_als.metadata.csv`:

| Mutation | Carriers | Sporadic ALS | Total | Power |
|----------|----------|--------------|-------|-------|
| C9orf72  | 21       | 298          | 319   | Limited |
| SOD1     | 17       | 298          | 315   | Limited |
| TARDBP   | ~5       | 298          | ~303  | Very limited |
| FUS      | ~3       | 298          | ~301  | Very limited |

## Implementation Details

### Function: `derive_genetic_phenotype()`

```python
def derive_genetic_phenotype(
    metadata: pd.DataFrame,
    mutation: str,
    mutation_col: str = "ClinReport_Mutations_Details",
    phenotype_col: str = "phenotype",
) -> tuple[pd.DataFrame, str, str]:
    """
    Derive binary genetic phenotype from mutation data.

    Returns:
        Tuple of (filtered_metadata, carrier_label, sporadic_label)
    """
```

**Logic:**
1. Filter to CASE samples only
2. Identify mutation carriers
3. Identify sporadic cases (no known mutations)
4. Create `genetic_phenotype` column
5. Validate sample counts
6. Print warnings if underpowered

### Integration Points

The implementation modifies the differential CLI flow:

1. **After metadata loading**: Call `derive_genetic_phenotype()` if `--genetic-contrast` specified
2. **Condition column**: Override to use `genetic_phenotype` instead of original `phenotype`
3. **Auto-contrast**: Create contrast name like `C9ORF72_vs_SPORADIC`
4. **Mode branching**: Split between protein-level and clique-level analysis
5. **Parameters**: Save `genetic_contrast` and `mode` in JSON output

## Output Files

### Clique Mode
- `clique_differential.csv`: Clique-level results
- `significant_cliques.csv`: Significant cliques (if any)
- `protein_differential.csv`: Protein-level results (if `--also-protein-level`)
- `analysis_parameters.json`: Metadata including `genetic_contrast` field

### Protein Mode
- `protein_differential.csv`: Protein-level results
- `significant_proteins.csv`: Significant proteins (if any)
- `analysis_parameters.json`: Metadata including `mode: "protein"`

### Permutation Mode
- `clique_differential_permutation.csv`: Permutation test results
- `null_distribution_summary.csv`: Null distribution statistics
- `significant_cliques.csv`: Significant cliques (if any)

## Error Handling

The implementation includes robust error handling:

- **Missing column**: Clear error if `ClinReport_Mutations_Details` not in metadata
- **No carriers**: Error with available mutations listed
- **No sporadic**: Error if no sporadic ALS samples found
- **Invalid mutation**: Error with mutation distribution shown

## Testing

Test the implementation with:

```bash
python test_genetic_contrast.py
```

Tests validate:
- Correct carrier/sporadic assignment
- Control exclusion
- Other mutation exclusion
- Sample count validation
- Warning generation for small samples
- Error handling for invalid mutations

## Notes

- The `--genetic-contrast` flag **overrides** any `--contrast` specifications
- Healthy controls (CTRL) are always excluded from genetic contrasts
- Carriers of other mutations are excluded from both groups
- The contrast direction is always: mutation carriers vs sporadic ALS
- For log2FC interpretation: positive = higher in carriers, negative = higher in sporadic
