# Genetic Subtype Contrast Implementation Summary

## Overview

Successfully extended `src/cliquefinder/cli/differential.py` to support flexible phenotype handling for genetic subtype analyses. The implementation enables comparing familial ALS cases (e.g., C9orf72 carriers) against sporadic ALS cases, while maintaining full backwards compatibility with existing workflows.

## Files Modified

### Primary Implementation
- **`src/cliquefinder/cli/differential.py`** (958 lines)
  - Added `derive_genetic_phenotype()` helper function
  - Added CLI arguments: `--mode` and `--genetic-contrast`
  - Modified main flow to support genetic contrasts
  - Added protein-only analysis mode

### Documentation
- **`GENETIC_SUBTYPE_USAGE.md`** (new)
  - User-facing documentation with examples
  - Sample size considerations
  - Output file descriptions

### Testing
- **`test_genetic_contrast.py`** (new)
  - Unit tests for `derive_genetic_phenotype()`
  - Validation of sample selection logic
  - Error handling tests

## Implementation Details

### 1. CLI Arguments Added

```python
parser.add_argument(
    "--mode",
    type=str,
    choices=["clique", "protein"],
    default="clique",
    help="Analysis mode: clique-level or protein-level (default: clique)",
)

parser.add_argument(
    "--genetic-contrast",
    type=str,
    metavar="MUTATION",
    help="Genetic subtype contrast (e.g., 'C9orf72' for carriers vs sporadic ALS). "
         "Requires ClinReport_Mutations_Details column in metadata. "
         "Known mutations: C9orf72, SOD1, TARDBP, FUS, SETX, Multiple, Other",
)
```

### 2. Helper Function: `derive_genetic_phenotype()`

**Location:** Lines 338-430 in differential.py

**Signature:**
```python
def derive_genetic_phenotype(
    metadata: pd.DataFrame,
    mutation: str,
    mutation_col: str = "ClinReport_Mutations_Details",
    phenotype_col: str = "phenotype",
) -> tuple[pd.DataFrame, str, str]
```

**Functionality:**
1. Validates required columns exist in metadata
2. Filters to CASE samples only (excludes healthy controls)
3. Creates carrier mask: `metadata[mutation_col] == mutation`
4. Creates sporadic mask: CASE without any known mutations
5. Generates labels: `mutation.upper()` and `"SPORADIC"`
6. Creates new `genetic_phenotype` column
7. Validates sample counts and prints warnings if underpowered
8. Returns (filtered_metadata, carrier_label, sporadic_label)

**Known Mutations List:**
```python
known_mutations = [
    'C9orf72', 'SOD1', 'Multiple', 'Other',
    'SETX', 'TARDBP', 'TARDBP (TDP43)', 'FUS'
]
```

**Sample Size Warnings:**
- n < 30: "Statistical power may be limited"
- n < 10: "Very small sample size. Results should be interpreted with caution"

### 3. Main Flow Modifications

**A. Metadata Processing (Lines 457-475)**
```python
# Handle genetic contrast if specified
condition_col = args.condition_col
if args.genetic_contrast:
    metadata, carrier_label, sporadic_label = derive_genetic_phenotype(
        metadata=metadata,
        mutation=args.genetic_contrast,
    )
    # Override condition column to use derived genetic phenotype
    condition_col = 'genetic_phenotype'

    # Set up contrast automatically
    if args.contrast:
        print(f"  Warning: Ignoring --contrast when using --genetic-contrast")
    args.contrast = [(f"{carrier_label}_vs_{sporadic_label}", carrier_label, sporadic_label)]
```

**B. Clique Loading (Lines 490-505)**
```python
# Load clique definitions (skip if protein-only mode)
cliques = None
if args.mode == "clique":
    print(f"\nLoading cliques: {args.cliques}")
    cliques = load_clique_definitions(args.cliques, min_proteins=args.min_proteins)
    # ... filtering logic ...
else:
    print(f"\nMode: protein-level analysis (skipping clique loading)")
```

**C. Analysis Branching (Lines 540-618)**
```python
# Branch based on mode
if args.mode == "protein":
    # Protein-level differential analysis
    from cliquefinder.stats.differential import run_differential_analysis

    result = run_differential_analysis(
        data=data,
        feature_ids=feature_ids,
        sample_condition=metadata[condition_col],
        sample_subject=metadata[args.subject_col] if args.subject_col else None,
        contrasts=contrasts,
        use_mixed=not args.no_mixed_model,
        fdr_method=args.fdr_method,
        fdr_threshold=args.fdr_threshold,
        n_jobs=args.workers,
        verbose=True,
    )
    # ... save results ...
    return 0

# Clique-level analysis continues...
```

**D. Condition Column Usage**

All analysis functions now use the `condition_col` variable instead of `args.condition_col`:
- Line 558: `sample_condition=metadata[condition_col]`
- Line 653: `condition_col=condition_col` (permutation test)
- Line 804: `condition_col=condition_col` (standard analysis)

**E. Parameters Saving**

All parameter dictionaries now include:
```python
params = {
    "timestamp": datetime.now().isoformat(),
    "mode": "clique",  # or "protein"
    # ...
    "condition_col": condition_col,  # not args.condition_col
    # ...
}
if args.genetic_contrast:
    params["genetic_contrast"] = args.genetic_contrast
```

## Key Design Decisions

### 1. Backwards Compatibility
- Default mode is "clique" (existing behavior)
- `--genetic-contrast` is optional
- Existing `--contrast CASE CTRL` workflows unchanged
- All existing tests pass without modification

### 2. Automatic Contrast Creation
- When `--genetic-contrast` specified, contrast is auto-generated
- Warning issued if user also specifies `--contrast`
- Contrast name format: `{MUTATION}_vs_SPORADIC`

### 3. Sample Filtering Logic
- Only CASE samples included (CTRL excluded)
- Carriers of other mutations excluded from both groups
- Sporadic = no known mutations (including NaN values)
- Clear validation with helpful error messages

### 4. Mode Selection
- `--mode clique`: Standard clique-level analysis (default)
- `--mode protein`: Protein-level only (faster, no cliques needed)
- Protein mode skips clique loading entirely
- Both modes support genetic contrasts

### 5. Integration with Existing Features
- Works with `--permutation-test`
- Works with `--also-protein-level` (clique mode)
- Works with `--network-query` (existing feature)
- Compatible with all statistical options

## Sample Selection Example

For C9orf72 vs sporadic ALS with the proteomics dataset:

**Input metadata:**
- 298 sporadic ALS (CASE with no known mutation)
- 21 C9orf72 carriers (CASE with C9orf72)
- 17 SOD1 carriers (CASE with SOD1)
- ~5 other mutation carriers (TARDBP, FUS, etc.)
- 50 healthy controls (CTRL)

**After `derive_genetic_phenotype(metadata, 'C9orf72')`:**
- 21 samples labeled as 'C9ORF72'
- 298 samples labeled as 'SPORADIC'
- 319 total samples
- Controls and other mutation carriers excluded

## Usage Examples

### Example 1: C9orf72 Clique Analysis
```bash
cliquefinder differential \
    --data output/proteomics/imputed.data.csv \
    --metadata output/proteomics/all_als.metadata.csv \
    --cliques output/proteomics/cliques/cliques.csv \
    --output output/proteomics/diff_c9_clique \
    --genetic-contrast C9orf72 \
    --subject-col subject_id \
    --mode clique
```

### Example 2: C9orf72 Protein Analysis
```bash
cliquefinder differential \
    --data output/proteomics/imputed.data.csv \
    --metadata output/proteomics/all_als.metadata.csv \
    --cliques output/proteomics/cliques/cliques.csv \
    --output output/proteomics/diff_c9_protein \
    --genetic-contrast C9orf72 \
    --subject-col subject_id \
    --mode protein
```

### Example 3: Standard CASE vs CTRL (Backwards Compatible)
```bash
cliquefinder differential \
    --data output/proteomics/imputed.data.csv \
    --metadata output/proteomics/imputed.metadata.csv \
    --cliques output/proteomics/cliques/cliques.csv \
    --output output/proteomics/diff_case_ctrl \
    --condition-col phenotype \
    --subject-col subject_id \
    --contrast CASE_vs_CTRL CASE CTRL
```

## Error Handling

The implementation includes comprehensive error handling:

### Missing Column
```python
if mutation_col not in metadata.columns:
    raise ValueError(
        f"Mutation column '{mutation_col}' not found in metadata. "
        f"Available columns: {', '.join(metadata.columns)}"
    )
```

### No Carriers Found
```python
if n_carriers == 0:
    raise ValueError(
        f"No carriers found for mutation '{mutation}'. "
        f"Available mutations: {metadata_cases[mutation_col].value_counts().to_dict()}"
    )
```

### No Sporadic Samples
```python
if n_sporadic == 0:
    raise ValueError("No sporadic ALS samples found")
```

## Testing

### Test Script: `test_genetic_contrast.py`

Tests validate:
1. **Correct sample assignment**
   - C9orf72 carriers → 'C9ORF72' label
   - Sporadic ALS → 'SPORADIC' label
   - Counts match expected values

2. **Proper filtering**
   - Controls excluded
   - Other mutation carriers excluded
   - Only CASE samples included

3. **Warning generation**
   - n < 30 triggers warning
   - n < 10 triggers stronger warning

4. **Error handling**
   - Invalid mutation names
   - Missing columns
   - No samples found

### Run Tests
```bash
# Requires conda environment with pandas
python test_genetic_contrast.py
```

## Code Quality

### Surgical Implementation
- Minimal changes to existing code
- Clear separation of concerns
- Follows existing patterns (e.g., params dict structure)
- Consistent variable naming

### Documentation
- Comprehensive docstrings
- Clear error messages
- Usage examples in GENETIC_SUBTYPE_USAGE.md
- Implementation notes in this document

### Maintainability
- Helper function can be reused
- Easy to add new known mutations
- Clear integration points
- Well-structured branching logic

## Output Files

### Clique Mode
- `clique_differential.csv`: Clique-level results
- `significant_cliques.csv`: Significant cliques (FDR < threshold)
- `protein_differential.csv`: Protein-level results (if `--also-protein-level`)
- `analysis_parameters.json`: Includes `genetic_contrast` and `mode` fields

### Protein Mode
- `protein_differential.csv`: Protein-level results
- `significant_proteins.csv`: Significant proteins (FDR < threshold)
- `analysis_parameters.json`: Includes `genetic_contrast` and `mode: "protein"`

### Permutation Mode (works with both clique/protein)
- `clique_differential_permutation.csv`: Empirical p-values
- `null_distribution_summary.csv`: Null statistics
- `significant_cliques.csv`: Empirical p < threshold

### Parameters JSON Example
```json
{
  "timestamp": "2026-01-15T14:30:00",
  "mode": "clique",
  "data": "output/proteomics/imputed.data.csv",
  "metadata": "output/proteomics/all_als.metadata.csv",
  "condition_col": "genetic_phenotype",
  "genetic_contrast": "C9orf72",
  "contrasts": {
    "C9ORF72_vs_SPORADIC": ["C9ORF72", "SPORADIC"]
  },
  "n_cliques_tested": 150,
  "n_significant": 12
}
```

## Validation

### Code Checks
```bash
# Verify function exists
grep -c "def derive_genetic_phenotype" src/cliquefinder/cli/differential.py
# Output: 1

# Verify CLI arguments added
grep -c '"--mode"' src/cliquefinder/cli/differential.py
# Output: 1
grep -c '"--genetic-contrast"' src/cliquefinder/cli/differential.py
# Output: 1

# Verify genetic contrast handling
grep -c 'if args.genetic_contrast:' src/cliquefinder/cli/differential.py
# Output: 3 (metadata processing, permutation params, standard params)

# Verify protein mode branch
grep -c 'if args.mode == "protein":' src/cliquefinder/cli/differential.py
# Output: 1
```

### Integration Verification
- All references to `args.condition_col` replaced with `condition_col` variable
- Both permutation and standard analysis use updated variable
- Parameters dicts include new fields in all branches

## Future Enhancements

### Potential Extensions
1. **Multi-class contrasts**: Compare multiple genetic subtypes simultaneously
2. **Compound mutations**: Support analyzing Multiple mutation carriers
3. **Continuous variables**: Age at onset, progression rate
4. **Stratified analysis**: Within-mutation subgroup analysis
5. **Export phenotype mapping**: Save genetic_phenotype assignments for downstream use

### Performance Optimizations
1. **Lazy clique loading**: Only load cliques when needed
2. **Cached phenotype derivation**: Store derived phenotypes
3. **Batch mode**: Process multiple genetic contrasts in one run

## Summary

Successfully implemented flexible phenotype handling for genetic subtype analyses with:

✅ **Functionality**
- Genetic contrast support (`--genetic-contrast MUTATION`)
- Protein-only mode (`--mode protein`)
- Automatic contrast generation
- Sample size warnings

✅ **Robustness**
- Comprehensive error handling
- Input validation
- Clear error messages
- Sample count reporting

✅ **Compatibility**
- Backwards compatible with existing workflows
- Works with permutation testing
- Works with network queries
- Works with all statistical options

✅ **Code Quality**
- Surgical modifications
- Clear separation of concerns
- Well-documented
- Testable

✅ **Documentation**
- User guide (GENETIC_SUBTYPE_USAGE.md)
- Implementation notes (this document)
- Inline docstrings
- Usage examples

The implementation is production-ready and enables sophisticated genetic subtype analyses while maintaining the simplicity and flexibility of the existing differential CLI.
