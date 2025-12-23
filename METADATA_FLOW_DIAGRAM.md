# Metadata Propagation Data Flow

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA SOURCES                              │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐         ┌──────────────────────────┐
│   Proteomics Matrix      │         │    RNA-seq Matrix        │
│   3,264 × 436            │         │   60,664 × 578           │
├──────────────────────────┤         ├──────────────────────────┤
│ Sample IDs:              │         │ Sample IDs:              │
│ CASE_NEUAA295HHE-9014-P  │         │ CASE-NEUAA295HHE-5310-T  │
│ CTRL_W14C179-7929-P_B2   │         │ CTRL-W14C179-8234-T      │
│                          │         │                          │
│ Metadata:                │         │ Metadata:                │
│ ✓ phenotype (from ID)    │         │ ✗ phenotype (missing)    │
│ ✓ Sex_predicted          │         │ ✗ Sex (missing)          │
│ ✓ Age, Site, etc.        │         │                          │
└──────────────────────────┘         └──────────────────────────┘
         │                                      │
         │                                      │
         └──────────────┬───────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    METADATA PROPAGATION PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: Extract Participant IDs
────────────────────────────────
Pattern: r'(NEU[A-Z0-9]+)'

Proteomics:                      RNA:
CASE_NEUAA295HHE-9014-P    →    NEUAA295HHE    ←    CASE-NEUAA295HHE-5310-T
CTRL_W14C179-7929-P_B2     →    W14C179        ←    CTRL-W14C179-8234-T

Step 2: Build Participant → Sample Mapping
───────────────────────────────────────────
{
  'NEUAA295HHE': 'CASE_NEUAA295HHE-9014-P_D3',
  'W14C179': 'CTRL_W14C179-7929-P_B2'
}

Step 3: Match RNA Samples to Proteomics
────────────────────────────────────────
For each RNA sample:
  1. Extract participant ID
  2. Look up in proteomics mapping
  3. If found: propagate metadata
  4. If not found: infer phenotype from ID prefix

Step 4: Create RNA Metadata DataFrame
──────────────────────────────────────
Index: RNA sample IDs
Columns:
  - phenotype: 'CASE' or 'CTRL' (from RNA sample ID prefix)
  - Sex: Copied from matched proteomics sample (or pd.NA)
  - participant_id: Extracted participant ID (provenance)
  - matched_proteomics_sample: Source sample ID (traceability)

                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      ENRICHED RNA DATASET                               │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ RNA Matrix with Propagated Metadata                                 │
├──────────────────────────────────────────────────────────────────────┤
│ Sample ID                         phenotype    Sex       matched     │
├──────────────────────────────────────────────────────────────────────┤
│ CASE-NEUAA295HHE-5310-T           CASE        Female    ✓           │
│ CTRL-W14C179-8234-T               CTRL        Male      ✓           │
│ CASE-NEUBB123XYZ-1234-T           CASE        NaN       ✗           │
│ ...                               ...         ...       ...          │
└──────────────────────────────────────────────────────────────────────┘
       │
       │  Stats:
       │  - 578 total RNA samples
       │  - 381 matched to proteomics (65.9%)
       │  - 578 with phenotype (100%)
       │  - 381 with Sex (65.9%)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               STRATIFIED EXPRESSION FILTERING                           │
└─────────────────────────────────────────────────────────────────────────┘

Stratify by: ['phenotype', 'Sex']

Groups:
  - CASE + Female:  ~190 samples  ✓ min_group_size=10
  - CASE + Male:    ~191 samples  ✓
  - CTRL + Female:  ~90 samples   ✓
  - CTRL + Male:    ~90 samples   ✓
  - Missing Sex:    ~197 samples  ✗ (excluded from stratification)

For each gene:
  Count samples with CPM >= 1.0 in each group
  If >= 10% of group has expression:
    Gene passes in that group
  If passes in ANY group:
    Gene included in final set

       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXPRESSION-VALIDATED GENES                           │
└─────────────────────────────────────────────────────────────────────────┘

~45,000 / 60,664 genes pass filter (74.2%)

These genes:
  - Expressed above threshold in at least one stratum
  - Biologically relevant in at least one phenotype/sex group
  - Used to annotate proteomics cliques (not filter proteomics)

       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 CROSS-MODAL INTEGRATION & CLIQUES                       │
└─────────────────────────────────────────────────────────────────────────┘

Proteomics genes (3,264) analyzed for cliques
RNA validation set (45,000) used for annotation
Sample alignment (381 matched participants) tracked
```

## Key Design Decisions

### 1. Phenotype Source: RNA Sample ID (Not Proteomics)

**Rationale:** RNA sample IDs use standardized format `CASE-NEU...` or `CTRL-NEU...`
- More reliable than proteomics format which varies (`CASE_` vs `CASE-`)
- Ensures consistency across entire RNA dataset

### 2. Sex Source: Proteomics Metadata

**Rationale:** Proteomics has curated `Sex_predicted` column from prior analysis
- Biological sex determination requires marker analysis
- Proteomics already performed this using protein-based classifiers
- Propagating saves redundant computation

### 3. Fallback to Inferred Values

**Rationale:** Enable analysis even with incomplete matching
- Unmatched RNA samples still get phenotype (from sample ID)
- Can participate in expression filtering with phenotype-only stratification
- User can add `--stratify-by phenotype` to avoid Sex requirement

### 4. Metadata Before External Override

**Rationale:** Two-stage enrichment enables flexible workflows
1. **Stage 1 (Propagation):** Use proteomics as baseline
2. **Stage 2 (External):** Override with authoritative clinical registry

## Example Scenarios

### Scenario 1: Standard Analysis (381 Matched)

```python
# Pipeline receives:
proteomics: 436 samples with Sex
RNA: 578 samples without Sex

# After propagation:
RNA: 578 samples
  - 381 with Sex (matched to proteomics)
  - 197 with Sex=NaN (no proteomics match)

# Expression filter:
Stratify by: ['phenotype', 'Sex']
Groups formed: 4 (CASE/CTRL × Female/Male)
Samples excluded: 197 (missing Sex)
Genes passing: ~45,000
```

### Scenario 2: Phenotype-Only Stratification

```python
# User runs:
--stratify-by phenotype

# Expression filter:
Stratify by: ['phenotype']
Groups formed: 2 (CASE, CTRL)
Samples excluded: 0
Genes passing: ~48,000
```

### Scenario 3: External Metadata Override

```python
# User provides:
--metadata clinical_registry.csv

# Pipeline flow:
1. Propagate from proteomics (381 matched)
2. Load external metadata (450 matched)
3. Merge: external overrides propagated
4. Final: 450 with authoritative Sex, 128 with proteomics Sex

# Expression filter:
Uses most complete metadata available
```

## Validation Checkpoints

1. **Post-Propagation:**
   - Check: All RNA samples have phenotype
   - Check: Match rate ~65-70% (381/578)
   - Check: No duplicates in metadata index

2. **Post-External Merge:**
   - Check: External metadata didn't remove propagated values
   - Check: Conflicts resolved (external wins)
   - Check: Metadata columns aligned with stratify_by

3. **Post-Expression Filter:**
   - Check: Groups formed match expected strata
   - Check: No empty groups
   - Check: Excluded samples logged

## Error Handling

| Issue | Detection | Resolution |
|-------|-----------|------------|
| No matches | `n_matched == 0` | Infer phenotype for all, continue with phenotype-only |
| Invalid pattern | `participant_pattern.search() returns None` | Sample treated as unmatched |
| Missing Sex column | Check `'Sex_predicted' in proteomics_metadata` | Fall back to `pd.NA` |
| Duplicate participants | First match kept | Log warning with count |
| Empty stratification groups | Check `group_size >= min_group_size` | Skip that group, warn user |
