# Quick Start: Metadata Propagation for Cross-Modal Analysis

## TL;DR

RNA-seq samples now automatically inherit metadata (phenotype, Sex) from matched proteomics samples. This enables stratified expression filtering without requiring external metadata files.

## What Changed

**Before:**
```
WARNING: Missing stratification columns in RNA metadata. Skipping expression filter.
```

**After:**
```
Propagating metadata from proteomics to RNA samples...
Metadata propagation: 381/578 RNA samples matched to proteomics (65.9%)
Applying stratified expression filter...
Expression filter: 45000/60664 genes passed (74.2%)
```

## Running the Pipeline

### Basic Usage (Metadata Auto-Propagated)

```bash
cliquefinder analyze \
  --input proteomics_data.csv \
  --rna-filter rna_counts.csv \
  --output results/ \
  --stratify-by phenotype Sex \
  --discover \
  --workers 4
```

**What happens:**
1. Loads proteomics (436 samples with Sex metadata)
2. Loads RNA-seq (578 samples, no metadata)
3. **Matches samples by participant ID** (NEUAA295HHE, etc.)
4. **Propagates Sex from proteomics to matched RNA samples**
5. Performs stratified expression filtering (by phenotype + Sex)
6. Continues with clique discovery

### With External Metadata (Optional Override)

```bash
cliquefinder analyze \
  --input proteomics_data.csv \
  --rna-filter rna_counts.csv \
  --metadata clinical_registry.csv \
  --output results/ \
  --stratify-by phenotype Sex Age \
  --discover
```

**What happens:**
1. Propagates metadata from proteomics (baseline)
2. Loads external metadata from clinical_registry.csv
3. **Merges external with propagated** (external takes precedence)
4. Uses most complete metadata for filtering

### Phenotype-Only Analysis (No Sex Required)

```bash
cliquefinder analyze \
  --input proteomics_data.csv \
  --rna-filter rna_counts.csv \
  --output results/ \
  --stratify-by phenotype \
  --discover
```

**Advantages:**
- Uses all 578 RNA samples (no samples excluded for missing Sex)
- Simpler stratification (2 groups instead of 4)
- Faster computation

## Understanding Match Rates

### Expected Output

```
Propagating metadata from proteomics to RNA samples...
Metadata propagation: 381/578 RNA samples matched to proteomics (65.9%)
RNA metadata summary: 578 samples with phenotype, 381 samples with Sex
```

### What This Means

| Statistic | Value | Explanation |
|-----------|-------|-------------|
| Total RNA samples | 578 | All samples in RNA-seq dataset |
| Matched to proteomics | 381 | Samples with matching participant ID |
| Match rate | 65.9% | Percentage with both RNA and proteomics |
| Samples with phenotype | 578 | All samples (inferred from sample ID) |
| Samples with Sex | 381 | Only matched samples get Sex from proteomics |

### Why Not 100% Match?

**Valid reasons for unmatched samples:**
1. RNA-only participants (no proteomics sample collected)
2. Proteomics-only participants (no RNA sample)
3. Different time points (same participant, different collection dates)
4. QC failures in one modality but not the other

**Unmatched samples are NOT excluded** - they still participate with phenotype-only metadata.

## Troubleshooting

### Issue: Low Match Rate (<50%)

**Check:**
```bash
# Examine sample ID formats
head -n 1 proteomics_data.csv
head -n 1 rna_counts.csv
```

**Expected formats:**
- Proteomics: `CASE_NEUAA295HHE-9014-P_D3`
- RNA: `CASE-NEUAA295HHE-5310-T_P010`
- Common participant: `NEUAA295HHE`

**If formats differ:** Participant ID pattern may need adjustment (contact maintainer)

### Issue: No Stratification Groups Formed

**Error message:**
```
WARNING: Missing stratification columns in RNA metadata: ['Sex']. Skipping expression filter.
```

**Solution 1:** Check that proteomics has Sex metadata
```python
import pandas as pd
prot = pd.read_csv('proteomics_data.csv', index_col=0)
print(prot.columns)  # Should include 'Sex' or 'Sex_predicted'
```

**Solution 2:** Use phenotype-only stratification
```bash
--stratify-by phenotype
```

### Issue: Expression Filter Returns Too Few Genes

**Symptoms:**
```
Expression filter: 5000/60664 genes passed (8.2%)
```

**Common causes:**
1. `--min-cpm` too high (default: 1.0)
2. `--min-prevalence` too strict (default: 0.10)
3. Small group sizes after stratification

**Adjustments:**
```bash
# Relax thresholds
--min-cpm 0.5 \
--min-prevalence 0.05

# Or reduce stratification
--stratify-by phenotype
```

## Checking Metadata Quality

### Inspect Propagated Metadata

After running the pipeline, check the logs:

```
RNA matrix metadata columns: ['phenotype', 'Sex', 'participant_id', 'matched_proteomics_sample']
```

**Columns explained:**
- `phenotype`: CASE or CTRL (from RNA sample ID prefix)
- `Sex`: Female, Male, or NaN (from matched proteomics)
- `participant_id`: NEUAA295HHE, etc. (for provenance)
- `matched_proteomics_sample`: Source proteomics sample ID (for tracing)

### Verify Stratification Groups

Look for this in the output:

```
Applying stratified expression filter...
Stratify by: ['phenotype', 'Sex']
Group: phenotype=CASE, Sex=Female: 190 samples
Group: phenotype=CASE, Sex=Male: 191 samples
Group: phenotype=CTRL, Sex=Female: 90 samples
Group: phenotype=CTRL, Sex=Male: 90 samples
Excluded (missing metadata): 197 samples
```

**Good signs:**
- All groups have >10 samples (meets `--min-group-size` default)
- Balanced case/control ratio
- Expected sex distribution

**Warning signs:**
- Any group <10 samples → may skip that stratum
- Extreme imbalance → check sample collection bias

## Advanced: Custom Participant ID Pattern

If your sample IDs use a different format, you can modify the pattern in the code.

**Current pattern:** `r'(NEU[A-Z0-9]+)'`

**Example alternatives:**

| Sample Format | Pattern | Example Match |
|---------------|---------|---------------|
| `SUBJ_12345-T1` | `r'SUBJ_(\d+)'` | `12345` |
| `PT-ABC123-RNA` | `r'PT-([A-Z0-9]+)'` | `ABC123` |
| `CASE_XYZ789_S1` | `r'CASE_([A-Z0-9]+)_'` | `XYZ789` |

**To customize:** Edit line 283 in `src/cliquefinder/cli/analyze.py`:
```python
participant_pattern: re.Pattern = re.compile(r'YOUR_PATTERN_HERE')
```

## Integration with Existing Workflows

### Workflow 1: Proteomics Primary, RNA Validation

```bash
# Use proteomics for discovery, RNA for validation
cliquefinder analyze \
  --input proteomics.csv \
  --rna-filter rna.csv \
  --stratify-by phenotype Sex \
  --discover \
  --min-targets 10
```

**Pipeline behavior:**
- Proteomics: All 3,264 proteins used for clique finding
- RNA: Expression filter creates validation gene set (~45k genes)
- Cliques annotated with RNA validation status

### Workflow 2: Strict Sample Alignment

```bash
# Only analyze samples present in BOTH modalities
cliquefinder analyze \
  --input proteomics.csv \
  --rna-filter rna.csv \
  --require-sample-alignment \
  --stratify-by phenotype Sex
```

**Pipeline behavior:**
- Filters to 381 matched participants only
- Both modalities use same sample set
- Stronger statistical power for matched pairs

### Workflow 3: Phenotype-Only, High Sensitivity

```bash
# Maximum gene coverage, simple stratification
cliquefinder analyze \
  --input proteomics.csv \
  --rna-filter rna.csv \
  --stratify-by phenotype \
  --min-prevalence 0.05 \
  --min-cpm 0.5
```

**Pipeline behavior:**
- All 578 RNA samples used (no Sex exclusions)
- Relaxed thresholds → more genes pass (~55k)
- Good for exploratory analysis

## FAQ

**Q: Do I need to provide external metadata anymore?**
A: No, metadata propagation from proteomics is now automatic. External metadata is optional for enrichment.

**Q: What if my RNA samples don't match proteomics format?**
A: Contact the maintainer to adjust the participant ID extraction pattern. The default pattern `r'(NEU[A-Z0-9]+)'` works for Answer ALS cohorts.

**Q: Can I see which RNA samples matched?**
A: Yes, check the `matched_proteomics_sample` column in the RNA metadata. Non-null values indicate successful matches.

**Q: What happens to unmatched RNA samples?**
A: They remain in the analysis with phenotype inferred from sample ID. Sex is set to NaN. They're only excluded from stratification groups that require Sex.

**Q: Does this affect proteomics analysis?**
A: No, proteomics clique discovery uses all 436 samples regardless of RNA matches. RNA data only affects expression-based annotation.

**Q: How do I verify the propagation worked?**
A: Check the logs for "Metadata propagation: X/Y RNA samples matched" and "RNA metadata summary: Z samples with phenotype, W samples with Sex"

## Support

For issues or questions:
1. Check the detailed implementation guide: `METADATA_PROPAGATION_IMPLEMENTATION.md`
2. Review the data flow diagram: `METADATA_FLOW_DIAGRAM.md`
3. Open an issue with your log output and sample ID examples
