# Configuration Usage Examples

## Quick Start

### 1. Basic Config Usage

```bash
# Use full config - all parameters from file
cliquefinder impute --config configs/proteomics_als.yaml
```

### 2. Config with CLI Overrides

```bash
# Use config but experiment with different threshold
cliquefinder impute --config configs/proteomics_als.yaml --threshold 2.0

# Use config but change output location
cliquefinder impute --config configs/proteomics_als.yaml --output results/experiment_v2

# Use config but switch to different method
cliquefinder impute --config configs/proteomics_als.yaml --method mad-z
```

### 3. Minimal Config

```bash
# Just specify input/output in config, use defaults for everything else
cliquefinder impute --config configs/minimal_example.yaml
```

## Common Workflows

### Research Workflow: Base Config + Experiments

Create a base configuration for your project:

```yaml
# configs/project_base.yaml
input: "data/cohort_proteomics.csv"
output: "results/imputed"

detection:
  method: "adjusted-boxplot"
  stratify_by: ["phenotype"]

phenotype:
  source_col: "SUBJECT_GROUP"
  case_values: ["ALS"]
  ctrl_values: ["Healthy Control"]
```

Then run experiments with different thresholds:

```bash
# Conservative threshold
cliquefinder impute --config configs/project_base.yaml \
  --threshold 2.0 \
  --output results/conservative

# Standard threshold
cliquefinder impute --config configs/project_base.yaml \
  --threshold 1.5 \
  --output results/standard

# Aggressive threshold
cliquefinder impute --config configs/project_base.yaml \
  --threshold 1.0 \
  --output results/aggressive
```

### Production Workflow: Locked Configuration

For reproducible production pipelines, lock all parameters in config:

```yaml
# configs/production_v1.yaml
input: "data/production_batch_2024_12.csv"
output: "results/production/batch_2024_12"

detection:
  method: "adjusted-boxplot"
  threshold: 1.5
  stratify_by: ["phenotype"]

imputation:
  strategy: "soft-clip"
  sharpness: 5.0

phenotype:
  source_col: "SUBJECT_GROUP"
  case_values: ["ALS", "ALS-FTD"]
  ctrl_values: ["Healthy Control"]
```

Run with no CLI overrides:

```bash
cliquefinder impute --config configs/production_v1.yaml
```

Version control the config file for reproducibility.

### Development Workflow: Interactive Exploration

Start with minimal config, add parameters as needed:

```bash
# Step 1: Test basic loading
cliquefinder impute \
  --config configs/minimal_example.yaml \
  --input data/test_sample.csv

# Step 2: Add method selection
cliquefinder impute \
  --config configs/minimal_example.yaml \
  --method adjusted-boxplot \
  --threshold 1.5

# Step 3: Add phenotype filtering
cliquefinder impute \
  --config configs/minimal_example.yaml \
  --method adjusted-boxplot \
  --phenotype-filter CASE

# Step 4: Save working parameters to new config
# (manually create config from working CLI args)
```

## Parameter Override Examples

### Override Detection Parameters

```bash
# Change detection method
cliquefinder impute --config pipeline.yaml --method iqr

# Change threshold
cliquefinder impute --config pipeline.yaml --threshold 2.5

# Add sex stratification
cliquefinder impute --config pipeline.yaml --group-cols phenotype Sex
```

### Override Imputation Parameters

```bash
# Switch to median imputation
cliquefinder impute --config pipeline.yaml --impute-strategy median

# Adjust soft-clip sharpness
cliquefinder impute --config pipeline.yaml --soft-clip-sharpness 10.0
```

### Override Phenotype Parameters

```bash
# Change case values
cliquefinder impute --config pipeline.yaml --case-values ALS ALS-FTD PLS

# Change control values
cliquefinder impute --config pipeline.yaml --ctrl-values "Healthy Control" HC
```

### Add Features Not in Config

Config files don't support all CLI features yet. You can add these via CLI:

```bash
# Enable sex classification
cliquefinder impute --config pipeline.yaml \
  --classify-sex \
  --sex-labels-col SEX

# Add phenotype filtering
cliquefinder impute --config pipeline.yaml \
  --phenotype-filter CASE

# Disable log transformation
cliquefinder impute --config pipeline.yaml \
  --no-log-transform

# Add clinical metadata
cliquefinder impute --config pipeline.yaml \
  --clinical-metadata data/clinical.csv \
  --clinical-id-col Participant_ID
```

## Config File Management

### Organizing Configs

```
configs/
├── README.md
├── minimal_example.yaml          # Minimal template
├── proteomics_als.yaml           # Recommended defaults
├── production_v1.yaml            # Production locked
├── experiments/
│   ├── threshold_sweep.yaml
│   ├── method_comparison.yaml
│   └── stratification_test.yaml
└── archive/
    └── deprecated_configs/
```

### Naming Conventions

- `{dataset}_{method}.yaml` - e.g., `proteomics_als.yaml`
- `{purpose}_v{version}.yaml` - e.g., `production_v1.yaml`
- `{experiment}_{date}.yaml` - e.g., `threshold_sweep_2024_12.yaml`

### Comments in Configs

Use YAML comments to document parameter choices:

```yaml
detection:
  method: "adjusted-boxplot"
  # Using 1.5 based on sensitivity analysis from 2024-12-15
  # Conservative threshold to minimize false positives
  threshold: 1.5

imputation:
  # soft-clip preserves rank order better than mad-clip
  # See lab notebook entry #142
  strategy: "soft-clip"
  sharpness: 5.0  # Tuned to match biological priors
```

## Troubleshooting

### Config Not Loading

```bash
# Check file exists
ls -lh configs/my_config.yaml

# Check YAML syntax
python3 -c "import yaml; print(yaml.safe_load(open('configs/my_config.yaml')))"
```

### Validation Errors

```
ERROR: Config file error: Invalid detection method 'invalid-method'
```

Check spelling and valid choices in error message.

### Required Fields Missing

```
ERROR: --input is required (via CLI or config file)
```

Add missing field to config or provide via CLI.

### CLI Override Not Working

Make sure to provide the flag explicitly:

```bash
# This works - explicit flag
cliquefinder impute --config pipeline.yaml --threshold 2.0

# This doesn't work - missing flag
cliquefinder impute --config pipeline.yaml 2.0
```

## Performance Tips

1. **Reuse configs** - Don't recreate for every run
2. **Version control** - Track parameter evolution
3. **Document choices** - Add comments explaining parameters
4. **Test incremental** - Start minimal, add complexity
5. **Validate early** - Check config loads before long runs

## Migration from CLI-Only

If you have existing CLI scripts, convert to config:

**Before (CLI-only):**
```bash
cliquefinder impute \
  --input data.csv \
  --output results/imputed \
  --method adjusted-boxplot \
  --threshold 1.5 \
  --impute-strategy soft-clip \
  --sharpness 5.0
```

**After (config + CLI):**
```yaml
# configs/my_pipeline.yaml
input: "data.csv"
output: "results/imputed"
detection:
  method: "adjusted-boxplot"
  threshold: 1.5
imputation:
  strategy: "soft-clip"
  sharpness: 5.0
```

```bash
cliquefinder impute --config configs/my_pipeline.yaml
```

Much cleaner and easier to maintain!
