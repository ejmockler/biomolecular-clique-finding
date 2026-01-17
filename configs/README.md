# CliqueFinder Configuration Files

This directory contains example configuration files for the CliqueFinder CLI.

## Overview

Configuration files provide a convenient way to specify pipeline parameters without lengthy command-line arguments. They support both YAML and JSON formats.

## Usage

### Basic Usage

```bash
# Use config file for all parameters
cliquefinder impute --config configs/proteomics_als.yaml

# Override specific parameters via CLI
cliquefinder impute --config configs/proteomics_als.yaml --threshold 2.0

# Mix config file with CLI args (CLI always wins)
cliquefinder impute --config configs/proteomics_als.yaml \
  --input my_data.csv \
  --method mad-z
```

### Priority Rules

When using both config files and CLI arguments:

1. **Explicitly provided CLI arguments** (highest priority)
2. **Config file values**
3. **CLI default values** (lowest priority)

This allows you to:
- Define base configurations in files
- Quickly experiment with different parameters via CLI
- Share reproducible configurations with collaborators

## Example Configurations

### proteomics_als.yaml

Optimal parameters for proteomics outlier detection in ALS cohort studies.

**Key features:**
- `adjusted-boxplot` method for skewed proteomics data
- Soft-clip imputation preserving rank order
- Phenotype-stratified detection (case/control aware)

**Recommended for:**
- Proteomic abundance data
- Mass spectrometry data
- Any skewed biomolecular measurements

### proteomics_als.json

Same configuration as YAML version, but in JSON format.

**Use when:**
- You prefer JSON syntax
- Integrating with JSON-based pipelines
- Programmatic config generation

## Configuration Schema

### Top-Level Fields

```yaml
input: "path/to/data.csv"          # Input data file
output: "path/to/output"           # Output base path
metadata: "path/to/metadata.csv"   # Optional external metadata
```

### Detection Section

```yaml
detection:
  method: "adjusted-boxplot"       # mad-z | iqr | adjusted-boxplot
  threshold: 1.5                   # Numeric threshold
  stratify_by: ["phenotype"]       # Metadata columns for stratification
```

**Methods:**
- `mad-z`: Symmetric, assumes normal distribution
- `iqr`: Symmetric, robust to outliers
- `adjusted-boxplot`: Asymmetric, accounts for skewness (recommended for proteomics)

**Threshold:**
- For `mad-z`: MAD-Z score (typical: 3.0-5.0)
- For `iqr` and `adjusted-boxplot`: IQR multiplier (typical: 1.5-2.0)

### Imputation Section

```yaml
imputation:
  strategy: "soft-clip"            # mad-clip | median | soft-clip | knn
  sharpness: 5.0                   # For soft-clip only (optional)
```

**Strategies:**
- `mad-clip`: Hard threshold at MAD-Z limit
- `median`: Replace with group median
- `soft-clip`: Smooth sigmoid clipping, preserves rank order
- `knn`: K-nearest neighbors imputation

### Phenotype Section

```yaml
phenotype:
  source_col: "SUBJECT_GROUP"      # Clinical column with phenotype
  case_values: ["ALS"]             # Values mapping to CASE
  ctrl_values: ["Healthy Control"] # Values mapping to CTRL
```

## Creating Your Own Config

1. Copy an example config:
   ```bash
   cp configs/proteomics_als.yaml configs/my_pipeline.yaml
   ```

2. Edit parameters for your data:
   ```yaml
   input: "my_data.csv"
   detection:
     threshold: 2.0  # Adjust as needed
   ```

3. Test the configuration:
   ```bash
   cliquefinder impute --config configs/my_pipeline.yaml
   ```

## Advanced Usage

### Minimal Config

You don't need to specify every parameter. This minimal config is valid:

```yaml
input: "data.csv"
output: "results/imputed"
```

All other parameters will use CLI defaults.

### Partial Overrides

Override just the parameters you need:

```yaml
# Use all defaults except method and threshold
detection:
  method: "adjusted-boxplot"
  threshold: 1.5
```

### CLI Override Examples

```bash
# Use config but change threshold
cliquefinder impute --config pipeline.yaml --threshold 2.0

# Use config but process only CASE samples
cliquefinder impute --config pipeline.yaml --phenotype-filter CASE

# Use config but disable log transformation
cliquefinder impute --config pipeline.yaml --no-log-transform
```

## Validation

The CLI automatically validates your configuration:

- Valid method/strategy choices
- Reasonable threshold ranges
- Proper file paths
- Required fields present

If validation fails, you'll see a clear error message:

```
ERROR: Config file error: Invalid detection method 'invalid-method'.
Choose from: mad-z, iqr, adjusted-boxplot
```

## Tips

1. **Version control your configs** - Track parameter changes over time
2. **Use descriptive names** - `proteomics_als_conservative.yaml` vs `config1.yaml`
3. **Document custom configs** - Add comments explaining parameter choices
4. **Share configs with collaborators** - Ensures reproducible analyses
5. **Test incrementally** - Start with minimal config, add parameters as needed

## Future Extensions

Planned features (not yet implemented):

- Multi-pass detection (residual, global cap)
- Sex classification parameters
- Expression filtering thresholds
- Batch effect correction settings

These will be added to the config schema in future releases.
