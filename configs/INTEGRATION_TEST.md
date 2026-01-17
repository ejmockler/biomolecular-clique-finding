# Configuration Integration Test Guide

This document shows how to test the configuration file support.

## Test 1: Basic Config Loading

Create a test config:

```yaml
# test_config.yaml
input: "test_data.csv"
output: "test_output"
detection:
  method: "adjusted-boxplot"
  threshold: 1.5
```

Run with config:

```bash
# This will load config, validate it, and merge with CLI defaults
cliquefinder impute --config test_config.yaml
```

Expected output:
```
Loading configuration from: test_config.yaml
  Configuration loaded successfully

======================================================================
  Phase 1: Outlier Detection and Imputation
======================================================================
Started: 2024-12-24 16:00:00

Loading: test_data.csv
...
```

## Test 2: CLI Override Priority

Test that CLI arguments override config values:

```yaml
# base_config.yaml
detection:
  method: "mad-z"
  threshold: 3.5
```

```bash
# This should use adjusted-boxplot (CLI) instead of mad-z (config)
cliquefinder impute --config base_config.yaml --method adjusted-boxplot --threshold 1.5
```

Verify in output that it uses `adjusted-boxplot` with threshold `1.5`.

## Test 3: Minimal Config

```yaml
# minimal.yaml
input: "data.csv"
output: "results"
```

```bash
cliquefinder impute --config minimal.yaml
```

Should use all default values for unspecified parameters.

## Test 4: JSON Format

```json
{
  "input": "data.csv",
  "output": "results",
  "detection": {
    "method": "adjusted-boxplot"
  }
}
```

```bash
cliquefinder impute --config test_config.json
```

Should work identically to YAML.

## Test 5: Validation Errors

Test that validation catches invalid parameters:

```yaml
# invalid_method.yaml
detection:
  method: "invalid-method"
```

```bash
cliquefinder impute --config invalid_method.yaml
```

Expected error:
```
ERROR: Config file error: Invalid detection method 'invalid-method'.
Choose from: mad-z, iqr, adjusted-boxplot
```

## Test 6: Missing Required Fields

```yaml
# missing_input.yaml
output: "results"
```

```bash
cliquefinder impute --config missing_input.yaml
```

Expected error:
```
ERROR: --input is required (via CLI or config file)
```

Can fix by providing via CLI:
```bash
cliquefinder impute --config missing_input.yaml --input data.csv
```

## Test 7: Complex Config with All Sections

```yaml
# full_config.yaml
input: "proteomics.csv"
output: "results/full_test"

detection:
  method: "adjusted-boxplot"
  threshold: 1.5
  stratify_by: ["phenotype", "Sex"]

imputation:
  strategy: "soft-clip"
  sharpness: 5.0

phenotype:
  source_col: "SUBJECT_GROUP"
  case_values: ["ALS", "ALS-FTD"]
  ctrl_values: ["Healthy Control", "Control"]
```

```bash
cliquefinder impute --config full_config.yaml
```

Should apply all specified parameters.

## Debugging Tips

### Check Config Loading

Add debug print to see what's loaded:

```python
# In run_impute() after config loading
if args.config:
    print(f"DEBUG: Loaded config: {config}")
    print(f"DEBUG: Merged args.method: {args.method}")
    print(f"DEBUG: Merged args.threshold: {args.threshold}")
```

### Verify Merge Logic

Test specific scenarios:

```bash
# Scenario 1: Config only
cliquefinder impute --config pipeline.yaml

# Scenario 2: CLI only
cliquefinder impute --input data.csv --output results --method mad-z

# Scenario 3: Config + CLI override
cliquefinder impute --config pipeline.yaml --threshold 2.0

# Scenario 4: Partial config + CLI complete
cliquefinder impute --config partial.yaml --input data.csv --output results
```

### Test Edge Cases

```bash
# Empty config file
echo "" > empty.yaml
cliquefinder impute --config empty.yaml --input data.csv --output results

# Config with nulls
cat > null_config.yaml << EOF
input: "data.csv"
output: "results"
metadata: null
EOF
cliquefinder impute --config null_config.yaml

# Unicode/special characters in paths
cat > special.yaml << EOF
input: "data with spaces.csv"
output: "résults/tést"
EOF
cliquefinder impute --config special.yaml
```

## Verification Checklist

- [ ] Config file loads without errors
- [ ] YAML format works
- [ ] JSON format works
- [ ] CLI arguments override config values
- [ ] Config values override CLI defaults
- [ ] Validation catches invalid methods
- [ ] Validation catches invalid thresholds
- [ ] Missing required fields show clear errors
- [ ] Partial configs work (unspecified params use defaults)
- [ ] Complex nested configs work (detection, imputation, phenotype sections)
- [ ] Path fields convert to Path objects
- [ ] List fields work (stratify_by, case_values, ctrl_values)

## Implementation Details

### Config Loading Flow

1. CLI parser creates args with defaults
2. If `--config` provided:
   - Load config file (YAML/JSON)
   - Validate config structure
   - Parse raw CLI args to detect explicit overrides
   - Merge config with args (CLI wins)
3. Validate required fields (input, output)
4. Proceed with normal execution

### Priority Resolution

```python
def _merge_value(cli_value, config_value, was_explicit):
    if was_explicit:
        return cli_value  # CLI wins
    if config_value is not None:
        return config_value  # Config wins
    return cli_value  # Default wins
```

### Extensibility

To add new config parameters:

1. Add to appropriate Config dataclass
2. Add mapping in merge_config_with_args()
3. Add validation in validate_config() if needed
4. Update example configs
5. Document in README.md

Example:
```python
# Add to DetectionConfig
@dataclass
class DetectionConfig:
    new_param: float = 1.0

# Add to merge logic
if 'new_param' in detection:
    merged.new_param = _merge_value(
        merged.new_param,
        detection['new_param'],
        'new_param' in explicit_args
    )

# Add validation
if 'new_param' in config.get('detection', {}):
    if config['detection']['new_param'] < 0:
        raise ValueError("new_param must be positive")
```
