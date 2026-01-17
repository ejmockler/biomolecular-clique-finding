"""
Configuration file support for CliqueFinder CLI.

Supports YAML and JSON config files with CLI argument override.
"""

import json
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DetectionConfig:
    """Outlier detection configuration."""
    method: str = "adjusted-boxplot"
    threshold: float = 1.5
    stratify_by: List[str] = field(default_factory=lambda: ["phenotype"])
    residual: Optional[Dict[str, Any]] = None
    global_cap: Optional[Dict[str, Any]] = None


@dataclass
class ImputationConfig:
    """Imputation configuration."""
    strategy: str = "soft-clip"
    sharpness: Optional[float] = None


@dataclass
class PhenotypeConfig:
    """Phenotype mapping configuration."""
    source_col: str = "SUBJECT_GROUP"
    case_values: List[str] = field(default_factory=lambda: ["ALS"])
    ctrl_values: List[str] = field(default_factory=lambda: ["Healthy Control"])


@dataclass
class ConfigSchema:
    """
    Complete configuration schema for cliquefinder impute command.

    Mirrors the CLI argument structure for consistency.
    """
    input: Optional[Path] = None
    output: Optional[Path] = None
    metadata: Optional[Path] = None
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    phenotype: Optional[PhenotypeConfig] = None


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Parameters:
        config_path: Path to config file (.yaml, .yml, or .json)

    Returns:
        Dictionary with configuration values

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is unsupported or invalid

    Examples:
        >>> config = load_config(Path("pipeline.yaml"))
        >>> print(config['detection']['method'])
        adjusted-boxplot
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()

    try:
        with open(config_path, 'r') as f:
            if suffix in ('.yaml', '.yml'):
                config = yaml.safe_load(f)
            elif suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config format: {suffix}. "
                    f"Use .yaml, .yml, or .json"
                )
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a dictionary/mapping at top level")

    return config


def _merge_value(cli_value: Any, config_value: Any, arg_name: str, was_explicitly_set: bool) -> Any:
    """
    Merge a single config value with CLI argument.

    Rules:
    - CLI args ALWAYS override config if explicitly set
    - If CLI arg not set, use config value
    - If neither set, keep CLI default

    Parameters:
        cli_value: Value from CLI args (may be default)
        config_value: Value from config file
        arg_name: Name of argument (for debugging)
        was_explicitly_set: Whether CLI arg was explicitly provided by user

    Returns:
        Merged value
    """
    # CLI argument explicitly set - always wins
    if was_explicitly_set:
        return cli_value

    # Config value available - use it
    if config_value is not None:
        return config_value

    # Fall back to CLI default
    return cli_value


def merge_config_with_args(config: Dict[str, Any], args: Namespace, cli_args: Optional[List[str]] = None) -> Namespace:
    """
    Merge config file values with CLI arguments.

    Priority (highest to lowest):
    1. Explicitly provided CLI arguments
    2. Config file values
    3. CLI argument defaults

    This allows users to:
    - Use config for base settings
    - Override specific values via CLI for quick experiments
    - Mix and match as needed

    Parameters:
        config: Configuration dictionary from load_config()
        args: Parsed CLI arguments (argparse.Namespace)
        cli_args: Raw CLI arguments list (for detecting explicit values)
                  If None, assumes all args are defaults

    Returns:
        Updated Namespace with merged values

    Examples:
        >>> config = load_config(Path("pipeline.yaml"))
        >>> args = parser.parse_args(["--input", "data.csv"])
        >>> merged = merge_config_with_args(config, args)
        >>> # args.input from CLI, args.method from config
    """
    # Track which CLI args were explicitly set
    explicit_args = set()
    if cli_args:
        # Parse cli_args to identify explicitly set arguments
        i = 0
        while i < len(cli_args):
            arg = cli_args[i]
            if arg.startswith('--'):
                # Long form argument
                arg_name = arg[2:].replace('-', '_')
                explicit_args.add(arg_name)
            elif arg.startswith('-') and len(arg) == 2:
                # Short form - need to map to long form
                # This is imperfect but handles common cases
                short_to_long = {
                    'i': 'input',
                    'o': 'output',
                    'f': 'format',
                }
                if arg[1] in short_to_long:
                    explicit_args.add(short_to_long[arg[1]])
            i += 1

    # Create a copy of args to modify
    merged = Namespace(**vars(args))

    # === Top-level simple arguments ===
    simple_mappings = {
        'input': 'input',
        'output': 'output',
        'metadata': 'metadata',
    }

    for config_key, arg_name in simple_mappings.items():
        if config_key in config:
            config_value = config[config_key]
            # Convert string paths to Path objects
            if config_value is not None and arg_name in ('input', 'output', 'metadata'):
                config_value = Path(config_value)

            was_explicit = arg_name in explicit_args
            merged_value = _merge_value(
                getattr(merged, arg_name),
                config_value,
                arg_name,
                was_explicit
            )
            setattr(merged, arg_name, merged_value)

    # === Detection section ===
    if 'detection' in config:
        detection = config['detection']

        # method
        if 'method' in detection:
            was_explicit = 'method' in explicit_args
            merged.method = _merge_value(
                merged.method,
                detection['method'],
                'method',
                was_explicit
            )

        # threshold
        if 'threshold' in detection:
            was_explicit = 'threshold' in explicit_args
            merged.threshold = _merge_value(
                merged.threshold,
                detection['threshold'],
                'threshold',
                was_explicit
            )

        # stratify_by (maps to group_cols)
        if 'stratify_by' in detection:
            was_explicit = 'group_cols' in explicit_args
            merged.group_cols = _merge_value(
                merged.group_cols,
                detection['stratify_by'],
                'group_cols',
                was_explicit
            )

        # residual detection (future extension point)
        if 'residual' in detection:
            residual = detection['residual']
            # Could map to additional CLI args if implemented
            # For now, store in merged for potential use
            if not hasattr(merged, '_config_residual'):
                merged._config_residual = residual

        # global_cap (future extension point)
        if 'global_cap' in detection:
            global_cap = detection['global_cap']
            if not hasattr(merged, '_config_global_cap'):
                merged._config_global_cap = global_cap

    # === Imputation section ===
    if 'imputation' in config:
        imputation = config['imputation']

        # strategy (maps to impute_strategy)
        if 'strategy' in imputation:
            was_explicit = 'impute_strategy' in explicit_args
            merged.impute_strategy = _merge_value(
                merged.impute_strategy,
                imputation['strategy'],
                'impute_strategy',
                was_explicit
            )

        # sharpness (maps to soft_clip_sharpness)
        if 'sharpness' in imputation:
            was_explicit = 'soft_clip_sharpness' in explicit_args
            merged.soft_clip_sharpness = _merge_value(
                merged.soft_clip_sharpness,
                imputation['sharpness'],
                'soft_clip_sharpness',
                was_explicit
            )

    # === Phenotype section ===
    if 'phenotype' in config:
        phenotype = config['phenotype']

        # source_col (maps to phenotype_source_col)
        if 'source_col' in phenotype:
            was_explicit = 'phenotype_source_col' in explicit_args
            merged.phenotype_source_col = _merge_value(
                merged.phenotype_source_col,
                phenotype['source_col'],
                'phenotype_source_col',
                was_explicit
            )

        # case_values
        if 'case_values' in phenotype:
            was_explicit = 'case_values' in explicit_args
            merged.case_values = _merge_value(
                merged.case_values,
                phenotype['case_values'],
                'case_values',
                was_explicit
            )

        # ctrl_values
        if 'ctrl_values' in phenotype:
            was_explicit = 'ctrl_values' in explicit_args
            merged.ctrl_values = _merge_value(
                merged.ctrl_values,
                phenotype['ctrl_values'],
                'ctrl_values',
                was_explicit
            )

    return merged


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and values.

    Performs basic validation:
    - Required sections present
    - Valid method/strategy choices
    - Reasonable threshold values

    Parameters:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate detection method if present
    if 'detection' in config and 'method' in config['detection']:
        valid_methods = ['mad-z', 'iqr', 'adjusted-boxplot']
        method = config['detection']['method']
        if method not in valid_methods:
            raise ValueError(
                f"Invalid detection method '{method}'. "
                f"Choose from: {', '.join(valid_methods)}"
            )

    # Validate imputation strategy if present
    if 'imputation' in config and 'strategy' in config['imputation']:
        valid_strategies = ['mad-clip', 'median', 'soft-clip', 'knn']
        strategy = config['imputation']['strategy']
        if strategy not in valid_strategies:
            raise ValueError(
                f"Invalid imputation strategy '{strategy}'. "
                f"Choose from: {', '.join(valid_strategies)}"
            )

    # Validate threshold ranges
    if 'detection' in config and 'threshold' in config['detection']:
        threshold = config['detection']['threshold']
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError(
                f"Detection threshold must be positive number, got: {threshold}"
            )

    # Validate sharpness if present
    if 'imputation' in config and 'sharpness' in config['imputation']:
        sharpness = config['imputation']['sharpness']
        if sharpness is not None:
            if not isinstance(sharpness, (int, float)) or sharpness <= 0:
                raise ValueError(
                    f"Imputation sharpness must be positive number, got: {sharpness}"
                )
