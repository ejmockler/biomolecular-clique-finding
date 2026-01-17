# Genetic Contrast Implementation - Code Reference

This document shows the key code sections implemented in `src/cliquefinder/cli/differential.py`.

## 1. CLI Arguments (Lines ~245-270)

```python
# Additional analysis
parser.add_argument(
    "--mode",
    type=str,
    choices=["clique", "protein"],
    default="clique",
    help="Analysis mode: clique-level or protein-level (default: clique)",
)
parser.add_argument(
    "--also-protein-level",
    action="store_true",
    help="Also run protein-level differential analysis (only for clique mode)",
)
parser.add_argument(
    "--workers", "-j",
    type=int,
    default=1,
    help="Parallel workers (default: 1)",
)

# Genetic subtype analysis
parser.add_argument(
    "--genetic-contrast",
    type=str,
    metavar="MUTATION",
    help="Genetic subtype contrast (e.g., 'C9orf72' for carriers vs sporadic ALS). "
         "Requires ClinReport_Mutations_Details column in metadata. "
         "Known mutations: C9orf72, SOD1, TARDBP, FUS, SETX, Multiple, Other",
)
```

## 2. Helper Function: derive_genetic_phenotype()

```python
def derive_genetic_phenotype(
    metadata: pd.DataFrame,
    mutation: str,
    mutation_col: str = "ClinReport_Mutations_Details",
    phenotype_col: str = "phenotype",
) -> tuple[pd.DataFrame, str, str]:
    """
    Derive binary genetic phenotype from mutation data.

    Creates a contrast between mutation carriers and sporadic ALS cases
    (excluding healthy controls).

    Args:
        metadata: Sample metadata DataFrame (indexed by sample ID).
        mutation: Mutation name to contrast (e.g., 'C9orf72', 'SOD1').
        mutation_col: Column containing mutation annotations.
        phenotype_col: Column containing CASE/CTRL labels.

    Returns:
        Tuple of (filtered_metadata, carrier_label, sporadic_label).
        The metadata has a new 'genetic_phenotype' column with labels.

    Raises:
        ValueError: If mutation column missing or no samples found.
    """
    if mutation_col not in metadata.columns:
        raise ValueError(
            f"Mutation column '{mutation_col}' not found in metadata. "
            f"Available columns: {', '.join(metadata.columns)}"
        )

    if phenotype_col not in metadata.columns:
        raise ValueError(
            f"Phenotype column '{phenotype_col}' not found in metadata."
        )

    # Known familial mutations
    known_mutations = ['C9orf72', 'SOD1', 'Multiple', 'Other', 'SETX', 'TARDBP', 'TARDBP (TDP43)', 'FUS']

    # Filter to ALS cases only (exclude healthy controls)
    case_mask = metadata[phenotype_col] == 'CASE'
    metadata_cases = metadata[case_mask].copy()

    if len(metadata_cases) == 0:
        raise ValueError("No CASE samples found in metadata")

    # Create carrier mask
    carrier_mask = metadata_cases[mutation_col] == mutation
    n_carriers = carrier_mask.sum()

    # Create sporadic mask (CASE without any known mutation)
    sporadic_mask = (
        ~metadata_cases[mutation_col].isin(known_mutations) |
        metadata_cases[mutation_col].isna()
    )
    n_sporadic = sporadic_mask.sum()

    if n_carriers == 0:
        raise ValueError(
            f"No carriers found for mutation '{mutation}'. "
            f"Available mutations: {metadata_cases[mutation_col].value_counts().to_dict()}"
        )

    if n_sporadic == 0:
        raise ValueError("No sporadic ALS samples found")

    # Create labels
    carrier_label = mutation.upper()
    sporadic_label = "SPORADIC"

    # Create derived phenotype column
    metadata_cases['genetic_phenotype'] = None
    metadata_cases.loc[carrier_mask, 'genetic_phenotype'] = carrier_label
    metadata_cases.loc[sporadic_mask, 'genetic_phenotype'] = sporadic_label

    # Filter to only samples with genetic phenotype assigned
    metadata_filtered = metadata_cases[
        metadata_cases['genetic_phenotype'].notna()
    ].copy()

    print(f"\nGenetic subtype contrast:")
    print(f"  Mutation: {mutation}")
    print(f"  Carriers ({carrier_label}): n={n_carriers}")
    print(f"  Sporadic ALS ({sporadic_label}): n={n_sporadic}")
    print(f"  Total samples: {len(metadata_filtered)}")

    # Warn if underpowered
    if n_carriers < 30 or n_sporadic < 30:
        print(f"  WARNING: Small sample size detected. Statistical power may be limited.")
        if n_carriers < 10 or n_sporadic < 10:
            print(f"  WARNING: Very small sample size (n<10). Results should be interpreted with caution.")

    return metadata_filtered, carrier_label, sporadic_label
```

## 3. Metadata Processing with Genetic Contrast

```python
# Load metadata
print(f"Loading metadata: {args.metadata}")
metadata = pd.read_csv(args.metadata, index_col=0)

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

# Align metadata with data
common_samples = [s for s in matrix.sample_ids if s in metadata.index]
if len(common_samples) < len(matrix.sample_ids):
    print(f"  Warning: {len(matrix.sample_ids) - len(common_samples)} samples missing from metadata")

metadata = metadata.loc[common_samples]
```

## 4. Clique Loading (Mode-Aware)

```python
# Load clique definitions (skip if protein-only mode)
cliques = None
if args.mode == "clique":
    print(f"\nLoading cliques: {args.cliques}")
    cliques = load_clique_definitions(args.cliques, min_proteins=args.min_proteins)
    print(f"  {len(cliques)} cliques loaded (min {args.min_proteins} proteins)")

    # Filter by coherence if specified
    if args.min_coherence:
        original_count = len(cliques)
        cliques = [c for c in cliques if c.coherence is None or c.coherence >= args.min_coherence]
        print(f"  Filtered by coherence >= {args.min_coherence}: {len(cliques)} remaining")
else:
    print(f"\nMode: protein-level analysis (skipping clique loading)")
```

## 5. Protein-Level Analysis Branch

```python
# Branch based on mode
if args.mode == "protein":
    # Protein-level differential analysis
    from cliquefinder.stats.differential import run_differential_analysis

    print("Running protein-level differential analysis...")
    print("=" * 70)

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

    # Save results
    protein_df = result.to_dataframe()
    protein_output = args.output / "protein_differential.csv"
    protein_df.to_csv(protein_output, index=False)
    print(f"\nProtein results: {protein_output}")

    # Save significant proteins
    sig_proteins = protein_df[protein_df['significant']]
    if len(sig_proteins) > 0:
        sig_output = args.output / "significant_proteins.csv"
        sig_proteins.to_csv(sig_output, index=False)
        print(f"Significant proteins: {sig_output}")

    # Save parameters
    params = {
        "timestamp": datetime.now().isoformat(),
        "mode": "protein",
        "data": str(args.data),
        "metadata": str(args.metadata),
        "condition_col": condition_col,
        "subject_col": args.subject_col,
        "contrasts": contrasts,
        "normalization": normalization.value,
        "imputation": imputation.value,
        "use_mixed_model": not args.no_mixed_model,
        "fdr_method": args.fdr_method,
        "fdr_threshold": args.fdr_threshold,
        "n_features_tested": result.n_features_tested,
        "n_significant": result.n_significant,
    }
    if args.genetic_contrast:
        params["genetic_contrast"] = args.genetic_contrast

    params_output = args.output / "analysis_parameters.json"
    with open(params_output, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Parameters: {params_output}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY (Protein-Level)")
    print("=" * 70)
    print(f"Proteins tested: {result.n_features_tested}")
    print(f"Significant (FDR < {args.fdr_threshold}): {result.n_significant}")

    if len(sig_proteins) > 0:
        print(f"\nTop significant proteins:")
        top = sig_proteins.nsmallest(10, 'adj_pvalue')[['feature_id', 'log2FC', 'adj_pvalue', 'contrast']]
        for _, row in top.iterrows():
            direction = "↑" if row['log2FC'] > 0 else "↓"
            print(f"  {row['feature_id']}: log2FC={row['log2FC']:.3f} {direction}, q={row['adj_pvalue']:.2e}")

    print(f"\nComplete! Duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0

# Clique-level analysis continues...
```

## 6. Clique Analysis with Updated condition_col

```python
# Clique-level analysis
if args.permutation_test:
    # ... permutation setup ...

    perm_results, null_df = run_permutation_clique_test(
        data=data,
        feature_ids=feature_ids,
        sample_metadata=metadata,
        clique_definitions=cliques,
        condition_col=condition_col,  # <-- Uses variable, not args.condition_col
        subject_col=args.subject_col,
        contrast=contrast_tuple,
        # ... other params ...
    )
else:
    # Standard BH FDR correction
    result = run_clique_differential_analysis(
        data=data,
        feature_ids=feature_ids,
        sample_metadata=metadata,
        clique_definitions=cliques,
        condition_col=condition_col,  # <-- Uses variable, not args.condition_col
        subject_col=args.subject_col,
        contrasts=contrasts,
        # ... other params ...
    )
```

## 7. Parameters Saving with Genetic Contrast

```python
# Save parameters (both permutation and standard modes)
params = {
    "timestamp": datetime.now().isoformat(),
    "mode": "clique",  # or "protein" in protein mode
    "data": str(args.data),
    "metadata": str(args.metadata),
    "cliques": str(args.cliques),
    "condition_col": condition_col,  # <-- Uses variable, not args.condition_col
    "subject_col": args.subject_col,
    "contrasts": contrasts,
    "summarization": summarization.value,
    "normalization": normalization.value,
    "imputation": imputation.value,
    "use_mixed_model": not args.no_mixed_model,
    "fdr_method": args.fdr_method,
    "fdr_threshold": args.fdr_threshold,
    "min_proteins": args.min_proteins,
    "n_cliques_tested": result.n_cliques_tested,
    "n_significant": result.n_significant,
}
if args.genetic_contrast:
    params["genetic_contrast"] = args.genetic_contrast

params_output = args.output / "analysis_parameters.json"
with open(params_output, "w") as f:
    json.dump(params, f, indent=2)
```

## Key Variable: condition_col

The `condition_col` variable is critical for maintaining backwards compatibility:

```python
# Initialize with default
condition_col = args.condition_col  # Default: "phenotype"

# Override if genetic contrast
if args.genetic_contrast:
    metadata, carrier_label, sporadic_label = derive_genetic_phenotype(...)
    condition_col = 'genetic_phenotype'  # Override to use derived column

# Use throughout analysis
# - Line 558: sample_condition=metadata[condition_col]
# - Line 653: condition_col=condition_col (permutation)
# - Line 804: condition_col=condition_col (standard)
# - Line 893: "condition_col": condition_col (params)
```

This ensures:
- Standard workflows use `args.condition_col` → "phenotype"
- Genetic contrasts use `condition_col` → "genetic_phenotype"
- All downstream code uses the correct column

## Sample Selection Logic

```python
# Known mutations to exclude from sporadic group
known_mutations = [
    'C9orf72',          # C9orf72 repeat expansion
    'SOD1',             # Superoxide dismutase 1
    'TARDBP',           # TAR DNA-binding protein
    'TARDBP (TDP43)',   # Alternative annotation
    'FUS',              # FUS RNA-binding protein
    'SETX',             # Senataxin
    'Multiple',         # Multiple mutations
    'Other'             # Other known mutations
]

# Carrier selection
carrier_mask = (metadata['phenotype'] == 'CASE') & \
               (metadata['ClinReport_Mutations_Details'] == mutation)

# Sporadic selection (CASE without any known mutation)
sporadic_mask = (metadata['phenotype'] == 'CASE') & \
                (~metadata['ClinReport_Mutations_Details'].isin(known_mutations) | \
                 metadata['ClinReport_Mutations_Details'].isna())
```

## Error Messages

```python
# Missing column
ValueError: Mutation column 'ClinReport_Mutations_Details' not found in metadata.
            Available columns: phenotype, subject_id, ...

# No carriers
ValueError: No carriers found for mutation 'INVALID'.
            Available mutations: {'C9orf72': 21, 'SOD1': 17, ...}

# No sporadic
ValueError: No sporadic ALS samples found
```

## Usage Pattern

```bash
# Genetic contrast (automatic contrast generation)
cliquefinder differential \
    --data data.csv \
    --metadata meta.csv \
    --cliques cliques.csv \
    --output output/ \
    --genetic-contrast C9orf72 \
    --subject-col subject_id

# Traditional contrast (manual specification)
cliquefinder differential \
    --data data.csv \
    --metadata meta.csv \
    --cliques cliques.csv \
    --output output/ \
    --condition-col phenotype \
    --subject-col subject_id \
    --contrast CASE_vs_CTRL CASE CTRL
```

## Integration Points

1. **CLI Argument Parsing** → New `--mode` and `--genetic-contrast` args
2. **Metadata Loading** → Call `derive_genetic_phenotype()` if needed
3. **Clique Loading** → Skip if `--mode protein`
4. **Analysis Execution** → Branch on mode, use `condition_col` variable
5. **Results Saving** → Include `genetic_contrast` and `mode` in params
6. **Error Handling** → Validate inputs, clear messages

All modifications follow existing code patterns and maintain backwards compatibility.
