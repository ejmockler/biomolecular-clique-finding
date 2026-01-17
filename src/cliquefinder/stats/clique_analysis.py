"""
Clique-level differential abundance analysis.

This module extends MSstats methodology to protein cliques (co-regulated groups).
Instead of analyzing individual proteins, we:

1. Aggregate protein abundances within each clique using Tukey's Median Polish
2. Test for differential abundance at the clique level
3. Account for biological replicates using linear mixed models

The key insight: treating correlated proteins as a single unit reduces
multiple testing burden and can detect coordinated changes that might
be missed at the individual protein level.

Statistical Framework:
    - Input: Protein-level log2 intensities + clique definitions
    - Summarization: Tukey's Median Polish (proteins → clique)
    - Model: log2(CliqueAbundance) ~ Condition + (1 | Subject)
    - Testing: Contrast-based hypothesis tests with FDR correction

References:
    - MSstats: Choi et al. (2014) Bioinformatics 30(17):2524-2526
    - Clique methodology: regulatory module co-expression analysis
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .summarization import (
    SummarizationMethod,
    summarize_clique,
    CliqueSummary,
    tukey_median_polish,
)
from .normalization import (
    NormalizationMethod,
    normalize,
    median_normalization,
)
from .missing import (
    ImputationMethod,
    impute_missing_values,
    analyze_missing_values,
)
from .differential import (
    run_differential_analysis,
    DifferentialResult,
    ProteinResult,
    fdr_correction,
)


@dataclass
class CliqueDefinition:
    """Definition of a protein clique.

    Attributes:
        clique_id: Unique identifier (typically regulator name)
        protein_ids: List of protein identifiers in the clique
        regulator: Regulator gene symbol (if applicable)
        condition: Condition where clique was identified
        coherence: Mean pairwise correlation in discovery condition
        n_indra_targets: Number of INDRA-validated targets
    """

    clique_id: str
    protein_ids: list[str]
    regulator: str | None = None
    condition: str | None = None
    coherence: float | None = None
    n_indra_targets: int | None = None


@dataclass
class CliqueDifferentialResult:
    """Result of clique-level differential analysis.

    Attributes:
        clique_id: Clique identifier
        regulator: Regulator gene (if applicable)
        n_proteins: Number of proteins in clique
        n_proteins_found: Number of proteins found in data
        protein_ids: List of proteins used
        summarization_method: How proteins were aggregated
        coherence: Intra-clique correlation
        log2_fc: Log2 fold change (clique level)
        se: Standard error
        t_value: t-statistic
        df: Degrees of freedom
        p_value: Raw p-value
        adj_p_value: FDR-adjusted p-value
        ci_lower: Lower 95% CI
        ci_upper: Upper 95% CI
        contrast: Contrast tested
        model_type: Fixed or mixed model
        issue: Warning/error message if any
    """

    clique_id: str
    regulator: str | None
    n_proteins: int
    n_proteins_found: int
    protein_ids: list[str]
    summarization_method: str
    coherence: float | None
    log2_fc: float
    se: float
    t_value: float
    df: float
    p_value: float
    adj_p_value: float | None
    ci_lower: float
    ci_upper: float
    contrast: str
    model_type: str
    issue: str | None = None

    def to_dict(self) -> dict:
        return {
            'clique_id': self.clique_id,
            'regulator': self.regulator,
            'n_proteins': self.n_proteins,
            'n_proteins_found': self.n_proteins_found,
            'summarization_method': self.summarization_method,
            'coherence': self.coherence,
            'log2FC': self.log2_fc,
            'SE': self.se,
            'tvalue': self.t_value,
            'df': self.df,
            'pvalue': self.p_value,
            'adj_pvalue': self.adj_p_value,
            'CI_lower': self.ci_lower,
            'CI_upper': self.ci_upper,
            'contrast': self.contrast,
            'model_type': self.model_type,
            'issue': self.issue,
            'proteins': ','.join(self.protein_ids),
        }


@dataclass
class CliqueAnalysisResult:
    """Complete clique-level analysis results.

    Attributes:
        clique_results: Per-clique differential results
        protein_results: Per-protein differential results (optional)
        contrasts_tested: Names of contrasts
        n_cliques_tested: Number of cliques analyzed
        n_significant: Number of significant cliques
        fdr_threshold: FDR threshold used
        preprocessing_params: Parameters used for preprocessing
    """

    clique_results: list[CliqueDifferentialResult]
    protein_results: DifferentialResult | None
    contrasts_tested: list[str]
    n_cliques_tested: int
    n_significant: int
    fdr_threshold: float
    preprocessing_params: dict = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert clique results to DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.clique_results])

    def significant_cliques(self, contrast: str | None = None) -> list[str]:
        """Get IDs of significant cliques."""
        results = self.clique_results
        if contrast:
            results = [r for r in results if r.contrast == contrast]
        return [r.clique_id for r in results if r.adj_p_value is not None and r.adj_p_value < self.fdr_threshold]

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Clique Differential Analysis Results",
            f"=" * 40,
            f"Cliques tested: {self.n_cliques_tested}",
            f"Significant (FDR < {self.fdr_threshold}): {self.n_significant}",
            f"Contrasts: {', '.join(self.contrasts_tested)}",
        ]
        return "\n".join(lines)


def load_clique_definitions(
    cliques_file: str | Path,
    min_proteins: int = 3,
) -> list[CliqueDefinition]:
    """
    Load clique definitions from cliquefinder output.

    Args:
        cliques_file: Path to cliques.csv or similar.
        min_proteins: Minimum proteins required per clique.

    Returns:
        List of CliqueDefinition objects.
    """
    df = pd.read_csv(cliques_file)

    cliques = []

    # Group by regulator (or clique_id if present)
    id_col = 'regulator' if 'regulator' in df.columns else 'clique_id'

    for clique_id, group in df.groupby(id_col):
        # Get protein list from clique_genes column
        if 'clique_genes' in group.columns:
            # Assuming comma-separated or set representation
            genes_str = group['clique_genes'].iloc[0]
            if isinstance(genes_str, str):
                # Handle various formats: {A,B,C} or "A,B,C" or "A;B;C"
                genes_str = genes_str.strip('{}[]"\'')
                proteins = [g.strip().strip("'\"") for g in genes_str.replace(';', ',').split(',')]
            else:
                proteins = []
        elif 'gene' in group.columns:
            proteins = group['gene'].unique().tolist()
        else:
            continue

        proteins = [p for p in proteins if p]  # Remove empty strings

        if len(proteins) < min_proteins:
            continue

        clique = CliqueDefinition(
            clique_id=str(clique_id),
            protein_ids=proteins,
            regulator=str(clique_id) if id_col == 'regulator' else None,
            condition=group['condition'].iloc[0] if 'condition' in group.columns else None,
            coherence=float(group['coherence'].iloc[0]) if 'coherence' in group.columns else None,
            n_indra_targets=int(group['n_indra_targets'].iloc[0]) if 'n_indra_targets' in group.columns else None,
        )
        cliques.append(clique)

    return cliques


def map_feature_ids_to_symbols(
    feature_ids: list[str],
    verbose: bool = True,
) -> dict[str, str]:
    """
    Map feature IDs (UniProt or Ensembl) to gene symbols using multi-source resolution.

    Uses GeneEntityResolver for confidence-ranked resolution from multiple sources:
    - HGNC (authoritative for human gene symbols)
    - Ensembl BioMart (authoritative for Ensembl IDs)
    - MyGeneInfo (comprehensive aggregator)
    - Alias resolution (for legacy/alternate names)

    Args:
        feature_ids: List of UniProt or Ensembl IDs.
        verbose: Print progress.

    Returns:
        Dict mapping symbol → feature_id (for reverse lookup).
    """
    # Detect ID type
    sample_ids = feature_ids[:100]
    ensembl_count = sum(1 for fid in sample_ids if fid.startswith('ENSG'))

    if ensembl_count > len(sample_ids) * 0.5:
        # Use the new entity resolver for Ensembl IDs
        try:
            from cliquefinder.validation.entity_resolver import GeneEntityResolver

            resolver = GeneEntityResolver()
            results = resolver.resolve_ensembl_ids(feature_ids, verbose=verbose)

            # Build symbol -> feature_id mapping including aliases
            symbol_to_feature = {}
            n_resolved = 0

            for eid, entity in results.items():
                if entity.is_resolved:
                    symbol_to_feature[entity.symbol] = eid
                    n_resolved += 1

                    # Also include aliases for broader matching
                    for alias in entity.aliases:
                        if alias and alias not in symbol_to_feature:
                            symbol_to_feature[alias] = eid

            if verbose:
                print(f"  Symbol mappings: {len(symbol_to_feature)} (from {n_resolved} resolved genes + aliases)")

            return symbol_to_feature

        except ImportError:
            if verbose:
                print("  Warning: entity_resolver not available, falling back to mygene")
            # Fall through to legacy mygene approach

    # Legacy mygene approach for UniProt IDs or as fallback
    try:
        import mygene
    except ImportError:
        raise ImportError("mygene package required for ID mapping. Install with: pip install mygene")

    if ensembl_count > len(sample_ids) * 0.5:
        scopes = 'ensembl.gene'
        type_name = 'Ensembl'
    else:
        scopes = 'uniprot'
        type_name = 'UniProt'

    mg = mygene.MyGeneInfo()

    if verbose:
        print(f"  Mapping {len(feature_ids)} {type_name} IDs to gene symbols...")

    results = mg.querymany(
        feature_ids,
        scopes=scopes,
        fields='symbol,alias',
        species='human',
        returnall=True,
    )

    symbol_to_feature = {}

    for hit in results.get('out', []):
        if 'symbol' in hit and 'query' in hit:
            symbol = hit['symbol']
            feature_id = hit['query']
            symbol_to_feature[symbol] = feature_id

            # Also add aliases
            aliases = hit.get('alias', [])
            if isinstance(aliases, str):
                aliases = [aliases]
            for alias in aliases:
                if alias and alias not in symbol_to_feature:
                    symbol_to_feature[alias] = feature_id

    if verbose:
        print(f"  Mapped {len(symbol_to_feature)} symbols/aliases")

    return symbol_to_feature


# Legacy alias for backwards compatibility
map_uniprot_to_symbols = map_feature_ids_to_symbols


def run_clique_differential_analysis(
    data: NDArray[np.float64],
    feature_ids: list[str],
    sample_metadata: pd.DataFrame,
    clique_definitions: list[CliqueDefinition],
    condition_col: str = "phenotype",
    subject_col: str | None = "subject_id",
    contrasts: dict[str, tuple[str, str]] | None = None,
    summarization_method: SummarizationMethod | str = SummarizationMethod.TUKEY_MEDIAN_POLISH,
    normalization_method: NormalizationMethod | str = NormalizationMethod.NONE,
    imputation_method: ImputationMethod | str = ImputationMethod.NONE,
    use_mixed_model: bool = True,
    fdr_method: str = "BH",
    fdr_threshold: float = 0.05,
    min_proteins_found: int = 2,
    also_run_protein_level: bool = False,
    map_ids: bool = True,
    n_jobs: int = 1,
    verbose: bool = True,
) -> CliqueAnalysisResult:
    """
    Run clique-level differential abundance analysis.

    This is the main entry point for MSstats-style analysis at the clique level.

    Pipeline:
        1. (Optional) Normalize data
        2. (Optional) Impute missing values
        3. For each clique:
           a. Extract protein data
           b. Summarize to clique level (Tukey's Median Polish)
           c. Fit linear mixed model
           d. Test contrasts
        4. Apply FDR correction across cliques
        5. (Optional) Also run protein-level analysis

    Args:
        data: 2D array (n_features, n_samples) of log2 intensities.
        feature_ids: List of protein identifiers.
        sample_metadata: DataFrame with sample information.
        clique_definitions: List of cliques to analyze.
        condition_col: Metadata column for condition labels.
        subject_col: Metadata column for subject IDs (None for no random effects).
        contrasts: Dict of contrasts to test (None for all pairwise).
        summarization_method: How to aggregate proteins within clique.
        normalization_method: Normalization to apply.
        imputation_method: Missing value imputation method.
        use_mixed_model: Whether to use mixed models.
        fdr_method: Multiple testing correction method.
        fdr_threshold: Significance threshold.
        min_proteins_found: Minimum proteins required in data.
        also_run_protein_level: Also run individual protein analysis.
        n_jobs: Parallel jobs.
        verbose: Print progress.

    Returns:
        CliqueAnalysisResult with all results.

    Example:
        >>> cliques = load_clique_definitions("output/cliques/cliques.csv")
        >>> result = run_clique_differential_analysis(
        ...     data=matrix.data,
        ...     feature_ids=list(matrix.feature_ids),
        ...     sample_metadata=matrix.sample_metadata,
        ...     clique_definitions=cliques,
        ...     condition_col="phenotype",
        ...     subject_col="subject_id",
        ...     contrasts={"CASE_vs_CTRL": ("CASE", "CTRL")},
        ... )
        >>> print(result.summary())
        >>> result.to_dataframe().to_csv("clique_differential.csv")
    """
    if isinstance(summarization_method, str):
        summarization_method = SummarizationMethod(summarization_method)

    n_features, n_samples = data.shape

    if verbose:
        print(f"Clique Differential Analysis")
        print(f"=" * 50)
        print(f"Data: {n_features} proteins × {n_samples} samples")
        print(f"Cliques: {len(clique_definitions)}")
        print(f"Summarization: {summarization_method.value}")

    # Check if we need ID mapping (cliques use symbols, data uses other IDs)
    # Sample some clique proteins and check if they're in feature_ids
    symbol_to_feature = {}
    if map_ids and len(clique_definitions) > 0:
        sample_proteins = []
        for clique in clique_definitions[:10]:
            sample_proteins.extend(clique.protein_ids[:3])

        # Check if sample proteins match feature_ids
        matches = sum(1 for p in sample_proteins if p in feature_ids)
        if matches < len(sample_proteins) * 0.5:
            # Low match rate - likely need ID mapping
            if verbose:
                print(f"\nID mapping required (cliques use symbols, data uses other IDs)")
            symbol_to_feature = map_feature_ids_to_symbols(feature_ids, verbose=verbose)

    # Create lookup dict - use symbol if mapped, otherwise original ID
    feature_to_idx = {f: i for i, f in enumerate(feature_ids)}

    # Also add symbol → idx mapping if we did ID conversion
    if symbol_to_feature:
        for symbol, feature_id in symbol_to_feature.items():
            if feature_id in feature_to_idx:
                feature_to_idx[symbol] = feature_to_idx[feature_id]

        if verbose:
            print(f"  Added {len(symbol_to_feature)} symbol mappings to lookup")

    # Preprocessing
    work_data = data.astype(np.float64).copy()

    # Normalization
    if normalization_method != NormalizationMethod.NONE:
        if verbose:
            print(f"Normalizing: {normalization_method}")
        norm_result = normalize(work_data, method=normalization_method)
        work_data = norm_result.data

    # Imputation
    if imputation_method != ImputationMethod.NONE:
        if verbose:
            analysis = analyze_missing_values(work_data)
            print(f"Missing values: {analysis.n_missing} ({analysis.missing_rate:.1%})")
            print(f"Imputing: {imputation_method}")
        imp_result = impute_missing_values(work_data, method=imputation_method)
        work_data = imp_result.data

    # Get condition and subject arrays
    sample_condition = sample_metadata[condition_col].values
    sample_subject = sample_metadata[subject_col].values if subject_col and subject_col in sample_metadata.columns else None

    # Get unique conditions for contrast building
    conditions = sorted(pd.Series(sample_condition).dropna().unique().tolist())

    if contrasts is None:
        # Default: first condition vs second
        if len(conditions) >= 2:
            contrasts = {f"{conditions[0]}_vs_{conditions[1]}": (conditions[0], conditions[1])}
        else:
            raise ValueError(f"Need at least 2 conditions, got {conditions}")

    contrast_names = list(contrasts.keys())

    if verbose:
        print(f"Conditions: {conditions}")
        print(f"Contrasts: {contrast_names}")
        print()

    # Analyze each clique
    clique_results: list[CliqueDifferentialResult] = []

    for i, clique in enumerate(clique_definitions):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Processing clique {i + 1}/{len(clique_definitions)}")

        # Find proteins in data
        protein_indices = []
        proteins_found = []
        for pid in clique.protein_ids:
            if pid in feature_to_idx:
                protein_indices.append(feature_to_idx[pid])
                proteins_found.append(pid)

        if len(proteins_found) < min_proteins_found:
            # Skip cliques with too few proteins
            continue

        # Extract clique protein data
        clique_data = work_data[protein_indices, :]

        # Summarize to clique level
        summary = summarize_clique(
            clique_data,
            proteins_found,
            clique.clique_id,
            method=summarization_method,
            compute_coherence=True,
        )

        # Run differential analysis on summarized clique
        clique_intensities = summary.sample_abundances

        # Single-feature differential analysis
        from .differential import differential_analysis_single, build_contrast_matrix

        contrast_matrix, _ = build_contrast_matrix(conditions, contrasts)

        result = differential_analysis_single(
            intensities=clique_intensities,
            condition=sample_condition,
            subject=sample_subject,
            feature_id=clique.clique_id,
            contrast_matrix=contrast_matrix,
            contrast_names=contrast_names,
            conditions=conditions,
            use_mixed=use_mixed_model,
        )

        # Convert to CliqueDifferentialResult
        for contrast_result in result.contrasts:
            clique_results.append(CliqueDifferentialResult(
                clique_id=clique.clique_id,
                regulator=clique.regulator,
                n_proteins=len(clique.protein_ids),
                n_proteins_found=len(proteins_found),
                protein_ids=proteins_found,
                summarization_method=summarization_method.value,
                coherence=summary.coherence,
                log2_fc=contrast_result.log2_fc,
                se=contrast_result.se,
                t_value=contrast_result.t_value,
                df=contrast_result.df,
                p_value=contrast_result.p_value,
                adj_p_value=None,  # Filled in below
                ci_lower=contrast_result.ci_lower,
                ci_upper=contrast_result.ci_upper,
                contrast=contrast_result.contrast_name,
                model_type=result.model_type.value,
                issue=result.issue,
            ))

    # Apply FDR correction per contrast
    for contrast in contrast_names:
        contrast_results = [r for r in clique_results if r.contrast == contrast]
        if not contrast_results:
            continue

        pvals = np.array([r.p_value for r in contrast_results])
        adj_pvals = fdr_correction(pvals, method=fdr_method)

        for r, adj_p in zip(contrast_results, adj_pvals):
            r.adj_p_value = float(adj_p) if not np.isnan(adj_p) else None

    # Count significant
    n_significant = sum(
        1 for r in clique_results
        if r.adj_p_value is not None and r.adj_p_value < fdr_threshold
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Cliques tested: {len(set(r.clique_id for r in clique_results))}")
        print(f"  Significant (FDR < {fdr_threshold}): {n_significant}")

    # Optional: protein-level analysis
    protein_results = None
    if also_run_protein_level:
        if verbose:
            print(f"\nRunning protein-level analysis...")
        protein_results = run_differential_analysis(
            data=work_data,
            feature_ids=feature_ids,
            sample_condition=sample_condition,
            sample_subject=sample_subject,
            contrasts=contrasts,
            use_mixed=use_mixed_model,
            fdr_method=fdr_method,
            fdr_threshold=fdr_threshold,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    return CliqueAnalysisResult(
        clique_results=clique_results,
        protein_results=protein_results,
        contrasts_tested=contrast_names,
        n_cliques_tested=len(set(r.clique_id for r in clique_results)),
        n_significant=n_significant,
        fdr_threshold=fdr_threshold,
        preprocessing_params={
            'normalization': normalization_method.value if hasattr(normalization_method, 'value') else str(normalization_method),
            'imputation': imputation_method.value if hasattr(imputation_method, 'value') else str(imputation_method),
            'summarization': summarization_method.value,
            'use_mixed_model': use_mixed_model,
            'fdr_method': fdr_method,
        },
    )


def compare_protein_vs_clique_results(
    clique_results: CliqueAnalysisResult,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Compare protein-level and clique-level results.

    Identifies cases where:
    - Clique is significant but individual proteins are not
    - Individual proteins are significant but clique is not

    Args:
        clique_results: Results from run_clique_differential_analysis.
        threshold: Significance threshold.

    Returns:
        DataFrame comparing protein and clique results.
    """
    if clique_results.protein_results is None:
        raise ValueError("Protein-level results not available. Run with also_run_protein_level=True")

    protein_df = clique_results.protein_results.to_dataframe()
    clique_df = clique_results.to_dataframe()

    comparisons = []

    for _, clique_row in clique_df.iterrows():
        clique_id = clique_row['clique_id']
        proteins = clique_row['proteins'].split(',')
        contrast = clique_row['contrast']

        # Get protein results for this clique's proteins
        protein_mask = protein_df['feature_id'].isin(proteins) & (protein_df['contrast'] == contrast)
        clique_proteins = protein_df[protein_mask]

        n_sig_proteins = (clique_proteins['adj_pvalue'] < threshold).sum()
        clique_sig = clique_row['adj_pvalue'] < threshold if pd.notna(clique_row['adj_pvalue']) else False

        comparisons.append({
            'clique_id': clique_id,
            'contrast': contrast,
            'n_proteins': len(proteins),
            'n_sig_proteins': n_sig_proteins,
            'clique_significant': clique_sig,
            'clique_log2FC': clique_row['log2FC'],
            'clique_pvalue': clique_row['adj_pvalue'],
            'mean_protein_log2FC': clique_proteins['log2FC'].mean() if len(clique_proteins) > 0 else np.nan,
            'coordinated_detection': clique_sig and n_sig_proteins == 0,  # Clique detects what proteins miss
            'individual_detection': not clique_sig and n_sig_proteins > 0,  # Proteins detect what clique misses
        })

    return pd.DataFrame(comparisons)


# =============================================================================
# Permutation-Based Significance Testing
# =============================================================================

@dataclass
class PermutationTestResult:
    """Result of permutation-based clique significance testing.

    The collaborator's approach: compare TF-defined cliques against
    random gene sets of the same size from the pool of all regulated genes.
    This controls for multiple testing without BH's independence assumption.

    Attributes:
        clique_id: TF/clique identifier
        observed_log2fc: Observed log2 fold change for the clique
        observed_pvalue: Observed p-value from MSstats
        observed_tvalue: Observed t-statistic
        null_log2fc_mean: Mean log2FC from permuted null
        null_log2fc_std: Std of log2FC from permuted null
        null_tvalue_mean: Mean t-statistic from permuted null
        empirical_pvalue: Proportion of permutations with |t| >= |observed t|
        empirical_pvalue_directional: One-sided (same direction as observed)
        n_permutations: Number of permutations run
        percentile_rank: Where observed t falls in null distribution (0-100)
        is_significant: True if empirical_pvalue < threshold
    """
    clique_id: str
    observed_log2fc: float
    observed_pvalue: float
    observed_tvalue: float
    null_log2fc_mean: float
    null_log2fc_std: float
    null_tvalue_mean: float
    empirical_pvalue: float
    empirical_pvalue_directional: float
    n_permutations: int
    percentile_rank: float
    is_significant: bool = False

    def to_dict(self) -> dict:
        return {
            'clique_id': self.clique_id,
            'observed_log2FC': self.observed_log2fc,
            'observed_pvalue': self.observed_pvalue,
            'observed_tvalue': self.observed_tvalue,
            'null_log2FC_mean': self.null_log2fc_mean,
            'null_log2FC_std': self.null_log2fc_std,
            'null_tvalue_mean': self.null_tvalue_mean,
            'empirical_pvalue': self.empirical_pvalue,
            'empirical_pvalue_directional': self.empirical_pvalue_directional,
            'n_permutations': self.n_permutations,
            'percentile_rank': self.percentile_rank,
            'is_significant': self.is_significant,
        }


def run_permutation_clique_test(
    data: NDArray[np.float64],
    feature_ids: list[str],
    sample_metadata: pd.DataFrame,
    clique_definitions: list[CliqueDefinition],
    condition_col: str = "phenotype",
    subject_col: str | None = "subject_id",
    contrast: tuple[str, str] = ("CASE", "CTRL"),
    summarization_method: SummarizationMethod | str = SummarizationMethod.TUKEY_MEDIAN_POLISH,
    n_permutations: int = 1000,
    use_mixed_model: bool = True,
    significance_threshold: float = 0.05,
    random_state: int | None = None,
    map_ids: bool = True,
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple[list[PermutationTestResult], pd.DataFrame]:
    """
    Permutation-based significance testing for clique differential abundance.

    Implements the collaborator's approach:
    1. Run MSstats on TF-defined cliques (observed)
    2. Generate null by sampling random gene sets of same sizes from
       the pool of all regulated genes
    3. Compute empirical p-values by comparing observed vs null

    This controls for multiple testing WITHOUT assuming independence,
    preserving the correlation structure of gene expression.

    Statistical Framework:
        H0: TF cliques are no more differentially abundant than random
            gene sets of the same size from regulated genes
        H1: TF cliques show coordinated differential abundance beyond chance

        Empirical p-value = (# permutations with |t| >= |observed t| + 1) / (N + 1)

    Args:
        data: 2D array (n_features, n_samples) of log2 intensities
        feature_ids: List of protein identifiers
        sample_metadata: DataFrame with sample information
        clique_definitions: List of TF cliques to test
        condition_col: Metadata column for condition labels
        subject_col: Metadata column for subject IDs
        contrast: Tuple of (test_condition, reference_condition)
        summarization_method: How to aggregate proteins within clique
        n_permutations: Number of permutations for null distribution
        use_mixed_model: Whether to use mixed models
        significance_threshold: Threshold for empirical p-value
        random_state: Random seed for reproducibility
        n_jobs: Parallel jobs for permutations
        verbose: Print progress

    Returns:
        Tuple of (list of PermutationTestResult, DataFrame with null distribution stats)

    Example:
        >>> results, null_df = run_permutation_clique_test(
        ...     data=matrix.data,
        ...     feature_ids=list(matrix.feature_ids),
        ...     sample_metadata=matrix.sample_metadata,
        ...     clique_definitions=cliques,
        ...     contrast=("CASE", "CTRL"),
        ...     n_permutations=1000,
        ... )
        >>> sig_cliques = [r for r in results if r.is_significant]
        >>> print(f"Significant cliques: {len(sig_cliques)}/{len(results)}")

    References:
        - Competitive gene set enrichment (Subramanian et al., PNAS 2005)
        - Permutation testing preserves correlation structure
    """
    if isinstance(summarization_method, str):
        summarization_method = SummarizationMethod(summarization_method)

    if random_state is not None:
        np.random.seed(random_state)

    n_features, n_samples = data.shape

    if verbose:
        print(f"Permutation-Based Clique Significance Testing")
        print(f"=" * 55)
        print(f"Cliques: {len(clique_definitions)}")
        print(f"Permutations: {n_permutations}")
        print(f"Contrast: {contrast[0]} vs {contrast[1]}")

    # Build feature index lookup
    feature_to_idx = {f: i for i, f in enumerate(feature_ids)}

    # ID mapping: cliques may use gene symbols while data uses UniProt/Ensembl IDs
    symbol_to_feature = {}
    if map_ids and len(clique_definitions) > 0:
        # Check if clique proteins match feature_ids
        sample_proteins = []
        for clique in clique_definitions[:10]:
            sample_proteins.extend(clique.protein_ids[:3])

        matches = sum(1 for p in sample_proteins if p in feature_to_idx)
        if matches < len(sample_proteins) * 0.5:
            if verbose:
                print(f"\nID mapping required (cliques use symbols, data uses other IDs)")
            symbol_to_feature = map_feature_ids_to_symbols(feature_ids, verbose=verbose)

            # Add symbol -> idx mappings
            for symbol, feature_id in symbol_to_feature.items():
                if feature_id in feature_to_idx:
                    feature_to_idx[symbol] = feature_to_idx[feature_id]

            if verbose:
                print(f"  Added {len(symbol_to_feature)} symbol mappings")

    # Collect all regulated genes (pool for permutation)
    all_regulated_genes: set[str] = set()
    clique_sizes: dict[str, int] = {}

    for clique in clique_definitions:
        # Find proteins present in data
        present_proteins = [p for p in clique.protein_ids if p in feature_to_idx]
        if len(present_proteins) >= 2:
            all_regulated_genes.update(present_proteins)
            clique_sizes[clique.clique_id] = len(present_proteins)

    regulated_genes_list = list(all_regulated_genes)

    if verbose:
        print(f"Regulated gene pool: {len(regulated_genes_list)} genes")
        if clique_sizes:
            print(f"Clique sizes: min={min(clique_sizes.values())}, max={max(clique_sizes.values())}")
        else:
            print(f"WARNING: No cliques have >= 2 proteins in data")

    if not clique_sizes:
        if verbose:
            print("No valid cliques found - returning empty results")
        return [], pd.DataFrame()

    # Get condition arrays
    sample_condition = sample_metadata[condition_col].values
    sample_subject = sample_metadata[subject_col].values if subject_col and subject_col in sample_metadata.columns else None

    conditions = sorted([contrast[0], contrast[1]])
    contrast_dict = {f"{contrast[0]}_vs_{contrast[1]}": contrast}

    # Helper function to run differential analysis on a gene set
    def analyze_gene_set(
        gene_set: list[str],
        set_id: str,
    ) -> tuple[float, float, float] | None:
        """Summarize and test a gene set, return (log2fc, pvalue, tvalue)."""
        # Get protein indices
        indices = [feature_to_idx[g] for g in gene_set if g in feature_to_idx]
        if len(indices) < 2:
            return None

        # Extract and summarize
        protein_data = data[indices, :]
        summary = summarize_clique(
            protein_data,
            gene_set,
            set_id,
            method=summarization_method,
            compute_coherence=False,
        )

        # Run differential analysis
        from .differential import differential_analysis_single, build_contrast_matrix

        contrast_matrix, _ = build_contrast_matrix(conditions, contrast_dict)

        try:
            result = differential_analysis_single(
                intensities=summary.sample_abundances,
                condition=sample_condition,
                subject=sample_subject,
                feature_id=set_id,
                contrast_matrix=contrast_matrix,
                contrast_names=[f"{contrast[0]}_vs_{contrast[1]}"],
                conditions=conditions,
                use_mixed=use_mixed_model,
            )

            if result.contrasts:
                c = result.contrasts[0]
                return (c.log2_fc, c.p_value, c.t_value)
        except Exception:
            pass

        return None

    # Step 1: Run observed analysis on actual cliques
    if verbose:
        print(f"\n[1/3] Running observed clique analysis...")

    observed_results: dict[str, tuple[float, float, float]] = {}

    for clique in clique_definitions:
        present_proteins = [p for p in clique.protein_ids if p in feature_to_idx]
        if len(present_proteins) < 2:
            continue

        result = analyze_gene_set(present_proteins, clique.clique_id)
        if result is not None:
            observed_results[clique.clique_id] = result

    if verbose:
        print(f"  Analyzed {len(observed_results)} cliques")

    # Step 2: Generate null distribution via permutation
    if verbose:
        print(f"\n[2/3] Generating null distribution ({n_permutations} permutations)...")

    # For each clique, we'll collect null statistics
    null_distributions: dict[str, list[tuple[float, float, float]]] = {
        cid: [] for cid in observed_results.keys()
    }

    for perm_idx in range(n_permutations):
        if verbose and (perm_idx + 1) % 100 == 0:
            print(f"  Permutation {perm_idx + 1}/{n_permutations}")

        # For each clique, sample random genes of same size
        for clique_id, size in clique_sizes.items():
            if clique_id not in observed_results:
                continue

            # Sample random genes from pool (without replacement)
            if size <= len(regulated_genes_list):
                random_genes = list(np.random.choice(
                    regulated_genes_list, size=size, replace=False
                ))
            else:
                random_genes = regulated_genes_list.copy()

            result = analyze_gene_set(random_genes, f"perm_{perm_idx}_{clique_id}")
            if result is not None:
                null_distributions[clique_id].append(result)

    # Step 3: Compute empirical p-values
    if verbose:
        print(f"\n[3/3] Computing empirical p-values...")

    permutation_results: list[PermutationTestResult] = []

    for clique_id, (obs_log2fc, obs_pval, obs_tval) in observed_results.items():
        null_stats = null_distributions[clique_id]

        if len(null_stats) < 10:
            # Too few permutations succeeded
            continue

        null_log2fc = np.array([s[0] for s in null_stats])
        null_tvals = np.array([s[2] for s in null_stats])

        # Two-sided empirical p-value: |t| >= |observed t|
        n_extreme = np.sum(np.abs(null_tvals) >= np.abs(obs_tval))
        empirical_pval = (n_extreme + 1) / (len(null_stats) + 1)

        # One-sided (directional): same sign and >= magnitude
        if obs_tval > 0:
            n_extreme_dir = np.sum(null_tvals >= obs_tval)
        else:
            n_extreme_dir = np.sum(null_tvals <= obs_tval)
        empirical_pval_dir = (n_extreme_dir + 1) / (len(null_stats) + 1)

        # Percentile rank
        percentile = 100 * np.mean(np.abs(null_tvals) < np.abs(obs_tval))

        permutation_results.append(PermutationTestResult(
            clique_id=clique_id,
            observed_log2fc=obs_log2fc,
            observed_pvalue=obs_pval,
            observed_tvalue=obs_tval,
            null_log2fc_mean=float(np.mean(null_log2fc)),
            null_log2fc_std=float(np.std(null_log2fc)),
            null_tvalue_mean=float(np.mean(null_tvals)),
            empirical_pvalue=empirical_pval,
            empirical_pvalue_directional=empirical_pval_dir,
            n_permutations=len(null_stats),
            percentile_rank=percentile,
            is_significant=empirical_pval < significance_threshold,
        ))

    # Create null distribution summary DataFrame
    null_summary_rows = []
    for clique_id, null_stats in null_distributions.items():
        if len(null_stats) > 0:
            null_log2fc = [s[0] for s in null_stats]
            null_tvals = [s[2] for s in null_stats]
            null_summary_rows.append({
                'clique_id': clique_id,
                'null_log2FC_mean': np.mean(null_log2fc),
                'null_log2FC_std': np.std(null_log2fc),
                'null_log2FC_5pct': np.percentile(null_log2fc, 5),
                'null_log2FC_95pct': np.percentile(null_log2fc, 95),
                'null_tvalue_mean': np.mean(null_tvals),
                'null_tvalue_std': np.std(null_tvals),
                'null_tvalue_5pct': np.percentile(null_tvals, 5),
                'null_tvalue_95pct': np.percentile(null_tvals, 95),
                'n_permutations': len(null_stats),
            })

    null_df = pd.DataFrame(null_summary_rows)

    # Summary
    n_significant = sum(1 for r in permutation_results if r.is_significant)

    if verbose:
        print(f"\nResults:")
        print(f"  Cliques tested: {len(permutation_results)}")
        print(f"  Significant (empirical p < {significance_threshold}): {n_significant}")
        print(f"  Significance rate: {100 * n_significant / len(permutation_results):.1f}%")

    return permutation_results, null_df


def run_matched_single_gene_comparison(
    data: NDArray[np.float64],
    feature_ids: list[str],
    sample_metadata: pd.DataFrame,
    clique_definitions: list[CliqueDefinition],
    condition_col: str = "phenotype",
    subject_col: str | None = "subject_id",
    contrast: tuple[str, str] = ("CASE", "CTRL"),
    n_random_samples: int = 100,
    use_mixed_model: bool = True,
    random_state: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compare clique-level vs matched single-gene differential analysis.

    Implements the collaborator's second suggestion:
    - Run MSstats on N cliques
    - Run MSstats on N randomly selected individual genes
    - Compare distributions of test statistics

    This tests whether clique aggregation improves signal detection
    compared to analyzing individual genes.

    Args:
        data: 2D array (n_features, n_samples) of log2 intensities
        feature_ids: List of protein identifiers
        sample_metadata: DataFrame with sample information
        clique_definitions: List of TF cliques
        condition_col: Metadata column for condition labels
        subject_col: Metadata column for subject IDs
        contrast: Tuple of (test_condition, reference_condition)
        n_random_samples: Number of random gene samples to compare
        use_mixed_model: Whether to use mixed models
        random_state: Random seed
        verbose: Print progress

    Returns:
        DataFrame comparing clique vs single-gene results with columns:
        - analysis_type: "clique" or "single_gene"
        - id: clique_id or gene_id
        - log2FC, pvalue, tvalue
        - abs_tvalue (for ranking)

    Example:
        >>> comparison_df = run_matched_single_gene_comparison(
        ...     data, feature_ids, metadata, cliques,
        ...     contrast=("CASE", "CTRL"),
        ... )
        >>> clique_df = comparison_df[comparison_df['analysis_type'] == 'clique']
        >>> gene_df = comparison_df[comparison_df['analysis_type'] == 'single_gene']
        >>> print(f"Median |t| cliques: {clique_df['abs_tvalue'].median():.2f}")
        >>> print(f"Median |t| genes: {gene_df['abs_tvalue'].median():.2f}")
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Build feature index
    feature_to_idx = {f: i for i, f in enumerate(feature_ids)}

    # Get condition arrays
    sample_condition = sample_metadata[condition_col].values
    sample_subject = sample_metadata[subject_col].values if subject_col and subject_col in sample_metadata.columns else None

    conditions = sorted([contrast[0], contrast[1]])
    contrast_dict = {f"{contrast[0]}_vs_{contrast[1]}": contrast}

    from .differential import differential_analysis_single, build_contrast_matrix
    contrast_matrix, _ = build_contrast_matrix(conditions, contrast_dict)

    results = []

    # Collect all regulated genes
    all_regulated: set[str] = set()
    for clique in clique_definitions:
        present = [p for p in clique.protein_ids if p in feature_to_idx]
        all_regulated.update(present)

    regulated_list = list(all_regulated)

    if verbose:
        print(f"Matched Single-Gene Comparison")
        print(f"=" * 40)
        print(f"Cliques: {len(clique_definitions)}")
        print(f"Regulated gene pool: {len(regulated_list)}")

    # Run clique-level analysis
    if verbose:
        print(f"\n[1/2] Running clique-level analysis...")

    for clique in clique_definitions:
        present_proteins = [p for p in clique.protein_ids if p in feature_to_idx]
        if len(present_proteins) < 2:
            continue

        indices = [feature_to_idx[p] for p in present_proteins]
        protein_data = data[indices, :]

        summary = summarize_clique(
            protein_data,
            present_proteins,
            clique.clique_id,
            method=SummarizationMethod.TUKEY_MEDIAN_POLISH,
            compute_coherence=False,
        )

        try:
            result = differential_analysis_single(
                intensities=summary.sample_abundances,
                condition=sample_condition,
                subject=sample_subject,
                feature_id=clique.clique_id,
                contrast_matrix=contrast_matrix,
                contrast_names=[f"{contrast[0]}_vs_{contrast[1]}"],
                conditions=conditions,
                use_mixed=use_mixed_model,
            )

            if result.contrasts:
                c = result.contrasts[0]
                results.append({
                    'analysis_type': 'clique',
                    'id': clique.clique_id,
                    'n_genes': len(present_proteins),
                    'log2FC': c.log2_fc,
                    'pvalue': c.p_value,
                    'tvalue': c.t_value,
                    'abs_tvalue': abs(c.t_value),
                })
        except Exception:
            pass

    n_cliques_tested = len([r for r in results if r['analysis_type'] == 'clique'])

    # Run single-gene analysis on matched number of random genes
    if verbose:
        print(f"\n[2/2] Running single-gene analysis (N={n_cliques_tested})...")

    # Sample same number of genes as cliques
    n_to_sample = min(n_cliques_tested, len(regulated_list))
    sampled_genes = np.random.choice(regulated_list, size=n_to_sample, replace=False)

    for gene in sampled_genes:
        if gene not in feature_to_idx:
            continue

        idx = feature_to_idx[gene]
        gene_intensities = data[idx, :]

        try:
            result = differential_analysis_single(
                intensities=gene_intensities,
                condition=sample_condition,
                subject=sample_subject,
                feature_id=gene,
                contrast_matrix=contrast_matrix,
                contrast_names=[f"{contrast[0]}_vs_{contrast[1]}"],
                conditions=conditions,
                use_mixed=use_mixed_model,
            )

            if result.contrasts:
                c = result.contrasts[0]
                results.append({
                    'analysis_type': 'single_gene',
                    'id': gene,
                    'n_genes': 1,
                    'log2FC': c.log2_fc,
                    'pvalue': c.p_value,
                    'tvalue': c.t_value,
                    'abs_tvalue': abs(c.t_value),
                })
        except Exception:
            pass

    df = pd.DataFrame(results)

    # Summary statistics
    if verbose and len(df) > 0:
        clique_df = df[df['analysis_type'] == 'clique']
        gene_df = df[df['analysis_type'] == 'single_gene']

        print(f"\nComparison Summary:")
        print(f"  Cliques tested: {len(clique_df)}")
        print(f"  Single genes tested: {len(gene_df)}")

        if len(clique_df) > 0 and len(gene_df) > 0:
            print(f"\n  Median |t-statistic|:")
            print(f"    Cliques: {clique_df['abs_tvalue'].median():.3f}")
            print(f"    Single genes: {gene_df['abs_tvalue'].median():.3f}")

            print(f"\n  Median |log2FC|:")
            print(f"    Cliques: {clique_df['log2FC'].abs().median():.3f}")
            print(f"    Single genes: {gene_df['log2FC'].abs().median():.3f}")

            # Mann-Whitney U test for difference
            from scipy.stats import mannwhitneyu
            stat, pval = mannwhitneyu(
                clique_df['abs_tvalue'],
                gene_df['abs_tvalue'],
                alternative='greater'
            )
            print(f"\n  Mann-Whitney U (cliques > genes): p={pval:.4f}")

    return df
