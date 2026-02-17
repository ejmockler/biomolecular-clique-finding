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
        direction: Classification of correlation signs in clique.
            POSITIVE = all edges have r > 0 (co-activation)
            NEGATIVE = all edges have r < 0 (anti-correlation/repression)
            MIXED = edges have both positive and negative correlations
            UNKNOWN = direction information not available
        signed_mean_correlation: Mean correlation preserving sign. Unlike
            coherence (which may use absolute values), this can be negative
            for anti-correlated cliques.
        signed_min_correlation: Minimum signed correlation (most negative or
            least positive).
        signed_max_correlation: Maximum signed correlation (most positive or
            least negative).
        n_positive_edges: Number of edges with r > 0.
        n_negative_edges: Number of edges with r < 0.
    """

    clique_id: str
    protein_ids: list[str]
    regulator: str | None = None
    condition: str | None = None
    coherence: float | None = None
    n_indra_targets: int | None = None
    direction: str = "unknown"
    signed_mean_correlation: float | None = None
    signed_min_correlation: float | None = None
    signed_max_correlation: float | None = None
    n_positive_edges: int = 0
    n_negative_edges: int = 0

    @property
    def is_coherent(self) -> bool:
        """True if clique has uniform correlation direction (positive or negative).

        Returns:
            True if direction is "positive" or "negative" (not "mixed" or "unknown").
            Coherent cliques are suitable for standard summarization methods like
            Tukey's Median Polish that assume all features move together.
        """
        return self.direction.lower() in ("positive", "negative")

    @property
    def is_mixed(self) -> bool:
        """True if clique has mixed correlation signs.

        Returns:
            True if direction is "mixed", indicating the clique contains both
            positively and negatively correlated edges. Mixed cliques violate
            the assumptions of additive summarization methods and may produce
            unreliable results.
        """
        return self.direction.lower() == "mixed"


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
        direction: Classification of correlation signs in clique.
            POSITIVE = all edges have r > 0 (co-activation)
            NEGATIVE = all edges have r < 0 (anti-correlation/repression)
            MIXED = edges have both positive and negative correlations
        signed_mean_correlation: Mean correlation preserving sign. Unlike
            coherence (which may use absolute values), this can be negative
            for anti-correlated cliques.
        signed_min_correlation: Minimum signed correlation (most negative or
            least positive).
        signed_max_correlation: Maximum signed correlation (most positive or
            least negative).
        n_positive_edges: Number of edges with r > 0.
        n_negative_edges: Number of edges with r < 0.
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
    direction: str = "unknown"
    signed_mean_correlation: float | None = None
    signed_min_correlation: float | None = None
    signed_max_correlation: float | None = None
    n_positive_edges: int = 0
    n_negative_edges: int = 0

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
            'direction': self.direction,
            'signed_mean_correlation': self.signed_mean_correlation,
            'signed_min_correlation': self.signed_min_correlation,
            'signed_max_correlation': self.signed_max_correlation,
            'n_positive_edges': self.n_positive_edges,
            'n_negative_edges': self.n_negative_edges,
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

    Supports both legacy CSVs (without direction info) and modern CSVs
    with correlation direction metadata from upstream clique discovery.

    Expected columns (all optional except regulator/clique_id and proteins):
        - regulator or clique_id: Clique identifier
        - clique_genes: Comma-separated protein IDs
        - direction: "positive", "negative", "mixed", "unknown" (optional)
        - n_positive_edges: Number of edges with r > 0 (optional)
        - n_negative_edges: Number of edges with r < 0 (optional)
        - signed_mean_correlation: Mean correlation preserving sign (optional)
        - signed_min_correlation: Minimum signed correlation (optional)
        - signed_max_correlation: Maximum signed correlation (optional)
        - coherence: Mean pairwise correlation (optional)
        - condition: Discovery condition (optional)
        - n_indra_targets: Number of INDRA-validated targets (optional)

    Args:
        cliques_file: Path to cliques.csv or similar.
        min_proteins: Minimum proteins required per clique.

    Returns:
        List of CliqueDefinition objects. If direction columns are missing,
        all cliques will have direction="unknown" and a warning is issued.

    Example:
        >>> # Load modern CSV with direction info
        >>> cliques = load_clique_definitions("output/cliques.csv")
        >>> coherent = [c for c in cliques if c.is_coherent]
        >>> mixed = [c for c in cliques if c.is_mixed]

        >>> # Load legacy CSV (warning issued, direction="unknown")
        >>> cliques_old = load_clique_definitions("legacy_cliques.csv")
    """
    df = pd.read_csv(cliques_file)

    # Check for direction columns and issue warning if missing
    has_direction = 'direction' in df.columns
    if not has_direction:
        warnings.warn(
            f"Direction columns not found in {cliques_file}. "
            "All cliques will be treated as 'unknown' direction. "
            "Re-run 'cliquefinder analyze' to generate direction info.",
            UserWarning,
            stacklevel=2,
        )

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

        # Extract new signed correlation fields with backward compatibility
        direction = 'unknown'
        signed_mean_correlation = None
        signed_min_correlation = None
        signed_max_correlation = None
        n_positive_edges = 0
        n_negative_edges = 0

        if 'direction' in group.columns:
            val = group['direction'].iloc[0]
            # Normalize to lowercase for consistency
            direction = str(val).lower() if pd.notna(val) else 'unknown'
        if 'signed_mean_correlation' in group.columns:
            val = group['signed_mean_correlation'].iloc[0]
            signed_mean_correlation = float(val) if pd.notna(val) else None
        if 'signed_min_correlation' in group.columns:
            val = group['signed_min_correlation'].iloc[0]
            signed_min_correlation = float(val) if pd.notna(val) else None
        if 'signed_max_correlation' in group.columns:
            val = group['signed_max_correlation'].iloc[0]
            signed_max_correlation = float(val) if pd.notna(val) else None
        if 'n_positive_edges' in group.columns:
            val = group['n_positive_edges'].iloc[0]
            n_positive_edges = int(val) if pd.notna(val) else 0
        if 'n_negative_edges' in group.columns:
            val = group['n_negative_edges'].iloc[0]
            n_negative_edges = int(val) if pd.notna(val) else 0

        clique = CliqueDefinition(
            clique_id=str(clique_id),
            protein_ids=proteins,
            regulator=str(clique_id) if id_col == 'regulator' else None,
            condition=group['condition'].iloc[0] if 'condition' in group.columns else None,
            coherence=float(group['coherence'].iloc[0]) if 'coherence' in group.columns else None,
            n_indra_targets=int(group['n_indra_targets'].iloc[0]) if 'n_indra_targets' in group.columns else None,
            direction=direction,
            signed_mean_correlation=signed_mean_correlation,
            signed_min_correlation=signed_min_correlation,
            signed_max_correlation=signed_max_correlation,
            n_positive_edges=n_positive_edges,
            n_negative_edges=n_negative_edges,
        )
        cliques.append(clique)

    return cliques


# Module-level cache for ID mappings (avoids redundant API calls in bootstrap)
_ID_MAPPING_CACHE: dict[tuple[str, ...], dict[str, str]] = {}


def map_feature_ids_to_symbols(
    feature_ids: list[str],
    verbose: bool = True,
    use_cache: bool = True,
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
        use_cache: If True, check/store results in module-level cache.
                   CRITICAL for bootstrap efficiency - avoids N redundant API calls.

    Returns:
        Dict mapping symbol → feature_id (for reverse lookup).
    """
    # Compute cache key (used for both lookup and storage)
    cache_key = tuple(sorted(feature_ids)) if use_cache else None

    # Check cache first (critical for bootstrap performance)
    if use_cache and cache_key in _ID_MAPPING_CACHE:
        if verbose:
            print("  Using cached ID mappings (skipping API calls)")
        return _ID_MAPPING_CACHE[cache_key].copy()

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

            # Store in cache for bootstrap reuse
            if use_cache:
                _ID_MAPPING_CACHE[cache_key] = symbol_to_feature.copy()

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

    # Store in cache for bootstrap reuse
    if use_cache and cache_key is not None:
        _ID_MAPPING_CACHE[cache_key] = symbol_to_feature.copy()

    return symbol_to_feature


def clear_id_mapping_cache() -> int:
    """Clear the module-level ID mapping cache.

    Returns:
        Number of entries cleared.
    """
    global _ID_MAPPING_CACHE
    count = len(_ID_MAPPING_CACHE)
    _ID_MAPPING_CACHE = {}
    return count


# Legacy alias for backwards compatibility
map_uniprot_to_symbols = map_feature_ids_to_symbols


def modules_to_clique_definitions(
    modules: list,
    symbol_to_feature: dict[str, str],
    min_genes_found: int = 3,
    verbose: bool = True,
) -> list[CliqueDefinition]:
    """
    Convert INDRA modules to CliqueDefinition objects for differential testing.

    Maps gene symbols from INDRA modules back to feature IDs (UniProt/Ensembl)
    present in the data, creating CliqueDefinition objects compatible with all
    differential testing methods (ROAST, permutation, MSstats).

    Args:
        modules: List of INDRAModule objects from discover_modules().
        symbol_to_feature: Dict mapping gene_symbol -> feature_id in data.
            Obtained from map_feature_ids_to_symbols().
        min_genes_found: Minimum genes that must map to feature IDs.
            Modules with fewer matches are skipped.
        verbose: Print progress information.

    Returns:
        List of CliqueDefinition objects, one per module with sufficient
        genes in the data.
    """
    clique_defs = []
    n_skipped = 0

    for module in modules:
        # Map gene symbols to feature IDs
        mapped_feature_ids = []
        for gene_name in module.indra_target_names:
            if gene_name in symbol_to_feature:
                mapped_feature_ids.append(symbol_to_feature[gene_name])

        if len(mapped_feature_ids) < min_genes_found:
            n_skipped += 1
            continue

        # Determine direction from INDRA edge types
        n_activated = len(module.activated_targets)
        n_repressed = len(module.repressed_targets)

        if n_activated > 0 and n_repressed == 0:
            direction = "positive"
        elif n_repressed > 0 and n_activated == 0:
            direction = "negative"
        elif n_activated > 0 and n_repressed > 0:
            direction = "mixed"
        else:
            direction = "unknown"

        clique_def = CliqueDefinition(
            clique_id=module.regulator_name,
            protein_ids=mapped_feature_ids,
            regulator=module.regulator_name,
            condition=None,
            coherence=None,
            n_indra_targets=len(module.indra_target_names),
            direction=direction,
            n_positive_edges=n_activated,
            n_negative_edges=n_repressed,
        )
        clique_defs.append(clique_def)

    if verbose:
        print(f"  Converted {len(clique_defs)} INDRA modules to gene set definitions")
        print(f"  Skipped {n_skipped} modules with < {min_genes_found} mapped genes")
        if clique_defs:
            sizes = [len(c.protein_ids) for c in clique_defs]
            print(f"  Gene set sizes: min={min(sizes)}, median={sorted(sizes)[len(sizes)//2]}, max={max(sizes)}")

    return clique_defs


def discover_gene_sets_from_indra(
    feature_ids: list[str],
    min_evidence: int = 2,
    min_targets: int = 10,
    max_targets: int | None = None,
    max_regulators: int | None = None,
    regulator_classes: set | None = None,
    stmt_types: list[str] | None = None,
    min_genes_found: int = 3,
    env_file: str | None = None,
    verbose: bool = True,
) -> list[CliqueDefinition]:
    """
    Discover INDRA regulatory modules and convert to CliqueDefinitions.

    One-step pipeline: maps feature IDs to gene symbols, discovers upstream
    regulators from INDRA CoGEx via a single batch reverse query, then
    converts to CliqueDefinition objects suitable for ROAST, permutation,
    or MSstats differential testing.

    Args:
        feature_ids: List of UniProt/Ensembl IDs from the data matrix.
        min_evidence: Minimum INDRA evidence count per relationship.
        min_targets: Minimum unique target genes in data for a regulator.
        max_targets: Maximum targets (exclude hub regulators). None = no limit.
        max_regulators: Maximum regulators to return. None = all.
        regulator_classes: Optional set of RegulatorClass enums to filter by.
        stmt_types: INDRA statement types (None = ALL_REGULATORY_TYPES).
        min_genes_found: Minimum mapped genes per module for inclusion.
        env_file: Path to .env file with INDRA credentials.
        verbose: Print progress.

    Returns:
        List of CliqueDefinition objects with protein_ids as feature IDs.
    """
    if verbose:
        print("Discovering INDRA gene sets for differential testing...")

    # Step 1: Map feature IDs to gene symbols
    symbol_to_feature = map_feature_ids_to_symbols(feature_ids, verbose=verbose)
    gene_symbols = list(symbol_to_feature.keys())

    if verbose:
        print(f"  Gene universe: {len(gene_symbols)} symbols from {len(feature_ids)} feature IDs")

    # Step 2: Initialize INDRA client and discover modules
    from cliquefinder.knowledge.cogex import CoGExClient, INDRAModuleExtractor
    from pathlib import Path

    client_kwargs = {}
    if env_file:
        client_kwargs["env_file"] = Path(env_file)

    client = CoGExClient(**client_kwargs)
    extractor = INDRAModuleExtractor(client)

    try:
        modules = extractor.discover_modules(
            gene_universe=gene_symbols,
            min_evidence=min_evidence,
            min_targets=min_targets,
            max_targets=max_targets,
            max_regulators=max_regulators,
            regulator_classes=regulator_classes,
            stmt_types=stmt_types,
        )

        if verbose:
            print(f"  Discovered {len(modules)} INDRA regulatory modules")

        # Step 3: Convert to CliqueDefinitions
        clique_defs = modules_to_clique_definitions(
            modules=modules,
            symbol_to_feature=symbol_to_feature,
            min_genes_found=min_genes_found,
            verbose=verbose,
        )

        return clique_defs

    finally:
        client.close()


def run_clique_differential_analysis(
    data: NDArray[np.float64],
    feature_ids: list[str],
    sample_metadata: pd.DataFrame,
    clique_definitions: list[CliqueDefinition],
    condition_col: str,
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
        condition_col: Metadata column for condition labels (REQUIRED - must be specified by user).
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
        ...     condition_col="treatment_group",
        ...     subject_col="subject_id",
        ...     contrasts={"treatment_vs_control": ("treatment", "control")},
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

        # Warn if clique has mixed correlation signs
        # Tukey Median Polish assumes additive structure (all features moving together)
        # Mixed-sign cliques (some positive, some negative edges) violate this assumption
        if clique.is_mixed:
            warnings.warn(
                f"Clique '{clique.clique_id}' has mixed correlation signs "
                f"({clique.n_positive_edges} positive, {clique.n_negative_edges} negative edges). "
                f"Tukey Median Polish summarization assumes co-directional features. "
                f"Results for this clique may be unreliable. Consider filtering by "
                f"direction='positive' or direction='negative' for robust analysis.",
                UserWarning,
                stacklevel=2,
            )

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
                direction=clique.direction,
                signed_mean_correlation=clique.signed_mean_correlation,
                signed_min_correlation=clique.signed_min_correlation,
                signed_max_correlation=clique.signed_max_correlation,
                n_positive_edges=clique.n_positive_edges,
                n_negative_edges=clique.n_negative_edges,
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
    condition_col: str,
    contrast: tuple[str, str],
    subject_col: str | None = "subject_id",
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
        condition_col: Metadata column for condition labels (REQUIRED - must be specified by user)
        contrast: Tuple of (test_condition, reference_condition) (REQUIRED - must be specified by user)
        subject_col: Metadata column for subject IDs
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
        ...     condition_col="treatment_group",
        ...     contrast=("treatment", "control"),
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
    condition_col: str,
    contrast: tuple[str, str],
    subject_col: str | None = "subject_id",
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
        condition_col: Metadata column for condition labels (REQUIRED - must be specified by user)
        contrast: Tuple of (test_condition, reference_condition) (REQUIRED - must be specified by user)
        subject_col: Metadata column for subject IDs
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
        ...     condition_col="treatment_group",
        ...     contrast=("group_a", "group_b"),
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

        # Warn if clique has mixed correlation signs
        if clique.is_mixed:
            warnings.warn(
                f"Clique '{clique.clique_id}' has mixed correlation signs "
                f"({clique.n_positive_edges} positive, {clique.n_negative_edges} negative edges). "
                f"Summarization results may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

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


# =============================================================================
# ROAST-Based Clique Analysis
# =============================================================================

def run_clique_roast_analysis(
    data: NDArray[np.float64],
    feature_ids: list[str],
    sample_metadata: pd.DataFrame,
    clique_definitions: list[CliqueDefinition],
    condition_column: str,
    conditions: list[str],
    contrast: tuple[str, str],
    n_rotations: int = 9999,
    seed: int | None = 42,
    use_gpu: bool = True,
    map_ids: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run ROAST rotation-based gene set test on cliques.

    ROAST (Rotation Gene Set Tests) is a self-contained gene set test that
    preserves inter-gene correlation structure. Unlike the MSstats-style
    approach (run_clique_differential_analysis), ROAST:

    - Tests gene sets directly without summarization
    - Detects bidirectional regulation (genes going UP and DOWN)
    - Produces exact p-values via rotation
    - Does NOT require FDR correction (raw p-values are valid)

    Statistical Foundation:
        ROAST projects expression data onto residual space via QR decomposition,
        then generates null distributions by rotating residual vectors on a
        hypersphere. This preserves the correlation structure among genes.

        The MSQ statistic is direction-agnostic, detecting sets where genes
        are differentially expressed regardless of direction - critical for
        transcription factors that both activate AND repress targets.

    Args:
        data: Expression matrix (n_features, n_samples), log2-transformed.
        feature_ids: Feature identifiers (UniProt IDs or gene symbols).
        sample_metadata: Sample metadata DataFrame.
        clique_definitions: List of CliqueDefinition objects.
        condition_column: Metadata column with condition labels.
        conditions: Two condition labels (e.g., ['CASE', 'CTRL']).
        contrast: Contrast to test (e.g., ('CASE', 'CTRL')).
        n_rotations: Number of rotations (higher = more precise p-values).
        seed: Random seed for reproducibility.
        use_gpu: Use GPU acceleration (requires MLX on Apple Silicon).
        map_ids: Map UniProt IDs to gene symbols if needed.
        verbose: Print progress.

    Returns:
        DataFrame with raw p-values (no FDR) for each clique:
        - feature_set_id: Clique regulator name
        - clique_genes: Comma-separated gene symbols
        - n_genes, n_genes_found: Clique size and matched genes
        - pvalue_{stat}_{alt}: Raw p-values for each statistic/alternative
        - observed_{stat}: Observed test statistics

    Example:
        >>> from cliquefinder.stats import run_clique_roast_analysis
        >>> results = run_clique_roast_analysis(
        ...     data=expression_matrix,
        ...     feature_ids=gene_ids,
        ...     sample_metadata=metadata,
        ...     clique_definitions=cliques,
        ...     condition_column='phenotype',
        ...     conditions=['CASE', 'CTRL'],
        ...     contrast=('CASE', 'CTRL'),
        ... )
        >>> # Top hits by bidirectional regulation
        >>> top = results.nsmallest(20, 'pvalue_msq_mixed')
        >>> # Candidates for follow-up (raw p < 0.01)
        >>> candidates = results[results['pvalue_msq_mixed'] < 0.01]

    References:
        Wu D et al. (2010) ROAST: rotation gene set tests for complex
        microarray experiments. Bioinformatics 26(17):2176-82.
    """
    from .rotation import RotationTestEngine, RotationTestConfig

    if len(conditions) != 2:
        raise ValueError(
            f"ROAST requires exactly 2 conditions, got {len(conditions)}: {conditions}"
        )

    if verbose:
        print("ROAST Clique Analysis")
        print("=" * 50)
        print(f"  Features: {len(feature_ids)}")
        print(f"  Samples: {data.shape[1]}")
        print(f"  Cliques: {len(clique_definitions)}")
        print(f"  Contrast: {contrast[0]} vs {contrast[1]}")
        print(f"  Rotations: {n_rotations}")
        print()

    # Build gene set dict from clique definitions
    clique_gene_symbols = {}
    for clique in clique_definitions:
        clique_gene_symbols[clique.regulator] = list(clique.protein_ids)

    # Handle ID mapping if feature_ids are UniProt
    if map_ids:
        # Check if IDs look like UniProt (not gene symbols)
        sample_ids = feature_ids[:100]
        looks_like_uniprot = sum(
            1 for fid in sample_ids
            if len(fid) == 6 and fid[0].isupper() and fid[1:].isalnum()
        ) > len(sample_ids) * 0.5

        if looks_like_uniprot:
            if verbose:
                print("Mapping UniProt IDs to gene symbols...")

            symbol_to_uniprot = map_feature_ids_to_symbols(feature_ids, verbose=verbose)

            # Convert clique genes (symbols) to UniProt IDs
            clique_uniprot = {}
            unmapped = set()
            for regulator, genes in clique_gene_symbols.items():
                mapped = []
                for gene in genes:
                    if gene in symbol_to_uniprot:
                        mapped.append(symbol_to_uniprot[gene])
                    else:
                        unmapped.add(gene)
                if mapped:
                    clique_uniprot[regulator] = mapped

            if verbose:
                print(f"  Cliques with mapped genes: {len(clique_uniprot)}/{len(clique_gene_symbols)}")
                print(f"  Unmapped genes: {len(unmapped)}")
                print()

            gene_sets = clique_uniprot
        else:
            gene_sets = clique_gene_symbols
    else:
        gene_sets = clique_gene_symbols

    # Initialize ROAST engine
    engine = RotationTestEngine(data, feature_ids, sample_metadata)

    if verbose:
        print("Fitting rotation model...")

    engine.fit(
        conditions=conditions,
        contrast=contrast,
        condition_column=condition_column,
    )

    if verbose:
        eb_d0 = engine._precomputed.eb_d0
        if eb_d0 and not np.isinf(eb_d0):
            print(f"  Empirical Bayes prior df: {eb_d0:.1f}")
        print()

    # Configure and run tests
    config = RotationTestConfig(
        n_rotations=n_rotations,
        use_gpu=use_gpu,
        seed=seed,
    )

    if verbose:
        print(f"Testing {len(gene_sets)} cliques...")

    results = engine.test_gene_sets(gene_sets, config=config, verbose=verbose)
    df = engine.results_to_dataframe(results)

    # Add original gene symbols for interpretability
    df['clique_genes'] = df['feature_set_id'].map(
        lambda x: ','.join(clique_gene_symbols.get(x, []))
    )

    # Reorder columns
    cols_first = ['feature_set_id', 'clique_genes', 'n_genes', 'n_genes_found']
    cols_rest = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + cols_rest]

    # Sort by MSQ p-value
    df = df.sort_values('pvalue_msq_mixed', na_position='last')

    if verbose:
        n_p01 = (df['pvalue_msq_mixed'] < 0.01).sum() if 'pvalue_msq_mixed' in df.columns else 0
        min_p = df['pvalue_msq_mixed'].min() if 'pvalue_msq_mixed' in df.columns else float('nan')
        print()
        print("Results (raw p-values, no FDR):")
        print(f"  Cliques with p < 0.01 (MSQ mixed): {n_p01}")
        print(f"  Minimum p-value: {min_p:.4f}")

    return df


def run_clique_roast_interaction_analysis(
    data: NDArray[np.float64],
    feature_ids: list[str],
    sample_metadata: pd.DataFrame,
    clique_definitions: list[CliqueDefinition],
    factor1_column: str,
    factor2_column: str,
    n_rotations: int = 9999,
    seed: int | None = 42,
    use_gpu: bool = True,
    map_ids: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run ROAST rotation-based gene set test for 2×2 factorial interaction.

    Tests whether the effect of factor2 differs across levels of factor1.
    For example, with Sex × Disease:
        (Male_CASE - Male_CTRL) - (Female_CASE - Female_CTRL)

    This tests whether the disease effect is different between males and females,
    which is the definition of a Sex × Disease interaction.

    Args:
        data: Expression matrix (n_features, n_samples), log2-transformed.
        feature_ids: Feature identifiers (UniProt IDs or gene symbols).
        sample_metadata: Sample metadata DataFrame.
        clique_definitions: List of CliqueDefinition objects.
        factor1_column: Metadata column for first factor (e.g., 'sex').
        factor2_column: Metadata column for second factor (e.g., 'phenotype').
        n_rotations: Number of rotations (higher = more precise p-values).
        seed: Random seed for reproducibility.
        use_gpu: Use GPU acceleration (requires MLX on Apple Silicon).
        map_ids: Map UniProt IDs to gene symbols if needed.
        verbose: Print progress.

    Returns:
        DataFrame with raw p-values (no FDR) for each clique.
        Columns are same as run_clique_roast_analysis.

    Example:
        >>> results = run_clique_roast_interaction_analysis(
        ...     data=expression_matrix,
        ...     feature_ids=gene_ids,
        ...     sample_metadata=metadata,
        ...     clique_definitions=cliques,
        ...     factor1_column='sex',
        ...     factor2_column='phenotype',
        ... )
        >>> # Top interaction effects
        >>> top = results.nsmallest(20, 'pvalue_msq_mixed')
    """
    from .rotation import RotationTestEngine, RotationTestConfig

    # Get factor labels
    factor1_labels = sample_metadata[factor1_column].values
    factor2_labels = sample_metadata[factor2_column].values

    # Validate 2×2 design
    levels1 = sorted(set(factor1_labels))
    levels2 = sorted(set(factor2_labels))

    if len(levels1) != 2:
        raise ValueError(
            f"Factor '{factor1_column}' must have exactly 2 levels, "
            f"got {len(levels1)}: {levels1}"
        )
    if len(levels2) != 2:
        raise ValueError(
            f"Factor '{factor2_column}' must have exactly 2 levels, "
            f"got {len(levels2)}: {levels2}"
        )

    if verbose:
        print("ROAST Interaction Analysis")
        print("=" * 50)
        print(f"  Features: {len(feature_ids)}")
        print(f"  Samples: {data.shape[1]}")
        print(f"  Cliques: {len(clique_definitions)}")
        print(f"  Factor 1: {factor1_column} ({levels1[0]} vs {levels1[1]})")
        print(f"  Factor 2: {factor2_column} ({levels2[0]} vs {levels2[1]})")
        print(f"  Testing: ({levels1[0]}_{levels2[0]} - {levels1[0]}_{levels2[1]}) - "
              f"({levels1[1]}_{levels2[0]} - {levels1[1]}_{levels2[1]})")
        print(f"  Rotations: {n_rotations}")
        print()

    # Build gene set dict from clique definitions
    clique_gene_symbols = {}
    for clique in clique_definitions:
        clique_gene_symbols[clique.regulator] = list(clique.protein_ids)

    # Handle ID mapping if feature_ids are UniProt
    if map_ids:
        sample_ids = feature_ids[:100]
        looks_like_uniprot = sum(
            1 for fid in sample_ids
            if len(fid) == 6 and fid[0].isupper() and fid[1:].isalnum()
        ) > len(sample_ids) * 0.5

        if looks_like_uniprot:
            if verbose:
                print("Mapping UniProt IDs to gene symbols...")

            symbol_to_uniprot = map_feature_ids_to_symbols(feature_ids, verbose=verbose)

            clique_uniprot = {}
            unmapped = set()
            for regulator, genes in clique_gene_symbols.items():
                mapped = []
                for gene in genes:
                    if gene in symbol_to_uniprot:
                        mapped.append(symbol_to_uniprot[gene])
                    else:
                        unmapped.add(gene)
                if mapped:
                    clique_uniprot[regulator] = mapped

            if verbose:
                print(f"  Cliques with mapped genes: {len(clique_uniprot)}/{len(clique_gene_symbols)}")
                print(f"  Unmapped genes: {len(unmapped)}")
                print()

            gene_sets = clique_uniprot
        else:
            gene_sets = clique_gene_symbols
    else:
        gene_sets = clique_gene_symbols

    # Initialize ROAST engine
    engine = RotationTestEngine(data, feature_ids, sample_metadata)

    if verbose:
        print("Fitting interaction model...")

    # Use fit_interaction for 2×2 factorial design
    engine.fit_interaction(
        factor1_column=factor1_column,
        factor2_column=factor2_column,
    )

    if verbose:
        eb_d0 = engine._precomputed.eb_d0
        if eb_d0 and not np.isinf(eb_d0):
            print(f"  Empirical Bayes prior df: {eb_d0:.1f}")
        print()

    # Configure and run tests
    config = RotationTestConfig(
        n_rotations=n_rotations,
        use_gpu=use_gpu,
        seed=seed,
    )

    if verbose:
        print(f"Testing {len(gene_sets)} cliques...")

    results = engine.test_gene_sets(gene_sets, config=config, verbose=verbose)
    df = engine.results_to_dataframe(results)

    # Add original gene symbols for interpretability
    df['clique_genes'] = df['feature_set_id'].map(
        lambda x: ','.join(clique_gene_symbols.get(x, []))
    )

    # Reorder columns
    cols_first = ['feature_set_id', 'clique_genes', 'n_genes', 'n_genes_found']
    cols_rest = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + cols_rest]

    # Sort by MSQ p-value
    df = df.sort_values('pvalue_msq_mixed', na_position='last')

    if verbose:
        n_p01 = (df['pvalue_msq_mixed'] < 0.01).sum() if 'pvalue_msq_mixed' in df.columns else 0
        min_p = df['pvalue_msq_mixed'].min() if 'pvalue_msq_mixed' in df.columns else float('nan')
        print()
        print("Results (raw p-values, no FDR):")
        print(f"  Cliques with p < 0.01 (MSQ mixed): {n_p01}")
        print(f"  Minimum p-value: {min_p:.4f}")

    return df
