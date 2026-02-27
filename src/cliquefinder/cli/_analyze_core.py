"""
Core analysis functions for stratified regulatory module analysis.

This module contains the refactored analysis logic from scripts/analyze_tf_regulatory_modules.py,
now available as a library for use by the CLI and programmatic access.

The analysis pipeline:
1. Map Ensembl gene IDs → gene symbols for CoGEx compatibility
2. Query INDRA CoGEx for regulator downstream targets (INDRA targets)
3. Run stratified correlation analysis across phenotype × sex strata
4. Find condition-specific correlation cliques (coherent modules)
5. Identify differential regulatory patterns (gained/lost in ALS)
"""

from __future__ import annotations

import logging
import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd

from cliquefinder import BioMatrix
from cliquefinder.validation.id_mapping import MyGeneInfoMapper
from cliquefinder.knowledge.cogex import (
    CoGExClient,
    INDRAModuleExtractor,
    INDRAModule,
    GeneId,
    RegulatorClass,
    get_regulator_class_genes,
)
from cliquefinder.knowledge.clique_validator import (
    CliqueValidator,
    CorrelationClique,
    ChildSetType2,
    DifferentialCliqueResult,
    GenePairDifferentialStat,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class StratifiedCliqueResult:
    """Results from clique analysis in a single stratum."""
    condition: str
    regulator_name: str
    indra_targets: Set[str]  # INDRA targets in gene universe (ALL targets, not filtered)
    coherent_clique: Optional[ChildSetType2]  # Correlation-validated subset
    all_cliques: List[CorrelationClique]
    n_samples: int
    rna_validated_targets: Optional[Set[str]] = None  # Targets with RNA expression validation (annotation, not filter)

    @property
    def coherent_genes(self) -> Set[str]:
        """Genes in the coherent clique (if found)."""
        if self.coherent_clique:
            return {g for g in self.coherent_clique.clique_genes}
        return set()

    @property
    def coherence_ratio(self) -> float:
        """Fraction of INDRA targets that form a correlated clique."""
        if not self.indra_targets:
            return 0.0
        return len(self.coherent_genes) / len(self.indra_targets)

    @property
    def rna_validation_ratio(self) -> float:
        """Fraction of coherent genes that are RNA-validated (for annotation)."""
        if not self.coherent_genes or self.rna_validated_targets is None:
            return 1.0  # If no RNA filter, assume all validated
        validated_coherent = self.coherent_genes & self.rna_validated_targets
        return len(validated_coherent) / len(self.coherent_genes) if self.coherent_genes else 0.0


@dataclass
class DifferentialResult:
    """Differential clique analysis between two conditions with FDR correction."""
    regulator_name: str
    case_condition: str
    ctrl_condition: str
    gained_cliques: List[CorrelationClique]  # Present in CASE, absent in CTRL
    lost_cliques: List[CorrelationClique]    # Present in CTRL, absent in CASE
    case_coherence: float
    ctrl_coherence: float
    # FDR-corrected statistics
    n_gene_pairs_tested: int = 0
    n_significant_pairs: int = 0
    fdr_threshold: float = 0.05
    significance_threshold: float = 0.7
    significant_gene_pairs: Optional[List[GenePairDifferentialStat]] = None
    # Effective tests (accounts for correlation structure)
    nominal_tests: int = 0
    effective_tests: float = 0.0

    @property
    def regulatory_rewiring_score(self) -> float:
        """Score indicating magnitude of regulatory rewiring."""
        return abs(self.case_coherence - self.ctrl_coherence)

    @property
    def fdr_support_ratio(self) -> float:
        """Fraction of gene pairs with FDR-significant differential correlation."""
        if self.n_gene_pairs_tested == 0:
            return 0.0
        return self.n_significant_pairs / self.n_gene_pairs_tested

    @property
    def effective_test_reduction(self) -> float:
        """Reduction factor showing correlation between tests (M_eff / M_nominal)."""
        if self.nominal_tests == 0:
            return 1.0
        return self.effective_tests / self.nominal_tests


@dataclass
class RegulatorAnalysisResult:
    """Complete analysis result for a single upstream regulator."""
    regulator_name: str
    regulator_id: GeneId
    indra_targets: Set[str]  # All INDRA targets in gene universe (NOT filtered by RNA)
    original_n_indra_targets: int  # Number of INDRA targets in universe
    stratified_results: Dict[str, StratifiedCliqueResult]
    differential_results: List[DifferentialResult]
    rna_validated_targets: Optional[Set[str]] = None  # Targets with RNA validation (annotation, not filter)

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        n_rna_validated = len(self.rna_validated_targets) if self.rna_validated_targets else None
        return {
            'regulator_name': self.regulator_name,
            'regulator_id': list(self.regulator_id),
            'n_indra_targets': len(self.indra_targets),
            'original_n_indra_targets': self.original_n_indra_targets,
            'n_rna_validated_targets': n_rna_validated,  # Annotation metadata
            'indra_targets': list(self.indra_targets),
            'conditions_analyzed': list(self.stratified_results.keys()),
            'stratified_summary': {
                cond: {
                    'n_samples': res.n_samples,
                    'n_coherent_genes': len(res.coherent_genes),
                    'coherence_ratio': res.coherence_ratio,
                    'n_cliques': len(res.all_cliques),
                    'rna_validation_ratio': res.rna_validation_ratio,  # NEW: annotation
                }
                for cond, res in self.stratified_results.items()
            },
            'differential_summary': [
                {
                    'comparison': f"{d.case_condition}_vs_{d.ctrl_condition}",
                    'gained_cliques': len(d.gained_cliques),
                    'lost_cliques': len(d.lost_cliques),
                    'rewiring_score': d.regulatory_rewiring_score,
                }
                for d in self.differential_results
            ]
        }


# =============================================================================
# Process Worker for Multiprocessing
# =============================================================================

# Module-level globals for worker processes (initialized via _init_worker)
_worker_validator = None
_worker_conditions = None
_worker_params = None


def _init_worker(mmap_path, feature_ids, sample_ids, sample_metadata,
                 quality_flags, stratify_by, min_samples, conditions,
                 min_correlation, min_clique_size, use_fast_maximum, correlation_method):
    """Initialize worker process with memory-mapped data.

    Instead of receiving the full matrix via pickle serialization (which
    duplicates ~300 MB per worker), workers load a memory-mapped .npy file.
    The OS shares the physical pages across all worker processes (copy-on-write),
    so total memory usage is O(1) regardless of worker count.

    Args:
        mmap_path: Path to a temporary .npy file containing the expression matrix.
            Loaded with ``np.load(mmap_path, mmap_mode='r')`` for read-only access.
        feature_ids: List of feature ID strings (gene names / Ensembl IDs).
        sample_ids: List of sample ID strings.
        sample_metadata: DataFrame of sample metadata (pickled normally — small).
        quality_flags: NumPy array of quality flags (small).
        stratify_by: Column name for stratification.
        min_samples: Minimum samples per stratum.
        conditions: List of condition labels.
        min_correlation: Minimum correlation threshold.
        min_clique_size: Minimum clique size.
        use_fast_maximum: Whether to use fast maximum clique algorithm.
        correlation_method: Correlation method ('pearson' or 'spearman').
    """
    global _worker_validator, _worker_conditions, _worker_params

    # Load expression matrix via memory mapping (OS shares physical pages, read-only)
    matrix_data = np.load(mmap_path, mmap_mode='r')

    # Reconstruct BioMatrix in this process
    matrix = BioMatrix(
        data=matrix_data,
        feature_ids=pd.Index(feature_ids),
        sample_ids=pd.Index(sample_ids),
        sample_metadata=sample_metadata,
        quality_flags=quality_flags
    )

    # Create validator for this process
    _worker_validator = CliqueValidator(
        matrix=matrix,
        stratify_by=stratify_by,
        min_samples=min_samples
    )
    _worker_validator.precompute_condition_data()

    _worker_conditions = conditions
    _worker_params = {
        'min_correlation': min_correlation,
        'min_clique_size': min_clique_size,
        'use_fast_maximum': use_fast_maximum,
        'correlation_method': correlation_method
    }


def _process_regulator_worker(args):
    """Worker function for ProcessPoolExecutor - analyzes one regulator."""
    regulator_name, regulator_id, indra_target_names, rna_filter_genes = args  # rna_filter_genes for target annotation

    global _worker_validator, _worker_conditions, _worker_params

    if _worker_validator is None:
        return None

    try:
        # Create a minimal INDRAModule-like object for compatibility
        class MinimalModule:
            def __init__(self, name, rid, targets):
                self.regulator_name = name
                self.regulator_id = rid
                self.indra_target_names = targets

        module = MinimalModule(regulator_name, regulator_id, indra_target_names)

        result = analyze_regulator_module(
            regulator_name=regulator_name,
            module=module,
            validator=_worker_validator,
            conditions=_worker_conditions,
            min_correlation=_worker_params['min_correlation'],
            min_clique_size=_worker_params['min_clique_size'],
            use_fast_maximum=_worker_params['use_fast_maximum'],
            correlation_method=_worker_params['correlation_method'],
            n_threads=1,  # No nested threading in process workers
            rna_filter_genes=rna_filter_genes  # Pass through for annotation (not filter)
        )
        return result
    except Exception as e:
        logger.error(f"Process worker failed for {regulator_name}: {e}")
        return None


# =============================================================================
# Gene ID Mapping
# =============================================================================

def build_gene_symbol_mapping(
    ensembl_ids: List[str],
    cache_dir: Optional[Path] = None,
    source_type: str = 'ensembl_gene' # NEW argument
) -> Dict[str, str]:
    """
    Map gene IDs to gene symbols.

    Args:
        ensembl_ids: List of IDs (ENSG... or symbols)
        cache_dir: Directory for caching mappings
        source_type: Input ID type ('ensembl_gene', 'symbol', etc.)

    Returns:
        Dict mapping Input ID → gene symbol
    """
    logger.info(f"Mapping {len(ensembl_ids)} {source_type} IDs to gene symbols...")

    if source_type == 'symbol':
        # Identity mapping for symbols
        return {gid: gid for gid in ensembl_ids}

    mapper = MyGeneInfoMapper(cache_dir=cache_dir)
    mapping = mapper.map_ids(
        ensembl_ids,
        source_type=source_type,
        target_type='symbol',
        species='human'
    )

    n_mapped = len(mapping)
    logger.info(f"Successfully mapped {n_mapped}/{len(ensembl_ids)} "
                f"({100*n_mapped/len(ensembl_ids):.1f}%) genes")

    return mapping


def create_symbol_indexed_matrix(
    matrix: BioMatrix,
    ensembl_to_symbol: Dict[str, str]
) -> Tuple[BioMatrix, Dict[str, str]]:
    """
    Create a new BioMatrix with gene symbols as feature IDs.

    Args:
        matrix: Original matrix with Ensembl IDs
        ensembl_to_symbol: Mapping from Ensembl to symbol

    Returns:
        Tuple of (new_matrix, symbol_to_ensembl_mapping)
    """
    # Filter to genes with symbol mapping
    mappable_mask = matrix.feature_ids.isin(ensembl_to_symbol.keys())
    mappable_ensembl = matrix.feature_ids[mappable_mask].tolist()

    # Get corresponding symbols
    symbols = [ensembl_to_symbol[eid] for eid in mappable_ensembl]

    # Handle duplicate symbols by keeping highest variance
    symbol_to_ensembl = {}
    seen_symbols = set()
    keep_indices = []

    # Calculate variance for each gene
    variances = np.var(matrix.data[mappable_mask], axis=1)

    # Sort by variance (descending) to keep highest variance gene per symbol
    sorted_indices = np.argsort(variances)[::-1]

    for idx in sorted_indices:
        symbol = symbols[idx]
        if symbol not in seen_symbols:
            seen_symbols.add(symbol)
            keep_indices.append(idx)
            symbol_to_ensembl[symbol] = mappable_ensembl[idx]

    # Sort indices back to original order
    keep_indices = sorted(keep_indices)

    # Subset data
    original_indices = np.where(mappable_mask)[0][keep_indices]
    new_data = matrix.data[original_indices]
    new_symbols = [symbols[i] for i in keep_indices]
    new_flags = matrix.quality_flags[original_indices]

    # Create new matrix
    new_matrix = BioMatrix(
        data=new_data,
        feature_ids=pd.Index(new_symbols),
        sample_ids=matrix.sample_ids,
        sample_metadata=matrix.sample_metadata,
        quality_flags=new_flags
    )

    logger.info(f"Created symbol-indexed matrix: {new_matrix.n_features} genes × "
                f"{new_matrix.n_samples} samples")

    return new_matrix, symbol_to_ensembl


# =============================================================================
# Stratified Analysis Pipeline
# =============================================================================

def _analyze_single_condition(
    regulator_name: str,
    indra_targets: Set[str],
    validator: CliqueValidator,
    condition: str,
    min_correlation: float,
    min_clique_size: int,
    use_fast_maximum: bool,
    correlation_method: str = "max",
    rna_validated_targets: Optional[Set[str]] = None  # Annotation metadata (not filter)
) -> Optional[StratifiedCliqueResult]:
    """
    Analyze a single condition for a regulator. Designed for parallel execution.

    Args:
        rna_validated_targets: Optional set of genes with RNA expression support.
            Used for annotation only - does NOT filter targets from analysis.

    Returns:
        StratifiedCliqueResult or None if analysis fails
    """
    try:
        if use_fast_maximum:
            # Use efficient greedy maximum clique algorithm - O(n * d^2)
            largest_clique = validator.find_maximum_clique(
                genes=indra_targets,
                condition=condition,
                min_correlation=min_correlation,
                method=correlation_method
            )
            cliques = [largest_clique] if largest_clique else []

            # Derive coherent module from the maximum clique
            coherent_module = None
            if largest_clique and largest_clique.size >= min_clique_size:
                coherent_module = ChildSetType2(
                    regulator_name=regulator_name,
                    indra_targets=indra_targets,
                    clique_genes=largest_clique.genes,
                    condition=condition,
                    correlation_threshold=min_correlation,
                    mean_correlation=largest_clique.mean_correlation,
                    direction=largest_clique.direction,
                    signed_mean_correlation=largest_clique.signed_mean_correlation,
                    signed_min_correlation=largest_clique.signed_min_correlation,
                    signed_max_correlation=largest_clique.signed_max_correlation,
                    n_positive_edges=largest_clique.n_positive_edges,
                    n_negative_edges=largest_clique.n_negative_edges,
                    edge_correlations=largest_clique.edge_correlations,
                )
        else:
            # Use full clique enumeration (exponential but exact)
            # exact=True ensures complete enumeration without silent truncation at 10k cliques
            # timeout_seconds=300.0 provides reasonable upper bound per condition
            cliques = validator.find_cliques(
                genes=indra_targets,
                condition=condition,
                min_correlation=min_correlation,
                min_clique_size=min_clique_size,
                method=correlation_method,
                exact=True,
                timeout_seconds=300.0
            )

            # Derive coherent module from largest clique
            coherent_module = None
            if cliques:
                largest_clique = cliques[0]
                coherent_module = ChildSetType2(
                    regulator_name=regulator_name,
                    indra_targets=indra_targets,
                    clique_genes=largest_clique.genes,
                    condition=condition,
                    correlation_threshold=min_correlation,
                    mean_correlation=largest_clique.mean_correlation,
                    direction=largest_clique.direction,
                    signed_mean_correlation=largest_clique.signed_mean_correlation,
                    signed_min_correlation=largest_clique.signed_min_correlation,
                    signed_max_correlation=largest_clique.signed_max_correlation,
                    n_positive_edges=largest_clique.n_positive_edges,
                    n_negative_edges=largest_clique.n_negative_edges,
                    edge_correlations=largest_clique.edge_correlations,
                )

        # Get sample count for this condition
        mask = validator._get_condition_mask(condition)
        n_samples = mask.sum()

        # Filter cliques by size for reporting
        valid_cliques = [c for c in cliques if c and c.size >= min_clique_size]

        return StratifiedCliqueResult(
            condition=condition,
            regulator_name=regulator_name,
            indra_targets=indra_targets,
            coherent_clique=coherent_module,
            all_cliques=valid_cliques,
            n_samples=n_samples,
            rna_validated_targets=rna_validated_targets  # Annotation metadata
        )

    except Exception as e:
        logger.warning(f"  {condition}: Analysis failed - {e}")
        return None


def analyze_regulator_module(
    regulator_name: str,
    module: INDRAModule,
    validator: CliqueValidator,
    conditions: List[str],
    min_correlation: float = 0.7,
    min_clique_size: int = 3,
    use_fast_maximum: bool = True,
    n_threads: int = 1,
    correlation_method: str = "max",
    rna_filter_genes: Optional[Set[str]] = None # NEW argument
) -> RegulatorAnalysisResult:
    """
    Analyze a single regulator's INDRA module across all conditions.

    Args:
        regulator_name: Regulator gene symbol
        module: INDRA module from CoGEx
        validator: CliqueValidator with expression data
        conditions: List of condition names to analyze
        min_correlation: Minimum correlation for clique edges
        min_clique_size: Minimum genes in a clique
        use_fast_maximum: If True, use greedy maximum clique algorithm O(n*d^2).
            If False, enumerate all cliques (exponential). Default: True.
        n_threads: Number of threads for parallel condition analysis.
            Default: 1 (sequential). Threads share memory, so safe for large matrices.
        rna_filter_genes: Optional set of RNA-expressed genes for target annotation.
            Used to annotate which targets have RNA expression support.
            Does NOT filter targets from analysis - all INDRA targets are retained.

    Returns:
        RegulatorAnalysisResult with stratified and differential analyses
    """
    # Get INDRA targets (target names from INDRA CoGEx)
    # IMPORTANT: We use ALL INDRA targets for clique finding, not just RNA-validated ones
    original_indra_targets = module.indra_target_names
    indra_targets = original_indra_targets  # Keep ALL targets (no filtering)

    # RNA validation as ANNOTATION (not filter) - tracks which targets have RNA support
    rna_validated_targets = None
    if rna_filter_genes is not None:
        rna_validated_targets = indra_targets & rna_filter_genes
        n_validated = len(rna_validated_targets)
        n_total = len(indra_targets)
        logger.info(
            f"RNA validation for {regulator_name}: "
            f"{n_validated}/{n_total} targets ({n_validated/n_total:.1%}) have RNA expression support. "
            f"(All {n_total} targets retained for clique finding)"
        )

    if not indra_targets:
        logger.warning(
            f"Regulator {regulator_name} has no INDRA targets in gene universe. Skipping."
        )
        return None

    logger.info(f"Analyzing {regulator_name}: {len(indra_targets)} INDRA targets (all retained)")

    # Stratified analysis - parallel across conditions using threads
    stratified_results = {}

    if n_threads > 1 and len(conditions) > 1:
        # Parallel execution using ThreadPoolExecutor (threads share memory)
        with ThreadPoolExecutor(max_workers=min(n_threads, len(conditions))) as executor:
            future_to_condition = {
                executor.submit(
                    _analyze_single_condition,
                    regulator_name,
                    indra_targets,  # ALL INDRA targets (not filtered)
                    validator,
                    condition,
                    min_correlation,
                    min_clique_size,
                    use_fast_maximum,
                    correlation_method,
                    rna_validated_targets  # Annotation metadata (not filter)
                ): condition
                for condition in conditions
            }

            for future in as_completed(future_to_condition):
                condition = future_to_condition[future]
                try:
                    result = future.result()
                    if result:
                        stratified_results[condition] = result
                        coherence = result.coherence_ratio
                        n_coherent = len(result.coherent_genes)
                        logger.info(f"  {condition}: {result.n_samples} samples, "
                                   f"coherent={n_coherent} genes, "
                                   f"coherence={coherence:.2f}")
                except Exception as e:
                    logger.warning(f"  {condition}: Analysis failed - {e}")
    else:
        # Sequential execution
        for condition in conditions:
            result = _analyze_single_condition(
                regulator_name,
                indra_targets,  # ALL INDRA targets (not filtered)
                validator,
                condition,
                min_correlation,
                min_clique_size,
                use_fast_maximum,
                correlation_method,
                rna_validated_targets  # Annotation metadata (not filter)
            )
            if result:
                stratified_results[condition] = result
                coherence = result.coherence_ratio
                n_coherent = len(result.coherent_genes)
                logger.info(f"  {condition}: {result.n_samples} samples, "
                           f"coherent={n_coherent} genes, "
                           f"coherence={coherence:.2f}")

    # Differential analysis (CASE vs CTRL within each sex if applicable)
    # Uses FDR-corrected differential correlation testing
    differential_results = []

    # Find CASE/CTRL pairs
    case_conditions = [c for c in conditions if 'CASE' in c]
    ctrl_conditions = [c for c in conditions if 'CTRL' in c]

    for case_cond in case_conditions:
        # Find matching CTRL condition
        case_suffix = case_cond.replace('CASE', '')
        matching_ctrl = f"CTRL{case_suffix}"

        if matching_ctrl in ctrl_conditions and case_cond in stratified_results and matching_ctrl in stratified_results:
            # Use FDR-corrected differential analysis
            fdr_result = validator.find_differential_cliques_with_stats(
                genes=indra_targets, # Use potentially filtered targets
                case_condition=case_cond,
                ctrl_condition=matching_ctrl,
                min_correlation=min_correlation,
                min_clique_size=min_clique_size,
                method=correlation_method,
                fdr_threshold=0.05,
                use_adaptive_threshold=False  # Use user-specified threshold
            )

            diff_result = DifferentialResult(
                regulator_name=regulator_name,
                case_condition=case_cond,
                ctrl_condition=matching_ctrl,
                gained_cliques=fdr_result.gained_cliques,
                lost_cliques=fdr_result.lost_cliques,
                case_coherence=stratified_results[case_cond].coherence_ratio,
                ctrl_coherence=stratified_results[matching_ctrl].coherence_ratio,
                # FDR statistics
                n_gene_pairs_tested=len(fdr_result.all_gene_pair_stats),
                n_significant_pairs=len(fdr_result.significant_gene_pairs),
                fdr_threshold=fdr_result.fdr_threshold,
                significance_threshold=fdr_result.significance_threshold,
                significant_gene_pairs=fdr_result.significant_gene_pairs,
                # Effective tests
                nominal_tests=fdr_result.nominal_tests,
                effective_tests=fdr_result.effective_tests
            )
            differential_results.append(diff_result)

            logger.info(f"  {case_cond} vs {matching_ctrl}: "
                       f"gained={len(fdr_result.gained_cliques)}, lost={len(fdr_result.lost_cliques)}, "
                       f"rewiring={diff_result.regulatory_rewiring_score:.2f}, "
                       f"FDR_sig={len(fdr_result.significant_gene_pairs)}/{len(fdr_result.all_gene_pair_stats)} pairs, "
                       f"M_eff={fdr_result.effective_tests:.0f}/{fdr_result.nominal_tests}")

    return RegulatorAnalysisResult(
        regulator_name=regulator_name,
        regulator_id=module.regulator_id,
        indra_targets=indra_targets,  # ALL INDRA targets (not filtered)
        original_n_indra_targets=len(original_indra_targets),
        stratified_results=stratified_results,
        differential_results=differential_results,
        rna_validated_targets=rna_validated_targets  # Annotation metadata (not filter)
    )


def run_stratified_analysis(
    matrix: BioMatrix,
    regulators: Optional[List[str]],
    cogex_client: CoGExClient,
    stratify_by: List[str],
    min_evidence: int = 2,
    min_correlation: float = 0.7,
    min_clique_size: int = 3,
    min_samples: int = 20,
    discover_mode: bool = False,
    min_targets: int = 10,
    max_targets: Optional[int] = None,
    max_regulators: Optional[int] = None,
    use_fast_maximum: bool = True,
    n_workers: int = 1,
    parallel_mode: str = "threads",
    correlation_method: str = "max",
    rna_filter_genes: Optional[Set[str]] = None,
    regulator_classes: Optional[Set[RegulatorClass]] = None,
    stmt_types: Optional[List[str]] = None,
) -> List[RegulatorAnalysisResult]:
    """
    Run stratified regulatory module analysis with optional parallelism.

    Supports two modes:
    1. Hand-picked regulators (default): Analyze specific regulators provided in `regulators`
    2. Discovery mode (--discover): Automatically discover regulators from INDRA that
       target genes in the dataset, without pre-specifying which TFs to analyze.

    Parallelism Strategy:
        - "threads": Use ThreadPoolExecutor for condition-level parallelism within each
          regulator. Threads share memory (no matrix duplication). Safe for 16GB RAM.
        - "processes": Use ProcessPoolExecutor for regulator-level parallelism. Each process
          gets a copy of the matrix. More memory intensive but true CPU parallelism.
        - "hybrid": Use processes for regulators + threads for conditions. Maximum parallelism
          but highest memory usage. Use with caution on limited RAM.

    Args:
        matrix: Expression matrix with gene symbols as feature IDs
        regulators: List of regulator gene symbols to analyze (ignored in discover_mode)
        cogex_client: Connected CoGEx client
        stratify_by: Metadata columns for stratification
        min_evidence: Minimum evidence count for CoGEx edges
        min_correlation: Minimum correlation for cliques
        min_clique_size: Minimum clique size
        min_samples: Minimum samples per stratum
        discover_mode: If True, discover regulators from INDRA instead of using hand-picked list
        min_targets: Minimum targets in universe to include regulator (discovery mode)
        max_targets: Maximum targets in universe to include regulator (discovery mode)
            Excludes hub regulators (e.g., TP53) that may produce non-specific modules
        max_regulators: Maximum number of top regulators to analyze (discovery mode)
        use_fast_maximum: If True, use greedy maximum clique algorithm O(n*d^2)
        n_workers: Number of parallel workers. Default: 1 (sequential)
        parallel_mode: "threads", "processes", or "hybrid". Default: "threads"
        correlation_method: Correlation method ("pearson", "spearman", "max").
            Default: "max" - uses max(|Pearson|, |Spearman|) for each gene pair to
            capture both linear and monotonic relationships.
        rna_filter_genes: Optional set of RNA-expressed genes for REGULATOR filtering (discovery mode).
            If provided, discovered regulators are filtered to those present in this set.
            Target genes are NOT filtered, only annotated for RNA presence.
            Biological rationale: Regulators not expressed in RNA are unlikely to be actively
            regulating in this experimental context.
        regulator_classes: Optional set of RegulatorClass enum members to restrict
            regulators by functional class (e.g., {RegulatorClass.TF} for transcription
            factors only). Composes with rna_filter_genes — regulators must pass both.
            Default: None (all functional classes).

    Returns:
        List of RegulatorAnalysisResult for each regulator
    """
    # Get gene universe from expression data
    gene_universe = list(matrix.feature_ids)
    logger.info(f"Gene universe: {len(gene_universe)} genes")

    # Extract INDRA modules from CoGEx
    # Use MyGeneInfoMapper for robust ID resolution fallback
    id_mapper = MyGeneInfoMapper(cache_dir=Path.home() / '.cache/biocore/id_mapping')
    extractor = INDRAModuleExtractor(cogex_client, id_mapper=id_mapper)

    if discover_mode:
        # Discovery mode: Find all regulators targeting genes in dataset
        logger.info(f"DISCOVERY MODE: Finding regulators from INDRA for genes in dataset...")
        modules = extractor.discover_modules(
            gene_universe=gene_universe,
            min_evidence=min_evidence,
            min_targets=min_targets,
            max_targets=max_targets,
            max_regulators=max_regulators,
            regulator_classes=regulator_classes,
            stmt_types=stmt_types,
        )
        logger.info(f"Discovered {len(modules)} regulators from INDRA")

        # Filter discovered regulators for RNA presence
        # CRITICAL: We filter REGULATORS, not their targets
        # Biological rationale: If a regulator is not expressed in RNA data,
        # it's unlikely to be actively regulating targets in this experimental context
        if rna_filter_genes is not None:
            pre_filter = len(modules)
            modules = [m for m in modules if m.regulator_name in rna_filter_genes]
            logger.info(
                f"RNA presence filter: {pre_filter} → {len(modules)} regulators "
                f"({pre_filter - len(modules)} regulators not expressed in RNA)"
            )
    else:
        # Hand-picked mode: Use specified regulators
        if not regulators:
            logger.error("No regulators specified. Use --regulators or --discover")
            return []
        modules = extractor.get_regulator_modules(
            regulators=regulators,
            gene_universe=gene_universe,
            min_evidence=min_evidence,
            stmt_types=stmt_types,
        )
        logger.info(f"Extracted {len(modules)} INDRA modules from CoGEx")

        # Apply regulator class filter in hand-picked mode
        if regulator_classes is not None:
            class_genes = get_regulator_class_genes(regulator_classes)
            pre_filter = len(modules)
            modules = [m for m in modules if m.regulator_name in class_genes]
            logger.info(
                f"Regulator class filter: {pre_filter} -> {len(modules)} modules "
                f"(classes: {[c.value for c in regulator_classes]})"
            )

    if not modules:
        logger.warning("No modules extracted - check regulator names and gene universe")
        return []

    # Create clique validator
    validator = CliqueValidator(
        matrix=matrix,
        stratify_by=stratify_by,
        min_samples=min_samples
    )

    # Get available conditions
    conditions = validator.get_available_conditions()
    logger.info(f"Stratification conditions: {conditions}")

    # Pre-compute condition data to enable efficient parallel access
    # This caches sliced matrices for each condition, avoiding redundant computation
    logger.info("Pre-computing condition data for parallel execution...")
    validator.precompute_condition_data()

    # MAJOR OPTIMIZATION: Precompute correlation matrices for union of all target genes
    # This reduces ~65,000 correlation computations (8,209 regulators × 4 conditions × 2 methods)
    # down to just 4 (one per condition). Expected speedup: 6 minutes → 1 minute
    logger.info("Collecting union of all target genes across regulators...")
    all_target_genes = set()
    for module in modules:
        all_target_genes.update(module.indra_target_names)

    logger.info(f"Precomputing correlation matrices for {len(all_target_genes)} genes "
               f"across {len(conditions)} conditions...")
    validator.precompute_correlation_matrices(
        genes=all_target_genes,
        conditions=conditions,
        method=correlation_method
    )

    # Log cache statistics
    cache_stats = validator.get_cache_stats()
    logger.info(f"Cache statistics: {cache_stats['n_precomputed_corr']} correlation matrices, "
               f"{cache_stats['precomputed_corr_mb']:.1f} MB, "
               f"{cache_stats['n_precomputed_genes']} genes indexed")

    # Determine parallelism strategy
    # With n_workers > 1, we parallelize at the REGULATOR level (not condition level)
    # This is more efficient: 4784 regulators >> 4 conditions
    use_regulator_parallelism = n_workers > 1 and len(modules) > 1

    # When parallelizing regulators, disable condition-level threading to avoid thread explosion
    # Each worker handles one regulator sequentially across its 4 conditions
    n_condition_threads = 1 if use_regulator_parallelism else n_workers

    logger.info(f"Parallelism: mode={parallel_mode}, workers={n_workers}, "
                f"regulator_parallel={use_regulator_parallelism}, "
                f"condition_threads={n_condition_threads}")

    # Analyze each regulator
    results = []

    if use_regulator_parallelism and parallel_mode == "processes":
        # Process-based parallelism - TRUE multi-core execution
        # Memory-mapped array: write matrix once, all workers share OS pages
        logger.info(f"Using {n_workers} PROCESS workers for true multi-core parallelism")

        # Prepare regulator args (tuples of picklable data)
        regulator_args = [
            (module.regulator_name, module.regulator_id, module.indra_target_names, rna_filter_genes) # Added rna_filter_genes
            for module in modules
        ]

        # Use 'spawn' context for clean process creation (avoids fork issues)
        ctx = mp.get_context('spawn')

        # Write expression matrix to a temporary .npy file for memory-mapping.
        # Workers load with mmap_mode='r' so the OS shares physical pages
        # across all processes (copy-on-write), avoiding per-worker copies.
        mmap_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
                mmap_path = tmp.name
                np.save(tmp, matrix.data)

            logger.info(
                f"Wrote expression matrix to mmap file: {mmap_path} "
                f"({matrix.data.nbytes / 1024 / 1024:.1f} MB)"
            )

            # Prepare worker init args — pass mmap path instead of raw matrix
            init_args = (
                mmap_path,  # path to memory-mapped .npy file
                list(matrix.feature_ids),  # list of strings
                list(matrix.sample_ids),  # list of strings
                matrix.sample_metadata,  # DataFrame - picklable (small)
                matrix.quality_flags,  # numpy array (small)
                stratify_by,
                min_samples,
                conditions,
                min_correlation,
                min_clique_size,
                use_fast_maximum,
                correlation_method
            )

            completed = 0
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=init_args
            ) as executor:
                futures = {
                    executor.submit(_process_regulator_worker, args): args
                    for args in regulator_args
                }

                for future in as_completed(futures):
                    args = futures[future]
                    completed += 1
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                        # Progress logging every 10 regulators or at milestones
                        if completed % 10 == 0 or completed == len(modules):
                            pct = 100 * completed / len(modules)
                            logger.info(f"Progress: {completed}/{len(modules)} regulators ({pct:.1f}%)")
                    except Exception as e:
                        logger.error(f"Failed to get result for {args[0]}: {e}")
        finally:
            # Clean up temporary mmap file
            if mmap_path and os.path.exists(mmap_path):
                os.unlink(mmap_path)
                logger.debug(f"Cleaned up mmap file: {mmap_path}")

    elif use_regulator_parallelism:
        # Thread-based parallelism for regulators (default)
        # NumPy releases GIL during computation, so threads provide partial speedup
        # All threads share the same validator (read-only after precompute)
        logger.info(f"Using {n_workers} THREAD workers for regulator-level parallelism")

        def process_regulator(args):
            """Worker function for parallel regulator processing."""
            idx, module = args
            try:
                return analyze_regulator_module(
                    regulator_name=module.regulator_name,
                    module=module,
                    validator=validator,
                    conditions=conditions,
                    min_correlation=min_correlation,
                    min_clique_size=min_clique_size,
                    use_fast_maximum=use_fast_maximum,
                    correlation_method=correlation_method,
                    n_threads=n_condition_threads,
                    rna_filter_genes=rna_filter_genes # Pass RNA filter genes
                )
            except Exception as e:
                logger.error(f"Failed to analyze {module.regulator_name}: {e}")
                return None

        # Submit all regulators to thread pool
        completed = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_regulator, (i, module)): (i, module)
                for i, module in enumerate(modules)
            }

            for future in as_completed(futures):
                idx, module = futures[future]
                completed += 1
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    # Progress logging every 10 regulators or at milestones
                    if completed % 10 == 0 or completed == len(modules):
                        pct = 100 * completed / len(modules)
                        logger.info(f"Progress: {completed}/{len(modules)} regulators ({pct:.1f}%)")
                except Exception as e:
                    logger.error(f"Failed to get result for {module.regulator_name}: {e}")
    else:
        # Sequential execution (n_workers=1)
        for i, module in enumerate(modules):
            try:
                logger.info(f"Processing regulator {i+1}/{len(modules)}: {module.regulator_name}")
                result = analyze_regulator_module(
                    regulator_name=module.regulator_name,
                    module=module,
                    validator=validator,
                    conditions=conditions,
                    min_correlation=min_correlation,
                    min_clique_size=min_clique_size,
                    use_fast_maximum=use_fast_maximum,
                    correlation_method=correlation_method,
                    n_threads=n_condition_threads,
                    rna_filter_genes=rna_filter_genes # Pass RNA filter genes
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {module.regulator_name}: {e}")

    return results


# =============================================================================
# Output and Reporting
# =============================================================================

def save_results(
    results: List[RegulatorAnalysisResult],
    output_dir: Path,
    parameters: dict
):
    """
    Save analysis results to flat, consolidated CSV files.

    Output files (all in output_dir):
    - analysis_parameters.json: Run parameters for reproducibility
    - regulators_summary.csv: One row per regulator with key metrics
    - cliques.csv: One row per (regulator, condition)
    - differential_rewiring.csv: One row per (regulator, comparison)
    - clique_genes.csv: Long-form (regulator, condition, gene) for network analysis
    - gene_clique_frequency.csv: Genes ranked by clique participation
    - top_rewiring_regulators.csv: Top 100 regulators by rewiring score
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save parameters JSON (lightweight, for reproducibility)
    with open(output_dir / 'analysis_parameters.json', 'w') as f:
        json.dump(parameters, f, indent=2, default=str)

    # Accumulate data for flat files
    summary_rows = []
    stratified_rows = []
    differential_rows = []
    clique_gene_rows = []
    gene_pair_rows = []  # For differential_gene_pairs.csv
    edge_rows = []  # For clique_edges.csv - per-edge correlation values

    for result in results:
        regulator = result.regulator_name
        n_indra_targets = len(result.indra_targets)

        # Track per-regulator aggregates
        max_clique_size = 0
        max_coherence = 0.0
        best_condition = ""
        conditions_with_cliques = 0
        max_rewiring = 0.0
        avg_rewiring = 0.0
        total_gained = 0
        total_lost = 0

        # Process stratified results
        for cond, sr in result.stratified_results.items():
            n_coherent = len(sr.coherent_genes)
            coherence = sr.coherence_ratio
            clique_genes_str = ','.join(sorted(sr.coherent_genes)) if sr.coherent_genes else ''

            # RNA validation annotation
            n_rna_validated = len(sr.rna_validated_targets) if sr.rna_validated_targets else None
            rna_validation_ratio = sr.rna_validation_ratio

            # Extract signed correlation fields from coherent_clique (ChildSetType2)
            cc = sr.coherent_clique
            direction = cc.direction.value if cc and hasattr(cc.direction, 'value') else 'unknown'
            signed_mean_corr = cc.signed_mean_correlation if cc else None
            signed_min_corr = cc.signed_min_correlation if cc else None
            signed_max_corr = cc.signed_max_correlation if cc else None
            n_pos_edges = cc.n_positive_edges if cc else 0
            n_neg_edges = cc.n_negative_edges if cc else 0

            stratified_rows.append({
                'regulator': regulator,
                'condition': cond,
                'n_samples': sr.n_samples,
                'n_indra_targets': n_indra_targets,
                'n_rna_validated_targets': n_rna_validated,  # Annotation: how many targets have RNA support
                'n_coherent_genes': n_coherent,
                'coherence_ratio': round(coherence, 6),
                'rna_validation_ratio': round(rna_validation_ratio, 4),  # Annotation: fraction of clique with RNA support
                'direction': direction,  # POSITIVE, NEGATIVE, MIXED, or unknown
                'signed_mean_correlation': round(signed_mean_corr, 6) if signed_mean_corr is not None else None,
                'signed_min_correlation': round(signed_min_corr, 6) if signed_min_corr is not None else None,
                'signed_max_correlation': round(signed_max_corr, 6) if signed_max_corr is not None else None,
                'n_positive_edges': n_pos_edges,
                'n_negative_edges': n_neg_edges,
                'clique_genes': clique_genes_str
            })

            # Long-form gene rows with RNA validation annotation
            for gene in sr.coherent_genes:
                is_rna_validated = (sr.rna_validated_targets is not None and
                                   gene in sr.rna_validated_targets)
                clique_gene_rows.append({
                    'regulator': regulator,
                    'condition': cond,
                    'gene': gene,
                    'rna_validated': is_rna_validated  # Annotation: does this gene have RNA support?
                })

            # Per-edge correlation values (for clique_edges.csv)
            if cc and cc.edge_correlations:
                for gene1, gene2, correlation in cc.edge_correlations:
                    edge_rows.append({
                        'regulator': regulator,
                        'condition': cond,
                        'gene1': gene1,
                        'gene2': gene2,
                        'correlation': round(correlation, 6),
                    })

            # Track max
            if n_coherent > 0:
                conditions_with_cliques += 1
            if n_coherent > max_clique_size:
                max_clique_size = n_coherent
                max_coherence = coherence
                best_condition = cond

        # Process differential results
        n_comparisons = len(result.differential_results)
        for dr in result.differential_results:
            rewiring = dr.regulatory_rewiring_score
            gained = len(dr.gained_cliques)
            lost = len(dr.lost_cliques)

            # Get sample sizes (available from StratifiedCliqueResult via stratified_results)
            # Or use dr.n_case_samples / dr.n_ctrl_samples if available in DifferentialResult
            # DifferentialResult doesn't store n_samples directly in previous version,
            # but we can get it from stratified_results
            n_case = result.stratified_results[dr.case_condition].n_samples if dr.case_condition in result.stratified_results else 0
            n_ctrl = result.stratified_results[dr.ctrl_condition].n_samples if dr.ctrl_condition in result.stratified_results else 0

            differential_rows.append({
                'regulator': regulator,
                'comparison': f"{dr.case_condition}_vs_{dr.ctrl_condition}",
                'n_case_samples': n_case, # NEW
                'n_ctrl_samples': n_ctrl, # NEW
                'gained_cliques': gained,
                'lost_cliques': lost,
                'case_coherence': round(dr.case_coherence, 6),
                'ctrl_coherence': round(dr.ctrl_coherence, 6),
                'rewiring_score': round(rewiring, 6),
                # FDR statistics
                'n_gene_pairs_tested': dr.n_gene_pairs_tested,
                'n_significant_pairs': dr.n_significant_pairs,
                'fdr_support_ratio': round(dr.fdr_support_ratio, 4),
                'fdr_threshold': dr.fdr_threshold,
                'correlation_threshold': dr.significance_threshold,
                # Effective tests (Nature Methods requirement)
                'nominal_tests': dr.nominal_tests,
                'effective_tests': round(dr.effective_tests, 2),
                'effective_test_reduction': round(dr.effective_test_reduction, 4),
            })

            # Process significant gene pairs for detailed output
            if dr.significant_gene_pairs:
                for stat in dr.significant_gene_pairs:
                    gene_pair_rows.append({
                        'regulator': regulator,
                        'comparison': f"{dr.case_condition}_vs_{dr.ctrl_condition}",
                        'gene1': stat.gene1,
                        'gene2': stat.gene2,
                        'r_case': round(stat.r_case, 4),
                        'r_ctrl': round(stat.r_ctrl, 4),
                        'delta_r': round(stat.delta_r, 4),
                        'z_score': round(stat.z_score, 4),
                        'p_value': stat.p_value, # Keep full precision for p-values
                        'q_value': stat.q_value,
                        'is_significant': stat.is_significant,
                        'ci_case_lower': round(stat.ci_case[0], 4),
                        'ci_case_upper': round(stat.ci_case[1], 4),
                        'ci_ctrl_lower': round(stat.ci_ctrl[0], 4),
                        'ci_ctrl_upper': round(stat.ci_ctrl[1], 4),
                    })

            max_rewiring = max(max_rewiring, rewiring)
            avg_rewiring += rewiring
            total_gained += gained
            total_lost += lost

        if n_comparisons > 0:
            avg_rewiring /= n_comparisons

        # RNA validation annotation for summary
        n_rna_validated = len(result.rna_validated_targets) if result.rna_validated_targets else None

        # Summary row
        summary_rows.append({
            'regulator': regulator,
            'n_indra_targets': n_indra_targets,
            'n_rna_validated_targets': n_rna_validated,  # Annotation: targets with RNA support
            'max_clique_size': max_clique_size,
            'max_coherence': round(max_coherence, 4),
            'best_condition': best_condition,
            'conditions_with_cliques': conditions_with_cliques,
            'max_rewiring_score': round(max_rewiring, 4),
            'avg_rewiring_score': round(avg_rewiring, 4),
            'total_gained_cliques': total_gained,
            'total_lost_cliques': total_lost
        })

    # Write flat files
    logger.info(f"Writing consolidated results to {output_dir}/")

    # 1. Regulators summary (sorted by max_rewiring_score descending)
    # FILTER: Only include regulators with at least one clique
    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        # Filter to regulators that have cliques in at least one condition
        summary_df = summary_df[summary_df['conditions_with_cliques'] > 0]
        summary_df = summary_df.sort_values('max_rewiring_score', ascending=False)
    logger.info(f"  regulators_summary.csv: {len(summary_df)} regulators with cliques (filtered from {len(summary_rows)} total)")
    summary_df.to_csv(output_dir / 'regulators_summary.csv', index=False)

    # 2. Stratified cliques
    # FILTER: Only include rows where cliques were found (coherent_genes > 0)
    stratified_df = pd.DataFrame(stratified_rows)
    if len(stratified_df) > 0:
        stratified_df = stratified_df[stratified_df['n_coherent_genes'] > 0]
    logger.info(f"  cliques.csv: {len(stratified_df)} rows with cliques (filtered from {len(stratified_rows)} total)")
    stratified_df.to_csv(output_dir / 'cliques.csv', index=False)

    # 3. Regulator Rewiring Stats (sorted by rewiring_score descending)
    # FILTER: Only include regulators that have at least one clique
    if differential_rows:
        differential_df = pd.DataFrame(differential_rows)
        # Get set of regulators with cliques from filtered summary
        regulators_with_cliques = set(summary_df['regulator'].values) if len(summary_df) > 0 else set()
        # Filter differential results to only those regulators
        differential_df = differential_df[differential_df['regulator'].isin(regulators_with_cliques)]
        differential_df = differential_df.sort_values('rewiring_score', ascending=False)
        logger.info(f"  regulator_rewiring_stats.csv: {len(differential_df)} rows with cliques (filtered from {len(differential_rows)} total)")
        differential_df.to_csv(output_dir / 'regulator_rewiring_stats.csv', index=False)

    # 4. Clique genes (long form)
    if clique_gene_rows:
        genes_df = pd.DataFrame(clique_gene_rows)
        genes_df.to_csv(output_dir / 'clique_genes.csv', index=False)
        logger.info(f"  clique_genes.csv: {len(clique_gene_rows)} gene memberships")

    # 5. Clique edges (per-edge correlation values)
    if edge_rows:
        edges_df = pd.DataFrame(edge_rows)
        edges_df.to_csv(output_dir / 'clique_edges.csv', index=False)
        logger.info(f"  clique_edges.csv: {len(edge_rows)} edges with correlation values")

    # 6. Gene Pair Stats (Detailed stats)
    # FILTER: Only include gene pairs from regulators with cliques
    if gene_pair_rows:
        pairs_df = pd.DataFrame(gene_pair_rows)
        # Get set of regulators with cliques from filtered summary
        regulators_with_cliques = set(summary_df['regulator'].values) if len(summary_df) > 0 else set()
        # Filter gene pair results to only those regulators
        pairs_df = pairs_df[pairs_df['regulator'].isin(regulators_with_cliques)]
        # Sort by q-value (most significant first)
        pairs_df = pairs_df.sort_values('q_value', ascending=True)
        logger.info(f"  gene_pair_stats.csv: {len(pairs_df)} significant pairs with cliques (filtered from {len(gene_pair_rows)} total)")
        pairs_df.to_csv(output_dir / 'gene_pair_stats.csv', index=False)

    # 6. Multiple Testing Report (NEW)
    # Aggregate M_eff statistics across all differential comparisons
    # Use filtered differential_df if available, otherwise use differential_rows
    if differential_rows:
        # Use filtered data if available
        filtered_differential_rows = differential_df.to_dict('records') if 'differential_df' in locals() and len(differential_df) > 0 else []
        reporting_rows = filtered_differential_rows if filtered_differential_rows else differential_rows

        total_nominal = sum(r['nominal_tests'] for r in reporting_rows if r['nominal_tests'] > 0)
        total_effective = sum(r['effective_tests'] for r in reporting_rows if r['effective_tests'] > 0)
        n_comparisons_with_tests = sum(1 for r in reporting_rows if r['nominal_tests'] > 0)

        # Compute aggregated statistics
        effective_test_ratios = [
            r['effective_test_reduction'] for r in reporting_rows
            if r['nominal_tests'] > 0 and r['effective_test_reduction'] > 0
        ]

        multiple_testing_report = {
            'summary': {
                'n_regulators_analyzed': len(summary_rows),
                'n_regulators_with_cliques': len(summary_df),
                'n_differential_comparisons': len(differential_rows),
                'n_differential_comparisons_with_cliques': len(reporting_rows),
                'n_comparisons_with_statistical_tests': n_comparisons_with_tests,
                'total_nominal_tests': total_nominal,
                'total_effective_tests': round(total_effective, 2),
                'overall_effective_ratio': round(total_effective / total_nominal, 4) if total_nominal > 0 else None,
            },
            'effective_test_statistics': {
                'mean_effective_ratio': round(sum(effective_test_ratios) / len(effective_test_ratios), 4) if effective_test_ratios else None,
                'median_effective_ratio': round(sorted(effective_test_ratios)[len(effective_test_ratios) // 2], 4) if effective_test_ratios else None,
                'min_effective_ratio': round(min(effective_test_ratios), 4) if effective_test_ratios else None,
                'max_effective_ratio': round(max(effective_test_ratios), 4) if effective_test_ratios else None,
            },
            'fdr_correction': {
                'method': 'benjamini_hochberg',
                'threshold': reporting_rows[0]['fdr_threshold'] if reporting_rows else 0.05,
                'total_significant_pairs': sum(r['n_significant_pairs'] for r in reporting_rows),
            },
            'interpretation': {
                'note': 'M_eff/M < 1 indicates correlated tests; lower values mean stronger correlation',
                'typical_range_gene_expression': '0.3-0.6 (moderate correlation due to co-regulation)',
                'recommendation': 'Report both nominal M and effective M_eff for transparency',
            }
        }

        with open(output_dir / 'multiple_testing_report.json', 'w') as f:
            json.dump(multiple_testing_report, f, indent=2)
        logger.info(f"  multiple_testing_report.json: M_eff aggregated statistics")

    # Print quick stats (using filtered summary_df)
    if len(summary_df) > 0:
        with_cliques = len(summary_df)  # All rows in summary_df have cliques by definition
        max_size = summary_df['max_clique_size'].max()
        max_rew = summary_df['max_rewiring_score'].max()
        logger.info(f"\nQuick stats:")
        logger.info(f"  Regulators with cliques: {with_cliques} (out of {len(summary_rows)} analyzed)")
        logger.info(f"  Max clique size: {max_size}")
        logger.info(f"  Max rewiring score: {max_rew:.4f}")


def print_summary(results: List[RegulatorAnalysisResult]):
    """Print analysis summary to stdout."""
    print("\n" + "=" * 70)
    print("REGULATORY MODULE ANALYSIS SUMMARY")
    print("=" * 70)

    for result in results:
        print(f"\n{result.regulator_name} ({result.regulator_id[0]}:{result.regulator_id[1]})")
        print("-" * 40)
        print(f"  INDRA targets: {len(result.indra_targets)}")

        # Stratified results
        print("\n  Stratified Coherence:")
        for cond, sr in result.stratified_results.items():
            print(f"    {cond:15s}: {sr.coherence_ratio:.2f} "
                  f"({len(sr.coherent_genes)}/{len(sr.indra_targets)} genes, "
                  f"n={sr.n_samples})")

        # Differential results
        if result.differential_results:
            print("\n  Differential Analysis:")
            for dr in result.differential_results:
                direction = "+" if dr.case_coherence > dr.ctrl_coherence else "-"
                print(f"    {dr.case_condition} vs {dr.ctrl_condition}: "
                      f"rewiring={dr.regulatory_rewiring_score:.2f} [{direction}]")
                if dr.gained_cliques:
                    print(f"      Gained: {len(dr.gained_cliques)} cliques")
                if dr.lost_cliques:
                    print(f"      Lost: {len(dr.lost_cliques)} cliques")
