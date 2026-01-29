"""
Unified co-expression module discovery.

Two paradigms, one code path:

1. REGULATORY VALIDATION (knowledge-guided):
   Gene universe = INDRA regulatory targets
   Question: "Which known targets co-express in this condition?"

2. DE NOVO DISCOVERY (data-driven):
   Gene universe = High-variance genes from expression data
   Question: "What coordinated expression programs exist?"

Both flow through the same CliqueValidator infrastructure:
    gene_universe → build_correlation_graph → find_cliques → modules

Engineering Design:
    - GeneUniverseSelector: Abstract interface for gene set selection
    - RegulatoryTargetUniverse: INDRA-constrained (existing workflow)
    - VarianceFilteredUniverse: Data-driven (new capability)
    - ModuleDiscovery: Unified orchestrator using CliqueValidator
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Literal, Iterator
import numpy as np
import pandas as pd
import logging

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.knowledge.clique_validator import (
    CliqueValidator,
    CorrelationClique,
    InsufficientSamplesError,
)

logger = logging.getLogger(__name__)

__all__ = [
    'GeneUniverseSelector',
    'RegulatoryTargetUniverse',
    'VarianceFilteredUniverse',
    'RNAValidatedUniverse',
    'CoexpressionModule',
    'ModuleDiscovery',
]


# =============================================================================
# Gene Universe Selection
# =============================================================================

class GeneUniverseSelector(ABC):
    """
    Abstract interface for selecting gene universes for clique discovery.

    The gene universe defines the search space for co-expression modules.
    Different selectors enable different discovery paradigms while using
    the same downstream clique-finding machinery.
    """

    @abstractmethod
    def get_gene_universe(self) -> Set[str]:
        """Return the set of genes to analyze."""
        pass

    @abstractmethod
    def get_label(self) -> str:
        """Return a human-readable label for this universe."""
        pass

    @property
    @abstractmethod
    def paradigm(self) -> str:
        """Return 'regulatory_validation' or 'de_novo_discovery'."""
        pass


class RegulatoryTargetUniverse(GeneUniverseSelector):
    """
    Gene universe from INDRA regulatory relationships.

    Used for validating known regulatory programs:
    "Do TP53 targets show coordinated expression in ALS?"

    Attributes:
        regulator: Upstream regulator name (TF, kinase, etc.)
        targets: Set of downstream target genes from INDRA
    """

    def __init__(self, regulator: str, targets: Set[str]):
        self.regulator = regulator
        self.targets = targets

    def get_gene_universe(self) -> Set[str]:
        return self.targets

    def get_label(self) -> str:
        return f"{self.regulator}_regulatory_targets"

    @property
    def paradigm(self) -> str:
        return "regulatory_validation"


class VarianceFilteredUniverse(GeneUniverseSelector):
    """
    Gene universe from expression variance filtering.

    Used for de novo co-expression discovery:
    "What coordinated programs exist in this dataset?"

    Selects top N genes by variance, ensuring computational tractability
    while capturing biologically variable (and thus interesting) genes.

    Attributes:
        matrix: Expression data
        n_genes: Number of top-variance genes to select
        condition: Optional condition for variance calculation
    """

    def __init__(
        self,
        matrix: BioMatrix,
        n_genes: int = 5000,
        percentile: Optional[float] = None,
        condition_mask: Optional[np.ndarray] = None,
        label: str = "high_variance",
    ):
        """
        Initialize variance-filtered gene universe.

        Args:
            matrix: Expression BioMatrix
            n_genes: Number of top-variance genes (default: 5000)
            percentile: Alternative to n_genes - top X percentile
            condition_mask: Optional boolean mask for condition-specific variance
            label: Human-readable label for this universe
        """
        self.matrix = matrix
        self.n_genes = n_genes
        self.percentile = percentile
        self.condition_mask = condition_mask
        self._label = label
        self._cached_universe: Optional[Set[str]] = None

    def get_gene_universe(self) -> Set[str]:
        if self._cached_universe is not None:
            return self._cached_universe

        # Compute variance (condition-specific if mask provided)
        if self.condition_mask is not None:
            data = self.matrix.data[:, self.condition_mask]
        else:
            data = self.matrix.data

        variances = np.var(data, axis=1)

        # Select top genes
        if self.percentile is not None:
            threshold = np.percentile(variances, 100 - self.percentile)
            selected_mask = variances >= threshold
        else:
            n_select = min(self.n_genes, len(variances))
            top_indices = np.argsort(variances)[-n_select:]
            selected_mask = np.zeros(len(variances), dtype=bool)
            selected_mask[top_indices] = True

        selected_genes = set(self.matrix.feature_ids[selected_mask].tolist())
        self._cached_universe = selected_genes

        logger.info(f"Variance-filtered universe: {len(selected_genes)} genes selected")
        return selected_genes

    def get_label(self) -> str:
        return self._label

    @property
    def paradigm(self) -> str:
        return "de_novo_discovery"


class ConnectedComponentUniverse(GeneUniverseSelector):
    """
    Gene universe from correlation graph connected components.

    For very large gene sets, first builds a sparse correlation graph
    and extracts connected components, then runs clique finding on
    each component separately. More efficient than global clique search.

    Attributes:
        parent_universe: The larger universe to partition
        component_genes: Genes in this specific component
        component_id: Identifier for this component
    """

    def __init__(
        self,
        component_genes: Set[str],
        component_id: int,
        parent_label: str,
    ):
        self.component_genes = component_genes
        self.component_id = component_id
        self.parent_label = parent_label

    def get_gene_universe(self) -> Set[str]:
        return self.component_genes

    def get_label(self) -> str:
        return f"{self.parent_label}_component_{self.component_id}"

    @property
    def paradigm(self) -> str:
        return "de_novo_discovery"


class RNAValidatedUniverse(GeneUniverseSelector):
    """
    Filters gene universe to those present in RNA dataset.

    Biological rationale: Only consider regulators with RNA-seq evidence,
    ensuring transcriptomal activity confirmation for proteomics findings.

    Can wrap another universe selector for composition:
        base_universe = VarianceFilteredUniverse(matrix, n_genes=5000)
        rna_universe = RNAValidatedUniverse(rna_genes, base_selector=base_universe)

    This enables cross-modal validation: genes must pass both criteria
    (e.g., high variance in proteomics AND measured in RNA-seq).

    Attributes:
        rna_genes: Set of gene symbols present in RNA data
        base_selector: Optional inner selector to compose with
        require_all_in_rna: If True, raise error if base returns genes not in RNA
        label: Human-readable label for this universe
    """

    def __init__(
        self,
        rna_genes: Set[str],
        base_selector: Optional[GeneUniverseSelector] = None,
        require_all_in_rna: bool = False,
        label: str = "rna_validated",
    ):
        """
        Initialize RNA-validated gene universe.

        Args:
            rna_genes: Set of gene symbols present in RNA data
            base_selector: Optional inner selector to compose with
            require_all_in_rna: If True, raise error if base returns genes not in RNA
            label: Human-readable label for this universe

        Examples:
            >>> # Direct usage - filter to RNA-measured genes only
            >>> rna_universe = RNAValidatedUniverse(rna_genes={'PGK1', 'BRCA1', 'TP53'})
            >>> genes = rna_universe.get_gene_universe()

            >>> # Composed usage - high variance proteins that are also in RNA
            >>> base = VarianceFilteredUniverse(matrix, n_genes=5000)
            >>> rna_universe = RNAValidatedUniverse(rna_genes, base_selector=base)
        """
        self.rna_genes = rna_genes
        self.base_selector = base_selector
        self.require_all_in_rna = require_all_in_rna
        self._label = label
        self._cached_universe: Optional[Set[str]] = None

    def get_gene_universe(self) -> Set[str]:
        """
        Returns intersection of:
        1. Base selector's universe (or all RNA genes if no base)
        2. RNA gene set

        Logs statistics about filtering to track cross-modal coverage.

        Returns:
            Set of gene symbols present in both universes

        Raises:
            ValueError: If require_all_in_rna=True and base genes missing from RNA
        """
        if self._cached_universe is not None:
            return self._cached_universe

        # Get base universe (or use all RNA genes)
        if self.base_selector is not None:
            base_genes = self.base_selector.get_gene_universe()
            logger.info(
                f"RNA validation: Base universe '{self.base_selector.get_label()}' "
                f"has {len(base_genes)} genes"
            )
        else:
            # No base selector: return all RNA genes
            base_genes = self.rna_genes
            logger.info(f"RNA validation: Using all {len(base_genes)} RNA genes as base")

        # Intersect with RNA genes
        validated_genes = base_genes & self.rna_genes

        # Check coverage
        n_missing = len(base_genes) - len(validated_genes)
        coverage_pct = 100 * len(validated_genes) / len(base_genes) if base_genes else 0

        logger.info(
            f"RNA validation: {len(validated_genes)}/{len(base_genes)} genes "
            f"({coverage_pct:.1f}%) present in RNA dataset"
        )

        if n_missing > 0:
            logger.info(
                f"RNA validation: {n_missing} genes from base universe "
                f"not measured in RNA-seq (filtered out)"
            )

        # Strict mode: error if genes missing from RNA
        if self.require_all_in_rna and n_missing > 0:
            missing_genes = base_genes - self.rna_genes
            raise ValueError(
                f"RNA validation failed: {n_missing} genes from base universe "
                f"not found in RNA dataset (require_all_in_rna=True). "
                f"Missing genes: {sorted(list(missing_genes))[:10]}..."
            )

        self._cached_universe = validated_genes
        return validated_genes

    def get_label(self) -> str:
        """Return human-readable label showing composition."""
        if self.base_selector is not None:
            return f"{self._label}({self.base_selector.get_label()})"
        return self._label

    @property
    def paradigm(self) -> str:
        """
        Return paradigm from base selector, or 'cross_modal_validation' if no base.

        Inherits paradigm when wrapping another selector (e.g., wrapping
        VarianceFilteredUniverse keeps 'de_novo_discovery' paradigm).
        """
        if self.base_selector is not None:
            return self.base_selector.paradigm
        return "cross_modal_validation"


# =============================================================================
# Module Results
# =============================================================================

@dataclass
class CoexpressionModule:
    """
    A discovered co-expression module.

    Unifies results from both paradigms with provenance tracking.

    Attributes:
        genes: Set of genes in this module
        condition: Experimental condition where module was found
        mean_correlation: Mean pairwise correlation within module
        min_correlation: Minimum pairwise correlation (coherence floor)
        size: Number of genes
        paradigm: 'regulatory_validation' or 'de_novo_discovery'
        universe_label: Label identifying the gene universe source
        regulator: For regulatory validation, the upstream regulator
        direction: Classification of correlation signs in module.
            POSITIVE = all edges have r > 0 (co-activation)
            NEGATIVE = all edges have r < 0 (anti-correlation/repression)
            MIXED = edges have both positive and negative correlations
        signed_mean_correlation: Mean correlation preserving sign. Unlike
            mean_correlation (which uses absolute values), this can be negative
            for anti-correlated modules.
        signed_min_correlation: Minimum signed correlation (most negative or
            least positive).
        signed_max_correlation: Maximum signed correlation (most positive or
            least negative).
        n_positive_edges: Number of edges with r > 0.
        n_negative_edges: Number of edges with r < 0.
    """
    genes: Set[str]
    condition: str
    mean_correlation: float
    min_correlation: float
    size: int
    paradigm: str
    universe_label: str
    regulator: Optional[str] = None
    direction: str = "unknown"
    signed_mean_correlation: float | None = None
    signed_min_correlation: float | None = None
    signed_max_correlation: float | None = None
    n_positive_edges: int = 0
    n_negative_edges: int = 0

    def to_dict(self) -> Dict:
        return {
            'genes': ','.join(sorted(self.genes)),
            'condition': self.condition,
            'mean_correlation': self.mean_correlation,
            'min_correlation': self.min_correlation,
            'size': self.size,
            'paradigm': self.paradigm,
            'universe_label': self.universe_label,
            'regulator': self.regulator or '',
            # NEW fields
            'direction': self.direction,
            'signed_mean_correlation': self.signed_mean_correlation,
            'signed_min_correlation': self.signed_min_correlation,
            'signed_max_correlation': self.signed_max_correlation,
            'n_positive_edges': self.n_positive_edges,
            'n_negative_edges': self.n_negative_edges,
        }


# =============================================================================
# Unified Module Discovery
# =============================================================================

class ModuleDiscovery:
    """
    Unified co-expression module discovery.

    Orchestrates both paradigms through the same CliqueValidator:

    1. Regulatory Validation:
       >>> universe = RegulatoryTargetUniverse("TP53", indra_targets)
       >>> modules = discovery.find_modules(universe, condition="CASE")

    2. De Novo Discovery:
       >>> universe = VarianceFilteredUniverse(matrix, n_genes=5000)
       >>> modules = discovery.find_modules(universe, condition="CASE")

    Both use identical clique-finding code paths.

    Attributes:
        validator: CliqueValidator instance with precomputed condition data
    """

    def __init__(self, validator: CliqueValidator, rna_filter_genes: Optional[Set[str]] = None):
        """
        Initialize with a configured CliqueValidator.

        Args:
            validator: CliqueValidator with expression data and stratification
            rna_filter_genes: Optional set of genes validated in RNA-seq data
        """
        self.validator = validator
        self.rna_filter_genes = rna_filter_genes

    @classmethod
    def from_matrix(
        cls,
        matrix: BioMatrix,
        stratify_by: Optional[List[str]] = None,
        min_samples: int = 20,
        rna_filter_genes: Optional[Set[str]] = None,
    ) -> 'ModuleDiscovery':
        """
        Convenience constructor from BioMatrix.

        Args:
            matrix: Expression data with sample metadata
            stratify_by: Metadata columns for condition stratification
            min_samples: Minimum samples per stratum
            rna_filter_genes: Optional set of genes validated in RNA-seq data

        Returns:
            Configured ModuleDiscovery instance
        """
        validator = CliqueValidator(
            matrix,
            stratify_by=stratify_by,
            min_samples=min_samples,
            precompute=True,
        )
        return cls(validator, rna_filter_genes=rna_filter_genes)

    def find_modules(
        self,
        universe: GeneUniverseSelector,
        condition: str,
        min_correlation: float = 0.7,
        min_module_size: int = 3,
        method: Literal["pearson", "spearman"] = "pearson",
        use_greedy: bool = True,
    ) -> List[CoexpressionModule]:
        """
        Find co-expression modules within a gene universe.

        Routes to CliqueValidator's clique-finding machinery.

        Args:
            universe: Gene universe selector (regulatory or de novo)
            condition: Experimental condition (e.g., "CASE", "CASE_Male")
            min_correlation: Minimum correlation threshold
            min_module_size: Minimum genes per module
            method: Correlation method
            use_greedy: If True, use O(n*d²) greedy; if False, enumerate all

        Returns:
            List of CoexpressionModule objects
        """
        genes = universe.get_gene_universe()

        # Filter to genes present in expression data
        available_genes = set(self.validator.matrix.feature_ids.tolist())
        genes = genes & available_genes

        if len(genes) < min_module_size:
            logger.warning(
                f"Universe '{universe.get_label()}' has only {len(genes)} genes "
                f"in expression data (need {min_module_size})"
            )
            return []

        logger.info(
            f"Finding modules in '{universe.get_label()}': "
            f"{len(genes)} genes, condition={condition}, r>={min_correlation}"
        )

        try:
            if use_greedy:
                # Single largest module (efficient)
                clique = self.validator.find_maximum_clique(
                    genes, condition, min_correlation, method
                )
                cliques = [clique] if clique and clique.size >= min_module_size else []
            else:
                # All maximal cliques (expensive, exact enumeration)
                # exact=True ensures complete enumeration without silent truncation at 10k cliques
                # timeout_seconds=300.0 provides reasonable upper bound per condition
                cliques = self.validator.find_cliques(
                    genes, condition, min_correlation, min_module_size, method,
                    exact=True,
                    timeout_seconds=300.0
                )
        except InsufficientSamplesError as e:
            logger.warning(f"Skipping condition '{condition}': {e}")
            return []

        # Convert to CoexpressionModule format
        modules = []
        for clique in cliques:
            module = CoexpressionModule(
                genes=clique.genes,
                condition=condition,
                mean_correlation=clique.mean_correlation,
                min_correlation=clique.min_correlation,
                size=clique.size,
                paradigm=universe.paradigm,
                universe_label=universe.get_label(),
                regulator=(
                    universe.regulator
                    if isinstance(universe, RegulatoryTargetUniverse)
                    else None
                ),
                # NEW: propagate signed stats
                direction=clique.direction.value if hasattr(clique.direction, 'value') else str(clique.direction),
                signed_mean_correlation=clique.signed_mean_correlation,
                signed_min_correlation=clique.signed_min_correlation,
                signed_max_correlation=clique.signed_max_correlation,
                n_positive_edges=clique.n_positive_edges,
                n_negative_edges=clique.n_negative_edges,
            )
            modules.append(module)

        logger.info(f"Found {len(modules)} modules in '{universe.get_label()}'")
        return modules

    def find_modules_all_conditions(
        self,
        universe: GeneUniverseSelector,
        min_correlation: float = 0.7,
        min_module_size: int = 3,
        method: Literal["pearson", "spearman"] = "pearson",
        use_greedy: bool = True,
    ) -> List[CoexpressionModule]:
        """
        Find modules across all available conditions.

        Args:
            universe: Gene universe selector
            min_correlation: Minimum correlation threshold
            min_module_size: Minimum genes per module
            method: Correlation method
            use_greedy: Use greedy algorithm

        Returns:
            List of modules across all conditions
        """
        all_modules = []

        for condition in self.validator.get_available_conditions():
            modules = self.find_modules(
                universe, condition, min_correlation,
                min_module_size, method, use_greedy
            )
            all_modules.extend(modules)

        return all_modules

    def discover_de_novo(
        self,
        n_genes: int = 5000,
        min_correlation: float = 0.7,
        min_module_size: int = 5,
        method: Literal["pearson", "spearman"] = "pearson",
        conditions: Optional[List[str]] = None,
        use_greedy: bool = True,
        partition_by_components: bool = True,
    ) -> List[CoexpressionModule]:
        """
        Convenience method for de novo discovery.

        Selects high-variance genes and finds co-expression modules.
        Optionally partitions by connected components for efficiency.

        Args:
            n_genes: Number of high-variance genes to analyze
            min_correlation: Minimum correlation threshold
            min_module_size: Minimum genes per module
            method: Correlation method
            conditions: Specific conditions (default: all)
            use_greedy: Use greedy algorithm
            partition_by_components: If True, find connected components first

        Returns:
            List of discovered modules

        Example:
            >>> discovery = ModuleDiscovery.from_matrix(matrix, ['phenotype', 'Sex'])
            >>> modules = discovery.discover_de_novo(n_genes=5000, min_correlation=0.8)
            >>> print(f"Found {len(modules)} de novo modules")
        """
        conditions = conditions or self.validator.get_available_conditions()
        all_modules = []

        for condition in conditions:
            logger.info(f"De novo discovery for condition: {condition}")

            # Get condition mask for condition-specific variance
            try:
                condition_mask = self.validator._get_condition_mask(condition)
            except (InsufficientSamplesError, ValueError):
                logger.warning(f"Skipping condition '{condition}': insufficient samples")
                continue

            # Create variance-filtered universe for this condition
            base_universe = VarianceFilteredUniverse(
                self.validator.matrix,
                n_genes=n_genes,
                condition_mask=condition_mask,
                label=f"de_novo_{condition}",
            )

            # Optionally wrap with RNA filter
            if self.rna_filter_genes is not None:
                universe = RNAValidatedUniverse(
                    rna_genes=self.rna_filter_genes,
                    base_selector=base_universe,
                    label="rna_validated",
                )
            else:
                universe = base_universe

            if partition_by_components:
                # Find connected components first, then cliques per component
                modules = self._discover_via_components(
                    universe, condition, min_correlation,
                    min_module_size, method, use_greedy
                )
            else:
                # Direct clique finding on full universe
                modules = self.find_modules(
                    universe, condition, min_correlation,
                    min_module_size, method, use_greedy
                )

            all_modules.extend(modules)

        return all_modules

    def _discover_via_components(
        self,
        universe: GeneUniverseSelector,
        condition: str,
        min_correlation: float,
        min_module_size: int,
        method: str,
        use_greedy: bool,
    ) -> List[CoexpressionModule]:
        """
        Partition gene universe into connected components, then find cliques.

        More efficient for large sparse correlation graphs.
        """
        import networkx as nx

        genes = universe.get_gene_universe()

        # Filter to available genes
        available = set(self.validator.matrix.feature_ids.tolist())
        genes = genes & available

        if len(genes) < min_module_size:
            return []

        # Build correlation graph
        logger.info(f"Building correlation graph for {len(genes)} genes...")
        try:
            G = self.validator.build_correlation_graph(
                list(genes), condition, min_correlation, method
            )
        except InsufficientSamplesError:
            return []

        # Find connected components
        components = list(nx.connected_components(G))
        logger.info(f"Found {len(components)} connected components")

        # Find cliques in each component
        all_modules = []
        for i, component in enumerate(components):
            if len(component) < min_module_size:
                continue

            comp_universe = ConnectedComponentUniverse(
                component_genes=component,
                component_id=i,
                parent_label=universe.get_label(),
            )

            modules = self.find_modules(
                comp_universe, condition, min_correlation,
                min_module_size, method, use_greedy
            )
            all_modules.extend(modules)

        return all_modules
