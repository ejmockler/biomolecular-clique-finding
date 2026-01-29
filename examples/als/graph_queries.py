"""
Knowledge Graph Query Functions for ALS Research.

This module provides gene-specific query functions for ALS research that
leverage the generic GraphQuery API. These functions are experiment-specific
and serve as examples of how to build custom queries for your research.

Key Pattern:
    These functions demonstrate how to:
    1. Use GraphQuery.neighbors() to get 1-hop neighbors
    2. Filter by relationship type (activation, inhibition, etc.)
    3. Split results by relationship for separate analysis
    4. Constrain to your experimental gene universe

Usage:
    >>> from cliquefinder.knowledge import INDRAKnowledgeSource
    >>> from examples.als.graph_queries import get_gene_neighbor_sets
    >>>
    >>> source = INDRAKnowledgeSource(env_file=".env")
    >>> gene_universe = set(matrix.feature_ids)
    >>>
    >>> # Get neighbors for any gene
    >>> c9_neighbors = get_gene_neighbor_sets("C9orf72", source, gene_universe)
    >>> sod1_neighbors = get_gene_neighbor_sets("SOD1", source, gene_universe)
    >>> tp53_neighbors = get_gene_neighbor_sets("TP53", source, gene_universe)
    >>>
    >>> # Convert to feature sets for analysis
    >>> activated_set = c9_neighbors["activated"].to_feature_set("C9_activated")
    >>> inhibited_set = c9_neighbors["inhibited"].to_feature_set("C9_inhibited")
"""

from typing import Set

from cliquefinder.knowledge.base import KnowledgeSource, KnowledgeEdge
from cliquefinder.knowledge.graph_query import GraphQuery, QueryResult


def get_gene_neighbor_sets(
    gene_name: str,
    source: KnowledgeSource,
    universe: Set[str] | None = None,
    min_evidence: int = 2,
    activation_rels: Set[str] | None = None,
    inhibition_rels: Set[str] | None = None,
) -> dict[str, QueryResult]:
    """
    Get gene neighbors split by relationship type.

    This is a generic version that works for any gene. It queries 1-hop
    neighbors and splits them into activation/inhibition categories based
    on relationship types.

    Args:
        gene_name: Gene symbol (e.g., "C9orf72", "SOD1", "TP53")
        source: KnowledgeSource (e.g., INDRAKnowledgeSource)
        universe: Optional set of genes in dataset (for filtering)
        min_evidence: Minimum evidence count (default: 2)
        activation_rels: Relationship types to consider as "activation"
            Default: {"IncreaseAmount", "Activation"}
        inhibition_rels: Relationship types to consider as "inhibition"
            Default: {"DecreaseAmount", "Inhibition"}

    Returns:
        Dict with keys: "activated", "inhibited", "all"
        Each value is a QueryResult containing:
            - entities: Set of neighboring gene symbols
            - edges: List of KnowledgeEdge objects
            - query_description: Human-readable description

    Example:
        >>> source = INDRAKnowledgeSource(env_file=".env")
        >>> gene_universe = set(matrix.feature_ids)
        >>>
        >>> # Get C9orf72 neighbors
        >>> neighbors = get_gene_neighbor_sets("C9orf72", source, gene_universe)
        >>> print(f"Activated: {len(neighbors['activated'])} genes")
        >>> print(f"Inhibited: {len(neighbors['inhibited'])} genes")
        >>>
        >>> # Convert to feature sets for permutation testing
        >>> activated_set = neighbors["activated"].to_feature_set("C9_activated")
        >>> inhibited_set = neighbors["inhibited"].to_feature_set("C9_inhibited")
        >>> all_neighbors_set = neighbors["all"].to_feature_set("C9_all_neighbors")

    Notes:
        - Uses bidirectional queries to capture both upstream and downstream neighbors
        - Filters to minimum evidence count to reduce noise
        - Constrains to gene universe if provided (recommended)
        - Results can be directly converted to FeatureSet for statistical analysis
    """
    # Default relationship categories
    if activation_rels is None:
        activation_rels = {"IncreaseAmount", "Activation"}
    if inhibition_rels is None:
        inhibition_rels = {"DecreaseAmount", "Inhibition"}

    # Build query
    base_query = GraphQuery.neighbors(gene_name, direction="bidirectional")

    if min_evidence > 1:
        base_query = base_query.filter_evidence(min_evidence)

    if universe is not None:
        base_query = base_query.constrain_to(universe)

    # Execute and split by relationship
    result = base_query.execute(source)
    by_rel = result.by_relationship()

    # Aggregate into activation/inhibition categories
    activated_entities: Set[str] = set()
    activated_edges: list[KnowledgeEdge] = []
    inhibited_entities: Set[str] = set()
    inhibited_edges: list[KnowledgeEdge] = []

    for rel_name, rel_result in by_rel.items():
        if rel_name in activation_rels:
            activated_entities.update(rel_result.entities)
            activated_edges.extend(rel_result.edges)
        elif rel_name in inhibition_rels:
            inhibited_entities.update(rel_result.entities)
            inhibited_edges.extend(rel_result.edges)

    return {
        "activated": QueryResult(
            entities=activated_entities,
            edges=activated_edges,
            query_description=f"{gene_name} activated neighbors",
        ),
        "inhibited": QueryResult(
            entities=inhibited_entities,
            edges=inhibited_edges,
            query_description=f"{gene_name} inhibited neighbors",
        ),
        "all": result,
    }


def get_c9orf72_neighbor_sets(
    source: KnowledgeSource,
    universe: Set[str] | None = None,
    min_evidence: int = 2,
) -> dict[str, QueryResult]:
    """
    Get C9orf72 neighbors split by relationship type.

    This is a convenience wrapper around get_gene_neighbor_sets() specifically
    for C9orf72. It's provided for backward compatibility with existing code
    that used the deprecated function from the core library.

    Args:
        source: KnowledgeSource (e.g., INDRAKnowledgeSource)
        universe: Optional set of genes in dataset (for filtering)
        min_evidence: Minimum evidence count (default: 2)

    Returns:
        Dict with keys: "activated", "inhibited", "all"
        Each value is a QueryResult

    Example:
        >>> source = INDRAKnowledgeSource(env_file=".env")
        >>> neighbors = get_c9orf72_neighbor_sets(source, gene_universe)
        >>>
        >>> # Convert to feature sets for analysis
        >>> activated_set = neighbors["activated"].to_feature_set("C9_activated")
        >>> inhibited_set = neighbors["inhibited"].to_feature_set("C9_inhibited")

    See Also:
        get_gene_neighbor_sets: Generic version that works for any gene
    """
    return get_gene_neighbor_sets("C9orf72", source, universe, min_evidence)


def get_als_gene_neighbor_sets(
    source: KnowledgeSource,
    universe: Set[str] | None = None,
    min_evidence: int = 2,
) -> dict[str, dict[str, QueryResult]]:
    """
    Get neighbors for all major ALS-associated genes.

    Queries neighbors for C9orf72, SOD1, TARDBP, and FUS - the four most
    commonly mutated genes in familial ALS.

    Args:
        source: KnowledgeSource (e.g., INDRAKnowledgeSource)
        universe: Optional set of genes in dataset (for filtering)
        min_evidence: Minimum evidence count (default: 2)

    Returns:
        Dict mapping gene name → neighbor results
        Example: {
            "C9orf72": {"activated": QueryResult, "inhibited": QueryResult, "all": QueryResult},
            "SOD1": {"activated": QueryResult, "inhibited": QueryResult, "all": QueryResult},
            ...
        }

    Example:
        >>> source = INDRAKnowledgeSource(env_file=".env")
        >>> gene_universe = set(matrix.feature_ids)
        >>>
        >>> # Get neighbors for all ALS genes
        >>> als_neighbors = get_als_gene_neighbor_sets(source, gene_universe)
        >>>
        >>> # Create feature sets for each gene's activated targets
        >>> feature_sets = []
        >>> for gene, neighbors in als_neighbors.items():
        ...     fs = neighbors["activated"].to_feature_set(f"{gene}_activated")
        ...     feature_sets.append(fs)
        >>>
        >>> # Run competitive permutation test
        >>> results = engine.run_competitive_test(feature_sets, design, pool)

    Notes:
        - Queries all genes in parallel (independent queries)
        - Useful for comparative analysis across ALS genes
        - Each gene's results can be analyzed separately or combined
    """
    als_genes = ["C9orf72", "SOD1", "TARDBP", "FUS"]

    results = {}
    for gene in als_genes:
        results[gene] = get_gene_neighbor_sets(
            gene_name=gene,
            source=source,
            universe=universe,
            min_evidence=min_evidence,
        )

    return results


# =============================================================================
# Example: Custom Relationship Categories
# =============================================================================

def get_gene_neighbors_custom_categories(
    gene_name: str,
    source: KnowledgeSource,
    relationship_categories: dict[str, Set[str]],
    universe: Set[str] | None = None,
    min_evidence: int = 2,
) -> dict[str, QueryResult]:
    """
    Get gene neighbors with custom relationship categories.

    This demonstrates how to create custom groupings of relationship types
    beyond the default activation/inhibition split.

    Args:
        gene_name: Gene symbol
        source: KnowledgeSource
        relationship_categories: Dict mapping category name → relationship types
            Example: {
                "positive_regulation": {"IncreaseAmount", "Activation", "Phosphorylation"},
                "negative_regulation": {"DecreaseAmount", "Inhibition", "Dephosphorylation"},
                "binding": {"Complex", "BindsTo"},
            }
        universe: Optional gene universe
        min_evidence: Minimum evidence count

    Returns:
        Dict mapping category name → QueryResult
        Plus "all" key with all neighbors

    Example:
        >>> categories = {
        ...     "transcriptional": {"IncreaseAmount", "DecreaseAmount"},
        ...     "post_translational": {"Phosphorylation", "Ubiquitination"},
        ...     "complex_formation": {"Complex"},
        ... }
        >>> neighbors = get_gene_neighbors_custom_categories(
        ...     "TP53", source, categories, gene_universe
        ... )
        >>> transcriptional_targets = neighbors["transcriptional"]
        >>> ptm_targets = neighbors["post_translational"]
    """
    # Build query
    base_query = GraphQuery.neighbors(gene_name, direction="bidirectional")

    if min_evidence > 1:
        base_query = base_query.filter_evidence(min_evidence)

    if universe is not None:
        base_query = base_query.constrain_to(universe)

    # Execute and split by relationship
    result = base_query.execute(source)
    by_rel = result.by_relationship()

    # Aggregate into custom categories
    categorized: dict[str, QueryResult] = {}

    for category_name, rel_types in relationship_categories.items():
        category_entities: Set[str] = set()
        category_edges: list[KnowledgeEdge] = []

        for rel_name, rel_result in by_rel.items():
            if rel_name in rel_types:
                category_entities.update(rel_result.entities)
                category_edges.extend(rel_result.edges)

        categorized[category_name] = QueryResult(
            entities=category_entities,
            edges=category_edges,
            query_description=f"{gene_name} {category_name} neighbors",
        )

    categorized["all"] = result
    return categorized
