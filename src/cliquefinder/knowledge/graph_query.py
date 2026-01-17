"""
Composable Graph Query Framework for Biological Knowledge Graphs.

Design Philosophy:
    This module provides a declarative, composable query interface for
    biological knowledge graphs that:

    1. Abstracts over specific databases (CoGEx, STRING, Reactome, etc.)
    2. Supports arbitrary graph traversal patterns (not just TF → targets)
    3. Composes naturally (chain operations, filter, project)
    4. Integrates with statistical analysis via FeatureSet projection

    The key insight: biological questions are graph queries:
    - "What does C9orf72 regulate?" → outgoing edges
    - "What regulates C9orf72?" → incoming edges
    - "What are C9orf72's 1-hop neighbors?" → bidirectional
    - "What pathways contain TP53?" → membership query
    - "What connects AKT1 to apoptosis?" → path query

Query Patterns:
    Neighbor queries (1-hop):
        - DOWNSTREAM: entity → targets (regulatory)
        - UPSTREAM: sources → entity (what regulates this?)
        - BIDIRECTIONAL: both directions (all neighbors)

    Path queries (n-hop):
        - SHORTEST_PATH: minimum hops between entities
        - ALL_PATHS: all paths up to max length

    Set operations:
        - INTERSECTION: entities in multiple result sets
        - UNION: entities in any result set
        - DIFFERENCE: entities in A but not B

Integration with Statistical Analysis:
    Query results project directly to FeatureSet for permutation testing:

    >>> query = (
    ...     GraphQuery.neighbors("C9orf72", direction="downstream")
    ...     .filter_relationship(["IncreaseAmount", "Activation"])
    ...     .filter_evidence(min_count=2)
    ... )
    >>> feature_set = query.execute(source).to_feature_set("C9orf72_activated")
    >>> # Now use in permutation test
    >>> results = engine.run_competitive_test([feature_set], design, pool)

Example - C9orf72 1-hop neighbor analysis:
    >>> from cliquefinder.knowledge import INDRAKnowledgeSource, GraphQuery
    >>>
    >>> source = INDRAKnowledgeSource(env_file=".env")
    >>>
    >>> # Get C9orf72 neighbors by relationship type
    >>> activated = (
    ...     GraphQuery.neighbors("C9orf72", direction="downstream")
    ...     .filter_relationship(["IncreaseAmount", "Activation"])
    ...     .execute(source)
    ... )
    >>>
    >>> inhibited = (
    ...     GraphQuery.neighbors("C9orf72", direction="downstream")
    ...     .filter_relationship(["DecreaseAmount", "Inhibition"])
    ...     .execute(source)
    ... )
    >>>
    >>> # Convert to feature sets for analysis
    >>> feature_sets = [
    ...     activated.to_feature_set("C9orf72_activated"),
    ...     inhibited.to_feature_set("C9orf72_inhibited"),
    ... ]

References:
    - Neo4j Cypher patterns
    - NetworkX graph traversal
    - Functional query composition (monadic patterns)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Generic,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
    runtime_checkable,
)
import logging

from .base import KnowledgeSource, KnowledgeEdge, RelationshipType

logger = logging.getLogger(__name__)


# =============================================================================
# Query Direction and Pattern Enums
# =============================================================================

class QueryDirection(Enum):
    """Direction of graph traversal."""

    DOWNSTREAM = auto()  # entity → targets (outgoing edges)
    UPSTREAM = auto()    # sources → entity (incoming edges)
    BIDIRECTIONAL = auto()  # both directions


class SetOperation(Enum):
    """Set operations for combining query results."""

    UNION = auto()       # A ∪ B
    INTERSECTION = auto()  # A ∩ B
    DIFFERENCE = auto()  # A - B


# =============================================================================
# Query Result Container
# =============================================================================

@dataclass
class QueryResult:
    """
    Result of a graph query.

    Contains both the entity set and the edges that produced it,
    enabling downstream analysis and provenance tracking.

    Attributes:
        entities: Set of entity IDs returned by query
        edges: List of edges traversed to reach those entities
        query_description: Human-readable description of query
        metadata: Additional query-specific information
    """

    entities: Set[str]
    edges: list[KnowledgeEdge] = field(default_factory=list)
    query_description: str = ""
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entities)

    def __iter__(self) -> Iterator[str]:
        return iter(self.entities)

    def __and__(self, other: "QueryResult") -> "QueryResult":
        """Set intersection: self & other."""
        return QueryResult(
            entities=self.entities & other.entities,
            edges=self.edges + other.edges,
            query_description=f"({self.query_description}) ∩ ({other.query_description})",
        )

    def __or__(self, other: "QueryResult") -> "QueryResult":
        """Set union: self | other."""
        return QueryResult(
            entities=self.entities | other.entities,
            edges=self.edges + other.edges,
            query_description=f"({self.query_description}) ∪ ({other.query_description})",
        )

    def __sub__(self, other: "QueryResult") -> "QueryResult":
        """Set difference: self - other."""
        return QueryResult(
            entities=self.entities - other.entities,
            edges=[e for e in self.edges if e.target not in other.entities],
            query_description=f"({self.query_description}) - ({other.query_description})",
        )

    def filter(self, predicate: Callable[[str], bool]) -> "QueryResult":
        """Filter entities by predicate."""
        filtered = {e for e in self.entities if predicate(e)}
        return QueryResult(
            entities=filtered,
            edges=[e for e in self.edges if e.target in filtered or e.source in filtered],
            query_description=f"filter({self.query_description})",
        )

    def constrain_to_universe(self, universe: Set[str]) -> "QueryResult":
        """Constrain to entities present in universe (e.g., your dataset)."""
        return QueryResult(
            entities=self.entities & universe,
            edges=[e for e in self.edges if e.target in universe],
            query_description=f"{self.query_description} ∩ universe",
            metadata={**self.metadata, 'universe_size': len(universe)},
        )

    def to_feature_set(self, set_id: str) -> "SimpleFeatureSet":
        """
        Project query result to FeatureSet for statistical analysis.

        This is the key integration point between knowledge graph queries
        and the permutation testing framework.

        Args:
            set_id: Identifier for the feature set

        Returns:
            SimpleFeatureSet compatible with PermutationTestEngine
        """
        from ..stats.permutation_framework import SimpleFeatureSet

        return SimpleFeatureSet(
            _id=set_id,
            _feature_ids=list(self.entities),
            _metadata={
                'query': self.query_description,
                'n_edges': len(self.edges),
                **self.metadata,
            },
        )

    def by_relationship(self) -> dict[str, "QueryResult"]:
        """
        Split result by relationship type.

        Useful for analyzing different relationship types separately:
        - C9orf72 activated targets vs inhibited targets
        - Phosphorylation targets vs binding partners

        Returns:
            Dict mapping relationship type → QueryResult
        """
        by_rel: dict[str, list[KnowledgeEdge]] = {}

        for edge in self.edges:
            rel_name = edge.relationship.value if hasattr(edge.relationship, 'value') else str(edge.relationship)
            if rel_name not in by_rel:
                by_rel[rel_name] = []
            by_rel[rel_name].append(edge)

        results = {}
        for rel_name, edges in by_rel.items():
            entities = {e.target for e in edges} | {e.source for e in edges}
            results[rel_name] = QueryResult(
                entities=entities,
                edges=edges,
                query_description=f"{self.query_description}[{rel_name}]",
            )

        return results


# =============================================================================
# Composable Query Builder
# =============================================================================

@dataclass
class GraphQuery:
    """
    Composable query builder for biological knowledge graphs.

    Provides a fluent interface for constructing graph queries that can be
    executed against any KnowledgeSource implementation.

    Design Pattern: Builder + Command
        - Builder: Fluent interface for constructing queries
        - Command: Deferred execution (query built, then executed)

    Examples:
        # Simple neighbor query
        >>> query = GraphQuery.neighbors("TP53", direction="downstream")
        >>> result = query.execute(source)

        # Filtered query with chaining
        >>> query = (
        ...     GraphQuery.neighbors("C9orf72", direction="bidirectional")
        ...     .filter_relationship(["IncreaseAmount", "Activation"])
        ...     .filter_evidence(min_count=3)
        ...     .constrain_to(gene_universe)
        ... )
        >>> result = query.execute(source)

        # Combined queries
        >>> activated = GraphQuery.neighbors("C9orf72").filter_relationship(["Activation"])
        >>> inhibited = GraphQuery.neighbors("C9orf72").filter_relationship(["Inhibition"])
        >>> combined = activated.union(inhibited)
        >>> result = combined.execute(source)
    """

    # Core query parameters
    seed_entity: str
    direction: QueryDirection = QueryDirection.DOWNSTREAM
    max_hops: int = 1

    # Filters (applied after query)
    relationship_filter: list[str] | None = None
    min_evidence: int = 1
    min_confidence: float = 0.0
    universe_constraint: Set[str] | None = None

    # Combined queries (for set operations)
    combined_with: list[tuple["GraphQuery", SetOperation]] = field(default_factory=list)

    # Query metadata
    description: str = ""

    # ==========================================================================
    # Factory Methods (entry points)
    # ==========================================================================

    @classmethod
    def neighbors(
        cls,
        entity: str,
        direction: Literal["downstream", "upstream", "bidirectional"] = "downstream",
    ) -> "GraphQuery":
        """
        Query neighbors of an entity.

        Args:
            entity: Seed entity (gene symbol, protein ID, etc.)
            direction: Which direction to traverse
                - "downstream": entity → targets (what does it regulate?)
                - "upstream": sources → entity (what regulates it?)
                - "bidirectional": both directions (all neighbors)

        Returns:
            GraphQuery builder for further customization

        Examples:
            >>> # What does TP53 regulate?
            >>> query = GraphQuery.neighbors("TP53", direction="downstream")
            >>>
            >>> # What regulates TP53?
            >>> query = GraphQuery.neighbors("TP53", direction="upstream")
            >>>
            >>> # All TP53 interaction partners
            >>> query = GraphQuery.neighbors("TP53", direction="bidirectional")
        """
        direction_map = {
            "downstream": QueryDirection.DOWNSTREAM,
            "upstream": QueryDirection.UPSTREAM,
            "bidirectional": QueryDirection.BIDIRECTIONAL,
        }

        return cls(
            seed_entity=entity,
            direction=direction_map[direction],
            max_hops=1,
            description=f"neighbors({entity}, {direction})",
        )

    @classmethod
    def regulators_of(cls, target: str) -> "GraphQuery":
        """
        Query upstream regulators of a target.

        Convenience method equivalent to neighbors(target, "upstream").

        Args:
            target: Target entity

        Returns:
            GraphQuery for regulators
        """
        return cls.neighbors(target, direction="upstream")

    @classmethod
    def targets_of(cls, regulator: str) -> "GraphQuery":
        """
        Query downstream targets of a regulator.

        Convenience method equivalent to neighbors(regulator, "downstream").

        Args:
            regulator: Regulator entity

        Returns:
            GraphQuery for targets
        """
        return cls.neighbors(regulator, direction="downstream")

    # ==========================================================================
    # Filter Methods (chainable)
    # ==========================================================================

    def filter_relationship(
        self,
        relationship_types: list[str] | list[RelationshipType],
    ) -> "GraphQuery":
        """
        Filter to specific relationship types.

        Args:
            relationship_types: List of relationship types to include
                Can be strings ("IncreaseAmount") or RelationshipType enums

        Returns:
            New GraphQuery with filter applied

        Examples:
            >>> # Only activation relationships
            >>> query = (
            ...     GraphQuery.neighbors("C9orf72")
            ...     .filter_relationship(["IncreaseAmount", "Activation"])
            ... )
            >>>
            >>> # Only inhibition
            >>> query = (
            ...     GraphQuery.neighbors("C9orf72")
            ...     .filter_relationship([RelationshipType.INHIBITS])
            ... )
        """
        # Normalize to strings
        normalized = []
        for rt in relationship_types:
            if hasattr(rt, 'value'):
                normalized.append(rt.value)
            else:
                normalized.append(str(rt))

        return GraphQuery(
            seed_entity=self.seed_entity,
            direction=self.direction,
            max_hops=self.max_hops,
            relationship_filter=normalized,
            min_evidence=self.min_evidence,
            min_confidence=self.min_confidence,
            universe_constraint=self.universe_constraint,
            combined_with=self.combined_with.copy(),
            description=f"{self.description}.filter_rel({normalized})",
        )

    def filter_evidence(self, min_count: int) -> "GraphQuery":
        """
        Filter to edges with minimum evidence count.

        Args:
            min_count: Minimum number of supporting evidence pieces
                - 2: High-confidence (default recommendation)
                - 5: Very high-confidence
                - 10: Extremely well-established

        Returns:
            New GraphQuery with filter applied
        """
        return GraphQuery(
            seed_entity=self.seed_entity,
            direction=self.direction,
            max_hops=self.max_hops,
            relationship_filter=self.relationship_filter,
            min_evidence=min_count,
            min_confidence=self.min_confidence,
            universe_constraint=self.universe_constraint,
            combined_with=self.combined_with.copy(),
            description=f"{self.description}.min_evidence({min_count})",
        )

    def filter_confidence(self, min_score: float) -> "GraphQuery":
        """
        Filter to edges with minimum confidence score.

        Args:
            min_score: Minimum confidence (0.0 - 1.0)

        Returns:
            New GraphQuery with filter applied
        """
        return GraphQuery(
            seed_entity=self.seed_entity,
            direction=self.direction,
            max_hops=self.max_hops,
            relationship_filter=self.relationship_filter,
            min_evidence=self.min_evidence,
            min_confidence=min_score,
            universe_constraint=self.universe_constraint,
            combined_with=self.combined_with.copy(),
            description=f"{self.description}.min_conf({min_score})",
        )

    def constrain_to(self, universe: Set[str]) -> "GraphQuery":
        """
        Constrain results to entities in a universe (e.g., your dataset).

        Args:
            universe: Set of valid entity IDs

        Returns:
            New GraphQuery with constraint applied

        Examples:
            >>> # Only neighbors present in our proteomics data
            >>> gene_universe = set(matrix.feature_ids)
            >>> query = (
            ...     GraphQuery.neighbors("C9orf72")
            ...     .constrain_to(gene_universe)
            ... )
        """
        return GraphQuery(
            seed_entity=self.seed_entity,
            direction=self.direction,
            max_hops=self.max_hops,
            relationship_filter=self.relationship_filter,
            min_evidence=self.min_evidence,
            min_confidence=self.min_confidence,
            universe_constraint=universe,
            combined_with=self.combined_with.copy(),
            description=f"{self.description}.constrain(universe[{len(universe)}])",
        )

    # ==========================================================================
    # Set Operation Methods (combinable)
    # ==========================================================================

    def union(self, other: "GraphQuery") -> "GraphQuery":
        """
        Combine with another query via set union.

        Args:
            other: Another GraphQuery

        Returns:
            Combined query (results = self ∪ other)
        """
        new_combined = self.combined_with.copy()
        new_combined.append((other, SetOperation.UNION))

        return GraphQuery(
            seed_entity=self.seed_entity,
            direction=self.direction,
            max_hops=self.max_hops,
            relationship_filter=self.relationship_filter,
            min_evidence=self.min_evidence,
            min_confidence=self.min_confidence,
            universe_constraint=self.universe_constraint,
            combined_with=new_combined,
            description=f"({self.description}) ∪ ({other.description})",
        )

    def intersection(self, other: "GraphQuery") -> "GraphQuery":
        """
        Combine with another query via set intersection.

        Args:
            other: Another GraphQuery

        Returns:
            Combined query (results = self ∩ other)
        """
        new_combined = self.combined_with.copy()
        new_combined.append((other, SetOperation.INTERSECTION))

        return GraphQuery(
            seed_entity=self.seed_entity,
            direction=self.direction,
            max_hops=self.max_hops,
            relationship_filter=self.relationship_filter,
            min_evidence=self.min_evidence,
            min_confidence=self.min_confidence,
            universe_constraint=self.universe_constraint,
            combined_with=new_combined,
            description=f"({self.description}) ∩ ({other.description})",
        )

    def difference(self, other: "GraphQuery") -> "GraphQuery":
        """
        Combine with another query via set difference.

        Args:
            other: Another GraphQuery

        Returns:
            Combined query (results = self - other)
        """
        new_combined = self.combined_with.copy()
        new_combined.append((other, SetOperation.DIFFERENCE))

        return GraphQuery(
            seed_entity=self.seed_entity,
            direction=self.direction,
            max_hops=self.max_hops,
            relationship_filter=self.relationship_filter,
            min_evidence=self.min_evidence,
            min_confidence=self.min_confidence,
            universe_constraint=self.universe_constraint,
            combined_with=new_combined,
            description=f"({self.description}) - ({other.description})",
        )

    # ==========================================================================
    # Execution
    # ==========================================================================

    def execute(self, source: KnowledgeSource) -> QueryResult:
        """
        Execute query against a knowledge source.

        Args:
            source: KnowledgeSource implementation (INDRA, STRING, etc.)

        Returns:
            QueryResult with entities and edges

        Examples:
            >>> source = INDRAKnowledgeSource(env_file=".env")
            >>> query = GraphQuery.neighbors("C9orf72", direction="bidirectional")
            >>> result = query.execute(source)
            >>> print(f"Found {len(result)} neighbors")
        """
        # Execute base query
        result = self._execute_single(source)

        # Apply combined queries
        for other_query, operation in self.combined_with:
            other_result = other_query._execute_single(source)

            if operation == SetOperation.UNION:
                result = result | other_result
            elif operation == SetOperation.INTERSECTION:
                result = result & other_result
            elif operation == SetOperation.DIFFERENCE:
                result = result - other_result

        # Apply universe constraint if specified
        if self.universe_constraint is not None:
            result = result.constrain_to_universe(self.universe_constraint)

        return result

    def _execute_single(self, source: KnowledgeSource) -> QueryResult:
        """Execute a single (non-combined) query."""
        all_edges: list[KnowledgeEdge] = []
        all_entities: Set[str] = set()

        # Map relationship filter to RelationshipType enums if needed
        rel_types = None
        if self.relationship_filter:
            rel_types = []
            for rt_str in self.relationship_filter:
                # Try to find matching RelationshipType
                for rt in RelationshipType:
                    if rt.value == rt_str or rt.name == rt_str:
                        rel_types.append(rt)
                        break

        # Query based on direction
        if self.direction in (QueryDirection.DOWNSTREAM, QueryDirection.BIDIRECTIONAL):
            # Outgoing edges: seed → targets
            edges = source.get_edges(
                source_entity=self.seed_entity,
                relationship_types=rel_types,
                min_evidence=self.min_evidence,
                min_confidence=self.min_confidence,
            )
            all_edges.extend(edges)
            all_entities.update(e.target for e in edges)

        if self.direction in (QueryDirection.UPSTREAM, QueryDirection.BIDIRECTIONAL):
            # Incoming edges: sources → seed
            # This requires reverse query support
            # For now, we use discover_regulators as a workaround
            try:
                # Get all regulators that target our seed entity
                modules = source.discover_regulators(
                    target_universe={self.seed_entity},
                    min_targets=1,
                    relationship_types=rel_types,
                    min_evidence=self.min_evidence,
                )

                for module in modules:
                    # Filter edges to only those targeting our seed
                    for edge in module.edges:
                        if edge.target == self.seed_entity:
                            all_edges.append(edge)
                            all_entities.add(edge.source)

            except Exception as e:
                logger.warning(f"Upstream query failed: {e}")

        return QueryResult(
            entities=all_entities,
            edges=all_edges,
            query_description=self.description,
            metadata={
                'seed_entity': self.seed_entity,
                'direction': self.direction.name,
            },
        )


# =============================================================================
# Convenience Functions for Common Queries
# =============================================================================

def get_c9orf72_neighbor_sets(
    source: KnowledgeSource,
    universe: Set[str] | None = None,
    min_evidence: int = 2,
) -> dict[str, QueryResult]:
    """
    Get C9orf72 neighbors split by relationship type.

    This implements the collaborator's request:
    "Grab all 1hop neighbors of c9orf72 that are inc/dec/act/inh"

    Args:
        source: KnowledgeSource (e.g., INDRAKnowledgeSource)
        universe: Optional set of genes in dataset (for filtering)
        min_evidence: Minimum evidence count

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
    """
    # Build queries
    base_query = GraphQuery.neighbors("C9orf72", direction="bidirectional")

    if min_evidence > 1:
        base_query = base_query.filter_evidence(min_evidence)

    if universe is not None:
        base_query = base_query.constrain_to(universe)

    # Execute and split by relationship
    result = base_query.execute(source)
    by_rel = result.by_relationship()

    # Aggregate into activation/inhibition categories
    activation_rels = {"IncreaseAmount", "Activation"}
    inhibition_rels = {"DecreaseAmount", "Inhibition"}

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
            query_description="C9orf72 activated neighbors",
        ),
        "inhibited": QueryResult(
            entities=inhibited_entities,
            edges=inhibited_edges,
            query_description="C9orf72 inhibited neighbors",
        ),
        "all": result,
    }


def query_to_feature_sets(
    entity: str,
    source: KnowledgeSource,
    universe: Set[str] | None = None,
    min_evidence: int = 2,
    split_by_relationship: bool = True,
) -> list:
    """
    Query neighbors and convert to FeatureSet objects for statistical analysis.

    This is the main integration point between knowledge graph queries
    and the permutation testing framework.

    Args:
        entity: Seed entity (e.g., "C9orf72", "TP53")
        source: KnowledgeSource
        universe: Optional gene universe for filtering
        min_evidence: Minimum evidence count
        split_by_relationship: If True, return separate sets per relationship type

    Returns:
        List of SimpleFeatureSet objects ready for permutation testing

    Example:
        >>> source = INDRAKnowledgeSource(env_file=".env")
        >>> feature_sets = query_to_feature_sets(
        ...     "C9orf72", source, gene_universe,
        ...     split_by_relationship=True
        ... )
        >>>
        >>> # Run permutation test
        >>> results = engine.run_competitive_test(feature_sets, design, pool)
    """
    from ..stats.permutation_framework import SimpleFeatureSet

    query = (
        GraphQuery.neighbors(entity, direction="bidirectional")
        .filter_evidence(min_evidence)
    )

    if universe is not None:
        query = query.constrain_to(universe)

    result = query.execute(source)

    if not split_by_relationship:
        return [result.to_feature_set(f"{entity}_neighbors")]

    # Split by relationship type
    by_rel = result.by_relationship()

    feature_sets = []
    for rel_name, rel_result in by_rel.items():
        if len(rel_result) > 0:
            fs = rel_result.to_feature_set(f"{entity}_{rel_name}")
            feature_sets.append(fs)

    return feature_sets
