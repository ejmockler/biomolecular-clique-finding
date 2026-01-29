"""
Knowledge-based validation and analysis modules.

This package provides tools for validating biological hypotheses using prior knowledge
from databases, literature, and curated gene sets. Includes correlation-based clique
finding for TF regulatory module validation and INDRA CoGEx knowledge graph queries.
"""

from cliquefinder.knowledge.clique_validator import (
    CorrelationClique,
    ChildSetType2,
    CliqueValidator,
    InsufficientSamplesError,
    GeneNotFoundError,
)

from cliquefinder.knowledge.clique_algorithms import (
    kcore_reduction,
    compute_degeneracy_ordering,
    estimate_clique_complexity,
)

from cliquefinder.knowledge.cogex import (
    GeneId,
    INDRAEdge,
    INDRAModule,
    CoGExClient,
    INDRAModuleExtractor,
)

from cliquefinder.knowledge.stability import (
    StableClique,
    bootstrap_clique_stability,
    filter_stable_cliques,
    summarize_stability,
)

from cliquefinder.knowledge.base import (
    RelationshipType,
    KnowledgeEdge,
    KnowledgeModule,
    KnowledgeSource,
    CompositeKnowledgeSource,
)

from cliquefinder.knowledge.indra_source import (
    INDRAKnowledgeSource,
)

from cliquefinder.knowledge.module_discovery import (
    GeneUniverseSelector,
    RegulatoryTargetUniverse,
    VarianceFilteredUniverse,
    ConnectedComponentUniverse,
    CoexpressionModule,
    ModuleDiscovery,
)

from cliquefinder.knowledge.cross_modal_mapper import (
    CrossModalMapping,
    CrossModalIDMapper,
)

from cliquefinder.knowledge.graph_query import (
    QueryDirection,
    QueryResult,
    GraphQuery,
    get_c9orf72_neighbor_sets,  # Deprecated: use examples.als.graph_queries
    query_to_feature_sets,
)

__all__ = [
    # Clique validation
    'CorrelationClique',
    'ChildSetType2',
    'CliqueValidator',
    'InsufficientSamplesError',
    'GeneNotFoundError',
    # Clique algorithms
    'kcore_reduction',
    'compute_degeneracy_ordering',
    'estimate_clique_complexity',
    # INDRA CoGEx
    'GeneId',
    'INDRAEdge',
    'INDRAModule',
    'CoGExClient',
    'INDRAModuleExtractor',
    # Bootstrap stability
    'StableClique',
    'bootstrap_clique_stability',
    'filter_stable_cliques',
    'summarize_stability',
    # Knowledge source plugin architecture
    'RelationshipType',
    'KnowledgeEdge',
    'KnowledgeModule',
    'KnowledgeSource',
    'CompositeKnowledgeSource',
    'INDRAKnowledgeSource',
    # Unified module discovery (both paradigms)
    'GeneUniverseSelector',
    'RegulatoryTargetUniverse',
    'VarianceFilteredUniverse',
    'ConnectedComponentUniverse',
    'CoexpressionModule',
    'ModuleDiscovery',
    # Cross-modal ID mapping
    'CrossModalMapping',
    'CrossModalIDMapper',
    # Composable graph queries
    'QueryDirection',
    'QueryResult',
    'GraphQuery',
    'get_c9orf72_neighbor_sets',
    'query_to_feature_sets',
]
