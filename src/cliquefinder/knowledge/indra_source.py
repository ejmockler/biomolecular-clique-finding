"""
INDRA CoGEx implementation of KnowledgeSource interface.
"""

from typing import List, Set, Optional
from cliquefinder.knowledge.base import (
    KnowledgeSource, KnowledgeEdge, KnowledgeModule, RelationshipType
)
from cliquefinder.knowledge.cogex import CoGExClient, INDRAModuleExtractor


class INDRAKnowledgeSource(KnowledgeSource):
    """
    INDRA CoGEx knowledge source for transcriptional regulation.

    Wraps CoGExClient to provide standardized KnowledgeSource interface.
    """

    def __init__(self, env_file: str = ".env"):
        self.client = CoGExClient(env_file=env_file)
        self.extractor = INDRAModuleExtractor(self.client)

    @property
    def name(self) -> str:
        return "INDRA-CoGEx"

    @property
    def supported_relationships(self) -> List[RelationshipType]:
        return [
            RelationshipType.INCREASES_EXPRESSION,
            RelationshipType.DECREASES_EXPRESSION,
            RelationshipType.REGULATES
        ]

    def get_edges(
        self,
        source_entity: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_evidence: int = 2,
        min_confidence: float = 0.0
    ) -> List[KnowledgeEdge]:
        # Resolve gene name to ID
        gene_id = self.extractor.resolve_gene_name(source_entity)
        if not gene_id:
            return []

        # Query all downstream targets from INDRA
        from cliquefinder.knowledge.cogex import ALL_REGULATORY_TYPES
        indra_edges = self.client.get_downstream_targets(
            regulator=gene_id,
            stmt_types=list(ALL_REGULATORY_TYPES),
            min_evidence=min_evidence
        )

        edges = []
        for indra_edge in indra_edges:
            rel_type = self._map_relationship(indra_edge.regulation_type)
            if relationship_types and rel_type not in relationship_types:
                continue

            edges.append(KnowledgeEdge(
                source=source_entity,
                target=indra_edge.target_name,
                relationship=rel_type,
                evidence_count=indra_edge.evidence_count,
                confidence=1.0,
                sources=list(indra_edge.source_counts_dict.keys()),
                metadata={'stmt_hash': indra_edge.stmt_hash}
            ))

        return edges

    def get_module(
        self,
        regulator: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_evidence: int = 2
    ) -> Optional[KnowledgeModule]:
        edges = self.get_edges(regulator, relationship_types, min_evidence)
        targets = {e.target for e in edges}

        if not targets:
            return None

        return KnowledgeModule(
            regulator=regulator,
            regulator_name=regulator,
            targets=targets,
            edges=edges,
            source_db=self.name
        )

    def discover_regulators(
        self,
        target_universe: Set[str],
        min_targets: int = 10,
        max_targets: Optional[int] = None,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_evidence: int = 2
    ) -> List[KnowledgeModule]:
        # Convert set to list for INDRA extractor
        gene_universe = list(target_universe)

        # Use INDRA's discover_modules method
        indra_modules = self.extractor.discover_modules(
            gene_universe=gene_universe,
            min_evidence=min_evidence,
            min_targets=min_targets,
            max_targets=max_targets
        )

        modules = []
        for indra_module in indra_modules:
            edges = []
            for indra_edge in indra_module.targets:  # targets is List[INDRAEdge]
                if indra_edge.target_name not in target_universe:
                    continue

                rel_type = self._map_relationship(indra_edge.regulation_type)
                if relationship_types and rel_type not in relationship_types:
                    continue

                edges.append(KnowledgeEdge(
                    source=indra_module.regulator_name,
                    target=indra_edge.target_name,
                    relationship=rel_type,
                    evidence_count=indra_edge.evidence_count,
                    confidence=1.0,
                    sources=list(indra_edge.source_counts_dict.keys()),
                    metadata={'stmt_hash': indra_edge.stmt_hash}
                ))

            targets = {e.target for e in edges}
            if len(targets) >= min_targets:
                modules.append(KnowledgeModule(
                    regulator=indra_module.regulator_name,
                    regulator_name=indra_module.regulator_name,
                    targets=targets,
                    edges=edges,
                    source_db=self.name
                ))

        modules.sort(key=lambda m: -m.n_targets)
        return modules

    def _map_relationship(self, indra_rel: str) -> RelationshipType:
        """Map INDRA regulation type to RelationshipType."""
        mapping = {
            'activation': RelationshipType.INCREASES_EXPRESSION,
            'repression': RelationshipType.DECREASES_EXPRESSION,
        }
        return mapping.get(indra_rel, RelationshipType.REGULATES)

    def close(self):
        """Close INDRA CoGEx connection."""
        self.client.close()
