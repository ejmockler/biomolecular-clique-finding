"""
Abstract interface for biological knowledge sources.

Enables plugin architecture for different databases:
- INDRA CoGEx (transcriptional regulation)
- STRING (protein-protein interactions)
- PhosphoSitePlus (kinase-substrate)
- KEGG (metabolic reactions)
- Reactome (pathways)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Any
from enum import Enum


class RelationshipType(Enum):
    """Types of biological relationships."""
    # Transcriptional
    INCREASES_EXPRESSION = "IncreaseAmount"
    DECREASES_EXPRESSION = "DecreaseAmount"
    REGULATES = "Regulates"

    # Protein-protein
    BINDS = "Binds"
    PHOSPHORYLATES = "Phosphorylation"
    UBIQUITINATES = "Ubiquitination"
    ACETYLATES = "Acetylation"

    # Metabolic
    CATALYZES = "Catalysis"
    INHIBITS = "Inhibition"
    PRODUCES = "Produces"
    CONSUMES = "Consumes"

    # Generic
    ASSOCIATED = "Association"


@dataclass
class KnowledgeEdge:
    """A relationship between biological entities."""
    source: str  # Source entity ID (gene, protein, metabolite)
    target: str  # Target entity ID
    relationship: RelationshipType
    evidence_count: int = 1
    confidence: float = 1.0
    sources: List[str] = field(default_factory=list)  # Database sources
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.source, self.target, self.relationship))


@dataclass
class KnowledgeModule:
    """A regulatory/interaction module from a knowledge source."""
    regulator: str  # Central regulator/hub
    regulator_name: str  # Human-readable name
    targets: Set[str]  # Target entities
    edges: List[KnowledgeEdge]  # All edges from regulator to targets
    source_db: str  # Source database name

    @property
    def n_targets(self) -> int:
        return len(self.targets)


class KnowledgeSource(ABC):
    """
    Abstract interface for biological knowledge databases.

    Implementations:
    - INDRAKnowledgeSource: Transcriptional regulation from INDRA CoGEx
    - STRINGKnowledgeSource: Protein-protein interactions
    - PhosphoSitePlusSource: Kinase-substrate relationships
    - KEGGKnowledgeSource: Metabolic reactions

    Usage:
        source = INDRAKnowledgeSource(env_file=".env")
        modules = source.get_modules_for_regulator("TP53", min_evidence=2)

        # Or discover regulators
        regulators = source.discover_regulators(gene_universe, min_targets=20)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this knowledge source."""
        pass

    @property
    @abstractmethod
    def supported_relationships(self) -> List[RelationshipType]:
        """Relationship types this source provides."""
        pass

    @abstractmethod
    def get_edges(
        self,
        source_entity: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_evidence: int = 2,
        min_confidence: float = 0.0
    ) -> List[KnowledgeEdge]:
        """
        Get all edges from a source entity.

        Args:
            source_entity: Source ID (gene symbol, UniProt, HMDB, etc.)
            relationship_types: Filter to specific relationships (None = all)
            min_evidence: Minimum evidence count (default 2 for high-confidence)
            min_confidence: Minimum confidence score

        Returns:
            List of KnowledgeEdge objects
        """
        pass

    @abstractmethod
    def get_module(
        self,
        regulator: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_evidence: int = 2
    ) -> Optional[KnowledgeModule]:
        """
        Get a regulatory module centered on a regulator.

        Args:
            regulator: Regulator ID
            relationship_types: Filter relationships
            min_evidence: Minimum evidence (default 2 for high-confidence)

        Returns:
            KnowledgeModule or None if no targets found
        """
        pass

    @abstractmethod
    def discover_regulators(
        self,
        target_universe: Set[str],
        min_targets: int = 10,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_evidence: int = 2
    ) -> List[KnowledgeModule]:
        """
        Discover regulators that target genes in the universe.

        Args:
            target_universe: Set of potential target IDs (e.g., genes in dataset)
            min_targets: Minimum targets in universe for a regulator
            relationship_types: Filter relationships
            min_evidence: Minimum evidence (default 2 for high-confidence)

        Returns:
            List of KnowledgeModule objects, sorted by n_targets descending
        """
        pass

    def filter_to_universe(
        self,
        module: KnowledgeModule,
        universe: Set[str]
    ) -> KnowledgeModule:
        """Filter module targets to those in the given universe."""
        filtered_targets = module.targets & universe
        filtered_edges = [e for e in module.edges if e.target in universe]

        return KnowledgeModule(
            regulator=module.regulator,
            regulator_name=module.regulator_name,
            targets=filtered_targets,
            edges=filtered_edges,
            source_db=module.source_db
        )


class CompositeKnowledgeSource(KnowledgeSource):
    """
    Combines multiple knowledge sources.

    Usage:
        composite = CompositeKnowledgeSource([
            INDRAKnowledgeSource(),
            STRINGKnowledgeSource()
        ])
        # Queries all sources, merges results
    """

    def __init__(self, sources: List[KnowledgeSource]):
        self.sources = sources

    @property
    def name(self) -> str:
        return f"Composite({','.join(s.name for s in self.sources)})"

    @property
    def supported_relationships(self) -> List[RelationshipType]:
        all_rels = set()
        for source in self.sources:
            all_rels.update(source.supported_relationships)
        return list(all_rels)

    def get_edges(
        self,
        source_entity: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_evidence: int = 2,
        min_confidence: float = 0.0
    ) -> List[KnowledgeEdge]:
        all_edges = []
        for source in self.sources:
            try:
                edges = source.get_edges(
                    source_entity, relationship_types, min_evidence, min_confidence
                )
                all_edges.extend(edges)
            except Exception as e:
                import logging
                logging.warning(f"Error querying {source.name}: {e}")
        return all_edges

    def get_module(
        self,
        regulator: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_evidence: int = 2
    ) -> Optional[KnowledgeModule]:
        all_targets = set()
        all_edges = []

        for source in self.sources:
            try:
                module = source.get_module(regulator, relationship_types, min_evidence)
                if module:
                    all_targets.update(module.targets)
                    all_edges.extend(module.edges)
            except Exception:
                pass

        if not all_targets:
            return None

        return KnowledgeModule(
            regulator=regulator,
            regulator_name=regulator,
            targets=all_targets,
            edges=all_edges,
            source_db=self.name
        )

    def discover_regulators(
        self,
        target_universe: Set[str],
        min_targets: int = 10,
        relationship_types: Optional[List[RelationshipType]] = None,
        min_evidence: int = 2
    ) -> List[KnowledgeModule]:
        regulator_modules: Dict[str, KnowledgeModule] = {}

        for source in self.sources:
            try:
                modules = source.discover_regulators(
                    target_universe, min_targets, relationship_types, min_evidence
                )
                for module in modules:
                    if module.regulator in regulator_modules:
                        # Merge
                        existing = regulator_modules[module.regulator]
                        existing.targets.update(module.targets)
                        existing.edges.extend(module.edges)
                    else:
                        regulator_modules[module.regulator] = module
            except Exception:
                pass

        # Filter by min_targets after merging
        result = [m for m in regulator_modules.values() if m.n_targets >= min_targets]
        result.sort(key=lambda m: -m.n_targets)
        return result
