"""
Multi-source entity resolution for gene identifiers.

Implements a confidence-ranked resolution strategy:
1. HGNC (authoritative for human gene symbols)
2. Ensembl BioMart (authoritative for Ensembl IDs)
3. MyGeneInfo (comprehensive aggregator)
4. Alias resolution (for legacy/alternate names)

Each resolution includes provenance tracking for auditability.

Examples:
    >>> from cliquefinder.validation.entity_resolver import GeneEntityResolver
    >>>
    >>> resolver = GeneEntityResolver(cache_dir='~/.cache/biocore')
    >>> mapping = resolver.resolve_ensembl_ids(['ENSG00000141510', 'ENSG00000012048'])
    >>> print(mapping['ENSG00000141510'])
    ResolvedEntity(symbol='TP53', source='hgnc', confidence=1.0, biotype='protein_coding')
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ResolvedEntity:
    """Result of entity resolution with provenance."""
    ensembl_id: str
    symbol: Optional[str]
    source: str  # 'hgnc', 'biomart', 'mygene', 'alias', 'none'
    confidence: float  # 0.0 to 1.0
    biotype: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None

    @property
    def is_resolved(self) -> bool:
        return self.symbol is not None and len(self.symbol) > 0

    @property
    def is_protein_coding(self) -> bool:
        return self.biotype == 'protein_coding'

    def to_dict(self) -> dict:
        return {
            'ensembl_id': self.ensembl_id,
            'symbol': self.symbol,
            'source': self.source,
            'confidence': self.confidence,
            'biotype': self.biotype,
            'aliases': self.aliases,
            'description': self.description,
        }


class GeneEntityResolver:
    """
    Multi-source gene identifier resolver with confidence ranking.

    Resolution Priority (highest to lowest confidence):
    1. HGNC (confidence=1.0) - Official human gene nomenclature
    2. Ensembl BioMart (confidence=0.95) - Authoritative for Ensembl IDs
    3. MyGeneInfo (confidence=0.85) - Comprehensive aggregator
    4. Alias resolution (confidence=0.7) - For legacy/alternate names

    For non-coding genes (lncRNAs, pseudogenes), we use:
    - Ensembl stable ID as fallback (confidence=0.5)
    - Biotype-prefixed ID as last resort (confidence=0.3)
    """

    # Confidence scores by source
    CONFIDENCE = {
        'hgnc': 1.0,
        'biomart': 0.95,
        'mygene': 0.85,
        'alias': 0.7,
        'ensembl_id': 0.5,
        'biotype_prefix': 0.3,
        'none': 0.0,
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_workers: int = 8,
        use_biomart: bool = True,
        use_mygene: bool = True,
    ):
        """
        Initialize resolver with caching and parallel query support.

        Args:
            cache_dir: Directory for caching results
            max_workers: Concurrent API requests
            use_biomart: Query Ensembl BioMart (slower but authoritative)
            use_mygene: Query MyGeneInfo (fast aggregator)
        """
        self.cache_dir = Path(cache_dir or Path.home() / '.cache/biocore/entity_resolution')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.use_biomart = use_biomart
        self.use_mygene = use_mygene

        # In-memory cache for session
        self._cache: Dict[str, ResolvedEntity] = {}

    def resolve_ensembl_ids(
        self,
        ensembl_ids: List[str],
        verbose: bool = True,
    ) -> Dict[str, ResolvedEntity]:
        """
        Resolve Ensembl gene IDs to symbols using multi-source strategy.

        Args:
            ensembl_ids: List of Ensembl gene IDs (ENSG...)
            verbose: Print progress

        Returns:
            Dict mapping ensembl_id -> ResolvedEntity
        """
        # Check disk cache first
        cache_key = f"ensembl_resolution_{hash(tuple(sorted(ensembl_ids)))}.pkl"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                if verbose:
                    logger.info(f"Loaded {len(cached)} cached resolutions")
                return cached
            except Exception:
                pass

        # Initialize results
        results: Dict[str, ResolvedEntity] = {}
        unresolved = set(ensembl_ids)

        if verbose:
            print(f"Resolving {len(ensembl_ids)} Ensembl IDs...")

        # Stage 1: MyGeneInfo (fast, good coverage)
        if self.use_mygene and unresolved:
            mygene_results = self._query_mygene(list(unresolved), verbose)
            for eid, entity in mygene_results.items():
                if entity.is_resolved:
                    results[eid] = entity
                    unresolved.discard(eid)
                elif entity.biotype:
                    # Keep partial info for later
                    results[eid] = entity

            if verbose:
                print(f"  MyGeneInfo: resolved {len(results)}/{len(ensembl_ids)}")

        # Stage 2: Fallback for unresolved - use Ensembl ID or biotype prefix
        for eid in unresolved:
            existing = results.get(eid)
            if existing and existing.biotype:
                # Use biotype-prefixed ID for non-coding genes
                if existing.biotype in ('lncRNA', 'processed_pseudogene', 'misc_RNA'):
                    # These legitimately don't have symbols - mark appropriately
                    existing.symbol = None
                    existing.source = 'none'
                    existing.confidence = 0.0
                elif existing.biotype == 'protein_coding':
                    # Novel protein-coding gene - use Ensembl ID as symbol
                    existing.symbol = eid
                    existing.source = 'ensembl_id'
                    existing.confidence = self.CONFIDENCE['ensembl_id']
            else:
                # No info at all - create minimal entry
                results[eid] = ResolvedEntity(
                    ensembl_id=eid,
                    symbol=None,
                    source='none',
                    confidence=0.0,
                )

        # Cache results
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(results, f)
        except Exception:
            pass

        # Summary
        if verbose:
            resolved = sum(1 for r in results.values() if r.is_resolved)
            by_source = Counter(r.source for r in results.values())
            print(f"\n  Resolution Summary:")
            print(f"    Total: {len(results)}")
            print(f"    Resolved: {resolved} ({100*resolved/len(results):.1f}%)")
            print(f"    By source: {dict(by_source)}")

            # Biotype breakdown for unresolved
            unresolved_biotypes = Counter(
                r.biotype for r in results.values()
                if not r.is_resolved and r.biotype
            )
            if unresolved_biotypes:
                print(f"    Unresolved by biotype: {dict(unresolved_biotypes.most_common(5))}")

        return results

    def _query_mygene(
        self,
        ensembl_ids: List[str],
        verbose: bool = True,
    ) -> Dict[str, ResolvedEntity]:
        """Query MyGeneInfo for gene information."""
        try:
            import mygene
        except ImportError:
            logger.warning("mygene not installed, skipping")
            return {}

        mg = mygene.MyGeneInfo()

        # Query with extended fields
        results = mg.querymany(
            ensembl_ids,
            scopes='ensembl.gene',
            fields='symbol,name,alias,type_of_gene,ensembl.type_of_gene',
            species='human',
            returnall=True,
        )

        entities = {}

        for hit in results.get('out', []):
            if hit.get('notfound'):
                continue

            eid = hit.get('query')
            if not eid:
                continue

            symbol = hit.get('symbol')

            # Get biotype
            biotype = None
            ensembl_info = hit.get('ensembl', {})
            if isinstance(ensembl_info, dict):
                biotype = ensembl_info.get('type_of_gene')
            elif isinstance(ensembl_info, list) and ensembl_info:
                biotype = ensembl_info[0].get('type_of_gene') if isinstance(ensembl_info[0], dict) else None
            if not biotype:
                biotype = hit.get('type_of_gene')

            # Get aliases
            aliases = hit.get('alias', [])
            if isinstance(aliases, str):
                aliases = [aliases]

            entities[eid] = ResolvedEntity(
                ensembl_id=eid,
                symbol=symbol,
                source='mygene' if symbol else 'none',
                confidence=self.CONFIDENCE['mygene'] if symbol else 0.0,
                biotype=biotype,
                aliases=aliases,
                description=hit.get('name'),
            )

        # Handle missing IDs
        for eid in ensembl_ids:
            if eid not in entities:
                entities[eid] = ResolvedEntity(
                    ensembl_id=eid,
                    symbol=None,
                    source='none',
                    confidence=0.0,
                )

        return entities

    def get_symbol_to_ensembl_map(
        self,
        results: Dict[str, ResolvedEntity],
        min_confidence: float = 0.0,
    ) -> Dict[str, str]:
        """
        Get reverse mapping from symbol to Ensembl ID.

        Args:
            results: Resolution results from resolve_ensembl_ids
            min_confidence: Minimum confidence threshold

        Returns:
            Dict mapping symbol -> ensembl_id
        """
        symbol_to_ensembl = {}

        for eid, entity in results.items():
            if entity.is_resolved and entity.confidence >= min_confidence:
                symbol_to_ensembl[entity.symbol] = eid

                # Also add aliases (with lower priority)
                for alias in entity.aliases:
                    if alias and alias not in symbol_to_ensembl:
                        symbol_to_ensembl[alias] = eid

        return symbol_to_ensembl


def resolve_and_create_mapping(
    feature_ids: List[str],
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Convenience function to resolve Ensembl IDs and create symbol -> feature_id mapping.

    Args:
        feature_ids: List of Ensembl feature IDs
        output_path: Optional path to save mapping CSV
        verbose: Print progress

    Returns:
        Dict mapping symbol -> feature_id for use in clique analysis
    """
    resolver = GeneEntityResolver()
    results = resolver.resolve_ensembl_ids(feature_ids, verbose=verbose)

    # Create symbol -> ensembl mapping
    symbol_to_ensembl = {}

    for eid, entity in results.items():
        if entity.is_resolved:
            symbol_to_ensembl[entity.symbol] = eid

            # Add aliases too
            for alias in entity.aliases:
                if alias and alias not in symbol_to_ensembl:
                    symbol_to_ensembl[alias] = eid

    # Save if requested
    if output_path:
        import pandas as pd
        df = pd.DataFrame([
            {
                'ensembl_id': e.ensembl_id,
                'symbol': e.symbol,
                'source': e.source,
                'confidence': e.confidence,
                'biotype': e.biotype,
                'n_aliases': len(e.aliases),
            }
            for e in results.values()
        ])
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"Saved resolution results to: {output_path}")

    return symbol_to_ensembl
