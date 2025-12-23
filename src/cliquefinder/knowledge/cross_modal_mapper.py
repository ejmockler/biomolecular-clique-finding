"""
Cross-Modal ID Mapping for Multi-Omics Integration

Maps between proteomics (gene symbols) and RNA-seq IDs (Ensembl, symbols, or
numeric indices) to enable cross-modal validation of regulatory modules.

Key Features:
    - Unified ID mapping: Convert between gene symbols, Ensembl IDs, etc.
    - INDRA validation: Optional validation via INDRA CoGEx knowledge graph
    - Caching: Disk-based caching for expensive ID mappings
    - Comprehensive statistics: Track mapping success rates and failures

Design Philosophy:
    - Pragmatic fallback: Use INDRA for validation, MyGeneInfo for conversion
    - Transparency: Log all mapping decisions and statistics
    - Robustness: Handle missing IDs gracefully with detailed diagnostics
    - Performance: Cache expensive queries to disk

Examples:
    >>> from cliquefinder.knowledge.cross_modal_mapper import CrossModalIDMapper
    >>>
    >>> # Simple mapping: both datasets use gene symbols
    >>> mapper = CrossModalIDMapper()
    >>> mapping = mapper.unify_ids(
    ...     protein_ids=['PGK1', 'BRCA1', 'TP53'],
    ...     rna_ids=['PGK1', 'BRCA1', 'MDM2'],
    ...     rna_id_type='symbol'
    ... )
    >>> print(mapping.common_genes)
    {'PGK1', 'BRCA1'}
    >>>
    >>> # Complex mapping: RNA uses Ensembl IDs
    >>> mapping = mapper.unify_ids(
    ...     protein_ids=['PGK1', 'BRCA1'],
    ...     rna_ids=['ENSG00000102144', 'ENSG00000012048'],
    ...     rna_id_type='ensembl_gene'
    ... )
    >>> print(mapping.common_genes)
    {'PGK1', 'BRCA1'}
    >>>
    >>> # With INDRA validation
    >>> mapper = CrossModalIDMapper(use_indra=True)
    >>> mapping = mapper.unify_ids(...)
"""

from __future__ import annotations

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import pickle
import hashlib
import json
import re

# Import existing infrastructure
from cliquefinder.validation.id_mapping import MyGeneInfoMapper

# Optional INDRA imports
try:
    from cliquefinder.knowledge.cogex import CoGExClient
    INDRA_AVAILABLE = True
except ImportError:
    INDRA_AVAILABLE = False
    CoGExClient = None

__all__ = [
    'CrossModalMapping',
    'CrossModalIDMapper',
    'SampleAlignment',
    'SampleAlignedCrossModalMapping',
    'SampleAlignedCrossModalMapper',
]

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CrossModalMapping:
    """
    Results of cross-modal ID mapping between proteomics and RNA data.

    Attributes:
        common_genes: Set of gene symbols present in both datasets
        protein_only: Set of gene symbols only in proteomics data
        rna_only: Set of gene symbols only in RNA data
        mapping_stats: Dictionary with detailed mapping statistics

    Statistics Dictionary Keys:
        - total_protein_ids: Number of input protein IDs
        - total_rna_ids: Number of input RNA IDs
        - rna_ids_mapped: Number of RNA IDs successfully mapped to symbols
        - rna_ids_failed: Number of RNA IDs that failed to map
        - mapping_rate: Proportion of RNA IDs successfully mapped (0.0-1.0)
        - common_count: Number of genes in both datasets
        - protein_only_count: Number of proteomics-only genes
        - rna_only_count: Number of RNA-only genes
        - overlap_rate: Proportion of protein genes found in RNA (0.0-1.0)
        - indra_validated: Number of genes validated via INDRA (if enabled)
        - indra_validation_rate: Proportion validated (if INDRA enabled)

    Scientific Interpretation:
        - High mapping_rate (>0.95): Good ID quality
        - Low mapping_rate (<0.80): ID format issues or species mismatch
        - High overlap_rate (>0.70): Strong cross-modal coverage
        - Low overlap_rate (<0.30): Datasets may measure different gene sets

    Examples:
        >>> mapping = mapper.unify_ids(protein_ids, rna_ids, 'ensembl_gene')
        >>> print(f"Found {len(mapping.common_genes)} common genes")
        >>> print(f"RNA mapping success: {mapping.mapping_stats['mapping_rate']:.1%}")
        >>> print(f"Cross-modal overlap: {mapping.mapping_stats['overlap_rate']:.1%}")
    """
    common_genes: Set[str]
    protein_only: Set[str]
    rna_only: Set[str]
    mapping_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_unified_genes(self) -> int:
        """Total number of unique genes across both datasets."""
        return len(self.common_genes) + len(self.protein_only) + len(self.rna_only)

    @property
    def overlap_rate(self) -> float:
        """Proportion of protein genes found in RNA data."""
        total_protein = len(self.common_genes) + len(self.protein_only)
        if total_protein == 0:
            return 0.0
        return len(self.common_genes) / total_protein

    def summary(self) -> str:
        """Generate human-readable summary of mapping results."""
        lines = [
            "Cross-Modal ID Mapping Summary",
            "=" * 50,
            f"Common genes (both datasets):  {len(self.common_genes):>6}",
            f"Proteomics-only genes:         {len(self.protein_only):>6}",
            f"RNA-only genes:                {len(self.rna_only):>6}",
            f"Total unified genes:           {self.total_unified_genes:>6}",
            "",
            "Mapping Statistics:",
            f"  RNA IDs mapped:              {self.mapping_stats.get('rna_ids_mapped', 0):>6} / "
            f"{self.mapping_stats.get('total_rna_ids', 0)} "
            f"({self.mapping_stats.get('mapping_rate', 0):.1%})",
            f"  Cross-modal overlap:         {self.overlap_rate:.1%}",
        ]

        if 'indra_validated' in self.mapping_stats:
            lines.append(
                f"  INDRA validated:             {self.mapping_stats['indra_validated']:>6} / "
                f"{len(self.common_genes)} "
                f"({self.mapping_stats.get('indra_validation_rate', 0):.1%})"
            )

        return "\n".join(lines)


class CrossModalIDMapper:
    """
    Map between proteomics and RNA-seq ID spaces for multi-omics integration.

    Strategy:
        1. If both use gene symbols → direct intersection (fast path)
        2. If RNA uses Ensembl/other → map via MyGeneInfo to symbols
        3. Optionally validate gene names via INDRA CoGEx knowledge graph
        4. Cache all expensive mappings to disk for reuse

    The mapper is designed to be robust to:
        - Different ID formats (symbols, Ensembl, Entrez, etc.)
        - Missing/deprecated gene names
        - Case sensitivity issues
        - Ambiguous mappings (takes first match)

    Usage:
        >>> # Basic usage without INDRA
        >>> mapper = CrossModalIDMapper()
        >>> mapping = mapper.unify_ids(
        ...     protein_ids=['PGK1', 'BRCA1'],
        ...     rna_ids=['ENSG00000102144', 'ENSG00000012048'],
        ...     rna_id_type='ensembl_gene'
        ... )
        >>>
        >>> # With INDRA validation and custom cache
        >>> mapper = CrossModalIDMapper(
        ...     use_indra=True,
        ...     cache_dir=Path('~/.cache/cross_modal')
        ... )
        >>> mapping = mapper.unify_ids(protein_ids, rna_ids, 'ensembl_gene')
        >>> print(mapping.summary())
    """

    def __init__(
        self,
        use_indra: bool = False,
        cache_dir: Optional[Path] = None,
        indra_client: Optional['CoGExClient'] = None
    ):
        """
        Initialize cross-modal ID mapper.

        Args:
            use_indra: Whether to validate gene names via INDRA CoGEx
                Default: False (INDRA is optional)
                Enable for additional validation against knowledge graph
            cache_dir: Directory for caching expensive mappings
                Default: ~/.cache/biocore/cross_modal
                Caches MyGeneInfo queries and INDRA validations
            indra_client: Optional pre-configured CoGExClient
                Default: None (create new client if use_indra=True)
                Useful for reusing existing INDRA connection

        Raises:
            ImportError: If use_indra=True but INDRA not available
        """
        self.use_indra = use_indra
        self.cache_dir = cache_dir or Path.home() / '.cache/biocore/cross_modal'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MyGeneInfo mapper (always available)
        self.mygene_mapper = MyGeneInfoMapper(
            cache_dir=self.cache_dir / 'mygene',
            max_workers=8
        )

        # Initialize INDRA client if requested
        self.indra_client = indra_client
        if self.use_indra:
            if not INDRA_AVAILABLE:
                raise ImportError(
                    "INDRA CoGEx not available. Install with: "
                    "pip install git+https://github.com/indralab/indra_cogex.git"
                )
            if self.indra_client is None:
                logger.info("Creating new CoGExClient for validation")
                self.indra_client = CoGExClient()
                # Test connection on first use (lazy)

        logger.info(
            f"Initialized CrossModalIDMapper: "
            f"INDRA validation={'enabled' if self.use_indra else 'disabled'}, "
            f"cache_dir={self.cache_dir}"
        )

    def _get_cache_key(self, *components: Any) -> str:
        """
        Generate cache key from components.

        Args:
            components: Hashable components to include in key

        Returns:
            Hexadecimal hash string for cache filename
        """
        # Create deterministic hash from components
        content = json.dumps(components, sort_keys=True)
        hash_obj = hashlib.sha256(content.encode())
        return hash_obj.hexdigest()[:16]  # First 16 chars sufficient

    def _load_cache(self, cache_key: str) -> Optional[Any]:
        """Load cached result if available."""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None

    def _save_cache(self, cache_key: str, data: Any) -> None:
        """Save result to cache."""
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def _map_ensembl_to_symbol(
        self,
        ensembl_ids: List[str],
        species: str = 'human'
    ) -> Dict[str, str]:
        """
        Map Ensembl gene IDs to gene symbols via MyGeneInfo.

        Uses disk cache to avoid redundant API calls.

        Args:
            ensembl_ids: List of Ensembl gene IDs (e.g., ['ENSG00000141510'])
            species: Species name for MyGeneInfo query
                Default: 'human'

        Returns:
            Dict mapping ensembl_id -> gene_symbol (only successful mappings)

        Examples:
            >>> mapping = mapper._map_ensembl_to_symbol(['ENSG00000141510'])
            >>> print(mapping)
            {'ENSG00000141510': 'TP53'}
        """
        if not ensembl_ids:
            return {}

        # Check cache
        cache_key = self._get_cache_key('ensembl_to_symbol', tuple(sorted(ensembl_ids)), species)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info(f"Loaded Ensembl->Symbol mapping from cache ({len(cached)} mappings)")
            return cached

        logger.info(f"Mapping {len(ensembl_ids)} Ensembl IDs to gene symbols via MyGeneInfo")

        # Query MyGeneInfo
        mapping = self.mygene_mapper.map_ids(
            source_ids=ensembl_ids,
            source_type='ensembl_gene',
            target_type='symbol',
            species=species
        )

        # Cache result
        self._save_cache(cache_key, mapping)

        logger.info(
            f"Mapped {len(mapping)}/{len(ensembl_ids)} Ensembl IDs to symbols "
            f"({len(mapping)/len(ensembl_ids):.1%} success rate)"
        )

        return mapping

    def _map_generic_to_symbol(
        self,
        source_ids: List[str],
        source_type: str,
        species: str = 'human'
    ) -> Dict[str, str]:
        """
        Map arbitrary ID type to gene symbols via MyGeneInfo.

        Supports: ensembl_gene, uniprot, entrez, or symbol (passthrough).

        Args:
            source_ids: List of source IDs
            source_type: ID type ('ensembl_gene', 'uniprot', 'entrez', 'symbol')
            species: Species name for MyGeneInfo query

        Returns:
            Dict mapping source_id -> gene_symbol

        Raises:
            ValueError: If source_type is unsupported
        """
        if source_type == 'symbol':
            # Passthrough - already symbols
            return {sid: sid for sid in source_ids}

        if source_type not in ['ensembl_gene', 'uniprot', 'entrez', 'symbol_alias']:
            raise ValueError(
                f"Unsupported source_type: {source_type}. "
                f"Supported: 'symbol', 'symbol_alias', 'ensembl_gene', 'uniprot', 'entrez'"
            )

        # Check cache
        cache_key = self._get_cache_key(f'{source_type}_to_symbol', tuple(sorted(source_ids)), species)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info(f"Loaded {source_type}->Symbol mapping from cache ({len(cached)} mappings)")
            return cached

        logger.info(f"Mapping {len(source_ids)} {source_type} IDs to gene symbols via MyGeneInfo")

        # Query MyGeneInfo
        mapping = self.mygene_mapper.map_ids(
            source_ids=source_ids,
            source_type=source_type,
            target_type='symbol',
            species=species
        )

        # Cache result
        self._save_cache(cache_key, mapping)

        logger.info(
            f"Mapped {len(mapping)}/{len(source_ids)} {source_type} IDs to symbols "
            f"({len(mapping)/len(source_ids):.1%} success rate)"
        )

        return mapping

    def _validate_via_indra(self, gene_names: List[str]) -> Set[str]:
        """
        Validate gene names exist in INDRA knowledge graph.

        This provides additional confidence that gene names are:
        1. Correctly formatted
        2. Recognized by INDRA's HGNC client
        3. Present in biological knowledge bases

        Args:
            gene_names: List of gene symbols to validate

        Returns:
            Set of gene symbols that were successfully validated

        Note:
            Validation failures don't necessarily mean the gene is invalid,
            just that INDRA couldn't resolve it. Use as a filter, not a blocker.
        """
        if not self.use_indra or self.indra_client is None:
            # INDRA validation disabled - return all as valid
            return set(gene_names)

        if not gene_names:
            return set()

        # Check cache
        cache_key = self._get_cache_key('indra_validation', tuple(sorted(gene_names)))
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info(f"Loaded INDRA validation from cache ({len(cached)} validated)")
            return cached

        logger.info(f"Validating {len(gene_names)} gene names via INDRA CoGEx")

        # Import INDRA module extractor for gene name resolution
        try:
            from cliquefinder.knowledge.cogex import INDRAModuleExtractor

            extractor = INDRAModuleExtractor(self.indra_client)
            validated = set()

            for gene_name in gene_names:
                gene_id = extractor.resolve_gene_name(gene_name)
                if gene_id is not None:
                    validated.add(gene_name)

            # Cache result
            self._save_cache(cache_key, validated)

            logger.info(
                f"INDRA validated {len(validated)}/{len(gene_names)} gene names "
                f"({len(validated)/len(gene_names):.1%} validation rate)"
            )

            return validated

        except Exception as e:
            logger.warning(f"INDRA validation failed: {e}. Returning all genes as valid.")
            return set(gene_names)

    def unify_ids(
        self,
        protein_ids: List[str],
        rna_ids: List[str],
        rna_id_type: str,
        species: str = 'human'
    ) -> CrossModalMapping:
        """
        Unify proteomics and RNA ID spaces to common gene symbols.

        Main entry point for cross-modal ID mapping. Handles:
        1. Direct intersection (if both use symbols)
        2. ID conversion via MyGeneInfo (if RNA uses Ensembl/etc.)
        3. Optional INDRA validation
        4. Comprehensive statistics collection

        Args:
            protein_ids: List of gene symbols from proteomics data
                Example: ['PGK1', 'BRCA1', 'TP53']
            rna_ids: List of IDs from RNA-seq data
                Format depends on rna_id_type
                Example (symbol): ['PGK1', 'BRCA1', 'MDM2']
                Example (ensembl): ['ENSG00000102144', 'ENSG00000012048']
            rna_id_type: Type of RNA IDs
                Options: 'symbol', 'ensembl_gene', 'uniprot', 'entrez'
            species: Species for ID mapping
                Default: 'human'

        Returns:
            CrossModalMapping with common genes, unique genes, and statistics

        Examples:
            >>> # Same ID type (fast path)
            >>> mapping = mapper.unify_ids(
            ...     protein_ids=['PGK1', 'BRCA1'],
            ...     rna_ids=['PGK1', 'MDM2'],
            ...     rna_id_type='symbol'
            ... )
            >>> print(mapping.common_genes)
            {'PGK1'}
            >>>
            >>> # Different ID types (requires conversion)
            >>> mapping = mapper.unify_ids(
            ...     protein_ids=['PGK1', 'BRCA1'],
            ...     rna_ids=['ENSG00000102144', 'ENSG00000012048'],
            ...     rna_id_type='ensembl_gene'
            ... )
        """
        logger.info(
            f"Unifying IDs: {len(protein_ids)} protein IDs, "
            f"{len(rna_ids)} RNA IDs (type={rna_id_type})"
        )

        # Detect protein ID type (UniProt accession vs gene symbol)
        # UniProt accessions: 6-10 alphanumeric chars, start with letter, uppercase
        # Examples: A0AVT1, P12345, Q9NR48
        import re
        uniprot_pattern = re.compile(r'^[A-Z][A-Z0-9]{5,9}$')
        sample_ids = protein_ids[:100]  # Sample first 100
        uniprot_matches = sum(1 for pid in sample_ids if uniprot_pattern.match(pid))
        protein_id_type = 'uniprot' if uniprot_matches > len(sample_ids) * 0.5 else 'symbol_alias'
        logger.info(f"Detected protein ID type: {protein_id_type} ({uniprot_matches}/{len(sample_ids)} match UniProt pattern)")

        # Normalize protein IDs
        # Use MyGeneInfo to resolve to gene symbols
        try:
            # Map protein IDs to canonical symbols
            protein_mapping = self._map_generic_to_symbol(protein_ids, protein_id_type, species)
            protein_symbols = set(protein_mapping.values())
            # Add unmapped ones as-is (upper case) as fallback
            mapped_sources = set(protein_mapping.keys())
            for pid in protein_ids:
                if pid not in mapped_sources:
                    protein_symbols.add(pid.upper())
            logger.info(f"Resolved {len(protein_ids)} protein IDs to {len(protein_symbols)} canonical symbols")
        except Exception as e:
            logger.warning(f"Failed to resolve protein IDs: {e}. Using raw symbols.")
            protein_symbols = set(pid.upper() for pid in protein_ids)

        # Map RNA IDs to symbols
        if rna_id_type == 'symbol':
            # Resolve aliases for RNA symbols too
            try:
                rna_mapping = self._map_generic_to_symbol(rna_ids, 'symbol_alias', species)
                # Add unmapped as fallback
                mapped_rna = set(rna_mapping.keys())
                for rid in rna_ids:
                    if rid not in mapped_rna:
                        rna_mapping[rid] = rid.upper()
                rna_symbols = set(rna_mapping.values())
                logger.info("Resolved RNA symbols (handling aliases)")
            except Exception as e:
                logger.warning(f"Failed to resolve RNA aliases: {e}. Using direct intersection.")
                rna_symbols = set(rid.upper() for rid in rna_ids)
                rna_mapping = {rid: rid.upper() for rid in rna_ids}
        else:
            # Need to map RNA IDs to symbols
            rna_mapping = self._map_generic_to_symbol(rna_ids, rna_id_type, species)
            # Normalize mapped symbols to uppercase
            rna_mapping = {k: v.upper() for k, v in rna_mapping.items()}
            rna_symbols = set(rna_mapping.values())

        # Calculate basic intersection (all symbols now uppercase)
        common_genes = protein_symbols & rna_symbols
        protein_only = protein_symbols - rna_symbols
        rna_only = rna_symbols - protein_symbols

        # Collect mapping statistics
        stats = {
            'total_protein_ids': len(protein_ids),
            'total_rna_ids': len(rna_ids),
            'rna_ids_mapped': len(rna_mapping),
            'rna_ids_failed': len(rna_ids) - len(rna_mapping),
            'mapping_rate': len(rna_mapping) / len(rna_ids) if rna_ids else 0.0,
            'common_count': len(common_genes),
            'protein_only_count': len(protein_only),
            'rna_only_count': len(rna_only),
            'overlap_rate': len(common_genes) / len(protein_symbols) if protein_symbols else 0.0,
        }

        # Optional INDRA validation
        if self.use_indra and common_genes:
            validated = self._validate_via_indra(list(common_genes))
            stats['indra_validated'] = len(validated)
            stats['indra_validation_rate'] = len(validated) / len(common_genes)

            # Update common genes to only validated ones
            # (optional - can be controlled by parameter)
            # For now, we just log but keep all common genes
            logger.info(
                f"INDRA validation: {len(validated)}/{len(common_genes)} common genes validated"
            )

        # Create result
        result = CrossModalMapping(
            common_genes=common_genes,
            protein_only=protein_only,
            rna_only=rna_only,
            mapping_stats=stats
        )

        # Log summary
        logger.info(f"\n{result.summary()}")

        return result

    def filter_to_common(
        self,
        protein_ids: List[str],
        rna_ids: List[str],
        rna_id_type: str,
        species: str = 'human'
    ) -> Set[str]:
        """
        Convenience method: Return only common gene symbols.

        Shortcut for unify_ids(...).common_genes

        Args:
            protein_ids: List of gene symbols from proteomics
            rna_ids: List of IDs from RNA-seq
            rna_id_type: Type of RNA IDs
            species: Species for ID mapping

        Returns:
            Set of gene symbols present in both datasets

        Examples:
            >>> common = mapper.filter_to_common(
            ...     protein_ids=['PGK1', 'BRCA1'],
            ...     rna_ids=['PGK1', 'MDM2'],
            ...     rna_id_type='symbol'
            ... )
            >>> print(common)
            {'PGK1'}
        """
        mapping = self.unify_ids(protein_ids, rna_ids, rna_id_type, species)
        return mapping.common_genes


@dataclass
class SampleAlignment:
    """Results of cross-modal sample alignment."""
    matched_participants: Set[str]
    proteomics_only: Set[str]
    rna_only: Set[str]
    n_proteomics_samples: int
    n_rna_samples: int

    @property
    def n_matched(self) -> int:
        return len(self.matched_participants)

    @property
    def match_rate_proteomics(self) -> float:
        return self.n_matched / len(self.matched_participants | self.proteomics_only) if self.matched_participants or self.proteomics_only else 0.0

    @property
    def match_rate_rna(self) -> float:
        return self.n_matched / len(self.matched_participants | self.rna_only) if self.matched_participants or self.rna_only else 0.0


@dataclass
class SampleAlignedCrossModalMapping(CrossModalMapping):
    """Extended mapping with sample alignment and expression validation."""
    sample_alignment: Optional[SampleAlignment] = None
    expression_validated_genes: Optional[Set[str]] = None
    expression_filter_stats: Optional[Dict[str, Any]] = None


class SampleAlignedCrossModalMapper(CrossModalIDMapper):
    """
    Extended mapper with sample alignment and expression filtering.

    Orchestrates the full cross-modal validation pipeline:
    1. Sample alignment by participant ID (optional)
    2. Expression filtering by CPM threshold (optional)
    3. ID mapping (always)
    4. Intersection of validated genes
    """

    def __init__(
        self,
        use_indra: bool = False,
        cache_dir: Optional[Path] = None,
        participant_pattern: str = r'(NEU[A-Z0-9]+)',
    ):
        super().__init__(use_indra, cache_dir)
        self.participant_pattern = re.compile(participant_pattern)

    def _extract_participant_id(self, sample_id: str) -> Optional[str]:
        """Extract participant ID using regex pattern."""
        match = self.participant_pattern.search(str(sample_id))
        return match.group(1) if match else None

    def align_samples(
        self,
        proteomics_sample_ids: List[str],
        rna_sample_ids: List[str],
    ) -> SampleAlignment:
        """Align samples between modalities by participant ID."""
        prot_participants = {
            self._extract_participant_id(sid)
            for sid in proteomics_sample_ids
        }
        prot_participants.discard(None)

        rna_participants = {
            self._extract_participant_id(sid)
            for sid in rna_sample_ids
        }
        rna_participants.discard(None)

        matched = prot_participants & rna_participants

        return SampleAlignment(
            matched_participants=matched,
            proteomics_only=prot_participants - rna_participants,
            rna_only=rna_participants - prot_participants,
            n_proteomics_samples=len(proteomics_sample_ids),
            n_rna_samples=len(rna_sample_ids),
        )

    def unify_with_expression_filter(
        self,
        protein_ids: List[str],
        proteomics_sample_ids: List[str],
        rna_ids: List[str],
        rna_sample_ids: List[str],
        rna_id_type: str,
        expressed_genes: Optional[Set[str]] = None,  # Pre-computed from StratifiedExpressionFilter
        species: str = 'human',
    ) -> SampleAlignedCrossModalMapping:
        """
        Full cross-modal validation: alignment + expression + ID mapping.

        Args:
            protein_ids: Gene symbols from proteomics features
            proteomics_sample_ids: Sample IDs from proteomics matrix
            rna_ids: Gene IDs from RNA matrix
            rna_sample_ids: Sample IDs from RNA matrix
            rna_id_type: ID type in RNA matrix ('ensembl_gene', 'symbol', etc.)
            expressed_genes: Set of gene symbols passing expression filter (optional)
            species: Species for ID mapping

        Returns:
            SampleAlignedCrossModalMapping with full provenance
        """
        # 1. Sample alignment
        sample_alignment = self.align_samples(proteomics_sample_ids, rna_sample_ids)
        logger.info(
            f"Sample alignment: {sample_alignment.n_matched} matched participants "
            f"({len(sample_alignment.proteomics_only)} prot-only, "
            f"{len(sample_alignment.rna_only)} rna-only)"
        )

        # 2. Standard ID mapping (uses parent class)
        base_mapping = self.unify_ids(protein_ids, rna_ids, rna_id_type, species)

        # 3. Apply expression filter if provided
        expression_validated = base_mapping.common_genes
        if expressed_genes is not None:
            expression_validated = base_mapping.common_genes & expressed_genes
            if len(base_mapping.common_genes) > 0:
                logger.info(
                    f"Expression filter: {len(expression_validated)}/{len(base_mapping.common_genes)} "
                    f"genes validated ({100*len(expression_validated)/len(base_mapping.common_genes):.1f}%)"
                )
            else:
                logger.warning(
                    f"No common genes between proteomics and RNA. Expression filter not applied."
                )

        return SampleAlignedCrossModalMapping(
            common_genes=base_mapping.common_genes,
            protein_only=base_mapping.protein_only,
            rna_only=base_mapping.rna_only,
            mapping_stats=base_mapping.mapping_stats,
            sample_alignment=sample_alignment,
            expression_validated_genes=expression_validated,
        )
