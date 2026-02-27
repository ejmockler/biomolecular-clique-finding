"""
Abstract interface and concrete implementations for biological annotation databases.

This module provides a pluggable architecture for accessing biological annotations
from multiple sources (GO, KEGG, DisGeNET, etc.) with a unified interface.

Design Principles:
    - Abstract interface: Easy to add new annotation sources
    - Caching: Expensive network/file I/O cached for performance
    - ID mapping: Handle multiple gene identifier systems (Ensembl, Entrez, Symbol)
    - Fail gracefully: Return empty sets for unknown IDs, don't crash

Biological Context:
    Different annotations answer different questions:
    - GO (Gene Ontology): Molecular functions, biological processes, cellular components
    - KEGG: Metabolic and signaling pathways
    - DisGeNET: Disease-gene associations (critical for ALS validation)
    - String-DB: Protein-protein interactions (for network coherence)

Examples:
    >>> from cliquefinder.validation.annotation_providers import GOAnnotationProvider
    >>>
    >>> # Initialize provider
    >>> provider = GOAnnotationProvider()
    >>>
    >>> # Get annotations for a gene
    >>> annotations = provider.get_annotations('ENSG00000141510')  # TP53
    >>> print(annotations['go_biological_process'])
    {'GO:0006915', 'GO:0042981', ...}  # apoptosis, stress response
    >>>
    >>> # Get human-readable names
    >>> provider.get_term_name('GO:0006915')
    'apoptotic process'
    >>>
    >>> # Get all genes annotated to a term
    >>> genes = provider.get_genes_for_term('GO:0006915')
    {'ENSG00000141510', 'ENSG00000...", ...}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Set, List, Optional
from pathlib import Path
import json
import gzip
import urllib.request
import logging
import warnings

logger = logging.getLogger(__name__)

__all__ = [
    'AnnotationProvider',
    'GOAnnotationProvider',
    'KEGGAnnotationProvider',
    'DisGeNETProvider',
    'CachedAnnotationProvider',
]


class AnnotationProvider(ABC):
    """
    Abstract interface for biological annotation databases.

    All annotation providers must implement this interface to work with
    the biological validation framework.

    Key Methods:
        get_annotations(gene_id): Get all annotations for a gene
        get_term_name(term_id): Get human-readable name for annotation term
        get_genes_for_term(term_id): Get all genes annotated to a term (reverse lookup)
        get_all_terms(): Get all available annotation terms
    """

    @abstractmethod
    def get_annotations(self, gene_id: str) -> Dict[str, Set[str]]:
        """
        Get annotations for a gene.

        Args:
            gene_id: Gene identifier (format depends on provider)

        Returns:
            Dictionary mapping annotation categories to sets of term IDs:
            {
                'go_biological_process': {'GO:0006915', 'GO:0042981', ...},
                'go_molecular_function': {'GO:0004674', ...},
                'go_cellular_component': {'GO:0005634', ...},
                'pathways': {'KEGG:hsa04010', 'Reactome:R-HSA-212436', ...},
                'diseases': {'ALS', 'Parkinson', ...}
            }

            Empty sets if no annotations found (don't crash on unknown IDs).

        Examples:
            >>> provider.get_annotations('ENSG00000141510')  # TP53
            {
                'go_biological_process': {'GO:0006915', 'GO:0042981'},
                'go_molecular_function': {'GO:0003677'},
                'pathways': {'KEGG:hsa04115'}
            }
        """
        pass

    @abstractmethod
    def get_term_name(self, term_id: str) -> str:
        """
        Get human-readable name for annotation term.

        Args:
            term_id: Annotation term ID (e.g., 'GO:0006915', 'KEGG:hsa04010')

        Returns:
            Human-readable term name

        Examples:
            >>> provider.get_term_name('GO:0006915')
            'apoptotic process'
            >>> provider.get_term_name('KEGG:hsa04010')
            'MAPK signaling pathway'
        """
        pass

    @abstractmethod
    def get_genes_for_term(self, term_id: str) -> Set[str]:
        """
        Get all genes annotated to a specific term (reverse lookup).

        Critical for enrichment testing - need to know population of genes
        that could have been in the term.

        Args:
            term_id: Annotation term ID

        Returns:
            Set of gene IDs annotated to this term

        Examples:
            >>> genes = provider.get_genes_for_term('GO:0006915')  # apoptosis
            >>> len(genes)
            1247
            >>> 'ENSG00000141510' in genes  # TP53 is in apoptosis
            True
        """
        pass

    @abstractmethod
    def get_all_terms(self) -> Set[str]:
        """
        Get all available annotation terms.

        Returns:
            Set of all term IDs in this annotation database

        Examples:
            >>> terms = provider.get_all_terms()
            >>> len(terms)
            45123
            >>> 'GO:0006915' in terms
            True
        """
        pass

    def get_background_genes(self) -> Set[str]:
        """
        Get all genes with at least one annotation (background set for enrichment).

        Default implementation: union of all genes across all terms.
        Override if database has more efficient approach.

        Returns:
            Set of all annotated gene IDs

        Examples:
            >>> background = provider.get_background_genes()
            >>> len(background)
            18523  # ~18k protein-coding genes in human genome
        """
        all_genes = set()
        for term in self.get_all_terms():
            all_genes.update(self.get_genes_for_term(term))
        return all_genes


class GOAnnotationProvider(AnnotationProvider):
    """
    Gene Ontology annotation provider.

    Downloads and parses GO annotations from:
    - http://current.geneontology.org/annotations/goa_human.gaf.gz
    - GO term definitions from http://purl.obolibrary.org/obo/go.obo

    Supports three GO namespaces:
    - biological_process: What biological processes the gene participates in
    - molecular_function: What molecular-level activities the gene performs
    - cellular_component: Where in the cell the gene product is located

    Args:
        annotation_file: Path to GOA annotation file (GAF format)
            If None, downloads from geneontology.org
        obo_file: Path to GO term definitions (OBO format)
            If None, downloads from geneontology.org
        cache_dir: Directory for caching downloaded files
            Default: ~/.cache/biocore/validation/go/
        id_type: Gene identifier type ('ensembl', 'entrez', 'symbol')
            Default: 'ensembl'

    Examples:
        >>> # Auto-download and cache
        >>> provider = GOAnnotationProvider()
        >>>
        >>> # Use local files
        >>> provider = GOAnnotationProvider(
        ...     annotation_file='goa_human.gaf',
        ...     obo_file='go.obo'
        ... )
        >>>
        >>> # Get annotations
        >>> annotations = provider.get_annotations('ENSG00000141510')
        >>> 'GO:0006915' in annotations['go_biological_process']
        True
    """

    def __init__(
        self,
        annotation_file: Optional[Path] = None,
        obo_file: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        id_type: str = 'ensembl'
    ):
        """Initialize GO annotation provider."""
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'biocore' / 'validation' / 'go'
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = cache_dir
        self.id_type = id_type

        # Load or download annotation file
        if annotation_file is None:
            annotation_file = self._download_goa()
        self.annotation_file = Path(annotation_file)

        # Load or download OBO file
        if obo_file is None:
            obo_file = self._download_obo()
        self.obo_file = Path(obo_file)

        # Parse files
        self._gene_to_terms: Dict[str, Dict[str, Set[str]]] = {}
        self._term_to_genes: Dict[str, Set[str]] = {}
        self._term_names: Dict[str, str] = {}
        self._term_namespaces: Dict[str, str] = {}

        self._parse_obo()
        self._parse_gaf()

    def _download_goa(self) -> Path:
        """Download GOA human annotation file if not cached."""
        cached_path = self.cache_dir / 'goa_human.gaf.gz'
        if cached_path.exists():
            return cached_path

        url = 'http://current.geneontology.org/annotations/goa_human.gaf.gz'
        print(f"Downloading GO annotations from {url}...")
        urllib.request.urlretrieve(url, cached_path)
        print(f"✓ Downloaded to {cached_path}")
        return cached_path

    def _download_obo(self) -> Path:
        """Download GO term definitions if not cached."""
        cached_path = self.cache_dir / 'go.obo'
        if cached_path.exists():
            return cached_path

        url = 'http://purl.obolibrary.org/obo/go.obo'
        print(f"Downloading GO term definitions from {url}...")
        urllib.request.urlretrieve(url, cached_path)
        print(f"✓ Downloaded to {cached_path}")
        return cached_path

    def _parse_obo(self) -> None:
        """Parse GO term definitions from OBO file."""
        print("Parsing GO term definitions...")
        with open(self.obo_file) as f:
            current_id = None
            current_name = None
            current_namespace = None

            for line in f:
                line = line.strip()

                if line == '[Term]':
                    # Save previous term
                    if current_id:
                        self._term_names[current_id] = current_name or current_id
                        self._term_namespaces[current_id] = current_namespace or 'unknown'
                    current_id = None
                    current_name = None
                    current_namespace = None

                elif line.startswith('id: GO:'):
                    current_id = line[4:]

                elif line.startswith('name: '):
                    current_name = line[6:]

                elif line.startswith('namespace: '):
                    current_namespace = line[11:]

            # Save last term
            if current_id:
                self._term_names[current_id] = current_name or current_id
                self._term_namespaces[current_id] = current_namespace or 'unknown'

        print(f"✓ Loaded {len(self._term_names)} GO terms")

    def _parse_gaf(self) -> None:
        """Parse gene-term associations from GAF file."""
        print("Parsing GO annotations...")

        # Open gzipped file
        import gzip
        with gzip.open(self.annotation_file, 'rt') as f:
            for line in f:
                # Skip comments
                if line.startswith('!'):
                    continue

                fields = line.strip().split('\t')
                if len(fields) < 13:
                    continue

                # Extract fields
                db_object_id = fields[1]  # Usually UniProt ID
                go_id = fields[4]  # GO:XXXXXXX
                evidence_code = fields[6]  # IEA, IDA, etc.

                # Get Ensembl ID from DB Object field (if available)
                # GAF format: field[1] is primary ID, field[10] contains synonyms
                gene_id = None
                if self.id_type == 'ensembl':
                    # Look for Ensembl ID in synonyms (field 10)
                    synonyms = fields[10].split('|')
                    for syn in synonyms:
                        if syn.startswith('ENSG'):
                            gene_id = syn
                            break
                elif self.id_type == 'entrez':
                    # Would need DB cross-reference parsing
                    pass
                elif self.id_type == 'symbol':
                    gene_id = fields[2]  # Gene symbol

                if not gene_id:
                    continue

                # Get namespace for this GO term
                namespace = self._term_namespaces.get(go_id, 'unknown')
                if namespace == 'unknown':
                    continue

                # Map namespace to category
                category_map = {
                    'biological_process': 'go_biological_process',
                    'molecular_function': 'go_molecular_function',
                    'cellular_component': 'go_cellular_component',
                }
                category = category_map.get(namespace)
                if not category:
                    continue

                # Add to gene -> terms mapping
                if gene_id not in self._gene_to_terms:
                    self._gene_to_terms[gene_id] = {
                        'go_biological_process': set(),
                        'go_molecular_function': set(),
                        'go_cellular_component': set(),
                    }
                self._gene_to_terms[gene_id][category].add(go_id)

                # Add to term -> genes mapping
                if go_id not in self._term_to_genes:
                    self._term_to_genes[go_id] = set()
                self._term_to_genes[go_id].add(gene_id)

        print(f"✓ Loaded annotations for {len(self._gene_to_terms)} genes")
        print(f"✓ Loaded annotations for {len(self._term_to_genes)} GO terms")

    def get_annotations(self, gene_id: str) -> Dict[str, Set[str]]:
        """Get GO annotations for a gene."""
        return self._gene_to_terms.get(gene_id, {
            'go_biological_process': set(),
            'go_molecular_function': set(),
            'go_cellular_component': set(),
        })

    def get_term_name(self, term_id: str) -> str:
        """Get human-readable name for GO term."""
        return self._term_names.get(term_id, term_id)

    def get_genes_for_term(self, term_id: str) -> Set[str]:
        """Get all genes annotated to a GO term."""
        return self._term_to_genes.get(term_id, set())

    def get_all_terms(self) -> Set[str]:
        """Get all GO term IDs."""
        return set(self._term_to_genes.keys())


class KEGGAnnotationProvider(AnnotationProvider):
    """
    KEGG pathway annotation provider.

    NOTE: This is a stub implementation. Full KEGG access requires license
    for programmatic access. For production use, either:
    1. Use KEGGREST R package via rpy2
    2. Download KEGG files manually and parse
    3. Use alternative pathway database (Reactome, WikiPathways)

    For demonstration purposes, returns empty annotations.
    """

    def __init__(self):
        warnings.warn(
            "KEGGAnnotationProvider is a stub. KEGG requires license for "
            "programmatic access. Consider using Reactome or WikiPathways instead.",
            UserWarning
        )
        self._data = {}

    def get_annotations(self, gene_id: str) -> Dict[str, Set[str]]:
        return {'pathways': set()}

    def get_term_name(self, term_id: str) -> str:
        return term_id

    def get_genes_for_term(self, term_id: str) -> Set[str]:
        return set()

    def get_all_terms(self) -> Set[str]:
        return set()


class DisGeNETProvider(AnnotationProvider):
    """
    Disease-gene association provider using DisGeNET.

    NOTE: This is a stub implementation. DisGeNET requires registration
    for API access. For production use, download from:
    https://www.disgenet.org/downloads

    For demonstration purposes, returns empty annotations.
    """

    def __init__(self):
        warnings.warn(
            "DisGeNETProvider is a stub. DisGeNET requires registration. "
            "Download manually from disgenet.org",
            UserWarning
        )
        self._data = {}

    def get_annotations(self, gene_id: str) -> Dict[str, Set[str]]:
        return {'diseases': set()}

    def get_term_name(self, term_id: str) -> str:
        return term_id

    def get_genes_for_term(self, term_id: str) -> Set[str]:
        return set()

    def get_all_terms(self) -> Set[str]:
        return set()


class CachedAnnotationProvider(AnnotationProvider):
    """
    Wrapper that caches expensive annotation lookups.

    Useful for repeated validation runs on same dataset.

    Args:
        provider: Underlying annotation provider to cache
        cache_file: Path to cache file (JSON format)

    Examples:
        >>> from cliquefinder.validation.annotation_providers import GOAnnotationProvider
        >>> provider = GOAnnotationProvider()
        >>> cached = CachedAnnotationProvider(provider, cache_file='go_cache.json')
        >>>
        >>> # First call: queries provider and caches
        >>> annotations = cached.get_annotations('ENSG00000141510')
        >>>
        >>> # Second call: loads from cache (much faster)
        >>> annotations = cached.get_annotations('ENSG00000141510')
    """

    def __init__(
        self,
        provider: AnnotationProvider,
        cache_file: Optional[Path] = None
    ):
        self.provider = provider
        self.cache_file = cache_file or (Path.home() / '.cache' / 'biocore' / 'annotations.json')
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Load cache if exists
        self._cache: Dict = {}
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    raw = json.load(f)
                self._cache = self._deserialize_cache(raw)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Corrupted cache file {self.cache_file}, ignoring: {e}")
                self._cache = {}

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize a cache value for JSON, converting sets to tagged lists."""
        if isinstance(value, set):
            return {"__set__": True, "values": sorted(value)}
        elif isinstance(value, dict):
            return {k: CachedAnnotationProvider._serialize_value(v) for k, v in value.items()}
        return value

    @staticmethod
    def _deserialize_value(value: Any) -> Any:
        """Deserialize a cache value from JSON, restoring sets from tagged lists."""
        if isinstance(value, dict):
            if value.get("__set__") is True and "values" in value:
                return set(value["values"])
            return {k: CachedAnnotationProvider._deserialize_value(v) for k, v in value.items()}
        return value

    @staticmethod
    def _deserialize_cache(raw: Dict) -> Dict:
        """Deserialize full cache dict from JSON."""
        return {k: CachedAnnotationProvider._deserialize_value(v) for k, v in raw.items()}

    def _save_cache(self):
        """Save cache to disk."""
        serializable = {k: self._serialize_value(v) for k, v in self._cache.items()}
        with open(self.cache_file, 'w') as f:
            json.dump(serializable, f, indent=2)

    def get_annotations(self, gene_id: str) -> Dict[str, Set[str]]:
        key = f"gene:{gene_id}"
        if key not in self._cache:
            self._cache[key] = self.provider.get_annotations(gene_id)
            self._save_cache()
        return self._cache[key]

    def get_term_name(self, term_id: str) -> str:
        key = f"term_name:{term_id}"
        if key not in self._cache:
            self._cache[key] = self.provider.get_term_name(term_id)
            self._save_cache()
        return self._cache[key]

    def get_genes_for_term(self, term_id: str) -> Set[str]:
        key = f"term:{term_id}"
        if key not in self._cache:
            self._cache[key] = self.provider.get_genes_for_term(term_id)
            self._save_cache()
        return self._cache[key]

    def get_all_terms(self) -> Set[str]:
        key = "all_terms"
        if key not in self._cache:
            self._cache[key] = self.provider.get_all_terms()
            self._save_cache()
        return self._cache[key]
