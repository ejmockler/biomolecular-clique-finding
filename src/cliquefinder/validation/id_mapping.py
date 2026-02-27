"""
Gene identifier mapping utilities

Handles conversion between identifier systems:
- Ensembl Gene ID (ENSG00000xxxxxx)
- Gene Symbol (TP53, SOD1, etc.)
- UniProt ID (P04637, etc.)
- Entrez Gene ID (7157, etc.)

Uses mygene.info API with caching for performance.

Examples:
    >>> from cliquefinder.validation.id_mapping import MyGeneInfoMapper
    >>>
    >>> mapper = MyGeneInfoMapper()
    >>> mapping = mapper.map_ids(
    ...     ['ENSG00000141510', 'ENSG00000012048'],
    ...     source_type='ensembl_gene',
    ...     target_type='symbol'
    ... )
    >>> print(mapping)
    {'ENSG00000141510': 'TP53', 'ENSG00000012048': 'BRCA1'}
"""

from __future__ import annotations

from typing import Dict, List, Set, Optional
from pathlib import Path
from abc import ABC, abstractmethod
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class IDMapper(ABC):
    """Abstract interface for gene ID mapping"""

    @abstractmethod
    def map_ids(
        self,
        source_ids: List[str],
        source_type: str,
        target_type: str
    ) -> Dict[str, str]:
        """
        Map gene IDs from source to target type

        Args:
            source_ids: List of source IDs
            source_type: 'ensembl_gene', 'symbol', 'uniprot', 'entrez'
            target_type: Same options as source_type

        Returns:
            Dict mapping source_id → target_id
            Unmapped IDs return None
        """
        pass


class MyGeneInfoMapper(IDMapper):
    """
    Uses mygene.info API for ID mapping with concurrent batch queries

    Advantages:
    - Comprehensive (all major ID types)
    - Fast batch queries with parallel execution
    - No authentication required
    - Up-to-date annotations

    Usage:
        mapper = MyGeneInfoMapper(max_workers=8)
        mapping = mapper.map_ids(
            ['ENSG00000141510', 'ENSG00000012048'],
            source_type='ensembl_gene',
            target_type='symbol'
        )
        # {'ENSG00000141510': 'TP53', 'ENSG00000012048': 'BRCA1'}
    """

    def __init__(self, cache_dir: Optional[Path] = None, max_workers: int = 8):
        """
        Initialize with optional caching and parallel query support

        Args:
            cache_dir: Directory for caching results (default: ~/.cache/biocore/id_mapping)
            max_workers: Number of concurrent API requests (default: 8)
        """
        import mygene
        self.mg = mygene.MyGeneInfo()
        self.cache_dir = cache_dir or Path.home() / '.cache/biocore/id_mapping'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self._lock = threading.Lock()

    def _query_batch(
        self,
        batch: List[str],
        source_field: str,
        target_field: str,
        species: str,
        batch_num: int,
        total_batches: int
    ) -> Dict[str, str]:
        """
        Query a single batch - thread-safe.

        Creates a new MyGeneInfo instance per thread to avoid concurrency issues.

        Args:
            batch: List of IDs to query in this batch
            source_field: mygene field for source IDs
            target_field: mygene field for target IDs
            species: Species to query
            batch_num: Current batch number (for logging)
            total_batches: Total number of batches (for logging)

        Returns:
            Dict mapping source_id → target_id for this batch
        """
        import mygene

        # Create new instance per thread for thread safety
        mg = mygene.MyGeneInfo()

        logger.debug(f"Querying batch {batch_num + 1}/{total_batches} ({len(batch)} IDs)")

        query_results = mg.querymany(
            batch,
            scopes=source_field,
            fields=target_field,
            species=species,
            returnall=True
        )

        results = {}
        for item in query_results['out']:
            source_id = item.get('query')

            # Handle nested fields (e.g., ensembl.gene)
            target_value = item
            for field_part in target_field.split('.'):
                if isinstance(target_value, dict):
                    target_value = target_value.get(field_part)
                else:
                    break

            # Handle cases where target_value is a list (multiple matches)
            if isinstance(target_value, list):
                # Take first match
                target_value = target_value[0] if target_value else None

            if target_value and source_id:
                results[source_id] = str(target_value)

        return results

    def map_ids(
        self,
        source_ids: List[str],
        source_type: str = 'ensembl_gene',
        target_type: str = 'symbol',
        species: str = 'human'
    ) -> Dict[str, str]:
        """
        Map IDs using mygene.info batch query

        Args:
            source_ids: List of source IDs to map
            source_type: Type of source IDs ('ensembl_gene', 'symbol', 'uniprot', 'entrez')
            target_type: Type to map to ('ensembl_gene', 'symbol', 'uniprot', 'entrez')
            species: Species to query (default: 'human')

        Returns:
            Dict mapping source_id → target_id (only successful mappings)
        """

        # Map type names to mygene fields
        # Use 'uniprot' (not 'uniprot.Swiss-Prot') to match both Swiss-Prot and TrEMBL
        type_map = {
            'ensembl_gene': 'ensembl.gene',
            'symbol': 'symbol',
            'symbol_alias': 'symbol,alias',  # Support for alias resolution
            'uniprot': 'uniprot',  # Matches both Swiss-Prot and TrEMBL
            'entrez': 'entrezgene'
        }

        source_field = type_map.get(source_type)
        target_field = type_map.get(target_type)

        if not source_field or not target_field:
            raise ValueError(f"Unsupported ID type: {source_type} or {target_type}")

        # Check cache
        cache_key = f"{source_type}_to_{target_type}_{hash(tuple(sorted(source_ids)))}.json"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Corrupted cache file {cache_path}, ignoring: {e}")

        # Query mygene.info in parallel batches (1000 IDs per batch)
        results = {}
        batch_size = 1000
        batches = [source_ids[i:i+batch_size] for i in range(0, len(source_ids), batch_size)]

        logger.info(f"Starting ID mapping: {len(source_ids)} IDs in {len(batches)} batches "
                   f"with {self.max_workers} workers")

        # Execute batch queries concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch queries
            futures = {
                executor.submit(
                    self._query_batch,
                    batch,
                    source_field,
                    target_field,
                    species,
                    i,
                    len(batches)
                ): i
                for i, batch in enumerate(batches)
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                batch_results = future.result()
                results.update(batch_results)
                completed += 1

                # Progress logging
                mapped_count = len(results)
                progress_pct = (completed / len(batches)) * 100
                logger.info(f"ID mapping progress: {completed}/{len(batches)} batches complete "
                           f"({progress_pct:.1f}%), {mapped_count}/{len(source_ids)} genes mapped")

        logger.info(f"ID mapping complete: {len(results)}/{len(source_ids)} genes successfully mapped "
                   f"({len(results)/len(source_ids)*100:.1f}%)")

        # Cache results
        with open(cache_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results


class BioMartMapper(IDMapper):
    """
    Uses Ensembl BioMart for ID mapping

    Advantages:
    - Official Ensembl source
    - Highly reliable for Ensembl IDs
    - Includes genomic coordinates

    Disadvantages:
    - Slower than mygene.info
    - Requires biomart Python package
    """

    def map_ids(
        self,
        source_ids: List[str],
        source_type: str = 'ensembl_gene',
        target_type: str = 'symbol'
    ) -> Dict[str, str]:
        """Map IDs using BioMart"""
        # Implementation using biomart package
        # (Specify if needed, but mygene.info is recommended)
        raise NotImplementedError("Use MyGeneInfoMapper for better performance")
