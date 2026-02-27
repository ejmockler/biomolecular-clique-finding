"""
INDRA CoGEx Knowledge Graph Client

This module provides access to the INDRA CoGEx Neo4j knowledge graph, which
integrates causal mechanistic knowledge from multiple sources (literature,
pathway databases, etc.).

Key Design Principles:
    - Generalized regulator queries: Not limited to TFs - any upstream gene
    - INDRA-native terminology: Results explicitly labeled as INDRA-sourced
    - Flexible relationship types: Supports any INDRA statement type
    - No hardcoded gene lists: Query any regulator dynamically

Terminology:
    - INDRA targets: Genes downstream of a regulator per INDRA CoGEx
    - Regulator: Any upstream gene (TF, kinase, signaling molecule, etc.)
    - Coherent modules: INDRA targets that show correlation in expression data

    INDRA CoGEx aggregates causal relationships from:
    - Text mining (Reach, Sparser, etc.)
    - Pathway databases (Reactome, PathwayCommons)
    - Direct experimental evidence (TRRUST, RegNetwork, PhosphoSitePlus)

    Each relationship has:
    - Statement type: IncreaseAmount, DecreaseAmount, Activation, Inhibition,
                      Phosphorylation, etc.
    - Evidence count: Number of supporting sources/papers
    - Source counts: Breakdown by database/text mining system

Engineering Design:
    - Lazy connection: Client only connects when first query is made
    - Credential fallback: Env vars → .env file → explicit parameters
    - Type safety: dataclasses for structured results
    - Gene name resolution: INDRA's HGNC client for robust name→ID mapping
    - Immutable results: All query results are frozen dataclasses

Examples:
    >>> from cliquefinder.knowledge.cogex import CoGExClient, INDRAModuleExtractor
    >>> from pathlib import Path
    >>>
    >>> # Initialize client with .env file
    >>> client = CoGExClient(env_file=Path("~/.env"))
    >>>
    >>> # Query any regulator's downstream targets
    >>> targets = client.get_downstream_targets(
    ...     regulator=("HGNC", "11998"),  # TP53
    ...     stmt_types=["IncreaseAmount", "DecreaseAmount"],
    ...     min_evidence=2
    ... )
    >>>
    >>> # Extract modules for multiple regulators (not just TFs)
    >>> extractor = INDRAModuleExtractor(client)
    >>> regulators = ["TP53", "AKT1", "MAPK1"]  # TF, kinase, kinase
    >>> gene_universe = ["MDM2", "CDKN1A", "BAX", "BBC3"]
    >>> modules = extractor.get_regulator_modules(
    ...     regulators=regulators,
    ...     gene_universe=gene_universe,
    ...     min_evidence=2
    ... )
"""

# Warning convention:
#   warnings.warn() -- user-facing (convergence, deprecated, sample size)
#   logger.warning() -- operator-facing (fallback, retry, missing data)

from __future__ import annotations

from enum import Enum
from typing import Tuple, List, Set, Optional, Dict, Literal
from dataclasses import dataclass, field
from pathlib import Path
import os
import json
import logging

# INDRA imports
try:
    from indra_cogex.client.neo4j_client import Neo4jClient
    from indra_cogex.representation import norm_id
    from indra.databases import hgnc_client
    INDRA_AVAILABLE = True
except ImportError:
    INDRA_AVAILABLE = False
    norm_id = None  # Placeholder

# Neo4j typed exceptions for defence-in-depth error classification (ARCH-4-NOTE)
try:
    from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
    _NEO4J_EXCEPTIONS_AVAILABLE = True
except ImportError:
    _NEO4J_EXCEPTIONS_AVAILABLE = False
    ServiceUnavailable = None  # type: ignore[assignment,misc]
    SessionExpired = None  # type: ignore[assignment,misc]
    TransientError = None  # type: ignore[assignment,misc]

__all__ = [
    'GeneId',
    'INDRAEdge',
    'INDRAModule',
    'CoGExClient',
    'INDRAModuleExtractor',
    'RegulatorClass',
    'get_regulator_class_genes',
    'ACTIVATION_TYPES',
    'REPRESSION_TYPES',
    'ALL_REGULATORY_TYPES',
    'PHOSPHORYLATION_TYPES',
    'STMT_TYPE_PRESETS',
    'resolve_stmt_types',
]

# Configure logging
logger = logging.getLogger(__name__)

# Type alias for gene identifiers
GeneId = Tuple[str, str]
"""
Gene identifier as (namespace, identifier) tuple.

Examples:
    - ("HGNC", "11998") for TP53
    - ("HGNC", "7553") for MYC
    - ("HGNC", "6204") for JUN
"""

# Statement type mappings
ACTIVATION_TYPES = {"IncreaseAmount", "Activation"}
"""INDRA statement types representing transcriptional activation."""

REPRESSION_TYPES = {"DecreaseAmount", "Inhibition"}
"""INDRA statement types representing transcriptional repression."""

ALL_REGULATORY_TYPES = ACTIVATION_TYPES | REPRESSION_TYPES
"""All INDRA statement types for TF regulatory relationships.

.. warning:: **Statement type conflation**

    This preset unions activators (IncreaseAmount, Activation) and
    repressors (DecreaseAmount, Inhibition) into a **single gene set**.
    When used for enrichment testing, conflating opposing regulatory
    directions can dilute directional signals: genes activated by a
    regulator and genes repressed by the same regulator will be pooled,
    partially cancelling the fold-change signal in a competitive test.

    **When to use this preset (``--stmt-types regulatory``):**
    - Exploratory analysis where you want the largest possible gene set
    - Self-contained tests (ROAST) that detect bidirectional regulation
    - When the research question is "does this regulator affect *any*
      downstream target?" regardless of direction

    **When to use directional presets instead:**
    - ``--stmt-types activation`` for genes *upregulated* by the regulator
    - ``--stmt-types repression`` for genes *downregulated* by the regulator
    - When testing a directional hypothesis (e.g., "C9ORF72 loss reduces
      expression of its activation targets")
    - When running competitive enrichment tests sensitive to sign coherence

    See also ``--strict-stmt-types`` CLI flag, which emits a warning when
    the mixed ``regulatory`` preset is used.
"""

PHOSPHORYLATION_TYPES = {"Phosphorylation"}
"""INDRA statement types for phosphorylation (kinase-substrate) relationships."""

STMT_TYPE_PRESETS: Dict[str, Set[str]] = {
    "regulatory": ALL_REGULATORY_TYPES,
    "activation": ACTIVATION_TYPES,
    "repression": REPRESSION_TYPES,
    "phosphorylation": PHOSPHORYLATION_TYPES,
}
"""Named presets for --stmt-types CLI argument."""


def resolve_stmt_types(value: Optional[str] = None) -> List[str]:
    """
    Resolve a statement type specifier to a list of INDRA statement type strings.

    Accepts named presets (case-insensitive) or comma-separated raw INDRA types.

    Args:
        value: Named preset ("regulatory", "activation", "repression",
               "phosphorylation") or comma-separated raw INDRA types
               (e.g., "IncreaseAmount,Phosphorylation").
               None defaults to ALL_REGULATORY_TYPES.

    Returns:
        List of INDRA statement type strings.

    Raises:
        ValueError: If value is empty or cannot be parsed.
    """
    if value is None:
        return list(ALL_REGULATORY_TYPES)

    key = value.strip().lower()
    if key in STMT_TYPE_PRESETS:
        return list(STMT_TYPE_PRESETS[key])

    raw = [t.strip() for t in value.split(",") if t.strip()]
    if not raw:
        raise ValueError(f"Could not parse stmt_types: '{value}'")
    return raw


class RegulatorClass(Enum):
    """
    Functional class of upstream regulators.

    Each member maps to a curated gene list from INDRA's hgnc_client,
    sourced from HUGO Gene Nomenclature Committee annotations:

        TF              ~1,672 transcription factors
        KINASE          protein kinases (phosphorylation writers)
        PHOSPHATASE     protein phosphatases (phosphorylation erasers)
        E3_LIGASE       curated subset of E3 ubiquitin ligases (10 genes)
        RECEPTOR_KINASE curated subset of receptor tyrosine kinases (20 genes)

    Usage:
        >>> genes = get_regulator_class_genes({RegulatorClass.TF})
        >>> "TP53" in genes
        True
    """
    TF = "tf"
    KINASE = "kinase"
    PHOSPHATASE = "phosphatase"
    E3_LIGASE = "e3_ligase"
    RECEPTOR_KINASE = "receptor_kinase"


# Curated subset of well-known E3 ubiquitin ligases.
# NOTE: This is NOT exhaustive — there are ~600+ E3 ligases in the human genome.
# This curated set covers major oncology/neurodegeneration-relevant E3 ligases.
_E3_LIGASE_GENES: Set[str] = {
    "MDM2", "TRIM21", "SMURF1", "SMURF2", "NEDD4",
    "ITCH", "PARKIN", "VHL", "CHIP", "RNF4",
}

# Curated subset of receptor tyrosine kinases (RTKs).
# These are kinases that also function as cell-surface receptors, a biologically
# distinct class from cytoplasmic kinases. Relevant for targeted therapy studies.
_RECEPTOR_KINASE_GENES: Set[str] = {
    "EGFR", "ERBB2", "FGFR1", "FGFR2", "FGFR3",
    "PDGFRA", "PDGFRB", "KIT", "FLT3", "MET",
    "RET", "ALK", "ROS1", "NTRK1", "NTRK2",
    "NTRK3", "IGF1R", "INSR", "VEGFR1", "VEGFR2",
}


def get_regulator_class_genes(classes: Set[RegulatorClass]) -> Set[str]:
    """
    Return union of gene symbols for the specified regulator classes.

    Uses INDRA's hgnc_client curated lists (loaded from bundled CSV resources,
    no Neo4j connection required).

    Args:
        classes: Set of RegulatorClass enum members.

    Returns:
        Set of gene symbols belonging to any of the specified classes.

    Raises:
        ImportError: If INDRA is not installed.
    """
    if not INDRA_AVAILABLE:
        raise ImportError("INDRA package required for regulator class lists")

    _CLASS_TO_LIST = {
        RegulatorClass.TF: hgnc_client.tfs,
        RegulatorClass.KINASE: hgnc_client.kinases,
        RegulatorClass.PHOSPHATASE: hgnc_client.phosphatases,
        RegulatorClass.E3_LIGASE: _E3_LIGASE_GENES,
        RegulatorClass.RECEPTOR_KINASE: _RECEPTOR_KINASE_GENES,
    }

    result: Set[str] = set()
    for cls in classes:
        gene_list = _CLASS_TO_LIST[cls]
        result.update(gene_list)
        logger.info(f"RegulatorClass.{cls.name}: {len(gene_list)} genes")

    logger.info(f"Combined regulator class filter: {len(result)} unique genes")
    return result


@dataclass(frozen=True)
class INDRAEdge:
    """
    Single regulator->target causal relationship from INDRA CoGEx.

    Represents a mechanistic causal relationship between an upstream regulator
    (TF, kinase, signaling molecule, etc.) and a downstream target gene,
    aggregated from multiple evidence sources.

    Attributes:
        regulator_id: Upstream regulator identifier (namespace, id)
        regulator_name: Human-readable regulator name (gene symbol)
        target_id: Target gene identifier (namespace, id)
        target_name: Human-readable target name (gene symbol)
        regulation_type: Direction of regulation ("activation" or "repression")
        evidence_count: Number of supporting evidence pieces
        stmt_hash: Unique INDRA statement hash (for provenance tracking)
        source_counts: JSON dict of evidence sources -> counts

    Scientific Interpretation:
        - evidence_count >= 2: Multiple independent sources support relationship
        - evidence_count >= 5: High-confidence relationship
        - evidence_count >= 10: Very well-established relationship

        Source counts breakdown (e.g., {"reach": 3, "sparser": 2, "trrust": 1})
        helps assess evidence diversity:
        - Multiple text mining systems: Robust text evidence
        - Includes databases (trrust, signor): Curated experimental evidence
    """
    regulator_id: GeneId
    regulator_name: str
    target_id: GeneId
    target_name: str
    regulation_type: Literal["activation", "repression"]
    evidence_count: int
    stmt_hash: int
    source_counts: str  # JSON string

    @property
    def source_counts_dict(self) -> Dict[str, int]:
        """Parse source_counts JSON string to dictionary."""
        try:
            return json.loads(self.source_counts)
        except (json.JSONDecodeError, TypeError):
            return {}

    @property
    def num_sources(self) -> int:
        """Number of distinct evidence sources supporting this relationship."""
        return len(self.source_counts_dict)


@dataclass(frozen=True)
class INDRAModule:
    """
    Complete regulatory module for a single upstream regulator.

    A regulatory module consists of an upstream regulator and all its validated
    downstream targets within a specific gene universe (genes in your dataset).

    Attributes:
        regulator_id: Regulator identifier (namespace, id)
        regulator_name: Human-readable regulator name (gene symbol)
        targets: List of all INDRA edges for this regulator

    Properties:
        indra_targets: Set of all downstream target gene IDs
        indra_target_names: Set of downstream target gene names
        activated_targets: Set of activated target gene IDs
        repressed_targets: Set of repressed target gene IDs

    Scientific Interpretation:
        Regulatory modules represent the "sphere of influence" of a regulator
        within your experimental context. Large modules (>50 targets) suggest:
        - Master regulators (e.g., TP53, MYC, AKT1)
        - Hub nodes in regulatory/signaling networks
        - Key drivers of cellular state transitions
    """
    regulator_id: GeneId
    regulator_name: str
    targets: List[INDRAEdge] = field(default_factory=list)

    @property
    def indra_targets(self) -> Set[GeneId]:
        """INDRA-sourced downstream target gene IDs."""
        return {edge.target_id for edge in self.targets}

    @property
    def indra_target_names(self) -> Set[str]:
        """INDRA-sourced downstream target names."""
        return {edge.target_name for edge in self.targets}

    @property
    def activated_targets(self) -> Set[GeneId]:
        """Target gene IDs that are activated by this regulator."""
        return {edge.target_id for edge in self.targets if edge.regulation_type == "activation"}

    @property
    def repressed_targets(self) -> Set[GeneId]:
        """Target gene IDs that are repressed by this regulator."""
        return {edge.target_id for edge in self.targets if edge.regulation_type == "repression"}


class CoGExClient:
    """
    Client for querying INDRA CoGEx Neo4j knowledge graph.

    Provides high-level interface to INDRA CoGEx for querying transcription factor
    regulatory relationships. Handles connection management, credential loading,
    and query construction.

    Connection Strategy:
        Lazy connection - client only connects when first query is made.
        Credential precedence:
        1. Explicit parameters (url, user, password)
        2. Environment variables (INDRA_NEO4J_URL, INDRA_NEO4J_USER, INDRA_NEO4J_PASSWORD)
        3. .env file specified in env_file parameter

    Usage:
        >>> # Initialize with .env file
        >>> client = CoGExClient(env_file=Path("~/.env"))
        >>>
        >>> # Initialize with explicit credentials
        >>> client = CoGExClient(
        ...     url="bolt://host:7687",
        ...     user="neo4j",
        ...     password="secret"
        ... )
        >>>
        >>> # Test connection
        >>> if client.ping():
        ...     print("Connected!")
        >>>
        >>> # Query downstream targets
        >>> targets = client.get_downstream_targets(
        ...     regulator=("HGNC", "11998"),
        ...     stmt_types=["IncreaseAmount", "DecreaseAmount"],
        ...     min_evidence=2
        ... )
    """

    def __init__(
        self,
        url: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        env_file: Optional[Path] = None
    ):
        """
        Initialize CoGEx client with lazy connection.

        Args:
            url: Neo4j bolt URL (e.g., "bolt://host:7687")
            user: Neo4j username
            password: Neo4j password
            env_file: Path to .env file with credentials
                Expected format:
                    INDRA_NEO4J_URL=bolt://...
                    INDRA_NEO4J_USER=neo4j
                    INDRA_NEO4J_PASSWORD=secret

        Raises:
            ImportError: If indra_cogex package not installed
        """
        if not INDRA_AVAILABLE:
            raise ImportError(
                "indra_cogex package required. Install with: "
                "pip install git+https://github.com/indralab/indra_cogex.git"
            )

        self._url = url
        self._user = user
        self._password = password
        self._env_file = env_file
        self._client: Optional[Neo4jClient] = None

    def _load_credentials(self) -> Tuple[str, str, str]:
        """
        Load Neo4j credentials with fallback logic.

        Precedence:
        1. Explicit parameters (self._url, self._user, self._password)
        2. Environment variables
        3. .env file

        Returns:
            Tuple of (url, user, password)

        Raises:
            ValueError: If credentials cannot be loaded from any source
        """
        # Try explicit parameters first
        if self._url and self._user and self._password:
            logger.info("Using explicit credentials")
            return self._url, self._user, self._password

        # Try environment variables
        url = os.getenv("INDRA_NEO4J_URL")
        user = os.getenv("INDRA_NEO4J_USER")
        password = os.getenv("INDRA_NEO4J_PASSWORD")

        if url and user and password:
            logger.info("Using credentials from environment variables")
            return url, user, password

        # Try .env file
        if self._env_file:
            env_path = Path(self._env_file).expanduser()
            if env_path.exists():
                try:
                    from dotenv import load_dotenv
                    load_dotenv(env_path)

                    url = os.getenv("INDRA_NEO4J_URL")
                    user = os.getenv("INDRA_NEO4J_USER")
                    password = os.getenv("INDRA_NEO4J_PASSWORD")

                    if url and user and password:
                        logger.info(f"Using credentials from .env file: {env_path}")
                        return url, user, password
                except ImportError:
                    logger.warning("python-dotenv not installed, cannot load .env file")

        raise ValueError(
            "Could not load INDRA CoGEx credentials. Please provide either:\n"
            "1. Explicit parameters: CoGExClient(url=..., user=..., password=...)\n"
            "2. Environment variables: INDRA_NEO4J_URL, INDRA_NEO4J_USER, INDRA_NEO4J_PASSWORD\n"
            "3. .env file: CoGExClient(env_file=Path('~/.env'))"
        )

    def _get_client(self, force_reconnect: bool = False) -> Neo4jClient:
        """
        Get or create Neo4j client (lazy initialization).

        Args:
            force_reconnect: If True, discard existing client and create a new one.
                Used by _execute_query to recover from dead connections.

        Returns:
            Connected Neo4jClient instance
        """
        if self._client is not None and not force_reconnect:
            return self._client

        url, user, password = self._load_credentials()
        self._client = Neo4jClient(url=url, auth=(user, password))
        logger.info(f"Connected to INDRA CoGEx at {url}")

        return self._client

    # Connection error keywords used by _execute_query to detect infrastructure
    # failures (as opposed to query logic errors like bad Cypher syntax).
    _CONNECTION_ERROR_KEYWORDS = (
        'connection', 'timeout', 'unavailable', 'refused', 'reset', 'broken',
    )

    def _execute_query(self, query: str, max_retries: int = 1, **params):
        """
        Execute a Cypher query with automatic reconnection on connection failure.

        Defence-in-depth strategy (ARCH-4-NOTE):
        1. If ``neo4j.exceptions`` typed classes are importable, catch
           ``ServiceUnavailable``, ``SessionExpired``, and ``TransientError``
           first — these are *definite* connection/transient failures.
        2. For any other ``Exception``, fall back to string-keyword matching
           against ``_CONNECTION_ERROR_KEYWORDS``.
        3. Non-connection errors (syntax, constraint, etc.) are never retried.

        Args:
            query: Cypher query string.
            max_retries: Number of retry attempts after a connection error
                (default 1, meaning at most 2 total attempts).
            **params: Query parameters forwarded to ``client.query_tx``.

        Returns:
            Query result rows from ``Neo4jClient.query_tx``.

        Raises:
            RuntimeError: If the query fails after all retries, or on a
                non-retryable error.
        """
        last_error = None

        for attempt in range(1 + max_retries):
            try:
                client = self._get_client()
                return client.query_tx(query, **params)
            except Exception as e:
                last_error = e

                # --- Classify the error ---
                is_connection_error = False

                # (a) Typed Neo4j exceptions — definite connection/transient
                if _NEO4J_EXCEPTIONS_AVAILABLE:
                    typed_classes = (ServiceUnavailable, SessionExpired, TransientError)
                    if isinstance(e, typed_classes):
                        is_connection_error = True

                # (b) Fallback: string-keyword heuristic
                if not is_connection_error:
                    error_str = str(e).lower()
                    is_connection_error = any(
                        word in error_str
                        for word in self._CONNECTION_ERROR_KEYWORDS
                    )

                if not is_connection_error:
                    raise  # Non-retryable (syntax, constraint, etc.)

                # Connection error — reset client and maybe retry
                logger.warning(
                    "Connection error on attempt %d/%d: %s",
                    attempt + 1, 1 + max_retries, e,
                )
                self._client = None  # force reconnect on next attempt

                if attempt >= max_retries:
                    raise RuntimeError(
                        f"Query failed after {1 + max_retries} attempts: {e}"
                    ) from e

        # Should never reach here, but satisfy type checkers
        raise RuntimeError(f"Query failed: {last_error}")  # pragma: no cover

    def ping(self) -> bool:
        """
        Test connection to INDRA CoGEx.

        Returns:
            True if connection successful, False otherwise

        Examples:
            >>> client = CoGExClient(env_file=Path("~/.env"))
            >>> if client.ping():
            ...     print("Connection successful")
            ... else:
            ...     print("Connection failed")
        """
        try:
            client = self._get_client()
            # Simple query to test connection
            result = client.query_tx("RETURN 1 as test")
            return len(result) > 0
        except Exception as e:
            logger.error(f"Failed to connect to INDRA CoGEx: {e}")
            return False

    def get_downstream_targets(
        self,
        regulator: GeneId,
        stmt_types: Optional[List[str]] = None,
        min_evidence: int = 2
    ) -> List[INDRAEdge]:
        """
        Query all downstream targets for any upstream regulator from INDRA.

        Executes Cypher query to find all regulator->target causal relationships
        in INDRA CoGEx. Works for any upstream gene (TF, kinase, etc.).

        Args:
            regulator: Upstream regulator identifier (namespace, id)
                Example: ("HGNC", "11998") for TP53
            stmt_types: List of INDRA statement types to query
                Options: "IncreaseAmount", "DecreaseAmount", "Activation", "Inhibition"
                Default: All regulatory types
            min_evidence: Minimum evidence count threshold
                Default: 2 for high-confidence relationships (≥2 independent sources)

        Returns:
            List of INDRAEdge objects (INDRA-sourced relationships)

        Examples:
            >>> # Query TP53 targets
            >>> targets = client.get_downstream_targets(("HGNC", "11998"))
            >>>
            >>> # Query AKT1 (kinase) targets
            >>> akt_targets = client.get_downstream_targets(("HGNC", "391"))
        """
        if stmt_types is None:
            stmt_types = list(ALL_REGULATORY_TYPES)

        # Normalize regulator ID to CURIE format for Neo4j query
        # INDRA CoGEx uses lowercase normalized CURIEs (e.g., "hgnc:11998")
        regulator_curie = norm_id(regulator[0], regulator[1])

        # Construct Cypher query
        query = """
            MATCH p=(reg:BioEntity)-[r:indra_rel]->(target:BioEntity)
            WHERE reg.id = $regulator_id
            AND r.stmt_type IN $stmt_types
            AND r.evidence_count >= $min_evidence
            RETURN reg.id as regulator_id, reg.name as regulator_name,
                   target.id as target_id, target.name as target_name,
                   r.stmt_type as stmt_type, r.evidence_count as evidence_count,
                   r.stmt_hash as stmt_hash, r.source_counts as source_counts
        """

        try:
            results = self._execute_query(
                query,
                regulator_id=regulator_curie,
                stmt_types=stmt_types,
                min_evidence=min_evidence
            )

            # Parse results into INDRAEdge objects
            edges = []
            for row in results:
                # Determine regulation type
                stmt_type = row[4]  # stmt_type column
                if stmt_type in ACTIVATION_TYPES:
                    reg_type = "activation"
                elif stmt_type in REPRESSION_TYPES:
                    reg_type = "repression"
                else:
                    logger.warning(f"Unknown statement type: {stmt_type}")
                    continue

                # Parse IDs (format: "hgnc:1234" -> ("HGNC", "1234"))
                # CoGEx returns lowercase normalized CURIEs, normalize to uppercase
                reg_id_str = row[0]  # regulator_id
                target_id_str = row[2]  # target_id

                reg_ns, reg_id_val = reg_id_str.split(":", 1)
                target_ns, target_id_val = target_id_str.split(":", 1)

                # Normalize namespace to uppercase for consistent ID comparison
                reg_ns = reg_ns.upper()
                target_ns = target_ns.upper()

                edge = INDRAEdge(
                    regulator_id=(reg_ns, reg_id_val),
                    regulator_name=row[1],  # regulator_name
                    target_id=(target_ns, target_id_val),
                    target_name=row[3],  # target_name
                    regulation_type=reg_type,
                    evidence_count=row[5],  # evidence_count
                    stmt_hash=row[6],  # stmt_hash
                    source_counts=row[7] if row[7] else "{}"  # source_counts
                )
                edges.append(edge)

            logger.info(f"Found {len(edges)} INDRA targets for {regulator[0]}:{regulator[1]}")
            return edges

        except Exception as e:
            logger.error(f"Failed to query downstream targets: {e}")
            raise RuntimeError(f"Query failed: {e}") from e

    # Chunk size for batching large CURIE lists in Neo4j queries
    CURIE_CHUNK_SIZE = 5000

    def discover_regulators(
        self,
        gene_universe: List[str],
        stmt_types: Optional[List[str]] = None,
        min_evidence: int = 2,
        min_targets: int = 5,
        regulator_types: Optional[Set[str]] = None,
        max_results: int = 100_000,
    ) -> Dict[str, List[INDRAEdge]]:
        """
        Discover all upstream regulators for genes in universe (reverse query).

        Instead of querying TF-by-TF with a hand-picked list, this performs a
        single batch Cypher query to find ALL upstream regulators that target
        genes in your dataset. Much more efficient for discovering relevant
        regulators from INDRA rather than starting with a predefined list.

        Query strategy: Single reverse query that finds all regulator->target
        edges where target is in gene_universe, then groups by regulator.

        Complexity: O(1) Neo4j queries regardless of gene universe size
        (vs O(n) queries for n hand-picked regulators)

        Args:
            gene_universe: List of gene symbols present in your dataset
            stmt_types: INDRA statement types to query
                Options: "IncreaseAmount", "DecreaseAmount", "Activation", "Inhibition"
                Default: All regulatory types
            min_evidence: Minimum evidence count per relationship
                Default: 2 for high-confidence relationships (≥2 independent sources)
            min_targets: Minimum number of UNIQUE target genes in universe to include regulator
                Default: 5 (filter out regulators with few targets in your data)
                This filters post-query to keep only significant regulators.
                NOTE: Counts unique target genes, not edges (a target can have multiple edges
                from different statement types like IncreaseAmount + Activation)
            regulator_types: Optional set of regulator namespaces to include
                Default: None (include all, typically HGNC for human genes)
                Example: {"HGNC"} to limit to human genes
            max_results: Maximum total result rows across all chunks.
                Default: 100,000. A warning is emitted if this limit is reached,
                indicating that results may be truncated.

        Returns:
            Dict mapping regulator symbol -> list of INDRAEdge objects
            Only includes regulators with >= min_targets targets in universe

        Scientific Interpretation:
            This reverse query approach discovers which regulators are actually
            relevant to your data, rather than starting with assumptions about
            which TFs matter. Regulators with many targets in your dataset are
            more likely to be biologically relevant.

        Examples:
            >>> # Discover all regulators for genes in dataset
            >>> gene_universe = ["MDM2", "CDKN1A", "BAX", "BBC3", ...]
            >>> regulators = client.discover_regulators(
            ...     gene_universe=gene_universe,
            ...     min_evidence=2,
            ...     min_targets=10  # Only regulators with 10+ targets
            ... )
            >>> for reg_name, edges in regulators.items():
            ...     print(f"{reg_name}: {len(edges)} targets in dataset")
        """
        if stmt_types is None:
            stmt_types = list(ALL_REGULATORY_TYPES)

        if not INDRA_AVAILABLE:
            raise ImportError("INDRA package required for gene name resolution")

        # Resolve gene universe to normalized CURIEs for Neo4j query
        # INDRA CoGEx uses lowercase normalized CURIEs (e.g., "hgnc:1234")
        target_curies = []
        name_to_curie = {}  # Map for reverse lookup

        for gene_name in gene_universe:
            # Try original name first, then upper-cased for case-insensitive resolution
            resolved = False
            for candidate in dict.fromkeys([gene_name, gene_name.upper()]):
                try:
                    hgnc_id = hgnc_client.get_current_hgnc_id(candidate)
                    if hgnc_id:
                        curie = norm_id("HGNC", hgnc_id)
                        target_curies.append(curie)
                        name_to_curie[gene_name] = curie
                        resolved = True
                        break
                except Exception:
                    continue
            if not resolved:
                logger.info("Could not resolve gene name to HGNC ID: %s", gene_name)

        if not target_curies:
            logger.warning("No genes in universe could be resolved to HGNC IDs")
            return {}

        logger.info(f"Resolved {len(target_curies)}/{len(gene_universe)} genes for reverse query")

        # Chunked Cypher query: batch large CURIE lists to avoid Neo4j
        # query size limits and unbounded memory usage (ARCH-14).
        query = """
            MATCH (reg:BioEntity)-[r:indra_rel]->(target:BioEntity)
            WHERE target.id IN $target_ids
            AND r.stmt_type IN $stmt_types
            AND r.evidence_count >= $min_evidence
            RETURN reg.id as regulator_id, reg.name as regulator_name,
                   target.id as target_id, target.name as target_name,
                   r.stmt_type as stmt_type, r.evidence_count as evidence_count,
                   r.stmt_hash as stmt_hash, r.source_counts as source_counts
        """

        try:
            # Chunk the CURIE list to avoid Neo4j query size limits
            # Uses _execute_query for auto-reconnection (ARCH-4 + ARCH-14)
            all_results = []
            chunk_size = self.CURIE_CHUNK_SIZE
            for i in range(0, len(target_curies), chunk_size):
                chunk = target_curies[i:i + chunk_size]
                chunk_results = self._execute_query(
                    query,
                    target_ids=chunk,
                    stmt_types=stmt_types,
                    min_evidence=min_evidence
                )
                all_results.extend(chunk_results)
                if len(all_results) >= max_results:
                    logger.warning(
                        "discover_regulators hit max_results limit (%d). "
                        "Results may be truncated. Consider increasing "
                        "max_results or narrowing the gene universe.",
                        max_results,
                    )
                    all_results = all_results[:max_results]
                    break

            if len(target_curies) > chunk_size:
                logger.info(
                    "Chunked %d CURIEs into %d batches of <= %d",
                    len(target_curies),
                    (len(target_curies) + chunk_size - 1) // chunk_size,
                    chunk_size,
                )

            # Group edges by regulator
            regulator_edges: Dict[str, List[INDRAEdge]] = {}

            for row in all_results:
                # Determine regulation type
                stmt_type = row[4]  # stmt_type column
                if stmt_type in ACTIVATION_TYPES:
                    reg_type = "activation"
                elif stmt_type in REPRESSION_TYPES:
                    reg_type = "repression"
                else:
                    continue  # Skip unknown statement types

                # Parse IDs (format: "hgnc:1234" -> ("HGNC", "1234"))
                reg_id_str = row[0]  # regulator_id
                target_id_str = row[2]  # target_id

                reg_ns, reg_id_val = reg_id_str.split(":", 1)
                target_ns, target_id_val = target_id_str.split(":", 1)

                # Normalize namespace to uppercase
                reg_ns = reg_ns.upper()
                target_ns = target_ns.upper()

                # Optional filter by regulator namespace
                if regulator_types is not None and reg_ns not in regulator_types:
                    continue

                regulator_name = row[1]  # regulator_name

                edge = INDRAEdge(
                    regulator_id=(reg_ns, reg_id_val),
                    regulator_name=regulator_name,
                    target_id=(target_ns, target_id_val),
                    target_name=row[3],  # target_name
                    regulation_type=reg_type,
                    evidence_count=row[5],  # evidence_count
                    stmt_hash=row[6],  # stmt_hash
                    source_counts=row[7] if row[7] else "{}"
                )

                if regulator_name not in regulator_edges:
                    regulator_edges[regulator_name] = []
                regulator_edges[regulator_name].append(edge)

            # Filter to regulators with minimum number of UNIQUE targets
            # IMPORTANT: Count unique target genes, not edges (a target can have multiple edges
            # from different statement types like IncreaseAmount + Activation to same gene)
            filtered_regulators = {}
            for name, edges in regulator_edges.items():
                unique_targets = {edge.target_id for edge in edges}  # Use target_id for uniqueness
                if len(unique_targets) >= min_targets:
                    filtered_regulators[name] = edges

            logger.info(
                f"Discovered {len(filtered_regulators)} regulators with >= {min_targets} unique targets "
                f"(from {len(regulator_edges)} total regulators)"
            )

            return filtered_regulators

        except Exception as e:
            logger.error(f"Failed to discover regulators: {e}")
            raise RuntimeError(f"Reverse query failed: {e}") from e

    def close(self):
        """Close Neo4j connection."""
        if self._client is not None:
            # Neo4jClient may use __del__ for cleanup instead of close()
            if hasattr(self._client, 'close'):
                self._client.close()
            self._client = None
            logger.info("Closed INDRA CoGEx connection")


class INDRAModuleExtractor:
    """
    Extract INDRA regulatory modules constrained to gene universe.

    High-level interface for extracting regulatory modules from INDRA CoGEx,
    with filtering to genes present in your experimental dataset.

    Works with ANY upstream regulator (TFs, kinases, signaling molecules, etc.),
    not just transcription factors.

    Key functionality:
    - Gene name resolution (symbol -> HGNC ID)
    - Multi-regulator batch queries
    - Universe filtering (keep only targets in your data)
    - Module construction with INDRA-native terminology

    Usage:
        >>> from cliquefinder.knowledge.cogex import CoGExClient, INDRAModuleExtractor
        >>> from pathlib import Path
        >>>
        >>> # Initialize
        >>> client = CoGExClient(env_file=Path("~/.env"))
        >>> extractor = INDRAModuleExtractor(client)
        >>>
        >>> # Define regulators (not just TFs!) and gene universe
        >>> regulators = ["TP53", "AKT1", "MAPK1"]  # TF + kinases
        >>> gene_universe = ["MDM2", "CDKN1A", "BAX", "BBC3"]
        >>>
        >>> # Extract modules
        >>> modules = extractor.get_regulator_modules(
        ...     regulators=regulators,
        ...     gene_universe=gene_universe,
        ...     min_evidence=2
        ... )
        >>>
        >>> # Analyze results
        >>> for module in modules:
        ...     print(f"{module.regulator_name}: {len(module.indra_targets)} INDRA targets")
    """

    def __init__(self, client: CoGExClient, id_mapper: Optional['MyGeneInfoMapper'] = None):
        """
        Initialize extractor with CoGEx client and optional ID mapper.

        Args:
            client: Connected CoGExClient instance
            id_mapper: Optional MyGeneInfoMapper for fallback ID resolution
        """
        self.client = client
        self.id_mapper = id_mapper
        self._gene_cache: Dict[str, Optional[GeneId]] = {}
        self._mygene_client = None  # Lazy singleton for MyGene.info client

    def _get_mygene_client(self):
        """
        Lazy singleton for MyGene.info client.

        Avoids instantiating a new ``mygene.MyGeneInfo()`` on every call to
        ``resolve_gene_name``, which would create redundant HTTP sessions.
        """
        if self._mygene_client is None:
            import mygene
            self._mygene_client = mygene.MyGeneInfo()
        return self._mygene_client

    def resolve_gene_name(self, name: str) -> Optional[GeneId]:
        """
        Resolve gene symbol OR UniProt accession to HGNC identifier.

        Results are cached in ``self._gene_cache`` so repeated lookups for the
        same gene name are O(1) dict lookups instead of redundant HGNC/UniProt
        queries.

        Resolution strategy (in order):
        1. INDRA HGNC client - for gene symbols (fast, authoritative)
        2. INDRA UniProt client - for UniProt accessions (direct mapping)
        3. MyGene.info batch API - fallback for aliases and edge cases

        Args:
            name: Gene symbol (e.g., "TP53") or UniProt accession (e.g., "P04637")

        Returns:
            GeneId tuple (namespace, id) or None if not found
            Example: ("HGNC", "11998") for TP53 or P04637
        """
        if name in self._gene_cache:
            return self._gene_cache[name]

        result = self._resolve_gene_name_uncached(name)
        self._gene_cache[name] = result
        return result

    def _resolve_gene_name_uncached(self, name: str) -> Optional[GeneId]:
        """
        Resolve gene symbol OR UniProt accession to HGNC identifier (uncached).

        This contains the actual resolution logic; ``resolve_gene_name`` adds a
        caching layer on top.
        """
        # Strategy 1: Try as gene symbol via INDRA HGNC client (fast, authoritative)
        # Try original name first, then upper-cased for case-insensitive resolution
        for candidate in dict.fromkeys([name, name.upper()]):
            try:
                hgnc_id = hgnc_client.get_current_hgnc_id(candidate)
                if hgnc_id:
                    return ("HGNC", hgnc_id)
            except Exception:
                pass

        # Strategy 2: Try as UniProt accession via INDRA uniprot_client (direct mapping)
        # UniProt accessions are 6-10 alphanumeric chars starting with letter
        import re
        if re.match(r'^[A-Z][A-Z0-9]{5,9}$', name):
            try:
                from indra.databases import uniprot_client
                hgnc_id = uniprot_client.get_hgnc_id(name)
                if hgnc_id:
                    return ("HGNC", hgnc_id)
            except Exception:
                pass

        # Strategy 3: Fallback to MyGene.info querymany (comprehensive, slower)
        # Use querymany with proper scopes for batch-friendly resolution
        if self.id_mapper:
            try:
                mg = self._get_mygene_client()

                # Use querymany (not query) with proper scopes
                # scopes='uniprot' for UniProt, 'symbol,alias' for gene names
                results = mg.querymany(
                    [name],
                    scopes='symbol,alias,uniprot',
                    fields='symbol,HGNC',
                    species='human',
                    verbose=False
                )

                if results and len(results) > 0:
                    hit = results[0]
                    if not hit.get('notfound'):
                        # Prefer HGNC ID if available
                        if 'HGNC' in hit:
                            hgnc_val = hit['HGNC']
                            if isinstance(hgnc_val, list):
                                hgnc_id = str(hgnc_val[0]) if hgnc_val else None
                            else:
                                hgnc_id = str(hgnc_val)
                            if hgnc_id:
                                return ("HGNC", hgnc_id)
                        # Fallback: resolve symbol to HGNC via hgnc_client
                        if 'symbol' in hit:
                            symbol = hit['symbol']
                            hgnc_id = hgnc_client.get_current_hgnc_id(symbol)
                            if hgnc_id:
                                return ("HGNC", hgnc_id)
            except Exception as e:
                logger.warning(f"MyGene fallback failed for {name}: {e}")

        logger.warning(f"Could not resolve gene name: {name}")
        return None

    def get_indra_targets(
        self,
        regulator: GeneId,
        gene_universe: Set[GeneId],
        min_evidence: int = 2,
        stmt_types: Optional[List[str]] = None,
    ) -> Set[GeneId]:
        """
        Get INDRA targets: downstream genes within gene universe.

        INDRA targets are genes that:
        1. Are downstream of the regulator in INDRA CoGEx
        2. Are present in the gene_universe (i.e., in your dataset)

        This is the "knowledge-based" gene set from INDRA.

        Args:
            regulator: Upstream regulator identifier (TF, kinase, etc.)
            gene_universe: Set of gene IDs present in your dataset
            min_evidence: Minimum evidence count (default 2 for high-confidence)

        Returns:
            Set of target gene IDs (subset of gene_universe)

        Examples:
            >>> regulator = ("HGNC", "11998")  # TP53
            >>> universe = {("HGNC", "1787"), ("HGNC", "581")}  # CDKN1A, BAX
            >>> targets = extractor.get_indra_targets(regulator, universe, min_evidence=2)
            >>> print(f"TP53 regulates {len(targets)} genes in dataset per INDRA")
        """
        # Query all downstream targets from INDRA
        edges = self.client.get_downstream_targets(
            regulator=regulator,
            stmt_types=stmt_types if stmt_types is not None else list(ALL_REGULATORY_TYPES),
            min_evidence=min_evidence
        )

        # Filter to gene universe
        targets_in_universe = {
            edge.target_id for edge in edges
            if edge.target_id in gene_universe
        }

        return targets_in_universe

    def get_regulator_modules(
        self,
        regulators: List[str],
        gene_universe: List[str],
        min_evidence: int = 2,
        stmt_types: Optional[List[str]] = None,
    ) -> List[INDRAModule]:
        """
        Extract INDRA regulatory modules for multiple upstream regulators.

        Main entry point for building regulatory modules. For each regulator:
        1. Resolve gene symbol -> HGNC ID
        2. Query downstream targets from INDRA CoGEx
        3. Filter targets to gene universe (genes in your dataset)
        4. Construct INDRAModule object

        Args:
            regulators: List of regulator gene symbols (e.g., ["TP53", "AKT1"])
            gene_universe: List of gene symbols in your dataset
            min_evidence: Minimum evidence count for relationships
                2 = require at least 2 independent sources (default, recommended)
                5 = high-confidence relationships only

        Returns:
            List of INDRAModule objects (one per regulator)
            Regulators that cannot be resolved or have no targets are omitted

        Examples:
            >>> # Extract modules for key regulators (not just TFs!)
            >>> regulators = ["TP53", "AKT1", "MAPK1"]
            >>> gene_universe = ["MDM2", "CDKN1A", "BAX", "BBC3", ...]
            >>> modules = extractor.get_regulator_modules(
            ...     regulators=regulators,
            ...     gene_universe=gene_universe,
            ...     min_evidence=2
            ... )
            >>>
            >>> # Summarize INDRA targets
            >>> for module in modules:
            ...     print(f"{module.regulator_name}: {len(module.indra_targets)} INDRA targets")
        """
        logger.info(f"Extracting INDRA regulatory modules for {len(regulators)} regulators")
        logger.info(f"Gene universe size: {len(gene_universe)}")

        # Resolve gene universe to IDs
        universe_ids: Set[GeneId] = set()
        for gene_name in gene_universe:
            gene_id = self.resolve_gene_name(gene_name)
            if gene_id:
                universe_ids.add(gene_id)

        logger.info(f"Resolved {len(universe_ids)}/{len(gene_universe)} genes in universe")

        # Extract modules for each regulator
        modules = []
        for reg_name in regulators:
            # Resolve regulator name
            reg_id = self.resolve_gene_name(reg_name)
            if not reg_id:
                logger.warning(f"Could not resolve regulator: {reg_name}")
                continue

            # Query downstream targets from INDRA
            try:
                edges = self.client.get_downstream_targets(
                    regulator=reg_id,
                    stmt_types=stmt_types if stmt_types is not None else list(ALL_REGULATORY_TYPES),
                    min_evidence=min_evidence
                )

                # Filter to gene universe
                filtered_edges = [
                    edge for edge in edges
                    if edge.target_id in universe_ids
                ]

                if filtered_edges:
                    module = INDRAModule(
                        regulator_id=reg_id,
                        regulator_name=reg_name,
                        targets=filtered_edges
                    )
                    modules.append(module)
                    logger.info(
                        f"  {reg_name}: {len(filtered_edges)} INDRA targets "
                        f"({len(module.activated_targets)} activated, "
                        f"{len(module.repressed_targets)} repressed)"
                    )
                else:
                    logger.warning(f"  {reg_name}: no INDRA targets in gene universe")

            except Exception as e:
                logger.error(f"Failed to extract module for {reg_name}: {e}")
                continue

        logger.info(f"Successfully extracted {len(modules)} INDRA regulatory modules")
        return modules

    def discover_modules(
        self,
        gene_universe: List[str],
        min_evidence: int = 2,
        min_targets: int = 10,
        max_targets: Optional[int] = None,
        max_regulators: Optional[int] = None,
        regulator_classes: Optional[Set[RegulatorClass]] = None,
        stmt_types: Optional[List[str]] = None,
    ) -> List[INDRAModule]:
        """
        Discover regulatory modules from INDRA without pre-specifying regulators.

        This is the RECOMMENDED workflow for unbiased discovery. Instead of
        starting with a hand-picked list of TFs, this method:
        1. Performs a single reverse query to find ALL regulators targeting genes
           in your dataset
        2. Filters to regulators with sufficient targets (min_targets)
        3. Optionally filters out regulators with too many targets (max_targets)
        4. Returns INDRAModule objects sorted by number of targets (descending)

        This approach:
        - Is unbiased (no predefined regulator assumptions)
        - Is efficient (single batch query vs O(n) queries)
        - Discovers regulators relevant to YOUR data

        Args:
            gene_universe: List of gene symbols present in your dataset
            min_evidence: Minimum evidence count per relationship
                Default: 2 for high-confidence relationships (≥2 independent sources)
            min_targets: Minimum UNIQUE target genes in universe to include regulator
                Default: 10 (filter out regulators with few targets)
                Higher values give more significant regulators.
                NOTE: Counts unique target genes, not edges (a target can have multiple edges)
            max_targets: Maximum UNIQUE target genes in universe to include regulator
                Default: None (no upper limit)
                Useful to exclude hub regulators (e.g., TP53 with 1000+ targets)
                that may produce non-specific modules.
                NOTE: Counts unique target genes, not edges
            max_regulators: Maximum number of top regulators to return
                Default: None (return all discovered regulators)
                Set to limit analysis to top N regulators by target count
            regulator_classes: Optional set of RegulatorClass enum members to
                restrict discovery to (e.g., {RegulatorClass.TF} for TFs only).
                Default: None (all functional classes).
                Applied after sort-by-targets, before max_targets/max_regulators.
            stmt_types: List of INDRA statement types to query.
                Default: None (uses ALL_REGULATORY_TYPES).
                Use resolve_stmt_types() for named presets.

        Returns:
            List of INDRAModule objects, sorted by number of unique targets (descending)

        Examples:
            >>> # Discover all significant regulators for genes in dataset
            >>> modules = extractor.discover_modules(
            ...     gene_universe=gene_universe,
            ...     min_evidence=2,
            ...     min_targets=20,  # Only regulators with 20+ targets
            ...     max_regulators=50  # Top 50 regulators
            ... )
            >>> for module in modules[:10]:
            ...     print(f"{module.regulator_name}: {len(module.indra_targets)} targets")
        """
        logger.info(f"Discovering regulatory modules from gene universe of {len(gene_universe)} genes")

        # Use the efficient reverse query
        regulator_edges = self.client.discover_regulators(
            gene_universe=gene_universe,
            stmt_types=stmt_types,
            min_evidence=min_evidence,
            min_targets=min_targets,
            regulator_types={"HGNC"}  # Limit to human genes
        )

        if not regulator_edges:
            logger.warning("No regulators discovered with specified criteria")
            return []

        # Convert to INDRAModule objects
        modules = []
        for reg_name, edges in regulator_edges.items():
            # Get regulator ID from first edge
            reg_id = edges[0].regulator_id

            module = INDRAModule(
                regulator_id=reg_id,
                regulator_name=reg_name,
                targets=edges
            )
            modules.append(module)

        # Sort by number of UNIQUE targets (descending)
        # Use indra_targets (set of unique target IDs) not targets (list of edges)
        modules.sort(key=lambda m: len(m.indra_targets), reverse=True)

        # Apply regulator class filter if specified (TF, kinase, phosphatase)
        if regulator_classes is not None:
            class_genes = get_regulator_class_genes(regulator_classes)
            pre_filter_count = len(modules)
            modules = [m for m in modules if m.regulator_name in class_genes]
            n_filtered = pre_filter_count - len(modules)
            if n_filtered > 0:
                logger.info(
                    f"Regulator class filter: kept {len(modules)}/{pre_filter_count} "
                    f"(removed {n_filtered} not in {[c.value for c in regulator_classes]})"
                )

        # Apply max_targets filter if specified (exclude hub regulators)
        # Count unique targets, not edges
        if max_targets is not None:
            pre_filter_count = len(modules)
            modules = [m for m in modules if len(m.indra_targets) <= max_targets]
            n_filtered = pre_filter_count - len(modules)
            if n_filtered > 0:
                logger.info(
                    f"Filtered out {n_filtered} hub regulators with >{max_targets} unique targets"
                )

        # Apply max_regulators limit if specified
        if max_regulators is not None and len(modules) > max_regulators:
            modules = modules[:max_regulators]
            logger.info(f"Limited to top {max_regulators} regulators")

        if modules:
            logger.info(
                f"Discovered {len(modules)} regulatory modules "
                f"(top regulator: {modules[0].regulator_name} with {len(modules[0].indra_targets)} unique targets)"
            )
        else:
            logger.warning("No regulatory modules found after filtering")

        return modules
