# INDRA CoGEx Integration Specification

## Overview

This specification defines the integration between our ALS transcriptomics analysis pipeline and INDRA CoGEx, a Neo4j-based knowledge graph containing INDRA statements about biological relationships.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TF-DRIVEN REGULATORY MODULE DISCOVERY                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. INDRA CoGEx                    2. ALS Expression Data                   │
│  ┌────────────────────┐            ┌─────────────────────────┐              │
│  │ TF → Target edges  │            │ 60,664 genes × 511 smpl │              │
│  │ (IncreaseAmount/   │            │ Stratified:             │              │
│  │  DecreaseAmount)   │            │   CASE×Male (219)       │              │
│  └─────────┬──────────┘            │   CASE×Female (129)     │              │
│            │                       │   CTRL×Male (38)        │              │
│            ▼                       │   CTRL×Female (125)     │              │
│  ┌─────────────────────────────────┴─────────────────────────┐              │
│  │               CHILD SET TYPE 1: INDRA-REPORTED            │              │
│  │  TF → {targets present in expression data}                │              │
│  │  Filter: gene must exist in our 60,664 features           │              │
│  └─────────────────────────────────┬─────────────────────────┘              │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │               CHILD SET TYPE 2: CORRELATION-VALIDATED       │            │
│  │  Subset of Type 1 where all pairs are correlated            │            │
│  │  Build stratified correlation matrix → find cliques         │            │
│  │  CASE-specific, CTRL-specific, sex-stratified               │            │
│  └─────────────────────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Neo4j Connection

### Credentials
```
URL: bolt://indra-cogex-lb-b954b684556c373c.elb.us-east-1.amazonaws.com:7687
User: neo4j
Password: (from ~/workspace/indra-cogex/.env)
```

### Environment Variables
```bash
INDRA_NEO4J_URL=bolt://indra-cogex-lb-b954b684556c373c.elb.us-east-1.amazonaws.com:7687
INDRA_NEO4J_USER=neo4j
INDRA_NEO4J_PASSWORD=<password>
```

## Core Interface Contracts

### Entity Identifiers

INDRA uses grounded identifiers as tuples:
```python
# HGNC identifiers for human genes
tp53: Tuple[str, str] = ("HGNC", "11998")
brca1: Tuple[str, str] = ("HGNC", "1100")

# Normalized to CURIE format internally
norm_id("HGNC", "11998")  # → "hgnc:11998"
```

### Relation Types

For TF→target regulatory relationships, we care about `indra_rel` edges with specific statement types:

| Statement Type | Meaning | Direction |
|---------------|---------|-----------|
| `IncreaseAmount` | TF activates/upregulates target | TF → Target |
| `DecreaseAmount` | TF represses/downregulates target | TF → Target |
| `Activation` | General activation | TF → Target |
| `Inhibition` | General inhibition | TF → Target |

### Relation Data Schema

Each `indra_rel` edge contains:
```python
{
    "stmt_type": str,           # e.g., "IncreaseAmount"
    "stmt_hash": int,           # Unique statement hash
    "evidence_count": int,      # Number of supporting evidences
    "source_counts": str,       # JSON: {"source1": count, ...}
    "has_database_evidence": bool,
    "stmt_json": str,           # Full INDRA statement as JSON
}
```

## Module Design

### biocore/knowledge/cogex.py

```python
"""
INDRA CoGEx client for regulatory network queries.

This module provides a high-level interface for querying TF→target
regulatory relationships from the INDRA CoGEx knowledge graph.

Design Decisions:
    - Uses environment variables for credentials (INDRA_NEO4J_*)
    - Lazy connection initialization
    - Returns pandas DataFrames for easy integration with BioMatrix
    - Caches TF lookups for performance

Examples:
    >>> from biocore.knowledge.cogex import RegulatoryModuleExtractor
    >>> extractor = RegulatoryModuleExtractor()
    >>> modules = extractor.get_tf_regulatory_modules(
    ...     tf_genes=["TP53", "MYC", "STAT3"],
    ...     gene_universe=matrix.feature_ids
    ... )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Literal
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

# Type aliases
GeneId = Tuple[str, str]  # (namespace, identifier), e.g., ("HGNC", "11998")
GeneName = str

@dataclass
class RegulatoryEdge:
    """A single TF→target regulatory relationship."""
    tf_id: GeneId
    tf_name: str
    target_id: GeneId
    target_name: str
    regulation_type: Literal["activation", "repression"]
    evidence_count: int
    stmt_hash: int
    source_counts: Dict[str, int] = field(default_factory=dict)

@dataclass
class RegulatoryModule:
    """A TF and its regulatory targets."""
    tf_id: GeneId
    tf_name: str
    targets: List[RegulatoryEdge]

    @property
    def target_genes(self) -> Set[str]:
        """Gene names of all targets."""
        return {e.target_name for e in self.targets}

    @property
    def activated_targets(self) -> Set[str]:
        """Targets that are activated by this TF."""
        return {e.target_name for e in self.targets if e.regulation_type == "activation"}

    @property
    def repressed_targets(self) -> Set[str]:
        """Targets that are repressed by this TF."""
        return {e.target_name for e in self.targets if e.regulation_type == "repression"}

class CoGExClient:
    """
    Low-level client for INDRA CoGEx Neo4j queries.

    Wraps the indra_cogex.client.Neo4jClient with our specific needs.
    Uses lazy initialization and connection pooling.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        env_file: Optional[Path] = None,
    ):
        """
        Initialize CoGEx client.

        Credentials are resolved in order:
        1. Explicit parameters
        2. Environment variables (INDRA_NEO4J_*)
        3. env_file if provided

        Connection is lazy - not established until first query.
        """
        self._url = url
        self._user = user
        self._password = password
        self._env_file = env_file
        self._client = None

    def _load_credentials(self) -> Tuple[str, str, str]:
        """Load Neo4j credentials from available sources."""
        # Implementation loads from env vars or .env file
        pass

    def _ensure_connected(self):
        """Ensure Neo4j client is connected."""
        if self._client is None:
            from indra_cogex.client.neo4j_client import Neo4jClient
            url, user, password = self._load_credentials()
            self._client = Neo4jClient(url=url, auth=(user, password))

    def get_tf_targets(
        self,
        tf: GeneId,
        stmt_types: List[str] = ["IncreaseAmount", "DecreaseAmount"],
        min_evidence: int = 1,
    ) -> List[RegulatoryEdge]:
        """
        Get all targets regulated by a transcription factor.

        Parameters
        ----------
        tf : GeneId
            Transcription factor identifier, e.g., ("HGNC", "11998") for TP53
        stmt_types : List[str]
            INDRA statement types to include
        min_evidence : int
            Minimum evidence count threshold

        Returns
        -------
        List[RegulatoryEdge]
            All regulatory edges from this TF to targets
        """
        pass

    def get_tfs_for_targets(
        self,
        targets: List[GeneId],
        stmt_types: List[str] = ["IncreaseAmount", "DecreaseAmount"],
    ) -> Dict[GeneId, List[RegulatoryEdge]]:
        """
        Get all TFs that regulate the given targets.

        Useful for finding shared upstream regulators.
        """
        pass

    def query_regulatory_subnetwork(
        self,
        genes: List[GeneId],
        include_mediated: bool = False,
    ) -> List[RegulatoryEdge]:
        """
        Get all regulatory relationships within a gene set.
        """
        pass

class RegulatoryModuleExtractor:
    """
    High-level interface for extracting TF regulatory modules.

    This is the main entry point for our analysis pipeline.
    Handles:
    - TF name → HGNC ID resolution
    - Filtering to genes present in expression data
    - Building RegulatoryModule objects

    Examples
    --------
    >>> extractor = RegulatoryModuleExtractor()
    >>> modules = extractor.get_tf_modules(
    ...     tfs=["TP53", "MYC", "STAT3"],
    ...     gene_universe={"TP53", "BRCA1", "CDK1", ...}  # genes in our data
    ... )
    >>> for module in modules:
    ...     print(f"{module.tf_name}: {len(module.targets)} targets in data")
    """

    def __init__(self, client: Optional[CoGExClient] = None):
        self._client = client or CoGExClient()
        self._gene_name_cache: Dict[str, GeneId] = {}

    def resolve_gene_name(self, name: str) -> Optional[GeneId]:
        """
        Resolve gene name to HGNC identifier.

        Uses INDRA's standardization machinery.
        """
        pass

    def get_tf_modules(
        self,
        tfs: List[str],
        gene_universe: Set[str],
        min_evidence: int = 2,
        include_activation: bool = True,
        include_repression: bool = True,
    ) -> List[RegulatoryModule]:
        """
        Get regulatory modules for TFs, filtered to genes in our data.

        Parameters
        ----------
        tfs : List[str]
            List of TF gene names (e.g., ["TP53", "MYC"])
        gene_universe : Set[str]
            Set of gene names present in expression data
            Used to filter targets to those we can analyze
        min_evidence : int
            Minimum evidence count for including an edge
        include_activation : bool
            Include TF→target activation edges
        include_repression : bool
            Include TF→target repression edges

        Returns
        -------
        List[RegulatoryModule]
            One module per TF with filtered targets
        """
        pass

    def get_child_set_type1(
        self,
        tf: str,
        gene_universe: Set[str],
    ) -> Set[str]:
        """
        Child Set Type 1: INDRA-reported targets present in our data.

        Returns gene names (not IDs) for easy matching with BioMatrix.
        """
        pass

    def get_shared_regulators(
        self,
        genes: List[str],
        min_targets: int = 2,
    ) -> List[RegulatoryModule]:
        """
        Find TFs that regulate multiple genes in the given set.

        Useful for identifying shared upstream regulation.
        """
        pass
```

### biocore/knowledge/clique_validator.py

```python
"""
Correlation-based clique validation for TF regulatory modules.

Given a set of TF targets (Child Set Type 1), this module identifies
subsets that form correlation cliques (Child Set Type 2).

Design Decisions:
    - Uses stratified correlation matrices (CASE/CTRL × Male/Female)
    - Finds maximal cliques using networkx
    - Supports differential co-expression analysis

Examples:
    >>> from biocore.knowledge.clique_validator import CliqueValidator
    >>> validator = CliqueValidator(matrix, stratify_by=["phenotype", "Sex"])
    >>>
    >>> # Find cliques within TF targets
    >>> cliques = validator.find_cliques(
    ...     genes=module.target_genes,
    ...     min_correlation=0.7,
    ...     condition="CASE"
    ... )
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional, Literal
from dataclasses import dataclass
import numpy as np
import pandas as pd
import networkx as nx

from biocore.core.biomatrix import BioMatrix

@dataclass
class CorrelationClique:
    """A set of genes that are all pairwise correlated."""
    genes: Set[str]
    condition: str  # e.g., "CASE", "CTRL", "CASE_Male"
    mean_correlation: float
    min_correlation: float
    size: int

    @property
    def is_maximal(self) -> bool:
        """True if no gene can be added while maintaining clique property."""
        return True  # By construction from networkx

@dataclass
class ChildSetType2:
    """
    Child Set Type 2: Correlation-validated subset of TF targets.

    This is a subset of Child Set Type 1 (INDRA-reported targets)
    where all pairs of genes are correlated above threshold.
    """
    tf_name: str
    parent_set: Set[str]  # Type 1: all INDRA targets in data
    clique_genes: Set[str]  # Type 2: correlated subset
    condition: str
    correlation_threshold: float
    mean_correlation: float

class CliqueValidator:
    """
    Validate TF regulatory modules by correlation structure.

    Builds correlation networks within gene sets and finds cliques
    to identify functionally coherent subsets.
    """

    def __init__(
        self,
        matrix: BioMatrix,
        stratify_by: List[str] = ["phenotype", "Sex"],
    ):
        """
        Initialize with expression matrix and stratification columns.

        Parameters
        ----------
        matrix : BioMatrix
            Expression matrix with sample metadata
        stratify_by : List[str]
            Metadata columns for stratification
        """
        self.matrix = matrix
        self.stratify_by = stratify_by
        self._correlation_cache: Dict[str, pd.DataFrame] = {}

    def compute_correlation_matrix(
        self,
        genes: List[str],
        condition: Optional[str] = None,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for gene subset within condition.

        Parameters
        ----------
        genes : List[str]
            Gene names to include
        condition : str, optional
            Stratification condition (e.g., "CASE", "CASE_Male")
            If None, uses all samples
        method : str
            Correlation method

        Returns
        -------
        pd.DataFrame
            Gene × Gene correlation matrix
        """
        pass

    def build_correlation_graph(
        self,
        genes: List[str],
        condition: str,
        min_correlation: float = 0.7,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> nx.Graph:
        """
        Build graph where edges connect correlated genes.

        Parameters
        ----------
        genes : List[str]
            Gene set to analyze
        condition : str
            Stratification condition
        min_correlation : float
            Minimum absolute correlation for edge

        Returns
        -------
        nx.Graph
            Correlation graph
        """
        pass

    def find_cliques(
        self,
        genes: Set[str],
        condition: str,
        min_correlation: float = 0.7,
        min_clique_size: int = 3,
    ) -> List[CorrelationClique]:
        """
        Find all maximal cliques in correlation graph.

        Parameters
        ----------
        genes : Set[str]
            Gene set to analyze (typically Child Set Type 1)
        condition : str
            Stratification condition
        min_correlation : float
            Minimum correlation for edge
        min_clique_size : int
            Minimum genes per clique

        Returns
        -------
        List[CorrelationClique]
            All maximal cliques meeting size threshold
        """
        pass

    def get_child_set_type2(
        self,
        tf_name: str,
        type1_genes: Set[str],
        condition: str,
        min_correlation: float = 0.7,
    ) -> List[ChildSetType2]:
        """
        Derive Child Set Type 2 from Type 1 for a TF.

        Returns all maximal cliques within the Type 1 gene set.
        """
        pass

    def find_differential_cliques(
        self,
        genes: Set[str],
        case_condition: str = "CASE",
        ctrl_condition: str = "CTRL",
        min_correlation: float = 0.7,
    ) -> Tuple[List[CorrelationClique], List[CorrelationClique]]:
        """
        Find cliques gained or lost in CASE vs CTRL.

        Returns
        -------
        gained : List[CorrelationClique]
            Cliques present in CASE but not CTRL
        lost : List[CorrelationClique]
            Cliques present in CTRL but not CASE
        """
        pass
```

## Data Flow

### Step 1: Extract TF Regulatory Modules from CoGEx

```python
from biocore.knowledge.cogex import RegulatoryModuleExtractor
from biocore.io.loaders import MatrixLoader

# Load expression data
matrix = MatrixLoader().load("imputed_matrix.h5")
gene_universe = set(matrix.feature_ids)

# Query CoGEx for TF targets
extractor = RegulatoryModuleExtractor()
modules = extractor.get_tf_modules(
    tfs=["TP53", "MYC", "STAT3", "NRF2", "HIF1A"],  # ALS-relevant TFs
    gene_universe=gene_universe,
    min_evidence=2,
)

# Child Set Type 1: INDRA targets in our data
for module in modules:
    type1_genes = module.target_genes & gene_universe
    print(f"{module.tf_name}: {len(type1_genes)} targets in data")
```

### Step 2: Validate Cliques in Expression Data

```python
from biocore.knowledge.clique_validator import CliqueValidator

validator = CliqueValidator(matrix, stratify_by=["phenotype", "Sex"])

for module in modules:
    type1_genes = module.target_genes & gene_universe

    # Find cliques in CASE samples
    case_cliques = validator.find_cliques(
        genes=type1_genes,
        condition="CASE",
        min_correlation=0.7,
        min_clique_size=5,
    )

    # Find cliques in CTRL samples
    ctrl_cliques = validator.find_cliques(
        genes=type1_genes,
        condition="CTRL",
        min_correlation=0.7,
        min_clique_size=5,
    )

    print(f"{module.tf_name}:")
    print(f"  CASE cliques: {len(case_cliques)}")
    print(f"  CTRL cliques: {len(ctrl_cliques)}")
```

### Step 3: Differential Analysis

```python
# Find disease-specific co-expression modules
for module in modules:
    gained, lost = validator.find_differential_cliques(
        genes=module.target_genes & gene_universe,
        case_condition="CASE",
        ctrl_condition="CTRL",
    )

    if gained:
        print(f"{module.tf_name} - Gained cliques in ALS:")
        for clique in gained:
            print(f"  {clique.genes}")
```

## Testing Strategy

### Unit Tests

```python
# tests/test_cogex_client.py
def test_credential_loading():
    """Test credentials load from env vars."""

def test_tf_target_query():
    """Test TF→target query returns expected structure."""

def test_gene_name_resolution():
    """Test gene name → HGNC ID resolution."""

# tests/test_clique_validator.py
def test_correlation_matrix_stratified():
    """Test correlation computed within strata."""

def test_find_cliques():
    """Test clique detection returns valid cliques."""

def test_differential_cliques():
    """Test gained/lost clique identification."""
```

### Integration Tests

```python
# tests/integration/test_cogex_live.py
@pytest.mark.integration
def test_cogex_connection():
    """Test live connection to CoGEx."""
    client = CoGExClient()
    assert client.ping()

@pytest.mark.integration
def test_tp53_targets():
    """Test querying TP53 targets returns expected results."""
    extractor = RegulatoryModuleExtractor()
    module = extractor.get_tf_modules(["TP53"], gene_universe={"CDKN1A", "BAX"})[0]
    assert "CDKN1A" in module.target_genes  # Known TP53 target
```

## Dependencies

```toml
# pyproject.toml additions
[project.dependencies]
indra = ">=1.0"
indra-cogex = ">=0.1"
neo4j = ">=5.0"
networkx = ">=3.0"
```

## Error Handling

```python
class CoGExConnectionError(Exception):
    """Failed to connect to INDRA CoGEx."""

class GeneResolutionError(Exception):
    """Failed to resolve gene name to identifier."""

class InsufficientDataError(Exception):
    """Not enough data for analysis (e.g., too few samples in stratum)."""
```

## Performance Considerations

1. **Lazy connection**: Neo4j client connects only on first query
2. **Query batching**: Batch TF queries to minimize round trips
3. **Correlation caching**: Cache stratified correlation matrices
4. **Gene universe filtering**: Filter at query time, not post-hoc
5. **Clique pruning**: Use minimum clique size to reduce enumeration
