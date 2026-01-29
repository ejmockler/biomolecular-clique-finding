"""
CliqueFinder - Regulatory Clique Discovery in Biomolecular Data

A framework for discovering regulatory cliques in expression data using
knowledge graphs. Applicable to any biomolecular dataset with expression
measurements and regulatory annotations.

Integrates expression data (proteomics, transcriptomics, metabolomics) with
regulatory knowledge (INDRA, pathway databases) to identify co-regulated
feature sets and test for differential abundance across experimental conditions.
"""

__version__ = "0.1.0"

from cliquefinder.core.biomatrix import BioMatrix
from cliquefinder.core.transform import Transform
from cliquefinder.core.quality import QualityFlag

__all__ = [
    "BioMatrix",
    "Transform",
    "QualityFlag",
]
