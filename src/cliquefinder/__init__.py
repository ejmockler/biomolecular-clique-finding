"""
CliqueFinder - Regulatory Clique Discovery for ALS Transcriptomics

A pipeline for discovering regulatory cliques in expression data using
INDRA CoGEx knowledge graphs. Designed for ALS transcriptomics analysis.
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
