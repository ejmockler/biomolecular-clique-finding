"""
Visualization module for biomolecular clique-finding pipeline.

This module provides publication-quality static figures (matplotlib/seaborn)
and interactive exploration tools (plotly) for:
- Quality control visualizations (outliers, imputation, sex classification)
- Correlation network graphs
- Clique membership and differential analysis
- Biological enrichment overlays

Architecture follows perceptual engineering principles:
- 4±1 chunk working memory at each view level
- Progressive disclosure (overview → stratum → detail)
- Consistent spatial layouts for spatial memory
- Domain conventions (CASE=blue, CTRL=orange, genes italicized)

Examples
--------
>>> from cliquefinder.viz import QCVisualizer, FigureCollection
>>>
>>> # Generate QC figures
>>> viz = QCVisualizer()
>>> fig = viz.plot_outlier_distribution(matrix_before, matrix_after)
>>> fig.save("figures/outliers.pdf")
>>>
>>> # Batch export
>>> collection = FigureCollection()
>>> collection.add("outliers", fig)
>>> collection.save_all(Path("figures/"), format="pdf")
"""

from cliquefinder.viz.core import Figure, FigureCollection
from cliquefinder.viz.styles import Palette, PALETTES, configure_style
from cliquefinder.viz.qc import QCVisualizer
from cliquefinder.viz.cliques import CliqueVisualizer
from cliquefinder.viz.id_mapper import get_gene_symbol, map_ids, format_feature_label

__all__ = [
    # Core
    "Figure",
    "FigureCollection",
    # Styles
    "Palette",
    "PALETTES",
    "configure_style",
    # Visualizers
    "QCVisualizer",
    "CliqueVisualizer",
    # ID mapping
    "get_gene_symbol",
    "map_ids",
    "format_feature_label",
]
