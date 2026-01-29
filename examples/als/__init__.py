"""
AnswerALS-specific analysis examples.

This directory contains examples demonstrating how to use the generic
permutation testing framework for AnswerALS-specific research questions.

The examples show how to:
- Define custom experimental designs from AnswerALS metadata
- Create genetic subtype comparisons (C9orf72 vs Sporadic)
- Apply the framework to proteomics data with proper metadata handling
- Query knowledge graphs for ALS-associated genes

Modules:
    genetic_contrasts: Experimental designs for genetic subtype comparisons
    graph_queries: Knowledge graph query functions for ALS genes
"""

from .genetic_contrasts import (
    create_c9_vs_sporadic_design,
    create_sod1_vs_sporadic_design,
    create_familial_vs_sporadic_design,
)

from .graph_queries import (
    get_gene_neighbor_sets,
    get_c9orf72_neighbor_sets,
    get_als_gene_neighbor_sets,
    get_gene_neighbors_custom_categories,
)

__all__ = [
    # Experimental designs
    'create_c9_vs_sporadic_design',
    'create_sod1_vs_sporadic_design',
    'create_familial_vs_sporadic_design',
    # Graph queries
    'get_gene_neighbor_sets',
    'get_c9orf72_neighbor_sets',
    'get_als_gene_neighbor_sets',
    'get_gene_neighbors_custom_categories',
]
