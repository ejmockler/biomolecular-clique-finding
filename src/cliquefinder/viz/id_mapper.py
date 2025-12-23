"""
UniProt to gene symbol mapping for visualizations.

Provides human-readable gene symbols instead of obscure biomolecular IDs.
Uses MyGene.info API with local caching for efficiency.

Examples
--------
>>> from cliquefinder.viz.id_mapper import get_gene_symbol, map_ids
>>> get_gene_symbol("P22314")
'UBA1'
>>> map_ids(["P22314", "P51784", "O43602"])
{'P22314': 'UBA1', 'P51784': 'USP11', 'O43602': 'DCX'}
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

# Hardcoded mappings for common proteins (avoids API calls)
# These are sex-linked proteins commonly used in sex classification
KNOWN_MAPPINGS = {
    # X-linked proteins (used in sex classification)
    "P22314": "UBA1",     # Ubiquitin-like modifier-activating enzyme 1
    "P51784": "USP11",    # Ubiquitin carboxyl-terminal hydrolase 11
    "O43602": "DCX",      # Doublecortin
    "Q9Y5S2": "CDC42BPB", # Serine/threonine-protein kinase MRCK beta
    "P41229": "KDM5C",    # Lysine-specific demethylase 5C

    # Common ALS-related proteins
    "P00441": "SOD1",     # Superoxide dismutase [Cu-Zn]
    "Q96QK1": "VPS35",    # Vacuolar protein sorting 35
    "Q13148": "TARDBP",   # TAR DNA-binding protein 43
    "P09651": "HNRNPA1",  # Heterogeneous nuclear ribonucleoprotein A1
    "Q9Y3I0": "RTCB",     # tRNA-splicing ligase RtcB homolog
    "P56192": "MARS1",    # Methionine--tRNA ligase

    # Y-linked proteins
    "Q9BZC1": "RPS4Y1",   # 40S ribosomal protein S4, Y-linked
    "P00156": "DDX3Y",    # ATP-dependent RNA helicase DDX3Y (actually DEAD box)
    "O14654": "IRS4",     # Insulin receptor substrate 4

    # Common housekeeping proteins
    "P68871": "HBB",      # Hemoglobin subunit beta
    "P60709": "ACTB",     # Actin, cytoplasmic 1
    "P07437": "TUBB",     # Tubulin beta chain
}


@lru_cache(maxsize=1000)
def get_gene_symbol(uniprot_id: str, fallback: bool = True) -> str:
    """
    Get gene symbol for a UniProt accession.

    Parameters
    ----------
    uniprot_id : str
        UniProt accession (e.g., "P22314")
    fallback : bool
        If True, return UniProt ID when lookup fails.
        If False, raise KeyError.

    Returns
    -------
    str
        Gene symbol (e.g., "UBA1")

    Examples
    --------
    >>> get_gene_symbol("P22314")
    'UBA1'
    >>> get_gene_symbol("UNKNOWN123")
    'UNKNOWN123'  # fallback to input
    """
    # Clean input
    uniprot_id = str(uniprot_id).strip().upper()

    # Check hardcoded mappings first
    if uniprot_id in KNOWN_MAPPINGS:
        return KNOWN_MAPPINGS[uniprot_id]

    # Try mygene API
    try:
        symbol = _query_mygene(uniprot_id)
        if symbol:
            return symbol
    except Exception as e:
        if not fallback:
            raise KeyError(f"Failed to resolve {uniprot_id}: {e}")

    # Fallback to original ID
    if fallback:
        return uniprot_id
    raise KeyError(f"No gene symbol found for {uniprot_id}")


def _query_mygene(uniprot_id: str) -> Optional[str]:
    """Query MyGene.info for gene symbol."""
    try:
        import mygene
        mg = mygene.MyGeneInfo()

        # Query by UniProt ID
        result = mg.query(f"uniprot:{uniprot_id}", fields="symbol", species="human")

        if result.get("hits"):
            return result["hits"][0].get("symbol")

        return None
    except ImportError:
        warnings.warn("mygene not installed. Using hardcoded mappings only.")
        return None
    except Exception:
        return None


def map_ids(uniprot_ids: list[str], fallback: bool = True) -> dict[str, str]:
    """
    Map multiple UniProt IDs to gene symbols.

    Parameters
    ----------
    uniprot_ids : list[str]
        List of UniProt accessions
    fallback : bool
        If True, use UniProt ID when lookup fails

    Returns
    -------
    dict[str, str]
        Mapping of UniProt ID -> gene symbol

    Examples
    --------
    >>> map_ids(["P22314", "P51784", "O43602"])
    {'P22314': 'UBA1', 'P51784': 'USP11', 'O43602': 'DCX'}
    """
    return {uid: get_gene_symbol(uid, fallback=fallback) for uid in uniprot_ids}


def format_feature_label(feature_id: str, italicize: bool = True) -> str:
    """
    Format feature ID for display in plots.

    Converts UniProt ID to gene symbol and optionally italicizes.

    Parameters
    ----------
    feature_id : str
        Feature identifier (UniProt accession or gene symbol)
    italicize : bool
        If True, wrap in matplotlib italic formatting

    Returns
    -------
    str
        Formatted label for matplotlib

    Examples
    --------
    >>> format_feature_label("P22314")
    '$\\\\mathit{UBA1}$'
    >>> format_feature_label("P22314", italicize=False)
    'UBA1'
    """
    symbol = get_gene_symbol(feature_id)

    if italicize:
        return f"$\\mathit{{{symbol}}}$"
    return symbol
