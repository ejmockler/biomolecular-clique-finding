"""
MyGene.info API Guide: Mapping UniProt Accessions to Gene Symbols

This script demonstrates the correct way to map UniProt accessions to gene symbols
using the MyGene.info API, addressing common issues and best practices.

Key Findings:
1. Use querymany() for batch operations (NOT query())
2. Use scopes='uniprot' (NOT 'uniprot.Swiss-Prot') for comprehensive coverage
3. Request fields='symbol' to get gene symbols (or 'symbol,name,entrezgene' for more info)
4. Use returnall=True to get detailed information about missing IDs
5. MyGene automatically batches large queries (>1000 IDs)

Author: Research based on MyGene.info v3.2.2 documentation and testing
Date: 2025-12-17
"""

import mygene
from typing import Dict, List, Tuple


def map_uniprot_to_symbols(
    uniprot_ids: List[str],
    species: str = 'human',
    verbose: bool = True
) -> Tuple[Dict[str, str], List[str]]:
    """
    Map UniProt accessions to gene symbols using MyGene.info API.

    This is the RECOMMENDED approach for proteomics data with UniProt accessions.

    Args:
        uniprot_ids: List of UniProt accession IDs (e.g., ['P07902', 'A0AVT1', 'P04637'])
        species: Species name for filtering (default: 'human')
        verbose: Print mapping statistics (default: True)

    Returns:
        mapping: Dictionary mapping {uniprot_id: gene_symbol} for successful mappings
        failed: List of UniProt IDs that failed to map

    Examples:
        >>> mapping, failed = map_uniprot_to_symbols(['P07902', 'P04637', 'A0AVT1'])
        >>> print(mapping)
        {'P07902': 'GALT', 'P04637': 'TP53', 'A0AVT1': 'UBA6'}
        >>> print(failed)
        []
    """
    mg = mygene.MyGeneInfo()

    # Query MyGene.info with batch operation
    results = mg.querymany(
        uniprot_ids,
        scopes='uniprot',  # Matches both Swiss-Prot and TrEMBL
        fields='symbol,name,entrezgene',  # Get gene symbols and additional info
        species=species,
        returnall=True,  # Get detailed results including missing IDs
        verbose=False  # Suppress MyGene's verbose output
    )

    # Build mapping dictionary
    mapping = {}
    for item in results['out']:
        if not item.get('notfound') and 'symbol' in item:
            mapping[item['query']] = item['symbol']

    # Collect failed IDs
    failed = list(set(results['missing'] + [
        item['query'] for item in results['out']
        if item.get('notfound') or 'symbol' not in item
    ]))

    # Print statistics
    if verbose:
        total = len(uniprot_ids)
        success = len(mapping)
        success_rate = (success / total * 100) if total > 0 else 0
        print(f"UniProt -> Gene Symbol Mapping:")
        print(f"  Total IDs: {total}")
        print(f"  Successful: {success} ({success_rate:.1f}%)")
        print(f"  Failed: {len(failed)}")
        if failed and len(failed) <= 10:
            print(f"  Failed IDs: {failed}")

    return mapping, failed


def map_uniprot_batch(
    uniprot_ids: List[str],
    species: str = 'human',
    batch_size: int = 1000
) -> Dict[str, str]:
    """
    Map large batches of UniProt IDs to gene symbols.

    MyGene.info automatically handles batching, but this function provides
    explicit control and progress reporting for very large datasets.

    Args:
        uniprot_ids: List of UniProt accessions
        species: Species name
        batch_size: Size of each batch (default: 1000, max recommended by MyGene)

    Returns:
        Complete mapping dictionary
    """
    mg = mygene.MyGeneInfo()

    mapping = {}
    total_batches = (len(uniprot_ids) + batch_size - 1) // batch_size

    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i+batch_size]
        batch_num = i // batch_size + 1

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} IDs)...")

        results = mg.querymany(
            batch,
            scopes='uniprot',
            fields='symbol',
            species=species,
            verbose=False
        )

        for item in results:
            if not item.get('notfound') and 'symbol' in item:
                mapping[item['query']] = item['symbol']

    return mapping


def demo_basic_mapping():
    """Demonstrate basic UniProt to gene symbol mapping."""
    print("="*70)
    print("DEMO 1: Basic UniProt -> Gene Symbol Mapping")
    print("="*70)

    # Example UniProt IDs from proteomics data
    uniprot_ids = [
        'P07902',  # GALT
        'A0AVT1',  # UBA6
        'P04637',  # TP53
        'P38398',  # BRCA1
        'P01308',  # INS
        'INVALID', # Invalid ID for testing
    ]

    mapping, failed = map_uniprot_to_symbols(uniprot_ids)

    print("\nMappings:")
    for uniprot, symbol in sorted(mapping.items()):
        print(f"  {uniprot:10s} -> {symbol}")

    if failed:
        print(f"\nFailed to map: {failed}")


def demo_querymany_vs_query():
    """Demonstrate the difference between querymany() and query()."""
    print("\n" + "="*70)
    print("DEMO 2: querymany() vs query() - Why querymany() is better")
    print("="*70)

    mg = mygene.MyGeneInfo()
    test_id = 'P07902'

    # Method 1: query() - Returns search results format
    print(f"\n1. Using query() - NOT RECOMMENDED for batch operations:")
    result1 = mg.query(test_id, scopes='uniprot', fields='symbol', species='human')
    print(f"   Result type: {type(result1)}")
    print(f"   Structure: {{'total': {result1.get('total')}, 'hits': [...]}}")
    print(f"   Requires parsing: result['hits'][0]['symbol']")

    # Method 2: querymany() - Returns direct list format
    print(f"\n2. Using querymany() - RECOMMENDED:")
    result2 = mg.querymany([test_id], scopes='uniprot', fields='symbol', species='human', verbose=False)
    print(f"   Result type: {type(result2)}")
    print(f"   Structure: [dict, dict, ...]")
    print(f"   Direct access: result[0]['symbol'] = {result2[0].get('symbol')}")

    print(f"\n   Key advantage: querymany() is designed for batch ID mapping!")


def demo_scope_comparison():
    """Demonstrate different scope parameters."""
    print("\n" + "="*70)
    print("DEMO 3: Scope Parameter Comparison")
    print("="*70)

    mg = mygene.MyGeneInfo()
    test_ids = ['P07902', 'A0AVT1', 'P04637']

    scopes_to_test = [
        ('uniprot', 'Recommended - matches both Swiss-Prot and TrEMBL'),
        ('uniprot.Swiss-Prot', 'Specific - only Swiss-Prot IDs'),
        ('symbol,uniprot', 'Mixed - accepts both gene symbols and UniProt IDs'),
    ]

    for scope, description in scopes_to_test:
        print(f"\nscopes='{scope}'")
        print(f"  Description: {description}")

        results = mg.querymany(
            test_ids,
            scopes=scope,
            fields='symbol',
            species='human',
            returnall=True,
            verbose=False
        )

        success = len([x for x in results['out'] if not x.get('notfound')])
        print(f"  Mapped: {success}/{len(test_ids)}")


def demo_fields_selection():
    """Demonstrate different field selections."""
    print("\n" + "="*70)
    print("DEMO 4: Field Selection Options")
    print("="*70)

    mg = mygene.MyGeneInfo()
    test_id = 'P07902'

    field_options = [
        ('symbol', 'Get only gene symbol'),
        ('symbol,name', 'Get symbol and full gene name'),
        ('symbol,name,entrezgene', 'Get multiple identifiers'),
        ('symbol,HGNC', 'Get gene symbol and HGNC ID'),
        ('all', 'Get all available fields (large response)'),
    ]

    for fields, description in field_options:
        print(f"\nfields='{fields}'")
        print(f"  Description: {description}")

        result = mg.querymany(
            [test_id],
            scopes='uniprot',
            fields=fields,
            species='human',
            verbose=False
        )[0]

        # Show what fields are returned
        returned_fields = [k for k in result.keys() if not k.startswith('_')]
        print(f"  Returned fields: {returned_fields[:5]}{'...' if len(returned_fields) > 5 else ''}")

        # Show key values
        if 'symbol' in result:
            print(f"    symbol: {result['symbol']}")
        if 'name' in result:
            print(f"    name: {result['name'][:50]}...")
        if 'entrezgene' in result:
            print(f"    entrezgene: {result['entrezgene']}")
        if 'HGNC' in result:
            print(f"    HGNC: {result['HGNC']}")


def demo_error_handling():
    """Demonstrate proper error handling."""
    print("\n" + "="*70)
    print("DEMO 5: Error Handling and Missing IDs")
    print("="*70)

    mg = mygene.MyGeneInfo()

    # Mix of valid and invalid IDs
    mixed_ids = [
        'P07902',   # Valid
        'P04637',   # Valid
        'INVALID1', # Invalid
        'P99999',   # Non-existent
        'FAKE',     # Invalid
    ]

    print(f"\nQuerying {len(mixed_ids)} IDs (some invalid)...")

    results = mg.querymany(
        mixed_ids,
        scopes='uniprot',
        fields='symbol',
        species='human',
        returnall=True,  # Important: get missing and duplicate info
        verbose=False
    )

    print(f"\nResults summary:")
    print(f"  Total queried: {len(mixed_ids)}")
    print(f"  Successful: {len([x for x in results['out'] if not x.get('notfound')])}")
    print(f"  Missing: {len(results['missing'])}")
    print(f"  Duplicates: {len(results['dup'])}")

    print(f"\nDetailed results:")
    for item in results['out']:
        query_id = item['query']
        if item.get('notfound'):
            print(f"  {query_id:10s} -> NOT FOUND")
        else:
            print(f"  {query_id:10s} -> {item.get('symbol', 'N/A')}")

    # Show how to extract only successful mappings
    successful_mapping = {
        item['query']: item['symbol']
        for item in results['out']
        if not item.get('notfound') and 'symbol' in item
    }

    print(f"\nSuccessful mappings only:")
    for uniprot, symbol in successful_mapping.items():
        print(f"  {uniprot} -> {symbol}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("MyGene.info API: UniProt to Gene Symbol Mapping Guide")
    print("="*70)
    print()
    print("This guide demonstrates the CORRECT way to map UniProt accessions")
    print("to gene symbols for proteomics data integration.")
    print()

    # Run all demos
    demo_basic_mapping()
    demo_querymany_vs_query()
    demo_scope_comparison()
    demo_fields_selection()
    demo_error_handling()

    print("\n" + "="*70)
    print("KEY RECOMMENDATIONS")
    print("="*70)
    print()
    print("1. Use querymany() for batch operations (NOT query())")
    print("2. Use scopes='uniprot' for comprehensive UniProt coverage")
    print("3. Use fields='symbol' to get gene symbols")
    print("4. Use returnall=True to get information about missing IDs")
    print("5. MyGene automatically handles batches >1000 IDs")
    print()
    print("Example code:")
    print("""
    import mygene
    mg = mygene.MyGeneInfo()

    results = mg.querymany(
        uniprot_ids,
        scopes='uniprot',
        fields='symbol,name,entrezgene',
        species='human',
        returnall=True,
        verbose=False
    )

    mapping = {
        item['query']: item['symbol']
        for item in results['out']
        if not item.get('notfound') and 'symbol' in item
    }
    """)
    print()
    print("="*70)


if __name__ == '__main__':
    main()
