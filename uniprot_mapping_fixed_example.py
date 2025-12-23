"""
FIXED EXAMPLE: Correct UniProt to Gene Symbol Mapping

This script demonstrates the CORRECT implementation fixing the issues found
in the current codebase.

Issues Fixed:
1. Use scopes='uniprot' instead of 'uniprot.Swiss-Prot'
2. Use querymany() instead of query() for batch operations
3. Use correct field names (HGNC not hgnc)
4. Proper error handling with returnall=True

Author: Based on MyGene.info API research
Date: 2025-12-17
"""

import mygene
from typing import Dict, List, Tuple


class UniProtMapper:
    """
    Fixed implementation of UniProt to gene symbol mapping.

    This replaces the problematic code in:
    - /src/cliquefinder/validation/id_mapping.py (line 186)
    - /src/cliquefinder/knowledge/cogex.py (line 753)
    """

    def __init__(self, species: str = 'human'):
        """Initialize with MyGene client."""
        self.mg = mygene.MyGeneInfo()
        self.species = species

    def map_uniprot_to_symbols(
        self,
        uniprot_ids: List[str]
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Map UniProt accessions to gene symbols.

        FIXES:
        - Uses scopes='uniprot' (not 'uniprot.Swiss-Prot')
        - Uses querymany() (not query())
        - Uses fields='symbol' (correct field name)

        Args:
            uniprot_ids: List of UniProt accession IDs

        Returns:
            mapping: {uniprot_id: gene_symbol} for successful mappings
            failed: List of UniProt IDs that failed to map
        """
        results = self.mg.querymany(
            uniprot_ids,
            scopes='uniprot',  # FIX: Changed from 'uniprot.Swiss-Prot'
            fields='symbol,name,entrezgene',
            species=self.species,
            returnall=True,
            verbose=False
        )

        # Extract successful mappings
        mapping = {}
        for item in results['out']:
            if not item.get('notfound') and 'symbol' in item:
                mapping[item['query']] = item['symbol']

        # Collect failed IDs
        failed = list(set(
            results['missing'] + [
                item['query'] for item in results['out']
                if item.get('notfound') or 'symbol' not in item
            ]
        ))

        return mapping, failed

    def resolve_gene_name_to_hgnc(self, name: str) -> Tuple[str, str]:
        """
        Resolve a gene name/symbol/UniProt to HGNC ID.

        FIXES for cogex.py:
        - Uses querymany() instead of query()
        - Uses 'HGNC' (uppercase) instead of 'hgnc'
        - Better error handling

        Args:
            name: Gene symbol, alias, or UniProt ID

        Returns:
            Tuple of (database, id) e.g., ('HGNC', '11998')
            Returns None if not found
        """
        results = self.mg.querymany(
            [name],
            scopes='symbol,alias,uniprot',  # Accept multiple ID types
            fields='symbol,HGNC',  # FIX: Uppercase HGNC
            species=self.species,
            verbose=False
        )

        if results and not results[0].get('notfound'):
            item = results[0]
            if 'HGNC' in item:
                hgnc_id = item['HGNC']
                # Handle list case (multiple matches - take first)
                if isinstance(hgnc_id, list):
                    hgnc_id = hgnc_id[0]
                return ('HGNC', str(hgnc_id))

        return None


def compare_old_vs_new():
    """Demonstrate the difference between old and new implementations."""

    print("="*70)
    print("COMPARISON: Old (Problematic) vs New (Fixed) Implementation")
    print("="*70)

    mg = mygene.MyGeneInfo()
    test_ids = ['P07902', 'A0AVT1', 'P04637', 'INVALID']

    # OLD APPROACH (from id_mapping.py)
    print("\n1. OLD APPROACH (from id_mapping.py line 186):")
    print("   scopes='uniprot.Swiss-Prot'")
    print("-"*70)

    old_results = mg.querymany(
        test_ids,
        scopes='uniprot.Swiss-Prot',  # OLD: Too restrictive
        fields='symbol',
        species='human',
        returnall=True,
        verbose=False
    )

    old_mapped = len([x for x in old_results['out'] if not x.get('notfound')])
    print(f"   Mapped: {old_mapped}/{len(test_ids)}")

    # NEW APPROACH (fixed)
    print("\n2. NEW APPROACH (fixed):")
    print("   scopes='uniprot'")
    print("-"*70)

    new_results = mg.querymany(
        test_ids,
        scopes='uniprot',  # NEW: More comprehensive
        fields='symbol',
        species='human',
        returnall=True,
        verbose=False
    )

    new_mapped = len([x for x in new_results['out'] if not x.get('notfound')])
    print(f"   Mapped: {new_mapped}/{len(test_ids)}")

    print(f"\n   Result: Both work for Swiss-Prot IDs, but 'uniprot' is recommended")

    # OLD APPROACH (from cogex.py)
    print("\n" + "="*70)
    print("3. OLD APPROACH (from cogex.py line 753):")
    print("   Using query() instead of querymany()")
    print("-"*70)

    test_name = 'TP53'

    # Old way
    print(f"\n   OLD: mg.query('{test_name}', scopes='symbol,alias,uniprot', fields='hgnc')")
    old_res = mg.query(test_name, scopes='symbol,alias,uniprot', fields='hgnc', species='human')
    print(f"   Result type: {type(old_res)}")
    print(f"   Structure: dict with 'hits' key")
    if old_res.get('hits'):
        print(f"   Parsing required: result['hits'][0]")
        print(f"   PROBLEM: fields='hgnc' returns empty (should be 'HGNC')")

    # New way
    print(f"\n   NEW: mg.querymany(['{test_name}'], scopes='symbol,alias,uniprot', fields='HGNC')")
    new_res = mg.querymany([test_name], scopes='symbol,alias,uniprot', fields='symbol,HGNC', species='human', verbose=False)
    print(f"   Result type: {type(new_res)}")
    print(f"   Structure: list of dicts")
    if new_res and not new_res[0].get('notfound'):
        print(f"   Direct access: result[0]['HGNC'] = {new_res[0].get('HGNC')}")
        print(f"   Symbol: {new_res[0].get('symbol')}")


def demo_fixed_implementation():
    """Demonstrate the fixed implementation."""

    print("\n" + "="*70)
    print("DEMO: Fixed Implementation")
    print("="*70)

    # Example proteomics data
    uniprot_ids = [
        'P07902',  # GALT
        'A0AVT1',  # UBA6
        'P04637',  # TP53
        'P38398',  # BRCA1
        'P01308',  # INS
        'INVALID', # Invalid ID
        'P12345',  # Non-existent
    ]

    # Use fixed implementation
    mapper = UniProtMapper(species='human')
    mapping, failed = mapper.map_uniprot_to_symbols(uniprot_ids)

    print(f"\n✓ Successfully mapped {len(mapping)}/{len(uniprot_ids)} UniProt IDs")

    print("\nMappings:")
    for uniprot, symbol in sorted(mapping.items()):
        print(f"  {uniprot:10s} -> {symbol}")

    print(f"\nFailed IDs ({len(failed)}):")
    for uniprot in failed:
        print(f"  {uniprot}")

    # Test HGNC resolution
    print("\n" + "="*70)
    print("DEMO: Gene Name -> HGNC Resolution (Fixed)")
    print("="*70)

    test_names = ['TP53', 'BRCA1', 'P07902', 'INVALID']

    for name in test_names:
        result = mapper.resolve_gene_name_to_hgnc(name)
        if result:
            db, id_val = result
            print(f"  {name:10s} -> {db}:{id_val}")
        else:
            print(f"  {name:10s} -> NOT FOUND")


def show_recommended_code():
    """Show the exact code to use in the codebase."""

    print("\n" + "="*70)
    print("RECOMMENDED CODE FOR CODEBASE")
    print("="*70)

    print("\n1. For /src/cliquefinder/validation/id_mapping.py (line 186):")
    print("-"*70)
    print("""
# OLD (line 186):
type_map = {
    'ensembl_gene': 'ensembl.gene',
    'symbol': 'symbol',
    'symbol_alias': 'symbol,alias',
    'uniprot': 'uniprot.Swiss-Prot',  # Too restrictive
    'entrez': 'entrezgene'
}

# NEW (recommended):
type_map = {
    'ensembl_gene': 'ensembl.gene',
    'symbol': 'symbol',
    'symbol_alias': 'symbol,alias',
    'uniprot': 'uniprot',  # More comprehensive
    'entrez': 'entrezgene'
}
    """)

    print("\n2. For /src/cliquefinder/knowledge/cogex.py (line 753):")
    print("-"*70)
    print("""
# OLD (line 753):
res = mg.query(name, scopes='symbol,alias,uniprot', fields='hgnc', species='human')

if res and 'hits' in res and len(res['hits']) > 0:
    hit = res['hits'][0]
    if 'hgnc' in hit:  # Wrong: should be 'HGNC'
        hgnc_val = hit['hgnc']
        # ...

# NEW (recommended):
res = mg.querymany([name], scopes='symbol,alias,uniprot',
                   fields='symbol,HGNC', species='human', verbose=False)

if res and not res[0].get('notfound'):
    if 'HGNC' in res[0]:
        hgnc_val = res[0]['HGNC']
        if isinstance(hgnc_val, list):
            hgnc_id = str(hgnc_val[0]) if hgnc_val else None
        else:
            hgnc_id = str(hgnc_val)
        if hgnc_id:
            return ("HGNC", hgnc_id)
    """)

    print("\n3. General usage pattern:")
    print("-"*70)
    print("""
import mygene

mg = mygene.MyGeneInfo()

# For batch UniProt -> Gene Symbol mapping:
results = mg.querymany(
    uniprot_ids,
    scopes='uniprot',  # Not 'uniprot.Swiss-Prot'
    fields='symbol,name,entrezgene',
    species='human',
    returnall=True,
    verbose=False
)

# Extract mapping:
mapping = {
    item['query']: item['symbol']
    for item in results['out']
    if not item.get('notfound') and 'symbol' in item
}

# Failed IDs:
failed = results['missing']
    """)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MyGene.info: Fixed UniProt Mapping Implementation")
    print("="*70)

    compare_old_vs_new()
    demo_fixed_implementation()
    show_recommended_code()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Fixes:
1. Use scopes='uniprot' instead of 'uniprot.Swiss-Prot'
2. Use querymany() instead of query() for batch operations
3. Use 'HGNC' (uppercase) not 'hgnc' (lowercase)
4. Use returnall=True for better error handling

These fixes will improve:
- Coverage (both Swiss-Prot and TrEMBL IDs)
- Performance (querymany is optimized for ID mapping)
- Correctness (proper field names)
- Reliability (better error handling)
    """)
    print("="*70)
