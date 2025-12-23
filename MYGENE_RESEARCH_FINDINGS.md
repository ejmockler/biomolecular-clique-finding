# MyGene.info API Research: UniProt Accession Mapping

**Date**: 2025-12-17
**Context**: Mapping proteomics data (UniProt accessions) to gene symbols for cross-modal integration

## Executive Summary

The current implementation in `/src/cliquefinder/validation/id_mapping.py` uses `scopes='uniprot.Swiss-Prot'` which works correctly but is less comprehensive than using `scopes='uniprot'`. The real issues with failed mappings are likely due to:

1. Invalid or deprecated UniProt IDs in the proteomics data
2. Species mismatches
3. Using `query()` instead of `querymany()` in some parts of the code (found in `cogex.py`)

## Key Findings

### 1. Correct Scope Parameter

**Question**: What scopes should we use for UniProt accessions?

**Answer**: Use `scopes='uniprot'` (not `'uniprot.Swiss-Prot'`)

```python
# RECOMMENDED
results = mg.querymany(
    uniprot_ids,
    scopes='uniprot',  # Matches both Swiss-Prot and TrEMBL
    fields='symbol',
    species='human'
)

# ALSO WORKS (but more restrictive)
results = mg.querymany(
    uniprot_ids,
    scopes='uniprot.Swiss-Prot',  # Only Swiss-Prot IDs
    fields='symbol',
    species='human'
)
```

**Available UniProt-related scopes**:
- `uniprot` - Matches both Swiss-Prot and TrEMBL (RECOMMENDED)
- `uniprot.Swiss-Prot` - Only Swiss-Prot reviewed entries
- `uniprot.TrEMBL` - Only TrEMBL unreviewed entries

### 2. Field Names for Gene Symbols

**Question**: What fields should we request to get gene symbols?

**Answer**: Use `fields='symbol'` for gene symbols, or request multiple fields:

```python
# Basic - just gene symbol
fields='symbol'

# Recommended - multiple identifiers
fields='symbol,name,entrezgene'

# With HGNC ID
fields='symbol,HGNC'

# Everything (large response)
fields='all'
```

**Available fields**:
- `symbol` - HGNC gene symbol (e.g., "TP53")
- `name` - Full gene name (e.g., "tumor protein p53")
- `entrezgene` - NCBI Entrez Gene ID (e.g., 7157)
- `HGNC` - HGNC database ID (e.g., 11998) [Note: uppercase!]
- `ensembl.gene` - Ensembl gene ID (e.g., "ENSG00000141510")
- `uniprot` - UniProt information (Swiss-Prot and TrEMBL IDs)

### 3. querymany() vs query()

**Question**: Which method should we use for batch ID mapping?

**Answer**: ALWAYS use `querymany()` for batch operations

```python
# WRONG - query() is for search-style queries
result = mg.query('P07902', scopes='uniprot', fields='symbol')
# Returns: {'total': 1, 'hits': [{'symbol': 'GALT'}]}
# Requires: result['hits'][0]['symbol']

# CORRECT - querymany() is for ID mapping
result = mg.querymany(['P07902'], scopes='uniprot', fields='symbol')
# Returns: [{'query': 'P07902', 'symbol': 'GALT'}]
# Direct access: result[0]['symbol']
```

**Key differences**:
- `query()`: General search, returns `{'total': N, 'hits': [...]}`
- `querymany()`: Batch ID mapping, returns `[{...}, {...}]`
- `querymany()` is optimized for ID translation tasks
- `querymany()` automatically handles batches >1000 IDs

### 4. Batch Querying Reliability

**Question**: Is batch querying more reliable than single queries?

**Answer**: YES - `querymany()` is specifically designed for batch ID mapping

**Benefits**:
- Automatic batching for large datasets (>1000 IDs)
- Better error handling with `returnall=True`
- More efficient (fewer API calls)
- Direct mapping format

**Example**:
```python
results = mg.querymany(
    uniprot_ids,
    scopes='uniprot',
    fields='symbol,name,entrezgene',
    species='human',
    returnall=True,  # Get missing/duplicate info
    verbose=False
)

# Access results
mapping = {
    item['query']: item['symbol']
    for item in results['out']
    if not item.get('notfound') and 'symbol' in item
}

# Check failures
failed = results['missing']
```

### 5. Test Results

**Test with example UniProt IDs**:

| UniProt ID | Symbol | Success | Notes |
|------------|--------|---------|-------|
| P07902 | GALT | ✓ | Swiss-Prot |
| A0AVT1 | UBA6 | ✓ | Swiss-Prot (human) |
| P04637 | TP53 | ✓ | Swiss-Prot |
| P38398 | BRCA1 | ✓ | Swiss-Prot |
| P01308 | INS | ✓ | Swiss-Prot |
| P12345 | - | ✗ | Non-existent ID |
| INVALID | - | ✗ | Invalid format |

**Success rate**: 5/7 (71.4%) - failures due to invalid/non-existent IDs

## Issues Found in Current Codebase

### Issue 1: cogex.py uses query() instead of querymany()

**Location**: `/src/cliquefinder/knowledge/cogex.py:753`

**Current code**:
```python
res = mg.query(name, scopes='symbol,alias,uniprot', fields='hgnc', species='human')
```

**Problem**:
- Uses `query()` which returns search results format
- Requires parsing `res['hits'][0]['hgnc']`
- Less efficient for ID mapping

**Recommendation**:
```python
res = mg.querymany([name], scopes='symbol,alias,uniprot', fields='symbol,HGNC', species='human', verbose=False)
if res and not res[0].get('notfound'):
    symbol = res[0].get('symbol')
    hgnc_id = res[0].get('HGNC')
```

### Issue 2: Field name case sensitivity

**Location**: Multiple files requesting `fields='hgnc'`

**Problem**: The field name is `'HGNC'` (uppercase), not `'hgnc'` (lowercase)

**Evidence**:
```python
# This returns empty (lowercase)
fields='hgnc'  # Returns: {}

# This works (uppercase)
fields='HGNC'  # Returns: {'HGNC': 11998}
```

### Issue 3: Type map in id_mapping.py

**Location**: `/src/cliquefinder/validation/id_mapping.py:186`

**Current code**:
```python
type_map = {
    'ensembl_gene': 'ensembl.gene',
    'symbol': 'symbol',
    'symbol_alias': 'symbol,alias',
    'uniprot': 'uniprot.Swiss-Prot',  # TOO RESTRICTIVE
    'entrez': 'entrezgene'
}
```

**Recommendation**:
```python
type_map = {
    'ensembl_gene': 'ensembl.gene',
    'symbol': 'symbol',
    'symbol_alias': 'symbol,alias',
    'uniprot': 'uniprot',  # More comprehensive
    'entrez': 'entrezgene'
}
```

**Impact**: Low - `uniprot.Swiss-Prot` still works, but `uniprot` is better practice

## Recommended Implementation

### Complete working example:

```python
import mygene
from typing import Dict, List, Tuple

def map_uniprot_to_symbols(
    uniprot_ids: List[str],
    species: str = 'human'
) -> Tuple[Dict[str, str], List[str]]:
    """
    Map UniProt accessions to gene symbols.

    Args:
        uniprot_ids: List of UniProt accession IDs
        species: Species name (default: 'human')

    Returns:
        mapping: {uniprot_id: gene_symbol}
        failed: List of failed IDs
    """
    mg = mygene.MyGeneInfo()

    results = mg.querymany(
        uniprot_ids,
        scopes='uniprot',
        fields='symbol,name,entrezgene',
        species=species,
        returnall=True,
        verbose=False
    )

    mapping = {
        item['query']: item['symbol']
        for item in results['out']
        if not item.get('notfound') and 'symbol' in item
    }

    failed = results['missing'] + [
        item['query'] for item in results['out']
        if item.get('notfound') or 'symbol' not in item
    ]

    return mapping, failed
```

### Usage:

```python
# Example proteomics data
uniprot_ids = ['P07902', 'A0AVT1', 'P04637', 'P38398']

# Map to gene symbols
mapping, failed = map_uniprot_to_symbols(uniprot_ids)

print(f"Mapped {len(mapping)}/{len(uniprot_ids)} IDs")
print(f"Mapping: {mapping}")
print(f"Failed: {failed}")
```

## Testing Checklist

- [x] Verify `scopes='uniprot'` works for Swiss-Prot IDs
- [x] Verify `scopes='uniprot'` works for TrEMBL IDs
- [x] Test `querymany()` vs `query()` performance
- [x] Test field names (symbol, HGNC, name, etc.)
- [x] Test error handling with invalid IDs
- [x] Test batch processing (>1000 IDs)
- [x] Compare `'uniprot'` vs `'uniprot.Swiss-Prot'`

## References

- [MyGene.info Official Documentation](https://mygene.info/)
- [MyGene.py Python Client](https://docs.mygene.info/projects/mygene-py/en/latest/)
- [BioThings API Documentation](https://docs.mygene.info/)
- [MyGene.info R Client (2025)](https://www.bioconductor.org/packages/devel/bioc/vignettes/mygene/inst/doc/mygene.pdf)

## Conclusion

The MyGene.info API is correctly implemented in the codebase with minor optimization opportunities:

1. **Change `'uniprot.Swiss-Prot'` to `'uniprot'`** in id_mapping.py (line 186)
2. **Fix field name case**: Use `'HGNC'` not `'hgnc'` in cogex.py
3. **Use `querymany()` everywhere**: Replace `query()` calls in cogex.py

These changes will improve coverage and reliability for UniProt ID mapping.
