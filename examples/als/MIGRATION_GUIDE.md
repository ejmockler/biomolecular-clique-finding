# Migration Guide: C9orf72-Specific Functions

## Overview

The C9orf72-specific function `get_c9orf72_neighbor_sets()` has been deprecated from the core library (`cliquefinder.knowledge.graph_query`) and moved to the examples directory. This guide shows how to update your code.

## What Changed

### Deprecated (Still Works, But Issues Warning)

```python
from cliquefinder.knowledge import get_c9orf72_neighbor_sets

# This still works but will issue a DeprecationWarning
neighbors = get_c9orf72_neighbor_sets(source, gene_universe, min_evidence=2)
```

### Recommended Approach 1: Use Generic Function

```python
from cliquefinder.knowledge import GraphQuery

# Use the generic GraphQuery API directly
query = (
    GraphQuery.neighbors("C9orf72", direction="bidirectional")
    .filter_evidence(min_evidence=2)
    .constrain_to(gene_universe)
)
result = query.execute(source)

# Split by relationship type
by_rel = result.by_relationship()

# Aggregate into activation/inhibition
activation_rels = {"IncreaseAmount", "Activation"}
inhibition_rels = {"DecreaseAmount", "Inhibition"}

activated_entities = set()
inhibited_entities = set()

for rel_name, rel_result in by_rel.items():
    if rel_name in activation_rels:
        activated_entities.update(rel_result.entities)
    elif rel_name in inhibition_rels:
        inhibited_entities.update(rel_result.entities)
```

### Recommended Approach 2: Use Example Functions

```python
from examples.als.graph_queries import get_gene_neighbor_sets

# Works for any gene (C9orf72, SOD1, TP53, etc.)
neighbors = get_gene_neighbor_sets(
    gene_name="C9orf72",
    source=source,
    universe=gene_universe,
    min_evidence=2,
)

# Access results
activated = neighbors["activated"]  # QueryResult
inhibited = neighbors["inhibited"]  # QueryResult
all_neighbors = neighbors["all"]    # QueryResult

# Convert to feature sets
activated_fs = activated.to_feature_set("C9_activated")
inhibited_fs = inhibited.to_feature_set("C9_inhibited")
```

### Backward Compatible (Same Interface)

```python
from examples.als.graph_queries import get_c9orf72_neighbor_sets

# Drop-in replacement - same interface as deprecated function
neighbors = get_c9orf72_neighbor_sets(source, gene_universe, min_evidence=2)
```

## Why the Change

The `get_c9orf72_neighbor_sets()` function is **experiment-specific** and doesn't belong in the core library because:

1. **Gene-specific**: Hardcoded to C9orf72 only
2. **Limited reusability**: Other researchers working with different genes need similar functions
3. **Better pattern exists**: The generic `GraphQuery` API is more flexible and composable

The generic `GraphQuery.neighbors()` API allows you to:
- Query neighbors for ANY gene (not just C9orf72)
- Combine multiple queries with set operations (union, intersection, difference)
- Chain filters for evidence, confidence, and relationship types
- Integrate seamlessly with permutation testing

## Benefits of the New Approach

### 1. Works for Any Gene

```python
# Query different genes with the same function
c9_neighbors = get_gene_neighbor_sets("C9orf72", source, gene_universe)
sod1_neighbors = get_gene_neighbor_sets("SOD1", source, gene_universe)
tp53_neighbors = get_gene_neighbor_sets("TP53", source, gene_universe)
```

### 2. Custom Relationship Categories

```python
from examples.als.graph_queries import get_gene_neighbors_custom_categories

# Define your own relationship groupings
custom_categories = {
    "transcriptional": {"IncreaseAmount", "DecreaseAmount"},
    "post_translational": {"Phosphorylation", "Ubiquitination"},
    "binding": {"Complex", "BindsTo"},
}

neighbors = get_gene_neighbors_custom_categories(
    "TP53", source, custom_categories, gene_universe
)
```

### 3. Batch Query Multiple Genes

```python
from examples.als.graph_queries import get_als_gene_neighbor_sets

# Query all major ALS genes at once
als_neighbors = get_als_gene_neighbor_sets(source, gene_universe)

# Returns: {"C9orf72": {...}, "SOD1": {...}, "TARDBP": {...}, "FUS": {...}}
```

### 4. Full Control with GraphQuery

```python
# Complex queries with chaining
query = (
    GraphQuery.neighbors("C9orf72", direction="downstream")
    .filter_relationship(["IncreaseAmount", "Activation"])
    .filter_evidence(min_count=3)
    .filter_confidence(min_score=0.8)
    .constrain_to(gene_universe)
)
result = query.execute(source)
```

## Migration Checklist

- [ ] Identify all uses of `get_c9orf72_neighbor_sets()`
- [ ] Choose migration approach:
  - [ ] Option A: Use generic `GraphQuery.neighbors()` API
  - [ ] Option B: Use `examples.als.graph_queries.get_gene_neighbor_sets()`
  - [ ] Option C: Copy the function to your own code
- [ ] Update imports
- [ ] Test with your data
- [ ] Remove any DeprecationWarnings from logs

## Questions?

See:
- `examples/als/graph_queries.py` - Generic implementations
- `examples/als/example_graph_queries.py` - Complete usage examples
- `src/cliquefinder/knowledge/graph_query.py` - Core GraphQuery API documentation

## Timeline

- **Now**: Deprecated function still works but issues warnings
- **Future release**: Function will be removed from core library
- **Always available**: Example implementations in `examples/als/graph_queries.py`
