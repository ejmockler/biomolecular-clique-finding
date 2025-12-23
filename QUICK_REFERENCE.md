# Quick Reference: Import Refactoring Patterns

## Find & Replace Rules (Use with care in IDE)

### Pattern 1: Direct Module Imports (Core Library)
```
Find:    from biocore\.
Replace: from cliquefinder.
```

### Pattern 2: Package Level Imports
```
Find:    from biocore import
Replace: from cliquefinder import
```

### Pattern 3: Remove sys.path manipulations in scripts
```
Find:    sys\.path\.insert\(0, str\(Path\(__file__\)\.parent(?:\.parent)?\)\)
Replace: (delete this line)
```

---

## File-by-File Checklist

### Libraries (alphabetical by module)

#### core/
- [ ] biocore/__init__.py (3 imports)
- [ ] biocore/core/__init__.py (3 imports)
- [ ] biocore/core/biomatrix.py (1 import)
- [ ] biocore/core/quality.py (0 biocore imports)
- [ ] biocore/core/transform.py (0 biocore imports)

#### quality/
- [ ] biocore/quality/__init__.py (2 imports)
- [ ] biocore/quality/imputation.py (5 imports) ⚠️ Has try/except for optional import
- [ ] biocore/quality/outliers.py (3 imports)
- [ ] biocore/quality/correlation_knn.py (1 import)

#### io/
- [ ] biocore/io/__init__.py (3 imports)
- [ ] biocore/io/loaders.py (2 imports)
- [ ] biocore/io/writers.py (1 import)
- [ ] biocore/io/metadata.py (1 import)

#### knowledge/
- [ ] biocore/knowledge/__init__.py (2 imports)
- [ ] biocore/knowledge/clique_validator.py (1 import)
- [ ] biocore/knowledge/cogex.py (0 biocore imports)
- [ ] biocore/knowledge/regulatory_coherence.py (0 biocore imports)

#### utils/
- [ ] biocore/utils/__init__.py (1 import)
- [ ] biocore/utils/correlation_matrix.py (1 import)

#### validation/
- [ ] biocore/validation/__init__.py (3 imports)
- [ ] biocore/validation/annotation_providers.py (0 biocore imports)
- [ ] biocore/validation/enrichment_tests.py (0 biocore imports)
- [ ] biocore/validation/id_mapping.py (0 biocore imports)

### Scripts (26 files) - All follow similar patterns
- [ ] scripts/*.py (25 scripts with imports + sys.path)
- [ ] scripts/archive/*.py (7 scripts with imports + sys.path)

### Tests (9 files)
- [ ] tests/conftest.py
- [ ] tests/test_*.py (8 test files)
- [ ] test_*.py (3 root-level test files)

---

## Critical Import Updates

### Most Changed Files (by import count)

1. **imputation.py** (5 imports) - Has try/except for optional dependency
   - Line count: ~700
   - Risk: MEDIUM (complex module with many code paths)
   
2. **impute_outliers.py** (6 imports) - Main entry point
   - Line count: ~425
   - Risk: HIGH (user-facing script, test it thoroughly)
   
3. **validation/__init__.py** (3 imports) - Package initialization
   - Line count: ~120
   - Risk: LOW (simple re-exports)

4. **conftest.py** (2 imports) - Test fixture setup
   - Line count: ~100+
   - Risk: MEDIUM (affects all tests)

---

## Verification Commands

### After Library Updates
```bash
# Test individual modules load
python -c "from cliquefinder.core import BioMatrix, QualityFlag, Transform; print('✓ Core imports OK')"
python -c "from cliquefinder.quality import OutlierDetector, Imputer; print('✓ Quality imports OK')"
python -c "from cliquefinder.io import load_csv_matrix, write_csv_matrix; print('✓ IO imports OK')"
python -c "from cliquefinder.knowledge import CliqueValidator; print('✓ Knowledge imports OK')"
python -c "from cliquefinder.validation import GOAnnotationProvider; print('✓ Validation imports OK')"
python -c "from cliquefinder.utils import get_correlation_matrix; print('✓ Utils imports OK')"
```

### After Script Updates
```bash
# Test main script works
python -m cliquefinder.cli.impute --help 2>&1 | head -5
python -m cliquefinder.cli.analyze --help 2>&1 | head -5
```

### After Test Updates
```bash
# Quick test run
pytest tests/test_full_pipeline.py -v -x
```

---

## Common Mistakes to Avoid

### ❌ WRONG: Incomplete import path
```python
from cliquefinder.core import quality  # Wrong - cliquefinder.core.quality is a module, not subpackage
```

### ✓ RIGHT: Complete import path
```python
from cliquefinder.core.quality import QualityFlag  # Correct
```

---

### ❌ WRONG: Forgetting sys.path removal
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # ← Keep this for old code
from cliquefinder import BioMatrix  # Old code after refactoring
```

### ✓ RIGHT: Clean imports without sys.path
```python
from cliquefinder import BioMatrix  # This works directly when installed
```

---

### ❌ WRONG: Relative imports (in CLI scripts)
```python
# In cliquefinder/cli/impute.py
from ..core import BioMatrix  # Complex relative imports
```

### ✓ RIGHT: Absolute imports (cleaner for CLI)
```python
# In cliquefinder/cli/impute.py
from cliquefinder.core import BioMatrix  # Explicit and clear
```

---

## Import Order Convention

After refactoring, follow PEP 8 import ordering in each file:

```python
# Standard library
import sys
from pathlib import Path
from typing import Optional

# Third-party
import numpy as np
import pandas as pd

# Local (cliquefinder)
from cliquefinder.core import BioMatrix, QualityFlag
from cliquefinder.quality import Imputer
```

---

## Handling Optional Dependencies

Example from **imputation.py** (keep as-is):

```python
try:
    from cliquefinder.utils.correlation_matrix import get_correlation_matrix
    _CORRELATION_CACHE_AVAILABLE = True
except ImportError:
    get_correlation_matrix = None
    _CORRELATION_CACHE_AVAILABLE = False
```

This pattern allows graceful degradation if optional module isn't installed.

---

## Testing the Refactoring

### Phase 1: Unit Tests
```bash
# Test each module independently
pytest tests/test_full_pipeline.py -v
pytest tests/test_optimization_equivalence.py -v
pytest tests/test_optimization_performance.py -v
```

### Phase 2: Integration Tests
```bash
# Test scripts work with new imports
python -m cliquefinder.cli.impute --input test_data.csv --output test_out
python -m cliquefinder.cli.analyze --input test_data.csv --output test_out
```

### Phase 3: Smoke Tests
```bash
# Quick import check
python -c "import cliquefinder; print(cliquefinder.__version__)"
```

---

## Rollback Strategy

If problems occur:

1. Keep old code in git branch
2. Test each module section independently
3. If a specific file has issues:
   ```bash
   git diff src/cliquefinder/core/biomatrix.py  # See what changed
   git checkout <old-hash> -- src/cliquefinder/core/biomatrix.py  # Revert just that file
   ```

---

## Estimated Time per Task

| Task | Estimate | Files |
|------|----------|-------|
| Library imports | 30 min | 15 |
| Script imports | 20 min | 26 |
| Test imports | 10 min | 9 |
| Verification | 15 min | All |
| **Total** | **75 min** | **50** |

---

## Key Success Metrics

- [ ] All internal imports use `cliquefinder` namespace
- [ ] No `from biocore` imports remain in source code
- [ ] All `sys.path.insert()` calls removed from scripts
- [ ] All tests pass with `pytest -v`
- [ ] `python -c "import cliquefinder"` works without errors
- [ ] Scripts can be run as `python -m cliquefinder.cli.*`
- [ ] No circular import errors
- [ ] Package installs cleanly: `pip install -e .`

