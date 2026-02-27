# Security Audit Findings

**Source:** Brutalist multi-agent review (Claude + Gemini), 2026-02-26
**Scope:** `src/cliquefinder/` — all Python source files

---

## S-1: `eval()` on Untrusted Data — Arbitrary Code Execution

### Status: VALID — CRITICAL

### Location
`viz/differential.py:377`

### Problem
```python
parsed = eval(val)  # val comes from DataFrame column
```
`val` originates from a DataFrame column value (string representation of a list or dict). `eval()` executes arbitrary Python code. If a malformed CSV contains crafted strings in that column, this is a remote code execution vector. A bare `except:` on line 380 masks any errors from malicious payloads.

### Impact
- Any user-provided CSV with a crafted column value can execute arbitrary Python
- The bare `except:` makes exploitation undetectable in logs
- Even without malicious intent, unexpected string formats cause silent failures

### Solution
```python
import ast
parsed = ast.literal_eval(val)
```
`ast.literal_eval()` safely evaluates strings containing only Python literals (strings, numbers, tuples, lists, dicts, booleans, None). It raises `ValueError` for anything else.

### Pitfalls
- `ast.literal_eval()` cannot parse NumPy arrays, sets, or custom objects — if `val` can contain those, use `json.loads()` instead
- If `val` contains single-quoted strings with nested quotes, `ast.literal_eval()` handles them but `json.loads()` does not (JSON requires double quotes)
- The bare `except:` on line 380 must also be narrowed to `except (ValueError, SyntaxError):`

### Priority: P0 — Fix today

---

## S-2: `pickle.load()` Without Integrity Verification — Deserialization RCE

### Status: VALID — CRITICAL

### Locations
1. `validation/id_mapping.py:203`
2. `validation/entity_resolver.py:140`
3. `validation/annotation_providers.py:510`
4. `knowledge/cross_modal_mapper.py:268`

### Problem
All four locations deserialize pickle cache files from disk without integrity checks. Python's `pickle.load()` can execute arbitrary code during deserialization via `__reduce__` methods. A compromised or tampered `.pkl` file in the cache directory executes code with the process's full permissions.

### Impact
- Cache poisoning: attacker writes a crafted `.pkl` to the cache directory → code execution on next load
- Supply chain risk: if cache files are shared between users or stored in shared filesystems
- In a controlled single-user environment the risk is lower but still violates defense-in-depth

### Solution

**Option A: Replace with JSON (preferred for simple structures)**
```python
import json
with open(cache_path) as f:
    data = json.load(f)
```
Requires ensuring all cached data is JSON-serializable (no NumPy arrays, sets, or custom objects in the cache).

**Option B: HMAC integrity check (if pickle is unavoidable)**
```python
import hashlib, hmac, pickle

SECRET = os.environ.get("CACHE_HMAC_KEY", "default-dev-key")

def safe_pickle_load(path):
    raw = path.read_bytes()
    stored_mac = raw[:32]
    payload = raw[32:]
    expected_mac = hmac.new(SECRET.encode(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(stored_mac, expected_mac):
        raise ValueError(f"Cache integrity check failed: {path}")
    return pickle.loads(payload)
```

**Option C: Use `numpy.load()` for array caches**
For caches that are primarily NumPy arrays, use `.npz` format with `allow_pickle=False`.

### Pitfalls
- JSON cannot serialize NumPy arrays directly — use `.tolist()` on save and `np.array()` on load
- JSON does not preserve Python `set` types — serialize as sorted lists
- `id_mapping.py` caches gene ID mappings (dict of str→str) — trivially JSON-serializable
- `entity_resolver.py` caches resolved entities — check for custom dataclass instances that need custom serialization
- `cross_modal_mapper.py` may cache DataFrames — use `df.to_json()` / `pd.read_json()`
- When migrating, delete existing `.pkl` caches to avoid loading stale data

### Priority: P0 — Fix this week

---

## S-3: Bare `except:` Clauses — Silent Error Masking

### Status: VALID — HIGH

### Locations
1. `stats/differential.py:826` — intercept coefficient extraction fallback
2. `viz/cliques.py:210`
3. `quality/outliers.py:719`
4. `viz/differential.py:380` — paired with the `eval()` issue (S-1)

### Problem
Bare `except:` catches ALL exceptions including `KeyboardInterrupt`, `SystemExit`, `MemoryError`, and `GeneratorExit`. This prevents:
- Graceful Ctrl+C termination
- Proper process shutdown
- Memory exhaustion detection
- Debugging of any unexpected error

The most dangerous is `differential.py:826`: on failure to find the intercept coefficient by name, it silently falls back to positional indexing. If the design matrix structure changes (e.g., different number of conditions), positional indexing extracts the **wrong coefficient** — producing incorrect log2FC and p-values with zero indication of error.

### Impact
- `differential.py:826`: Wrong statistical results without any error or warning
- Other locations: Silent swallowing of unexpected errors, unresponsive Ctrl+C

### Solution
Replace each bare `except:` with specific exception types:

```python
# differential.py:826 — replace:
except:
    coef = df.iloc[0]
# with:
except (KeyError, IndexError) as e:
    logger.warning(f"Intercept lookup by name failed, using positional: {e}")
    coef = df.iloc[0]
```

For each location, identify the specific exceptions that the code is trying to handle:
- `differential.py:826`: `KeyError` (missing column name), `IndexError` (empty DataFrame)
- `viz/cliques.py:210`: Likely `ValueError`, `KeyError`
- `quality/outliers.py:719`: Likely `ValueError`, `np.linalg.LinAlgError`
- `viz/differential.py:380`: `ValueError`, `SyntaxError` (from `ast.literal_eval` after S-1 fix)

### Pitfalls
- After narrowing exceptions, previously-masked errors will surface — run full test suite after changes
- The `differential.py:826` positional fallback may be load-bearing for some design matrix configurations — add a test that verifies the named lookup works for all supported designs before removing the fallback entirely
- Consider adding `logger.debug()` even for expected exceptions to aid future debugging

### Priority: P0 — Fix today
