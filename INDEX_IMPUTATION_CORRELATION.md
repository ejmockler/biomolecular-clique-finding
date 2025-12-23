# Documentation Index - Imputation & Correlation Architecture

## Overview
Complete architectural mapping of imputation and correlation analysis systems in the Biomolecular Clique Finding project.

## Primary Documents (Created Today)

### 1. ARCHITECTURE_IMPUTATION_CORRELATION.md
**Purpose:** Comprehensive technical architecture guide  
**Size:** 26 KB, 800 lines  
**Audience:** Senior engineers, architects, technical leads

**Contents:**
- Executive summary
- Complete file inventory with line ranges
- Detailed class and function documentation
- Strategy patterns analysis
- Quality flag system design
- Data flow diagrams
- Extension points for new implementations
- Architecture decisions with rationale
- Known limitations and future work
- Implementation task breakdown with time estimates
- Testing considerations

**Best For:**
- Understanding design decisions
- Deep technical review
- Mentoring junior engineers
- Long-term maintenance planning

---

### 2. QUICK_REFERENCE_IMPUTATION_CORRELATION.txt
**Purpose:** Quick lookup and task dispatch guide  
**Size:** 12 KB, 309 lines  
**Audience:** Implementation teams, task managers, reviewers

**Contents:**
- One-page file inventory
- Strategy patterns quick guide
- Correlation type handling (Pearson vs Spearman)
- Quality flag system quick reference
- Complete workflows with CLI examples
- Task dispatch guide with time estimates
- Key parameters and defaults
- Testing checklist for new implementations
- File reference table

**Best For:**
- Quick lookups during implementation
- Task assignment with time estimates
- On-boarding new team members
- Code review checklist

---

## How to Use This Documentation

### If You're Adding a New Imputation Strategy:
1. Read Quick Reference Section 2
2. Refer to Architecture Sections 1.1, 5.1
3. Follow the 4-step process outlined
4. Use testing checklist from Quick Reference

### If You're Adding Spearman Correlation:
1. Read Quick Reference Section 3
2. Refer to Architecture Section 4.2
3. Track file modifications (4 files)
4. Check testing checklist

### If You're Reviewing Code:
1. Use Quick Reference Section 9 (Testing Checklist)
2. Check against Architecture Section 7 (Task Breakdown)
3. Verify new code handles quality flags properly

### If You're On-boarding:
1. Start with Quick Reference Section 1 (File Overview)
2. Read relevant Architecture sections
3. Review workflow diagrams in Architecture Sections 6.1, 6.2
4. Use file reference table (Architecture Section 9)

### If You're Planning Implementation:
1. Check Quick Reference Section 7 (Task Dispatch)
2. Review time estimates and file counts
3. Refer to Architecture Section 7 (Detailed Breakdown)
4. Plan with team using task dependencies

---

## Quick Navigation Map

### Components By Use Case

**Understanding Imputation:**
- Quick Ref: Section 2
- Architecture: Sections 1.1, 1.2, 6.1
- Files: quality/imputation.py, quality/correlation_knn.py

**Understanding Correlation:**
- Quick Ref: Section 3
- Architecture: Sections 2.1, 2.2, 6.2
- Files: utils/correlation_matrix.py, knowledge/clique_validator.py

**Understanding CLI:**
- Quick Ref: Sections 5, 6
- Architecture: Sections 3.1, 3.2, 3.3
- Files: cli/impute.py, cli/analyze.py, cli/_analyze_core.py

**Understanding Quality Flags:**
- Quick Ref: Section 4
- Architecture: Section 1.2
- Files: core/quality.py

**Understanding Outlier Detection:**
- Quick Ref: Section 5
- Architecture: Section 1.3
- Files: quality/outliers.py

---

## File Reference Summary

| Component | File | Size | Complexity |
|-----------|------|------|------------|
| **Imputation Orchestration** | quality/imputation.py | 689 lines | Medium |
| **KNN+Correlation Implementation** | quality/correlation_knn.py | 1035 lines | High |
| **Outlier Detection** | quality/outliers.py | 386 lines | Medium |
| **Quality Flags** | core/quality.py | 138 lines | Low |
| **Correlation Computation** | utils/correlation_matrix.py | 688 lines | High |
| **Clique Finding** | knowledge/clique_validator.py | ~400+ lines | High |
| **Impute CLI** | cli/impute.py | 145 lines | Low |
| **Analyze CLI** | cli/analyze.py | 207 lines | Medium |
| **Core Analysis** | cli/_analyze_core.py | 1021 lines | Very High |
| **Correlation Tests** | stats/correlation_tests.py | 221 lines | Medium |

---

## Implementation Roadmap

### Tier 1: Easy (< 3 hours)
- [ ] Add new quality flag
- [ ] Make correlation type configurable in CLI
- [ ] Add parameter to existing imputation strategy

### Tier 2: Medium (3-6 hours)
- [ ] Add new imputation strategy (SVD, EM, etc.)
- [ ] Add Spearman correlation support
- [ ] Add distance metric options to KNN

### Tier 3: Complex (6+ hours)
- [ ] Implement approximate correlation (for speed)
- [ ] Add multi-method comparison framework
- [ ] Implement sensitivity analysis suite

---

## Key Insights from Architecture Review

### Design Patterns Used
1. **Strategy Pattern:** Imputation strategies (knn_correlation, radius_correlation, median)
2. **Bitwise Flags:** Quality tracking (IntFlag enum)
3. **Caching Pattern:** Correlation matrix (SHA256, memmap, metadata)
4. **Transform Pattern:** Pipeline (OutlierDetector, Imputer)

### Known Limitations
1. **Correlation Type:** Only Pearson implemented (Spearman would help for count data)
2. **Distance Metrics:** Only Pearson correlation distance for KNN
3. **CLI Configuration:** Correlation method not exposed via CLI (hardcoded)
4. **Cache Portability:** Not portable across systems (absolute paths)

### Recommended Improvements
1. End-to-end correlation method configuration
2. Alternative distance metric benchmarking
3. Expose cache configuration to CLI
4. Better error handling in imputation failures
5. Built-in sensitivity analysis

---

## Document Maintenance

### Update When:
- New imputation strategies added → Update Section 1.1, Table 2
- New correlation methods implemented → Update Section 4, Table 3
- New quality flags added → Update Section 1.2
- CLI arguments changed → Update CLI entry sections
- Design patterns change → Update Section 5, Section 10

### Quarterly Review:
- Check for obsolete references
- Update performance benchmarks
- Review new extension points
- Update time estimates based on actual implementation

---

## Contact & Support

For questions about specific components:
- **Imputation logic:** See `/src/cliquefinder/quality/imputation.py` docstrings
- **Correlation computation:** See `/src/cliquefinder/utils/correlation_matrix.py` docstrings
- **CLI design:** See individual `register_parser()` functions
- **Quality flags:** See `/src/cliquefinder/core/quality.py` module docstring

---

## Document Versions

Created: 2025-12-11  
Architecture Version: 1.0  
Last Updated: 2025-12-11  
Review Status: Complete

Next Review: 2025-03-11 (Quarterly)
