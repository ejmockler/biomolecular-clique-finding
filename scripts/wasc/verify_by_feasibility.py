"""Independent re-derivation of the WASC v1.0.5 C17 B-feasibility table.

Computes, for each pre-registered B value:
  - empirical p-value floor = 1 / (B + 1)
  - BY rank-k raw-p threshold = q * k / (N * H_N)
  - smallest testable rank k_min = ceil(floor / rank1_threshold)
  - untestable ranks (count) = k_min - 1
  - maximum testable rejections = N - (k_min - 1)

Runs as a one-shot script and prints the table.  A test in
tests/wasc/test_feasibility.py asserts this matches the v1.0.5 C17′ table.
"""
from __future__ import annotations

import math

# Locked v1.0.5 binding values (q, N) per spec §6 + E4 freeze.
Q = 0.10
N = 944

# H_N = harmonic number; computed exactly from N.
H_N = sum(1.0 / k for k in range(1, N + 1))

# Pre-registered B values per locked_bounds_v1.json.
B_VALUES = {
    "sensitivity_tier": 999,
    "primary":          9999,
    "floor_tie_rerun":  99999,
}


def feasibility_at_B(B: int) -> dict:
    """Return feasibility dict for a single B value."""
    floor = 1.0 / (B + 1)
    rank1_threshold = Q / (N * H_N)
    k_min = math.ceil(floor / rank1_threshold)
    n_untestable = k_min - 1
    n_testable = N - n_untestable
    return {
        "B": B,
        "floor": floor,
        "rank1_threshold": rank1_threshold,
        "k_min": k_min,
        "n_untestable": n_untestable,
        "n_testable": n_testable,
    }


def main() -> int:
    print(f"WASC v1.0.5 C17′ B-feasibility table")
    print(f"  N = {N}")
    print(f"  q = {Q}")
    print(f"  H_N = {H_N:.4f}")
    print(f"  rank-1 raw-p threshold = q/(N*H_N) = {Q / (N * H_N):.4e}")
    print()
    print(f"{'tier':<22} {'B':>6}  {'floor':>9}  {'k_min':>6}  "
          f"{'untestable':>11}  {'testable':>9}")
    for tier, B in B_VALUES.items():
        r = feasibility_at_B(B)
        print(f"{tier:<22} {B:>6}  {r['floor']:>9.2e}  "
              f"{r['k_min']:>6}  {r['n_untestable']:>11}  {r['n_testable']:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
