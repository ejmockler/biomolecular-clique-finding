"""Debug script to trace median polish differences."""
import numpy as np
from cliquefinder.stats.summarization import tukey_median_polish
from cliquefinder.stats.permutation_gpu import batched_median_polish_gpu

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True)

# Generate test data
data = np.random.randn(5, 10) * 2 + 15

print("Input data shape:", data.shape)
print("Input data (first 3 cols):")
print(data[:, :3])

# Sequential result
print("\n" + "="*60)
print("SEQUENTIAL RESULT")
print("="*60)
seq_result = tukey_median_polish(data, max_iter=10, eps=0.01)
print(f"Overall: {seq_result.overall:.6f}")
print(f"Row effects: {seq_result.row_effects}")
print(f"Col effects (first 5): {seq_result.col_effects[:5]}")
print(f"Sample abundances (first 5): {seq_result.sample_abundances[:5]}")
print(f"Iterations: {seq_result.iterations}")
print(f"Converged: {seq_result.converged}")

# Batched result
print("\n" + "="*60)
print("BATCHED RESULT")
print("="*60)
batch_input = data[np.newaxis, :, :]  # (1, 5, 10)
batch_result = batched_median_polish_gpu(batch_input, max_iter=10, eps=0.01, use_gpu=False)
print(f"Sample abundances (first 5): {batch_result[0, :5]}")

# Compare
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
diff = batch_result[0] - seq_result.sample_abundances
print(f"Max absolute difference: {np.max(np.abs(diff)):.6f}")
print(f"Differences (first 5): {diff[:5]}")

# Manual step-by-step for debugging
print("\n" + "="*60)
print("MANUAL STEP-BY-STEP (NumPy)")
print("="*60)

residuals = data.copy()
row_effects = np.zeros(5)
col_effects = np.zeros(10)

for iteration in range(10):
    print(f"\nIteration {iteration + 1}:")

    # Row sweep
    row_medians = np.nanmedian(residuals, axis=1)
    print(f"  Row medians: {row_medians}")
    residuals = residuals - row_medians[:, np.newaxis]
    row_effects = row_effects + row_medians
    print(f"  Row effects: {row_effects}")

    # Column sweep
    col_medians = np.nanmedian(residuals, axis=0)
    print(f"  Col medians (first 3): {col_medians[:3]}")
    residuals = residuals - col_medians[np.newaxis, :]
    col_effects = col_effects + col_medians
    print(f"  Col effects (first 3): {col_effects[:3]}")

    # Check convergence
    max_adjustment = max(np.max(np.abs(row_medians)), np.max(np.abs(col_medians)))
    print(f"  Max adjustment: {max_adjustment:.6f}")

    if max_adjustment < 0.01:
        print(f"  Converged!")
        break

# Extract overall
overall = np.nanmedian(row_effects)
print(f"\nOverall (from row_effects): {overall:.6f}")
row_effects = row_effects - overall
print(f"Adjusted row effects: {row_effects}")

# Final abundances
final = overall + col_effects
print(f"Final abundances (first 5): {final[:5]}")
print(f"Sequential (first 5):       {seq_result.sample_abundances[:5]}")
print(f"Match: {np.allclose(final, seq_result.sample_abundances)}")
