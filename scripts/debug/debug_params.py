
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

try:
    from cliquefinder.quality.outliers import compute_medcouple
    from cliquefinder.quality.imputation import soft_clip
except ImportError:
    print("Could not import cliquefinder modules. Make sure you are in the project root.")
    sys.exit(1)

def analyze_file(path_str):
    path = Path(path_str)
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"\nEvaluating {path.name}...")
    try:
        # specific to this dataset, it seems headerless or weird
        # Try loading with and without header
        df = pd.read_csv(path, index_col=0)
        print(f"  Shape: {df.shape}")
        
        # Check if it looks like numeric data
        if df.shape[1] > 10 and df.select_dtypes(include=np.number).shape[1] == df.shape[1]:
            data = df.values
            
            # 1. Check for pile-up (clipped values)
            # Flatten and check histogram
            flat = data.flatten()
            flat = flat[~np.isnan(flat)]
            
            p01, p99 = np.percentile(flat, [1, 99])
            print(f"  Range (1st-99th): {p01:.2f} to {p99:.2f}")
            
            # Check for values exactly at thresholds (e.g. 5.12, assuming scaling)
            # Actually, standardizing first (Robust scale) serves better
            median = np.median(flat)
            mad = np.median(np.abs(flat - median))
            z_scores = (flat - median) / (mad * 1.4826)
            
            print(f"  MAD-Z Range: {z_scores.min():.2f} to {z_scores.max():.2f}")
            
            n_gt_5 = (z_scores > 4.9).sum()
            n_gt_3 = (z_scores > 3.4).sum()
            print(f"  Values > 3.5 MAD-Z: {n_gt_3} ({100*n_gt_3/len(z_scores):.2f}%)")
            print(f"  Values > 5.0 MAD-Z: {n_gt_5} ({100*n_gt_5/len(z_scores):.2f}%)")
            
            # Check pile-up at 3.5 or 5.0
            # Bin the Z-scores
            hist, bins = np.histogram(z_scores, bins=100, range=(-6, 6))
            # print("  Z-score Histogram (bins of 0.12):")
            # for i in range(len(hist)):
            #     if hist[i] > 0 and abs(bins[i]) > 3:
            #         print(f"    {bins[i]:.1f}: {hist[i]}")

            # 2. Skewness Analysis (Medcouple)
            # Compute medcouple on a random sample of features to save time
            n_feats = min(100, data.shape[0])
            indices = np.random.choice(data.shape[0], size=n_feats, replace=False)
            medcouples = []
            for idx in indices:
                row = data[idx, :]
                row = row[~np.isnan(row)]
                if len(row) > 10:
                    medcouples.append(compute_medcouple(row))
            
            mc_arr = np.array(medcouples)
            print(f"  Medcouple (Skewness) Summary (n={n_feats}):")
            print(f"    Mean: {mc_arr.mean():.2f}")
            print(f"    Min/Max: {mc_arr.min():.2f} / {mc_arr.max():.2f}")
            print(f"    % Right Skewed (>0.1): {(mc_arr > 0.1).mean()*100:.1f}%")
            
            # 3. Simulate Parameters
            # Take one highly skewed feature
            if len(mc_arr) > 0:
                max_skew_idx = indices[np.argmax(np.abs(mc_arr))]
                skew_row = data[max_skew_idx, :]
                skew_row = skew_row[~np.isnan(skew_row)]
                
                print(f"\n  Simulation on heavily skewed feature (MC={np.max(np.abs(mc_arr)):.2f}):")
                
                # Standard params: threshold=3.5, k=5.0
                med = np.median(skew_row)
                mad_val = np.median(np.abs(skew_row - med))
                upper = med + 3.5 * mad_val * 1.4826
                lower = med - 3.5 * mad_val * 1.4826
                
                print(f"    Bounds (±3.5 MAD): [{lower:.2f}, {upper:.2f}]")
                
                # Soft clip k=5
                clipped_k5 = soft_clip(skew_row, lower, upper, sharpness=5.0)
                # Soft clip k=1
                clipped_k1 = soft_clip(skew_row, lower, upper, sharpness=1.0)
                
                def get_clipped_count(arr, l, u):
                    # In soft clip, values are asymptotic, so check how many are "near" bound
                    # e.g. within 1% of range
                    r = u - l
                    return ((arr > u - 0.01*r) | (arr < l + 0.01*r)).sum()

                print(f"    Original outliers (>bounds): {((skew_row > upper) | (skew_row < lower)).sum()}")
                print(f"    Compressed near bounds (k=5): {get_clipped_count(clipped_k5, lower, upper)} (Hard wall effect)")
                print(f"    Compressed near bounds (k=1): {get_clipped_count(clipped_k1, lower, upper)} (Smoother)")
                
    except Exception as e:
        print(f"  Error reading: {e}")

# Check files
files = [
    'output/proteomics/imputed.data.csv',
    'numeric__AALS-RNAcountsMatrix.csv',
    'aals_cohort1-6_counts_merged.csv'
]

for f in files:
    analyze_file(f)
