
import pandas as pd
import numpy as np
from pathlib import Path

def create_sporadic_metadata():
    input_path = "aals_dataportal_datatable.csv"
    output_path = "data/sporadic_metadata.csv"
    
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found.")
        return

    print("Loading metadata...")
    df = pd.read_csv(input_path)
    
    # Logic to identify Familial ALS
    # 1. ClinReport_Mutations_Details is not NaN/Empty
    # 2. HasVariant is not NaN/Empty (and not just "None")
    
    # Helper to check if a value indicates a mutation
    def has_mutation(row):
        # Check ClinReport
        cr = row.get('ClinReport_Mutations_Details')
        if pd.notna(cr) and str(cr).strip() != '':
            return True
            
        # Check HasVariant
        hv = row.get('HasVariant')
        if pd.notna(hv) and str(hv).strip() != '':
            return True
            
        return False

    # Apply logic
    print("Filtering Familial ALS...")
    mask_als = df['SUBJECT_GROUP'] == 'ALS'
    mask_mutation = df.apply(has_mutation, axis=1)
    
    familial_indices = df[mask_als & mask_mutation].index
    sporadic_indices = df[mask_als & ~mask_mutation].index
    
    print(f"  Total ALS: {mask_als.sum()}")
    print(f"  Familial ALS (to be relabeled): {len(familial_indices)}")
    print(f"  Sporadic ALS (to be kept): {len(sporadic_indices)}")
    
    # Update SUBJECT_GROUP for Familial ALS
    # We rename them so they don't match the 'ALS' case-value in impute.py
    df.loc[familial_indices, 'SUBJECT_GROUP'] = 'Familial ALS'
    
    # Verify
    print("\nUpdated SUBJECT_GROUP distribution:")
    print(df['SUBJECT_GROUP'].value_counts())
    
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    create_sporadic_metadata()
