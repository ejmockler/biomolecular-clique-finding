
import pandas as pd
import sys

def inspect_metadata():
    try:
        df = pd.read_csv("aals_dataportal_datatable.csv")
        print("Columns:", list(df.columns))
        
        # Check potential columns for ALS Type
        candidates = ['SUBJECT_GROUP', 'SUBJECT_SUBGROUP', 'ClinReport_Mutations_Details', 'HasVariant', 'Gene_Mutation', 'Diagnosis']
        for col in candidates:
            if col in df.columns:
                print(f"\n--- {col} Value Counts ---")
                print(df[col].value_counts(dropna=False).head(20))
                
        # Check cross-tabulation of Group vs Mutation
        if 'ClinReport_Mutations_Details' in df.columns and 'SUBJECT_GROUP' in df.columns:
            print("\n--- ALS with Mutations ---")
            als_df = df[df['SUBJECT_GROUP'] == 'ALS']
            has_mut = als_df['ClinReport_Mutations_Details'].notna() & (als_df['ClinReport_Mutations_Details'] != '')
            print(f"Total ALS: {len(als_df)}")
            print(f"ALS with Mutation Details: {has_mut.sum()}")
            print(f"ALS without Mutation Details (Sporadic?): {(~has_mut).sum()}")
            
            # Check HasVariant
            if 'HasVariant' in df.columns:
                 print(f"\nHasVariant value counts in ALS:")
                 print(als_df['HasVariant'].value_counts(dropna=False))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_metadata()
