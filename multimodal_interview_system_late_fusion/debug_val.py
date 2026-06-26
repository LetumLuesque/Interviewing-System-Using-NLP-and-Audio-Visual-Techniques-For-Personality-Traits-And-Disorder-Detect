import pandas as pd
import os

def debug_val():
    seumld_path = r"D:\Grad\Datasets\SEUMLD\SEUMLD\Labels\Fine-grained-labels.csv"
    print(f"Checking Labels at: {seumld_path}")
    
    if os.path.exists(seumld_path):
        try:
            df = pd.read_csv(seumld_path)
            print("Columns:", df.columns.tolist())
            
            target_col = None
            if 'label' in df.columns: target_col = 'label'
            elif 'Label' in df.columns: target_col = 'Label'
            elif 'deception' in df.columns: target_col = 'deception'
            elif 'Deception' in df.columns: target_col = 'Deception'
            elif 'Class' in df.columns: target_col = 'Class'
            
            if target_col:
                print(f"Found Label Column: '{target_col}'")
                print("Value Counts:")
                print(df[target_col].value_counts())
                print(f"Unique values: {df[target_col].unique()}")
            else:
                print("!!! NO LABEL COLUMN FOUND !!!")
                print("First 5 rows:")
                print(df.head())
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        print(f"File not found: {seumld_path}")

if __name__ == "__main__":
    debug_val()
