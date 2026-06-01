import os
import pandas as pd

def download_dataset_with_pandas():
    url = "https://raw.githubusercontent.com/611noorsaeed/Resume-Screening-App/main/UpdatedResumeDataSet.csv"
    target_dir = "data"
    target_file = os.path.join(target_dir, "UpdatedResumeDataSet.csv")
    
    print("Stage 1: Initializing directory checks...")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created folder path: '{target_dir}'")
        
    print("Stage 2: Streaming dataset from remote mirror via Pandas...")
    try:
        # Stream raw CSV straight into a dataframe dataframe
        df = pd.read_csv(url)
        
        print("Stage 3: Verifying structural integrity...")
        print(f"Rows found: {df.shape[0]} | Columns found: {df.shape[1]}")
        print(f"Column names detected: {list(df.columns)}")
        
        # Save to disk
        df.to_csv(target_file, index=False)
        print(f"Success! File explicitly locked and written to: {target_file}")
        
    except Exception as e:
        print(f"Core Engine Error during download pipeline: {e}")

if __name__ == "__main__":
    download_dataset_with_pandas()