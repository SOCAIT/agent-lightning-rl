import pandas as pd
import json
import os
from pathlib import Path

# Set paths
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / "data"
input_jsonl = data_dir / "fitness_scenarios.jsonl"
train_parquet = data_dir / "fitness_scenarios_train.parquet"
val_parquet = data_dir / "fitness_scenarios_val.parquet"

def process_and_save_data():
    if not input_jsonl.exists():
        print(f"Error: Input file {input_jsonl} not found.")
        return

    # Load JSONL data
    with open(input_jsonl, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Process data to flatten context or ensure it's in the format expected by the agent
    # The LitNutritionAgent.rollout expects 'input_question' and 'context'
    # The training loop reads parquet into a list of dicts.
    
    # Let's inspect the data structure based on previous reads:
    # {"id": ..., "context": {...}, "input_question": ..., "split": ...}
    
    # We can just load this into a DataFrame and split it.
    df = pd.DataFrame(data)
    
    # Filter for train and val/test splits
    # Assuming 'train' split exists. If 'val' doesn't exist, we might split 'train'
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] != 'train'] # treat test/val as validation
    
    # If val is empty, just take a small chunk of train
    if val_df.empty:
        print("No validation split found in data. Using 10% of train data for validation.")
        val_df = train_df.sample(frac=0.1, random_state=42)
        train_df = train_df.drop(val_df.index)
        
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Save to Parquet
    # We need to ensure columns that are dicts (like 'context') are stored correctly.
    # PyArrow handles JSON-like structures in object columns, but explicit conversion can be safer if needed.
    # For now, default pandas to_parquet should work if pyarrow is installed.
    
    try:
        train_df.to_parquet(train_parquet, engine='pyarrow')
        val_df.to_parquet(val_parquet, engine='pyarrow')
        print(f"Successfully saved:\n  {train_parquet}\n  {val_parquet}")
    except Exception as e:
        print(f"Error saving parquet files: {e}")

if __name__ == "__main__":
    process_and_save_data()
