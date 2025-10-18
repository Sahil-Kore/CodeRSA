import numpy as np
import os
import pandas as pd

from pathlib import Path

target_name = "cartesian_product.jsonl"
base_dir = Path("./instructions")

for file_path in base_dir.rglob(target_name):
    full_df = pd.read_json(file_path, lines=True)

    input_chunk_dir = file_path.parent
    os.makedirs(input_chunk_dir, exist_ok=True)

    num_chunks = 8
    df_chunks = np.array_split(full_df, num_chunks)

    for i, chunk_df in enumerate(df_chunks):
        chunk_path = os.path.join(input_chunk_dir, f"chunk_to_process_{i}.jsonl")
        chunk_df.to_json(chunk_path, orient="records", lines=True)
        print(f"Saved chunk {i} to {chunk_path}")