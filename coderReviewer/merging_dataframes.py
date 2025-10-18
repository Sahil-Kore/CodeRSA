import pandas as pd 
from pathlib import Path
import os 

BASE_DIR = Path("./instructions")

OUTPUT_DIR = Path("./merged_files")

col_subset = ["task_id" , "completion" , "literal_listener", "log_probs_c_given_i","log_sum_probs", "pragmatic_speaker"]

os.makedirs(OUTPUT_DIR , exist_ok=True)

def process_and_merge_model_chunks(model_dir :Path):
    print(model_dir.name)
    
    probabilities_dir= model_dir/ "probabilities"
    if not probabilities_dir.exists():
        print(f"No directory found {probabilities_dir}")
        return 

    print(f"Processing model :{model_dir.name}")

    processed_chunks = []
    
    chunk_files = sorted(probabilities_dir.glob("chunk_to_process*.jsonl"))
    if not chunk_files :
        print(f"No chunk file found in {probabilities_dir}")
    
    for file_path in chunk_files:
        df = pd.read_json(file_path,lines = True)

        df["log_sum_probs"] = df.groupby("completion")["log_probs_c_given_i"].transform("sum")
        df["pragmatic_speaker"] = -df ["log_probs_c_given_i"]/df["log_sum_probs"]

        processed_chunks.append(df)
    
    merged_df = pd.concat(processed_chunks , ignore_index=True)

    output_path = OUTPUT_DIR/ f"{model_dir.name}_final_scores.jsonl"
    merged_df[col_subset].to_json(output_path ,orient="records" , lines = True)



if __name__ == "__main__":
    for model_directory in BASE_DIR.iterdir():
        if model_directory.is_dir():
            process_and_merge_model_chunks(model_directory)
        
    print("completed")
    