import pandas as pd 
from pathlib import Path

target_name = "generations.jsonl"
base_dir = Path("./instructions")

for file_path in base_dir.rglob(target_name):
    df = pd.read_json(file_path , lines = True)
    
    all_pairs = []
    for task_id ,group_df in df.groupby("task_id"):
        base_df = group_df.rename(columns = {"generated_instructions" : "original_instructions"})
        instructions_df = group_df[["generated_instructions"]].reset_index(drop = True)

        base_df["key"] = 1
        instructions_df["key"]= 1
        
        task_pairs = pd.merge(base_df , instructions_df, on ="key").drop('key' , axis = 1)
        task_pairs["task_id"] = task_id
        
        all_pairs.append(task_pairs)
        
        

    pairs_df = pd.concat(all_pairs , ignore_index = True)

    pairs_df.rename(columns = {'generated_instructions' : 'instruction'} , inplace = True)

    new_path = file_path.parent / "cartesian_product.jsonl"
    pairs_df.to_json(new_path , orient= "records", lines = True)
