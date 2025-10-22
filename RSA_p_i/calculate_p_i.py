import pandas as pd

import os 
os.makedirs("p_i",exist_ok= True)

files = ["../RSA/pragmatic_speaker/ds.jsonl" , "../RSA/pragmatic_speaker/tl.jsonl"]



cols_to_select = ["task_id" , "completion" , "instruction" , "log_probs_c_given_i"]
for file in files :
    df = pd.read_json(file , lines = True)
    
    new_df = df[cols_to_select]
    
    unique_instructions_df = df[["task_id" , "instruction"]].drop_duplicates().copy()
    unique_instructions_df["instruction_length"] = unique_instructions_df["instruction"].str.len()
    unique_instructions_df["length_weight"] = 1.0 / unique_instructions_df["instruction_length"]

    denominator = unique_instructions_df.groupby("task_id")["length_weight"].transform("sum")
    unique_instructions_df["p_i"] = unique_instructions_df["length_weight"]/ denominator
    
    
    final_df = new_df.merge(
        unique_instructions_df[["task_id" , "instruction" , "p_i"]],
        on = ["task_id" , "instruction"],
        how = "left"
    )
    
    print(final_df.head())
    
    if "ds"  in file :
        final_df.to_json("./p_i/ds_p_i.jsonl", lines = True , orient = "records")
    else :
        final_df.to_json("./p_i/tl_p_i.jsonl", lines = True , orient = "records")