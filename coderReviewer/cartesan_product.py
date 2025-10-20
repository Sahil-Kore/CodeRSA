import pandas as pd 
import os       
from human_eval.data import read_problems

problems = read_problems()

files = ["./instructions/deepseek/generations.jsonl", './instructions/tinyLlama/generations.jsonl']

for file in files :
    cartesian_df_list  = []
    df = pd.read_json(file,lines = True)

    for task_id, group_df in df.groupby("task_id"):
        instructions = group_df["generated_instructions"].to_list()
        instructions.append(problems[task_id]["prompt"])
        instr_df = pd.DataFrame({"instruction" : instructions})
        
        group_df = group_df.drop(["generated_instructions"] , axis = 1)
        
        group_df["key"] = 1
        instr_df["key"] = 1
        cartesian_df = pd.merge(group_df , instr_df, on = "key").drop("key" , axis = 1)
        
        cartesian_df_list.append(cartesian_df)

    final_df = pd.concat(cartesian_df_list , axis =0 , ignore_index= True)
    dir_name = os.path.dirname(file)
    base_name = os.path.splitext(os.path.basename(file))[0]
    output_path = os.path.join(dir_name , f"{base_name}_cartesian.jsonl")
    final_df .to_json(output_path , orient = "records", lines = True)
    