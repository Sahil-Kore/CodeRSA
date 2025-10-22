import pandas as pd 


files = ["../coder/results/deepseek/eval.jsonl"]

for file in files :
    
    df = pd.read_json(file, lines = True)
    sub_df_list = []
    for task_id , group_df in df.groupby("task_id"):
        sub_df = group_df.sample(3 , random_state=42)
        sub_df_list.append(sub_df)
        
    
    final_df = pd.concat(sub_df_list, ignore_index= True)
    
    if "deepseek" in file:
        final_df.to_json("./ds_random_exp.jsonl" , lines = True , orient = "records")
    else:
        final_df.to_json("./tl_random_exp.jsonl" , lines = True , orient = "records")
        