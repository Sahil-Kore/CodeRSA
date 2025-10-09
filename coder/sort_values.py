import pandas as pd 
import os 

with os.scandir("./results") as entries:
    for entry in entries :
        if entry.is_dir():
            path = os.path.relpath(entry.path)
            df = pd.read_json(f"{path}/eval.jsonl" , lines = True)
            df['scores'] = df['completion'].apply(lambda x :x[1])
            df['completion']  = df['completion'].apply(lambda x :x[0])
            sorted_df = df.groupby("task_id" , group_keys= False).apply(lambda x : x.sort_values(by = "scores" , ascending= False))
            top_5 = df.groupby("task_id" , group_keys= False).apply(lambda x : x.sort_values(by = "scores" , ascending= False).head(5))
            top_5[["task_id","completion"]].to_json(f"{path}/eval_sorted.jsonl" , lines= True , orient = "records")