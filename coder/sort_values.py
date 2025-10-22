import pandas as pd 
import os 
from pathlib import Path
with os.scandir("./results") as entries:
    for entry in entries :
        print(entry)
        if entry.is_dir():
            path = os.path.relpath(entry.path)
            file_path =Path(path) / "eval.jsonl"
            df = pd.read_json(file_path , lines = True)
            print(df)
            top_3 = df.groupby("task_id" , group_keys= False).apply(lambda x : x.sort_values(by = "literal_listener" , ascending= False).head(3))
            top_3[["task_id","completion"]].to_json(f"{path}/top_3.jsonl" , lines= True , orient = "records")