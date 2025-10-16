import pandas as pd 
import os 
import numpy as np



def sum_c_i_prime(x):
    scores = x.apply(lambda x : x[1])
    logs =  np.log(scores)
    score_per_task  = np.sum(logs)
    return score_per_task

    
with os.scandir("./results") as entries:
    for entry in entries:
        if entry.is_dir():
            path = os.path.relpath(entry.path)

            df = pd.read_json(f"{path}/eval.jsonl" , lines = True)
            df["sum_c_i_prime"] = df.groupby("task_id")['completion'].transform(sum_c_i_prime)
            print(df["sum_c_i_prime"])
            break
        
        
