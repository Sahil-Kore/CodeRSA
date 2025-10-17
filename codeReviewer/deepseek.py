import pandas as pd 
import os 
import numpy as np



def calculate_sum_c_i_prime(x:pd.Series):
    literal_listener_scores = x.to_numpy()
    logs =  np.log(literal_listener_scores)
    sum_c_i_prime  = np.sum(logs)
    return sum_c_i_prime

    
with os.scandir("./results") as entries:
    for entry in entries:
        if entry.is_dir():
            path = os.path.relpath(entry.path)

            df = pd.read_json(f"{path}/eval.jsonl" , lines = True)
            df["sum_c_i_prime"] = df.groupby("task_id")['literal_listener'].transform(calculate_sum_c_i_prime)
            print(df["sum_c_i_prime"])
            df["pragmatic_speaker"] = -   df ["literal_listener"] / df["sum_c_i_prime"]
            break
        
        
