import pandas as pd 
import os 
import numpy as np
files = ["./log_probs/deepseek_log_probs.jsonl" , "./log_probs/tinyLlama_log_probs.jsonl"]

os.makedirs("pragmatic_speaker", exist_ok= True)

for file in files :
    df = pd.read_json(file , lines = True)
    sum_log_probs = df.groupby("completion")['log_probs_c_given_i'].transform('sum')
    print(type(sum_log_probs))
    df["pragmatic_speaker_score"] = - df["log_probs_c_given_i"] / sum_log_probs
    
    if "deepseek" in file:
        df.to_json("./pragmatic_speaker/ds.jsonl", lines = True , orient = "records")
    
    else :
        df.to_json("./pragmatic_speaker/tl.jsonl" , lines= True, orient= "records")