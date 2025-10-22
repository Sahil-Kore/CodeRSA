import pandas as pd
import os
import numpy as np

os.makedirs("./pragmatic_speaker", exist_ok= True)

files = ["./p_i/ds_p_i.jsonl" , "./p_i/tl_p_i.jsonl"]


for file in files :
    df = pd.read_json(file , lines= True)

    df["log_p_i"] = np.log(df["p_i"])
    
    #numerator
    df["score_term"] = df ['log_probs_c_given_i'] + df["log_p_i"]

    
    denominator = df.groupby("completion")["score_term"].transform("sum")
    df["pragmatic_speaker_score"] = - df ["score_term"] / denominator

    if "ds" in file :
        df.to_json("./pragmatic_speaker/ds_scores.jsonl" , lines = True , orient = "records")
    else :
        df.to_json("./pragmatic_speaker/tl_scores.jsonl" , lines = True, orient = "records")