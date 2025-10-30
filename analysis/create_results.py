import pandas as pd 

ds_results={
    'random':0.3719512195121951,
    'coder':0.35365853658536583,
    'CodeRSA':0.3719512195121951,
    "CodeRSA_p_i":0.3760162601626016
}


tl_results={
    'random':0.07926829268292683,
    'coder':0.07520325203252032,
    'CodeRSA':0.08943089430894309 ,
    "CodeRSA_p_i":0.08943089430894309
}

ds_df = pd.DataFrame(ds_results , index = ["pass@1"])

tl_df = pd.DataFrame(tl_results, index = ["pass@1"])

ds_df.to_json("./ds_conc.jsonl" , lines = True , orient="records")
tl_df.to_json("./tl_conc.jsonl" , lines = True , orient="records")
