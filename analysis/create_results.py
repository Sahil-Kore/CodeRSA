import pandas as pd 

ds_results={
    'random':[0.3719512195121951,0.4329268292682927],
    'coder':[0.35365853658536583, 0.40853658536585363],
    'CodeRSA':[0.3719512195121951 , 0.4146341463414634],
    "CodeRSA_p_i":[0.3760162601626016 , 0.4146341463414634]
}


tl_results={
    'random':[0.07926829268292683, 0.0975609756097561],
    'coder':[0.07520325203252032,0.10365853658536585],
    'CodeRSA':[0.08943089430894309 , 0.0975609756097561],
    "CodeRSA_p_i":[0.08943089430894309, 0.0975609756097561]
}

ds_df = pd.DataFrame(ds_results , index = ["pass@1" , "pass@3"])

tl_df = pd.DataFrame(tl_results, index = ["pass@1" , "pass@3"])

ds_df.to_json("./ds_conc.jsonl" , lines = True , orient="records")
tl_df.to_json("./tl_conc.jsonl" , lines = True , orient="records")
