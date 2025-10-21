import pandas as pd
from human_eval.data import read_problems

problems = read_problems()
prompts = [problems[tasks]["prompt"] for tasks in problems.keys()]
files = ["../pragmatic_speaker/ds.jsonl", "../pragmatic_speaker/tl.jsonl"]
cols_subset = ["task_id" , "completion"]
for file in files :
    df = pd.read_json(file , lines = True)
    score_column = "pragmatic_speaker_score"

    listener_df = df[df["instruction"].isin(prompts)].copy()

    top_completions = listener_df.sort_values(
        by = ["instruction" , score_column],
        ascending= [True ,False]
    ).groupby("instruction").head(3)

    if "ds" in file:
        top_completions[cols_subset].to_json("./ds_top3.jsonl" ,lines =True , orient="records")
    else:
        top_completions[cols_subset].to_json("./tl_top3.jsonl" , lines = True , orient = "records")