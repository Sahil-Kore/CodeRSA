from human_eval.data import write_jsonl , read_problems

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)


from tqdm import tqdm
import typing

BatchGenerator = typing.Callable[[PreTrainedModel , PreTrainedTokenizer , str , int], list[str]]

def filter_code (completion:str)-> str:
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def fix_indents(text : str) -> str:
    return text.replace("\t", "    ")

def split_batch(samples: list[str] , size = 4):
    mini_batches = []
    
    for i in range ( 0 , len(samples) , size ):
        mini_batches.append(samples[i : i + size])
        
    return mini_batches



def run_eval(
    model:PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples_per_task:int,
    out_path:str,
    generate_batch_completion: BatchGenerator,
    format_tabs : bool =False
):
    problems = read_problems()
    
    samples = []
    pbar = tqdm(total = len(problems) * num_samples_per_task)

    for task_id in problems:
        if format_tabs:
            prompt = problems[task_id]["prompt"].replace("    ", "\t")
        
        else:
            prompt = problems[task_id]["prompt"]

        
        batch_completion = generate_batch_completion(
            model , tokenizer , prompt, num_samples_per_task
        )
        
        for sample in batch_completion:
            result = dict (
                task_id= task_id,
                completion = sample
            )
            
            samples += [result]
        
        pbar.update(num_samples_per_task)
    
    write_jsonl(out_path , samples)