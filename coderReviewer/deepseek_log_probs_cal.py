import torch 
from tqdm import tqdm 

import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
@torch.inference_mode()

def get_log_prob_for_batch(instructions: list[str] , completions :list[str],model , tokenizer):
    prompts = [inst + comp for inst, comp in zip(instructions , completions )]

    prompt_inputs = tokenizer(prompts, padding = True, return_tensors = "pt").to(model.device)
    instruction_inputs = tokenizer(instructions, padding = True , return_tensors = "pt").to(model.device)

    instruction_lengths = instruction_inputs.attention_mask.sum(dim = 1)

    outputs = model(**prompt_inputs)
    logits = outputs.logits 
    
    log_probs = torch.nn.functional.log_softmax(logits , dim = -1)

    labels = prompt_inputs.input_ids
    
    batch_log_probs = []
    for i in range (len(instructions)):
        
        completion_logits = log_probs[i , instruction_lengths[i]-1:-1 , :]
        completions_labels = labels [i , instruction_lengths[i]:].unsqueeze(-1)

        gathered_log_probs = completion_logits.gather(dim = -1 , index = completions_labels)

        total_log_probs = gathered_log_probs.sum().item()
        batch_log_probs.append(total_log_probs)
    
    return batch_log_probs

tokenizer = AutoTokenizer("deepseek-ai/deepseek-coder-1.3b-instruct")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    dtype=torch.float16,
    quantization_config=quantization_config,
    attn_implementation="sdpa",
    device_map="auto",
).eval()


if tokenizer.pad_token is None :
    tokenizer.pad_token = tokenizer.eos_token

pairs_df = pd.read_json("./instructions/deepseek/cartesian_product.jsonl" , lines = True)
batch_size = 8

results = []

for i in tqdm(range(0 , len(pairs_df) , batch_size)):
    batch = pairs_df.iloc[i:i+batch_size]
    
    instructions = batch["instruction"].tolist()
    completions = batch["completion"].tolist()

    log_probs = get_log_prob_for_batch(instructions, completions , model , tokenizer)
    results.extend(log_probs)
    

pairs_df['log_probs_c_given_i'] = results 

output_path = "./instructions/deepseek/probabilites.jsonl"
pairs_df.to_json(output_path , orient = "records" , lines = True)

