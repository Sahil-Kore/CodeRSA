import torch 
import pandas as pd 
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from tqdm import tqdm 
import os 

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")

quant_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    dtype = torch.float16,
    quantization_config = quant_config,
    device_map = "auto"
).eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
    

@torch.inference_mode()
def get_log_probs_batch(instructions, completions, model, tokenizer, max_length=1024):

    device = model.device

    # Concatenate instructions + completions
    prompts = [i + c for i, c in zip(instructions, completions)]

    # Tokenize
    prompt_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(device)
    instr_inputs = tokenizer(instructions, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(device)

    instr_lengths = instr_inputs.attention_mask.sum(dim=1)  # number of tokens in each instruction

    outputs = model(**prompt_inputs)
    logits = outputs.logits  # [batch, seq_len, vocab_size]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    labels = prompt_inputs.input_ids
    batch_log_probs = []

    for i in range(len(instructions)):
        # Slice logits for completion tokens only
        completion_logits = log_probs[i, instr_lengths[i]-1:-1, :]
        completion_labels = labels[i, instr_lengths[i]:].unsqueeze(-1)

        gathered_log_probs = completion_logits.gather(dim=-1, index=completion_labels)
        batch_log_probs.append(gathered_log_probs.sum().item())

    return batch_log_probs



df = pd.read_json("./instructions/deepseek/generations_cartesian.jsonl", lines=True)
batch_size = 8
results = []

for start in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[start:start+batch_size]
    instructions = batch["instruction"].tolist()
    completions = batch["completion"].tolist()

    batch_log_probs = get_log_probs_batch(instructions, completions, model, tokenizer)
    results.extend(batch_log_probs)

df["log_probs_c_given_i"] = results

os.makedirs("./log_probs" ,  exist_ok= True)

df.to_json("./log_probs/deepseek_log_probs.jsonl", orient="records", lines=True)

