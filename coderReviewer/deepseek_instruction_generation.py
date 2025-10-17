from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import os

import torch
import pandas as pd

from tqdm import tqdm

torch.cuda.empty_cache()

out_path = "./instructions/deepseek/generations.jsonl"
os.makedirs("./instructions/deepseek", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
)

model = (
    AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        dtype=torch.float16,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
        device_map ="auto" 
    )
    .eval()
)


def get_prompt(completion):
    messages = [
        # 1. System Prompt: Sets the overall goal for the model.
        {
            "role": "system",
            "content": "You are an expert programmer. Your task is to take a Python code snippet (a completion) and generate the corresponding natural language prompt and docstring that would describe its function.",
        },
        # 2. One-Shot Example (HumanEval/0): This demonstrates the expected input/output format.
        {
            "role": "user",
            "content": """```python
    numbers.sort()
    for i in range(len(numbers) - 1):
        diff = numbers[i+1] - numbers[i]
        if diff < threshold:
            return True
    return False
    ```""",
        },
        {
            "role": "assistant",
            "content": """Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True""",
        },
        # 3. Final User Request: This is where you insert the new completion.
        {"role": "user", "content": f"```python\n{completion}\n```"},
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_instructions(df:pd.DataFrame):
    df["generated_instructions"] = None
    for task_id ,group_df in tqdm(df.groupby("task_id") , desc = "Generating instructions"):
        prompts = group_df["instruction_generation_prompt"].to_list()
        inputs = tokenizer(
            prompts,
            return_tensors = "pt",
            padding = True,
            truncation = True
        ).to(model.device)                
        seq_len = inputs["input_ids"].shape[1]
        generated_output = model.generate(
            **inputs,
            use_cache = True,
            max_new_tokens = 512,
            temperature = 0.2,
            top_p = 0.95,
            do_sample = True,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,
        )
        new_tokens = generated_output[: , seq_len:]

        decoded_instructions = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens = True
        )
        df.loc[group_df.index , "generated_instructions"] = decoded_instructions
                
def run():
    df = pd.read_json("../coder/results/deepseek/eval.jsonl", lines=True)
    df["instruction_generation_prompt"] = df["completion"].apply(get_prompt)
    generate_instructions(df)
    
    df.drop(["instruction_generation_prompt"] , axis = 1).to_json(out_path , lines =True , orient = "records")


if __name__ == "__main__":

    run()
