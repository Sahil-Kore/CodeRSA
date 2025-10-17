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

out_path = "./instructions/phi/generations.jsonl"
os.makedirs("./instructions/phi", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    dtype=torch.float16,
    quantization_config=quantization_config,
    attn_implementation="sdpa",
    device_map="auto",
).eval()


def get_prompt(completion):
    one_shot_example = """Instruct: You are an expert programmer. Your task is to take a Python code snippet (a completion) and generate the corresponding natural language prompt and docstring that would describe its function.
    Code: ```python
    numbers.sort()
    for i in range(len(numbers) - 1):
        diff = numbers[i+1] - numbers[i]
        if diff < threshold:
            return True
    return False
    ```
    Output:Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """

    final_prompt = f"""{one_shot_example}\nInstruct: You are an expert programmer. Your task is to take a Python code snippet (a completion) and generate the corresponding natural language prompt and docstring that would describe its function.
    Code:```python\n{completion}```\nOutput:"""
    return final_prompt


def generate_instructions(df: pd.DataFrame):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    df["generated_instructions"] = None
    for task_id, group_df in tqdm(
        df.groupby("task_id"), desc="Generating instructions"
    ):
        prompts = group_df["instruction_generation_prompt"].to_list()
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        seq_len = inputs["input_ids"].shape[1]
        generated_output = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated_output[:, seq_len:]

        decoded_instructions = tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )
        df.loc[group_df.index, "generated_instructions"] = decoded_instructions


def run():
    df = pd.read_json("../coder/results/phi/eval.jsonl", lines=True)
    df["instruction_generation_prompt"] = df["completion"].apply(get_prompt)
    generate_instructions(df)

    df.drop(["instruction_generation_prompt"], axis=1).to_json(
        out_path, lines=True, orient="records"
    )


if __name__ == "__main__":
    run()
