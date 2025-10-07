import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    BitsAndBytesConfig,
)

from Core.evaluation import filter_code, run_eval, fix_indents

import os
import torch

torch.cuda.empty_cache()


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]

    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=384,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    batch_completion = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids], skip_special_tokens=True
    )

    return [filter_code(fix_indents(completion)) for completion in batch_completion]


if __name__ == "__main__":
    num_sample_per_task = 8
    out_path = "results/phi2/eval.jsonl"
    os.makedirs("results/phi2", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
    ).eval()

    run_eval(
        model, tokenizer, num_sample_per_task, out_path, generate_batch_completion, True
    )
