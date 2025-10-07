from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from Core.evaluation import filter_code, run_eval, fix_indents

import os
import torch


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
        max_new_tokens=512,
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
    num_sample_per_task = 10
    out_path = "results/codeparrot-10/eval.jsonl"
    os.makedirs("results/codeparrot-10", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small")

    model = torch.compile(
        AutoModelForCausalLM.from_pretrained(
            "codeparrot/codeparrot-small",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .eval()
        .to("cuda")
    )

    run_eval(
        model, tokenizer, num_sample_per_task, out_path, generate_batch_completion, True
    )
