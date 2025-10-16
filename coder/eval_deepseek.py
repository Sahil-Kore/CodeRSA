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
) -> list[tuple[str, float]]:
    input_batch = [prompt for _ in range(batch_size)]

    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_output = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        return_dict_in_generate=True,
    )

    batch_completions_text = tokenizer.batch_decode(
        generated_output.sequences[:, input_ids_cutoff:], skip_special_tokens=True
    )

    batch_scores = []
    epsilon = 1e-45
    for i in range(batch_size):
        sequence_log_prob = 0.0
        sequence_ids = generated_output.sequences[i, input_ids_cutoff:]
        for j, token_id in enumerate(sequence_ids):
            logits_for_token = generated_output.scores[j][i, :].to("cpu", dtype=torch.float32)
            probs = torch.nn.functional.softmax(logits_for_token , dim = -1 )
            sequence_log_prob += torch.log(probs[token_id] + epsilon).item()
            
        batch_scores.append(sequence_log_prob)

    processed_results = [
        (filter_code(fix_indents(completion)), score)
        for completion, score in zip(batch_completions_text, batch_scores)
    ]
    return processed_results


if __name__ == "__main__":
    num_sample_per_task = 8
    out_path = "results/deepseek/eval.jsonl"
    os.makedirs("results/deepseek", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct"
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            dtype=torch.float16,
            quantization_config=quantization_config,
            attn_implementation="sdpa",
        )
        .eval()
        .to("cuda")
    )

    run_eval(
        model, tokenizer, num_sample_per_task, out_path, generate_batch_completion, True
    )
