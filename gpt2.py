from transformers import pipeline

# 1. Create the text-generation pipeline
# This will download the GPT-2 model and tokenizer automatically on the first run.
print("Loading the GPT-2 model...")
generator = pipeline('text-generation', model='gpt2')
print("Model loaded successfully.")

# 2. Define your prompt
prompt = "The future of artificial intelligence "
prompt2 = "who is SolidGoldMagikarp"

# 3. Generate text with specific parameters
print(f"\nGenerating text for the prompt: '{prompt}'")
outputs = generator(
    prompt2,
    max_length=75,              
    num_return_sequences=1,   
    temperature=1,            
    do_sample=True              
)

# 4. Print the generated outputs
print("\n--- Generated Sequences ---")
for i, output in enumerate(outputs):
    # The output is a dictionary, and the generated text is in the 'generated_text' key.
    print(f"{i+1}: {output['generated_text']}")
    print("-" * 20)
