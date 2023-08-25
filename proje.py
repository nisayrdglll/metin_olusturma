from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

input_prompt = "The development of science and technology has greatly impacted human life."
input_ids = tokenizer.encode(input_prompt, return_tensors="pt")

output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
