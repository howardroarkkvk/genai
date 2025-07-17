from transformers import AutoTokenizer

model_id='openai-community/gpt2'
tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
tokenizer.pad_token=tokenizer.eos_token
print(tokenizer)


texts = [
    "how are you?",
    "Akwirw ier",
    "Hello, do you like tea? In the sunlit terraces of someunknownPlace.",
]

# batch encode plus
inputs=tokenizer.batch_encode_plus(texts,padding='max_length',truncation=True,max_length=10)

print('batch encode plus',inputs)


inputs=tokenizer(texts,padding='max_length',truncation=True,max_length=10)
print('just batch encode',inputs)



for input in inputs['input_ids']:
    print(input)
    output=tokenizer.decode(input,skip_special_tokens=True)
    print(output)