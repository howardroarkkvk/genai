from transformers import AutoTokenizer

model_id='openai-community/gpt2'
tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
print(tokenizer)

texts=['the last he painted, you know, Mrs. Gisburn said with pardonable pride.']

for text in texts:
    print(text)
    print('ids')
    ids=tokenizer.encode(text)
    print('after encodeing',ids)
    output=tokenizer.decode(ids)
    print('output after decoding is :',output)

# text=''.join([tokenizer.decode(id) for id in ids])
# print(text)