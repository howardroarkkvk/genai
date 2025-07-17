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

# padding = true

# for text in texts:
#     print('input text to encode is :',text)
#     input=tokenizer.encode(text,padding=True,truncation=True,max_length=10)
#     print('after encode the value is ',input)
#     output=tokenizer.decode(input)
#     print('output after decode is ',output)

# padding till max length

# for text in texts:
#     print('input text to encode is :',text)
#     input=tokenizer.encode(text,padding='max_length',truncation=True,max_length=10)
#     print('after encode the value is ',input)
#     output=tokenizer.decode(input)
#     print('output after decode is ',output)

# padding till max length ignoring the padding in the output
# for text in texts:
#     print('input text to encode is :',text)
#     input=tokenizer.encode(text,padding='max_length',truncation=True,max_length=10)
#     print('after encode the value is ',input)
#     output=tokenizer.decode(input,skip_special_tokens=True)
#     print('output after decode is ',output)


for text in texts:
    print('input text to encode is :',text)
    input=tokenizer.encode(text,padding='max_length',truncation=True,max_length=10,return_tensors='pt')
    print(input)