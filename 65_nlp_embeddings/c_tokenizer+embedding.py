import torch
from torch.nn import Embedding
from transformers import AutoTokenizer

model_id='openai-community/gpt2'
tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
tokenizer.pad_token=tokenizer.eos_token

torch.manual_seed(42)
embed_dim=50
embed=Embedding(tokenizer.vocab_size,embed_dim)
print(embed.weight,embed.weight.shape)


sentences = [
    "Life is short, eat dessert first",
    "I love ice cream",
    "I love chocolate cake",
]

inputs=tokenizer(sentences,padding='max_length',truncation=True,max_length=10,return_tensors='pt')
print(inputs['input_ids'],inputs['input_ids'].shape)
embedded_output=embed(inputs['input_ids'])
print(embedded_output,embedded_output.shape)
x=torch.mean(embedded_output,dim=1)
print(x,x.shape)
# for input in inputs['input_ids']:
#     embedded_output=embed(input)
#     print(embedded_output,embedded_output.shape)

