import torch
import torch.nn as nn
from transformers import AutoTokenizer

model_id='openai-community/gpt2'
tokenizer=AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token=tokenizer.eos_token

sentence='A fluffy blue creature roamed the verdant forest'
inputs=tokenizer(sentence,padding='max_length',truncation=True,max_length=10,return_tensors='pt')
print(inputs['input_ids'])

torch.manual_seed(42)
embed_dim=50
embed=nn.Embedding(tokenizer.vocab_size,embed_dim)
embedded_sentence=embed(inputs['input_ids'])
print(f"embedded sentence is :\n {embedded_sentence} \n it's shape is {embedded_sentence.shape} ")

attention=nn.MultiheadAttention(embed_dim=embed_dim,num_heads=1,batch_first=True)
attn_output,attn_weight=attention(embedded_sentence,embedded_sentence,embedded_sentence)
print(f" attention output shape is : \n {attn_output.shape}")
print(f"attention weight: \n {attn_weight} and attention weight shape is : \n {attn_weight.shape}")



