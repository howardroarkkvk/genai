import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from a_sha_mha import *

tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token=tokenizer.eos_token

sentence='A fluffy blue creature roamed the verdant forest'
print('Input sentence is :',sentence)
inputs=tokenizer(sentence,padding='max_length',max_length=10,truncation=True,return_tensors='pt')
print('tokenized input is ',inputs)
torch.manual_seed(42)
embed_dim=6
embed=nn.Embedding(tokenizer.vocab_size,embed_dim)
embedded_sentence=embed(inputs['input_ids'])
print('Embedded sentence is :',embedded_sentence,embedded_sentence.shape)

attention=SingleHeadSelfAttention(embed_dim,k_dim=3,v_dim=3)
output=attention(embedded_sentence)
print(f'attention output is :',output,output.shape)

