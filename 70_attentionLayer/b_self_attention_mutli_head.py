import torch # for defining tensors...
import torch.nn as nn # this is for linear functions 
import torch.nn.functional as F # this is for activations
from transformers import AutoTokenizer # this is to import autotokenizer for converting the words to tokens
from a_sha_mha import * 

tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token=tokenizer.eos_token

sentence = "A fluffy blue creature roamed the verdant forest."

inputs=tokenizer(sentence,padding='max_length',max_length=10,truncation=True,return_tensors='pt')
print('Output of tokenizer is :',inputs['input_ids'],inputs['input_ids'].shape)

torch.manual_seed(42)
embed_dim=6
embed=nn.Embedding(tokenizer.vocab_size,embed_dim)
embedded_sentence=embed(inputs['input_ids'])
print('Embedded sentence is :',embedded_sentence,embedded_sentence.shape)


attention=MultiHeadSelfAttention(emb_dim=embed_dim,n_heads=3)
output=attention(embedded_sentence)
print('output is :',output,output.shape)






