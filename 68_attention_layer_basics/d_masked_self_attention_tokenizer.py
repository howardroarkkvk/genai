import torch 
import torch.nn as nn
from transformers import AutoTokenizer


model_id='openai-community/gpt2'
tokenizer=AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token=tokenizer.eos_token

sentence="Life is short, eat dessert first"
inputs=tokenizer(sentence, padding='max_length',truncation=True,max_length=10,return_tensors='pt')
print(inputs['input_ids'])
seq_len=inputs['input_ids'].shape[1]

print('sequence length is ',seq_len)

embed=nn.Embedding(tokenizer.vocab_size,50)
output=embed(inputs['input_ids'])
output= output.permute(1,0,2)
print(output,output.shape)


attention=nn.MultiheadAttention(50,num_heads=1)#,batch_first=True
mask=torch.triu(torch.ones(seq_len,seq_len)).bool()
print(mask)
attn_ouput,attn_output_weights=attention(output,output,output,attn_mask=mask)
print(attn_ouput.shape)
print(attn_output_weights.shape)
