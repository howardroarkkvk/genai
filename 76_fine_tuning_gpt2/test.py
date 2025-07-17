import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from c_models import * 
from a_data_preparation import *

prompt = "What is the capital of india"
tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
input=tokenizer(prompt,return_tensors='pt')
input_to_model=input['input_ids']
print(input_to_model,input_to_model.shape)
model=GPT2Model.from_pretrained('gpt2',len(tokenizer))
logits=model(input_to_model)
print(logits,logits.shape)
logits=logits[:,-1,:]
probs=torch.softmax(logits,dim=-1)
print(probs,probs.shape)
id_next=torch.argmax(probs,dim=-1)
print(id_next,id_next.shape)
prompt={'system': 'reverse this array',
            'user':"[10,20,30,40,50]",
            'assistant':None}

output=chat_format(prompt)
print(output)