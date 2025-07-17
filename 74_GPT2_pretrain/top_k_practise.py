import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
input=torch.tensor([[2,9,5,4,3],[3,4,5,6,7]])
print(input,input.shape)

embedder=nn.Embedding(10,8)
print(embedder)
output=embedder(input)
# print(output,output.shape)

logits=output
print('logits are:',logits,logits.shape)

temperature = 5.0
logits=logits[:,-1,:] # here we are indexing the logits whose shape changes to 2d from 3d by keeping the last element for each sentence....now no more sentences, it's (1word sentence, embeddings.)..
print('logits without temp change:',logits,logits.shape) # it displays the last word from each sentence...
logits=logits/temperature
print('logits with temp change change:',logits,logits.shape) # it displays the last word from each sentence...

top_k=5
logits_size=logits.size(-1)
print(logits_size)
values,indices=torch.topk(logits,min(logits_size,top_k))
print(values,values.shape)
print(indices,indices.shape)
x=values[:,[-1]]
print(x)
logits[logits<x]=-float("Inf")
print(logits, logits.shape)

probs=torch.softmax(logits,dim=-1)
print(probs)
# ids=torch.argmax(probs,dim=-1)
ids=torch.multinomial(probs,num_samples=1)
print(ids)


