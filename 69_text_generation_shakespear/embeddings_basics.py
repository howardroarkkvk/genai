import torch
import torch.nn as nn
import torch.nn.functional as F

embed=nn.Embedding(5,7)
input=torch.tensor([1,2])
print(input,input.shape)
output=embed(input)
print(output,output.shape)

x=torch.arange(0,10).unsqueeze(0)
print(x,x.shape)