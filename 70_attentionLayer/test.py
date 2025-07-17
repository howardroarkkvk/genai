import torch
import torch.nn as nn
import torch.nn.functional as F


layer1=nn.Linear(6,3)
x=torch.randn(10,6)
print(x,x.shape)
output=layer1(x)
print(output,output.shape)

for i ,j in layer1.named_parameters():
    print(i,j)
