import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# class TextGenerationModel4(nn.Module): # it extends nn.Module, we have to define forward method ....also we have to use super().__init__()
torch.manual_seed(42)
model=nn.Linear(1,6)
x=torch.randn(1)
print('value of x is',x)
y=model(x)
print(y)
print('model parameters:\n',model.parameters())
for i in model.parameters():
    print(i)
print('model state dict:\n',model.state_dict())
print('model named parameters:\n',model.named_parameters())
for name,param in model.named_parameters():
    print(f'name :\n {name}')
    print(f'param :\n {param}')
    print(f'param shape : {param.shape}')


