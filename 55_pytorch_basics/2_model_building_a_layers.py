import torch
import torch.nn as nn

torch.manual_seed(0)
x=torch.randn(5)
print('random tensor',x)

# 5 features as input and 1 output , single neuron layer
lin=nn.Linear(5,1)
print('lin weight',lin.weight)
print('lin x',lin(x))

# lin1=nn.Linear(5,2)
# print(lin1.weight)
# print(lin1(x))






