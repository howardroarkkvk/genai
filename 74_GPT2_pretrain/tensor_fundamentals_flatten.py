import torch
import torch.nn as nn
import torch.nn.functional as F

input=torch.randn(2,5,6)
print(f'input is of shape {input.shape} : \n {input}')
output=input.flatten(0,1)
print(f'output is of shape {output.shape} : \n {output}')