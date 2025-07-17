import torch
import torch.nn as nn
import torch.nn.functional as F

x=torch.randn(4,4)
print(x)
y=x.split(2, dim=0)
print(y,y[0].shape)

z=x.split(2, dim=-1)
print(z,z[0].shape)