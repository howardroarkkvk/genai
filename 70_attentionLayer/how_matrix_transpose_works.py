import torch
import torch.nn as nn
x=torch.randn(1,3,3)
print(x,x.shape)
# x=torch.tensor([[[1,2,3],[3,4,5],[5,6,7]]])
# print(x,x.shape)
y=x.transpose(0,1)
print(y,y.shape)

# z=x.transpose(-2,-1)
# print(z,z.shape)