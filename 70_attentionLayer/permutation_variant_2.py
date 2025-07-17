import torch
import torch.nn as nn
import torch.nn.functional as F

model=nn.Sequential(nn.Linear(3,4),nn.ReLU(),nn.Linear(4,1))

x1=torch.tensor([1.0,2.0,3.0])
print(x1.shape)
x2=torch.tensor([2.0,1.0,3.0])

print(model(x1))
print(model(x2)) # becuase model is initiated with different weights for the 2 columns hence the outputs are not same....