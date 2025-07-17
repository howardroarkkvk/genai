import torch
import torch.nn as nn
import torch.nn.functional as F
from b_data_loader import * 
from d_model import * 


# input=torch.tensor([2.0,1.0,0.5])
input=torch.randn(5,3,15)
print(input.size(0))

v,indices=torch.topk(input,min(10,input.size(-1)))
print(v,v.shape)
# print(indices)
print(v[:,[-1]])

# output=F.softmax(input,dim=0)
# print(output)
# log_output=-torch.log(output)
# print(log_output)