import torch
import torch.nn as nn
import torch.nn.functional as F

x=torch.randn(2,2)
# y=torch.flatten(x)#,start_dim=1,end_dim=
print('shape of input',x.shape,x)
# print('shape after flatten',y.shape,y)


# x1=torch.randn(3,2,2,2)
# y1=torch.flatten(x1,start_dim=1)#,start_dim=1,end_dim=
# print('shape of input1',x1.shape,x1)
# print('shape after flatten',y1.shape,y1)


# x2=torch.randn(3,2,2,2)
# y2=torch.flatten(x1,start_dim=2)#,start_dim=1,end_dim=
# print('shape of input2',x2.shape,x2)
# print('shape after flatten',y2.shape,y2)

# x3=torch.randn(3,2,2,2)
# y3=torch.flatten(x1,start_dim=3)#,start_dim=1,end_dim=
# print('shape of input3',x3.shape,x3)
# print('shape after flatten',y3.shape,y3)