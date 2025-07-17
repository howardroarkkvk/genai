import torch
import torch.nn.functional  as F

data=torch.randn(2)
print(data)


print(F.relu(data))

# relu basically pass +ve values as is and converts the negative values in array to zero.

data1=torch.randn(4)
print(data1)


print(F.relu(data1))


data2=torch.randn(5)
print(data2)
print(F.softmax(data2))

