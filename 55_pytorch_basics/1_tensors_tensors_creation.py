import torch


#  converting a list to tensor...
list1=[1,2,3]
t1=torch.Tensor(list1)
print(t1)


# 2 dimensional tensor
list2=[[1,2,3],[4,5,6]]
t2=torch.Tensor(list2)
print(t2)

list3=[[[1,2,3],[2,3,4]],[[4,5,6],[7,8,9]]]
t3=torch.Tensor(list3)
print(t3)

# This creates a single dimension tensor with same values every time, seed is like mentioning to the system to generate same values in the matrix..
torch.manual_seed(42)
r1=torch.randn(3)
print(r1)

r2=torch.randn(2,2)
print(r2)

r3=torch.ones(5)
print(r3)

r4=torch.ones(2,2)
print(r4)


r5=torch.ones(2,2,2)
print(r5)
