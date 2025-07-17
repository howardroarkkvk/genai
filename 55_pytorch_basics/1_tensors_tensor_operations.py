import torch

# indexing in py torch...

x=torch.initial_seed()
print(x)

y=torch.manual_seed(43)
print(y)

r1=torch.randn(2,3)
print(r1)
# print(r1[0])
# print(r1[0,-1])
# print(r1[0,2])
# print(r1[1][1])

a=torch.randn(3)
print('a is :', a)
b=torch.randn(3)
print('b is :',b)
# addition in torch
print(a+b)
print(torch.add(a,b))

# multiplication in torch
print(torch.multiply(a,b))
# sum of all elements of a list
print(torch.sum(a))

# how to concatenate 2 lists

print(torch.cat((a,b),dim=0))

c=torch.randn(2,2)
print('c is :',c)
d=torch.randn(2,2)
print('d is :',d)
# in order to add column wise use dim, but for the dim to work on cat on the column wise the tensors have to be of 2 dimensions, otherwise it throws error
print(torch.cat([c,d],dim=1))


# reshaping of a tensor

e=torch.randn(6)
print('e is : ',e)
print(e.reshape(6,-1))
print(e.reshape(1,6))
# print(e.reshape(-1,,6))
f=e.reshape(1,6)


g=torch.randn(2,3)
print('g is :', g)

print(g.view(1,-1))
print(g.view(-1,1))

print(g.view(3,-1))
print(g.view(6,-1))
print(g.view(-1,6))



h =torch.randn(2,3,4)

print('h is :',h)

print(h.view(1,-1).shape)

print(h.view(6,4))
