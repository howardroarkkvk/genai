import torch

x=torch.randn(4,6)
print(f'x is of shape :--->{x.shape} \n {x}') # it has 2 rows 4 columns


y=x[0:1,:]
print(f'y is of shape : --->{y.shape} \n {y}') # it prints first row all columns values with size of 1 *6

y1=x[[0,1],[4,5]]
print(f'y1 is of shape : --->{y1.shape} \n {y1}') # it prints first row 5th column and 2nd row 6th column...

y2=x[[0,1],[1]]
print(f'y2 is of shape : --->{y2.shape} \n {y2}') # it prints first row second row 2nd column value....

y3=x[:,[1]]
print(f'y2 is of shape : --->{y3.shape} \n {y3}') # it prints first row second row 2nd column value....

