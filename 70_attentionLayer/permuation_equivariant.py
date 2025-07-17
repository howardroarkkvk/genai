import torch
import torch.nn as nn
import torch.nn.functional as F


class PermEquivariate(nn.Module):
    def __init__(self):
        super().__init__()
        self.output=nn.Linear(1,4)
    
    def forward(self,x):
        return self.output(x)
        



if __name__=='__main__':
    x=torch.tensor([[[1.0],[2.0],[3.0]]])
    print(x,x.shape)
    x_perm=x[:,[2,0,1]]
    print(x_perm,x_perm.shape)
    torch.manual_seed(42)
    perm_equivariate=PermEquivariate()

    o1=perm_equivariate(x)
    print(o1,o1.shape)
    o2=perm_equivariate(x_perm)
    print(o2,o2.shape)



