import torch.nn as nn
import torch.nn.functional as F


class CreditScoreModel(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.input_layer=nn.Linear(input_dim,hidden_dim)
        self.hidden_layer1=nn.Linear(hidden_dim,hidden_dim)
        self.hidden_layer2=nn.Linear(hidden_dim,hidden_dim)
        self.output_layer=nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x=F.relu(self.input_layer(x))
        x=F.relu(self.hidden_layer1(x))
        x=F.relu(self.hidden_layer2(x))
        x=F.softmax(self.output_layer(x),dim=1)
        return x