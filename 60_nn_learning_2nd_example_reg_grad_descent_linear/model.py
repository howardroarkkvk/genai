import torch.nn as nn
import torch.nn.functional as F


class RegressionModel2(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super().__init__()
        self.input_layer=nn.Linear(input_dim,hidden_dim)
        self.hidden_layer1=nn.Linear(hidden_dim,hidden_dim)
        self.hidden_layer2=nn.Linear(hidden_dim,hidden_dim)
        self.output_layer=nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        # print('first input',x)
        x=self.input_layer(x)
        # print('x after input layer',x)
        x=self.hidden_layer1(x)
        # print('x after hidden layer1',x)
        x=self.hidden_layer2(x)
        # print('x after hidden layer2',x)
        x=self.output_layer(x)
        # print('x after output Layer',x)
        return x