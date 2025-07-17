import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(43)
lin_model1=nn.Linear(3,4)
input=torch.randn(1,3)
print('input is :',input,input.shape)
lin_model1_output=lin_model1(input)
print('Linear model 1 output:',lin_model1_output,lin_model1_output.shape)

relu_model=nn.ReLU()
relu_output=relu_model(lin_model1_output)
print('Relu function ouput',relu_output,relu_output.shape)
lin_model2=nn.Linear(4,1)
lin_model2_output=lin_model2(relu_output)
print('Linear model 2 output:',lin_model2_output,lin_model2_output.shape)


seq_model=nn.Sequential(lin_model1,relu_model,lin_model2)
output_from_seq_model=seq_model(input)
print('ouput from sequential model',output_from_seq_model,output_from_seq_model.shape)