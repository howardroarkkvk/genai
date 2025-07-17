import torch
import torch.nn as nn
import torch.nn.functional as F

def model():
    element_fn=nn.Sequential(nn.Linear(1,4),nn.ReLU())
    aggregate_fn=nn.Sequential(nn.Linear(4,4),nn.ReLU(),nn.Linear(4,1))
    return  element_fn,aggregate_fn   

def fun_1(x:list):
    # torch.manual_seed(42)
    # x=torch.randn(3,1)
    element_fn,aggregate_fn=model()
    print(x)
    for i in x:
        print(i)
        
        element_fn_ouput=element_fn(i)
        print('elemental function output is  :',element_fn_ouput,element_fn_ouput.shape)

        sum_output=element_fn_ouput.sum(dim=1)#torch.sum(element_fn_ouput,dim=0)
        print('sum output is :',sum_output,sum_output.shape)


        aggregate_ouput=aggregate_fn(sum_output)
        print('aggregate output is',aggregate_ouput,aggregate_ouput.shape)

list1=[]
x=torch.tensor([[1.0],[2.0],[3.0]])
x=x.unsqueeze(0)
list1.append(x)
y=torch.tensor([[2.0],[1.0],[3.0]])
y=y.unsqueeze(0)
list1.append(y)
print('list1 is :',list1)
fun_1(list1)