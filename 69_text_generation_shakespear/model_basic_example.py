import torch
import torch.nn as nn
import torch.nn.functional as F


model=nn.Linear(1,6)
print(model.parameters())
input=torch.randn(1)
print(input)
output=model(input)
print(output)
# class TextGenerationModel3(nn.Module):
for param in model.parameters():
    print(f' param:{param.shape}')
print('Total no. of elements')
print('****************************')
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

print(' This is inside named parameters:')
print('----------------------------------')
for name, param in model.named_parameters():
    print(f'name:{name}\nparameter is {param},\nparam shape:{param.shape}')
    if param.requires_grad:
        print(f"{name}:{param.shape}, total_elements={param.numel()}")

# model=nn.Linear(2,6)
# input=torch.randn(2)
# print(input)
# output=model(input)
# print(output)
# print(param_count(model))
