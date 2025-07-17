import torch
import torch.nn as nn

batch_size,seq_len,embed_dim=2,5,10
layer_norm=nn.LayerNorm(embed_dim)
print(layer_norm.state_dict())


input=torch.randn(batch_size,seq_len,embed_dim)
print(input,input.shape)
# for i in range(input.shape[0]):
#     for j in range(input.shape[1]):
#         inner_tensor=input[i,j,:]
#         var=inner_tensor.var(unbiased=True)
#         stdev=inner_tensor.std(unbiased=True)
#         print(f'in for loop {var.item()} , {stdev.item()}',)
#         var_inner_tensor=inner_tensor-var.item()
#         std_var_inner_tensor=torch.div(var_inner_tensor,stdev.item())
#         print('inner tensor is :',inner_tensor)
#         print('std and var applied inner tensor',std_var_inner_tensor)

output=layer_norm(input)
print(output,output.shape)
# for i in range(output.shape[0]):
#     for j in range(output.shape[1]):
#         outer_tensor=output[i,j,:]
#         var=outer_tensor.var(unbiased=True)
#         stdev=outer_tensor.std(unbiased=True)
#         mean=outer_tensor.mean()
#         print(f'in for loop {var.item()} , {stdev.item()}, mean: {mean}')


# why layernorm
# 🏗️ Why is this helpful?
# It makes the model more stable and faster to train.

# It prevents any one feature from dominating the learning.

# Especially useful in architectures like Transformers, where each word/token is processed independently.

# 🔬 Analogy with Baking:
# Think of a recipe where you're mixing different ingredients (features).
# If one ingredient (say, salt) is 10x more than the others, it ruins the dish.

# So, you normalize the ingredients — convert all to the same scale — before combining. That’s LayerNorm for features.