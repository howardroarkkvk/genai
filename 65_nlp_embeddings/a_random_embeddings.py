import torch
import torch.nn as nn

torch.manual_seed(32)

vocab_size=10
embed_size=5
embedding=nn.Embedding(vocab_size,embed_size)
print(embedding.weight)

# here we are 
inp_ids=torch.tensor([2])
output=embedding(inp_ids)
print(output,output.shape)

# inp_ids=torch.tensor([3,2,8])
# output=embedding(inp_ids)
# print(output,output.shape)


# inp_ids=torch.tensor([[3,2,8],[1,4,7]])
# output=embedding(inp_ids)
# print(output,output.shape)
