import torch
import torch.nn as nn

torch.manual_seed(42)
vocab_size=10
embed_dim=5

weights=torch.randn(vocab_size,embed_dim)
print("weights \n",weights)

embedding=nn.Embedding.from_pretrained(weights)
print('embedding weight \n',embedding.weight)


inp_ids = torch.tensor([3])
output = embedding(inp_ids)
print(output, output.shape)


inp_ids = torch.tensor([3, 2, 8])
output = embedding(inp_ids)
print(output, output.shape)

inp_ids = torch.tensor([[3, 2, 8], [1, 4, 5]])
output = embedding(inp_ids)
print(output, output.shape)