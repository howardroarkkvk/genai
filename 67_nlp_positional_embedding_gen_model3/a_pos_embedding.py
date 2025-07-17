import torch.nn as nn
import torch

torch.manual_seed(42)
context_length,embed_dim=10,5
embedding=nn.Embedding(context_length,embed_dim)
print(embedding.state_dict())


positions =torch.arange(0,context_length)
print(positions)
output=embedding(positions)
print(output)