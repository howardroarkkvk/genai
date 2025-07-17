import torch
import torch.nn as nn

torch.manual_seed(123)
vocab_size, embed_dim = 10, 6
embed = nn.Embedding(vocab_size, embed_dim)

inp_ids = torch.tensor([1, 2, 8])
output = embed(inp_ids)
print(output, output.shape)

attention = nn.MultiheadAttention(embed_dim, num_heads=1)
attn_output, attn_output_weights = attention(output, output, output)
print(attn_output, attn_output.shape)
print(attn_output_weights, attn_output_weights.shape)

attention = nn.MultiheadAttention(embed_dim, num_heads=2)
attn_output, attn_output_weights = attention(output, output, output)
print(attn_output, attn_output.shape)
print(attn_output_weights, attn_output_weights.shape)