import torch
import torch.nn as nn

torch.manual_seed(42)
vocab_size,embed_dim=10,6

embed=nn.Embedding(vocab_size,embed_dim)
inp_ids=torch.tensor([3,2,8])

output=embed(inp_ids)
print('this is embed output : ',output,output.shape)

attention=nn.MultiheadAttention(embed_dim,num_heads=1)#,batch_first=True
mask=torch.triu(torch.ones(3,3),diagonal=1).bool()
print(mask)

attn_output,attn_weights=attention(output,output,output,attn_mask=mask)
print(attn_output,attn_output.shape)
print(attn_weights,attn_weights.shape)

attention=nn.MultiheadAttention(embed_dim,num_heads=2)#,batch_first=True
attn_output,attn_weights=attention(output,output,output,attn_mask=mask)
print(attn_output,attn_output.shape)
print(attn_weights,attn_weights.shape)