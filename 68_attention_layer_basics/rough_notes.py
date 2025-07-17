import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

torch.manual_seed(42)
seq_len=1
batch_size=3
embed_dim=8
num_heads=2

x=torch.tensor([[1,5,10,14],[5,14,10,20]])
print(x)

embed=nn.Embedding(10000,embed_dim)
embed_output=embed(x)
print(embed_output,embed_output.shape)

mha=nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,batch_first=True) # typically MultiheadAttention takes batch as 2nd input, hence batch_first has to be givner
attn_output,attn_weights=mha(embed_output,embed_output,embed_output)

print('attn output  is :\n', attn_output,attn_output.shape)

print('attn weights are :\n', attn_weights,attn_weights.shape)
tokens=['i','love','my','country']

for val in range(0,seq_len):
    attn_weights_numpy=attn_weights[val].detach().numpy().reshape(4,4) # detach() is to remove the bias from weights....
    print(attn_weights_numpy)

    plt.figure(figsize=(6,6))
    sns.heatmap(attn_weights_numpy,xticklabels=tokens,yticklabels=tokens,cmap='viridis',annot=True)
    plt.title('Attention Heatmap')
    plt.xlabel("Key (attended to)")
    plt.ylabel("query (attending)")
    plt.show()

    df=pd.DataFrame(attn_weights_numpy,columns=['I','sat','on','wall'],index=['I','sat','on','wall'])
    print(df)


