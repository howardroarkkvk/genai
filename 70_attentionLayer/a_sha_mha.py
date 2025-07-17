import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SingleHeadSelfAttention(nn.Module):
    def __init__(self,emb_dim,k_dim,v_dim): # the dimension of query aand key are same hence only one is passed as input... and the other is derived
        super().__init__()
        self.W_q=nn.Linear(emb_dim,k_dim)
        self.W_k=nn.Linear(emb_dim,k_dim)
        self.W_v=nn.Linear(emb_dim,v_dim)
    

    def forward(self,x,mask=False):
        # why mask is given in forward not in the init.....i think while calling the model object, we can provide mask parameter, hence it is given at the forward instead of init
        print('input x and its shape is :',x,x.shape)
        Q=self.W_q(x)
        print('Q and its shape is :',Q,Q.shape)
        K=self.W_k(x)
        print('K and its shape is :',K,K.shape)
        V=self.W_v(x)
        print('V and its shape is :',V,V.shape)
        attn_scores=torch.matmul(Q,K.transpose(-2,-1)/math.sqrt(K.shape[2])) # we are transposing over the last 2 dimensions why because to multiply same matrices we transpose 1 to get the dot productof the 2 vectors...if we consider each row of the first matrix as a vector and after transpose 1st column as the vector...
        # print(f'attenion scores after multiplying Q,K and normalizing: {attn_scores},{attn_scores.shape}')
        if mask:
            print('x shape is :',x.shape[1])
            mask_matrix=torch.triu(torch.ones(x.shape[1],x.shape[1]),diagonal=1).bool()# this creates a matrix of size equal to shape * Shape where shape = no. of words in a sentence.... with all 1's in the upper triangle area
            print(f'mask matrix: {mask_matrix},{mask_matrix.shape}')
            attn_scores=attn_scores.masked_fill(mask=mask_matrix,value=-torch.inf)  # maksed fill, this tells what to fill in the matrix where the value = True...with minus infinity
        # print(f'attenion scores after mask filling: {attn_scores},{attn_scores.shape}')
        attn_weights=torch.softmax(attn_scores,dim=-1)
        print(f'attention weights',attn_weights,attn_weights.shape)
        attn_output=torch.matmul(attn_weights,V)
        print(f'final attention output is {attn_output} :its shape {attn_output.shape}')
        return attn_output
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,emb_dim,n_heads):
        super().__init__()
        assert emb_dim%n_heads==0,'embedding dimensoin should be divisible by number of heads'
        self.attention_heads=nn.ModuleList([SingleHeadSelfAttention(emb_dim,emb_dim//n_heads,emb_dim//n_heads) for _ in range(n_heads)])
        self.W_o=nn.Linear(emb_dim,emb_dim)

    def forward(self,x,mask=False):
        head_outputs=[attn_head(x,mask) for attn_head in self.attention_heads]
        print('head outputs is :',head_outputs,len(head_outputs),type(head_outputs))
        for i in range(len(self.attention_heads)):

            print('head output 0: ',head_outputs[i][0])
            # print('head output 1: ',head_outputs[1][0])
            # print('head output 2: ',head_outputs[2][0])
            # tuple_head_output=tuple(head_outputs)
        concat_output=torch.cat(head_outputs,dim=-1)#(head_outputs[0][0],head_outputs[1][0],head_outputs[2][0])
        print(concat_output,concat_output.shape)
        output=self.W_o(concat_output)
        return output



if __name__=='__main__':
    sha=SingleHeadSelfAttention(6,3,3)
    x=torch.randn(1,3,6)
    print(f'input to sha is :{x},{x.shape}')
    y=sha(x,mask=True)
    print('for multihead output...')
    mha=MultiHeadSelfAttention(6,3)
    output=mha(x,mask=True)


    


