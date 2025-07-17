import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer
from a_sha_mha import * 



# we have defined the model....
class TextGenerationModel4(nn.Module):
    def __init__(self,vocab_size,context_length,embed_dim,hidden_dim,n_heads):
        super().__init__()
        self.token_embedding=nn.Embedding(vocab_size,embed_dim)
        self.pos_embedding=nn.Embedding(context_length,embed_dim)
        self.attn=MultiHeadSelfAttention(emb_dim=embed_dim,n_heads=n_heads)
        self.hidden_layer=nn.Linear(embed_dim,hidden_dim)
        self.output_layer=nn.Linear(hidden_dim,vocab_size)


    def forward(self,x):
        print('Input data is :',x , x.shape)
        tok_embeds=self.token_embedding(x)
        print('token embeddings:', tok_embeds,tok_embeds.shape)
        pos_embeds=self.pos_embedding(torch.arange(0,x.shape[1]).unsqueeze(0))
        print('position embeddings:', pos_embeds,pos_embeds.shape)
        embeds=tok_embeds+pos_embeds
        print('embds which is sum of token + position embeddings :',embeds,embeds.shape)
        x=self.attn(embeds,True)
        print('attention layer ouput',x,x.shape)
        x=x+embeds
        print('Adding the input of attention layer to its output:',x, x.shape)
        x=F.relu(self.hidden_layer(x))
        print('After hidden layer',x,x.shape)
        logits=self.output_layer(x)
        print('logits are :',logits,logits.shape)
        return logits

# inferencing the model

if __name__=='__main__':

    tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
    tokenizer.pad_token=tokenizer.eos_token

    prompt = 'Life is short, eat dessert first'
    context_length=10
    n_heads=1
    model=TextGenerationModel4(tokenizer.vocab_size,context_length,128,64,n_heads)
    inputs=tokenizer(prompt,return_tensors='pt')
    input=inputs['input_ids']
    print('Inputs to the model are ids(converted from words):',input,input.shape)
    response=[]
    for i in range(10):
        input=input[:,-context_length:] # this is to get the input till context lenght always.... meaning is context length is 10  and if the number of words are more it will print the last 10 words as input
        print('input changed using context lenght....')
        logits=model(input)
        logits=logits[:,-1,:] # pick the last word vector
        probs=torch.softmax(logits,dim=-1) # softmax is applied on the 50257 embed dimension....
        id_next=torch.argmax(probs,dim=-1)
        next_word=tokenizer.decode(id_next,skip_special_tokens=True)
        response.append(next_word)
        input=torch.cat((input,id_next.unsqueeze(0)),dim=1)
    print(''.join(response))



        



    
