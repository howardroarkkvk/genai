import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

class TextGenerationModule4(nn.Module):

    def __init__(self,vocab_size,context_len,embed_dim,hidden_dim,num_heads):
        # in init mostly we have initialized all the classes for token embedding, position embedding, hidden layer, output layer and also the multi head attention
        super().__init__()
        self.token_embedding=nn.Embedding(vocab_size,embed_dim)
        self.position_embedding=nn.Embedding(context_len,embed_dim)
        self.hidden_layer=nn.Linear(embed_dim,hidden_dim)
        self.output_layer=nn.Linear(hidden_dim,vocab_size)
        self.attn=nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,batch_first=True)

    def forward(self,x):
        print('the actual input the user gives:\n',x,'\n',x.shape)
        tok_embedding=self.token_embedding(x)
        print(f"token embedding is :\n {tok_embedding}, it's shape is {tok_embedding.shape}")
        pos_embedding=self.position_embedding(torch.arange(0,x.shape[1])).unsqueeze(0)
        print(f"Postion embedding is :\n {pos_embedding}, it's shape is {pos_embedding.shape}")
        total_embedding=tok_embedding+pos_embedding
        print(f"total embedding:\n",total_embedding,total_embedding.shape)
        mask=torch.triu(torch.ones(x.shape[1],x.shape[1]),diagonal=1).bool()
        print(f'masking matrix is ', mask)
        x=self.attn(total_embedding,total_embedding,total_embedding,attn_mask=mask)[0] # self.attn return one the actual data(output), 2 the weights... hence [0]
        print(f'After attention the output is \n,{x} and shape is \n{x.shape}')
        x=F.relu(self.hidden_layer(x))
        print(f'hidden layer output is {x}, {x.shape}')
        logits=self.output_layer(x)
        print(f'final output as logits is {logits}, {logits.shape}')
        return logits

if __name__=='__main__':
    model_id='openai-community/gpt2'
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token=tokenizer.eos_token

    context_length=10
    prompt="life is short, eat dessert first"
    inputs=tokenizer(prompt,return_tensors='pt')

    input=inputs['input_ids']
    print('input fed to the whole model is :\n',input)
    model=TextGenerationModule4(tokenizer.vocab_size,context_len=context_length,embed_dim=128,hidden_dim=64,num_heads=1)

    response=[]
    for i in range(3):
        # input=input[:,-context_length:]# it displays 
        logits=model(input)
        print('logits shape before apply softmax:',logits.shape)
        logits=logits[:,-1,:] # this will pick the last word embedding of the logits...
        print('logits post the selectoin of last word',logits)
        probs=torch.softmax(logits,dim=-1) # softmax converts the 50257 dimensions in to probabillities.....
        print('probs shape is :',probs.shape)
        prediction_word_id=torch.argmax(probs,dim=-1) # argmax picks the id from 50257 values which is of highest probability....
        print('preidction word id is :',prediction_word_id)
        word=tokenizer.decode(prediction_word_id,skip_special_tokens=True)
        response.append(word)
        input=torch.cat((input,prediction_word_id.unsqueeze(0)),dim=1)
    print(" ".join(response))







