import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

class TextGenerationModel2(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.hidden_layer=nn.Linear(embedding_dim,hidden_dim)
        self.output_layer=nn.Linear(hidden_dim,vocab_size)


    def forward(self,x):
        print(x,x.shape)
        x=self.embedding(x)
        print('After embedding',x.shape)
        x=F.relu(self.hidden_layer(x))
        print('After hidden layer',x.shape)
        logits=self.output_layer(x)
        print(logits,logits.shape)
        return logits
    

if __name__=='__main__':
    model_id='openai-community/gpt2'
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token=tokenizer.eos_token

    model=TextGenerationModel2(tokenizer.vocab_size,128,64)
    prompt=['Life is short, eat dessert first' ]
    inputs=tokenizer(prompt,return_tensors='pt')
    input=inputs['input_ids']
    print('input shape is ',input.shape)
    response=[]
    for i in range(10):
        logits=model(input)
        logits_new=logits[:,-1,:]
        print('logits is :',logits_new,logits_new.shape)
        probs=torch.softmax(logits_new,dim=-1)
        print(probs,probs.shape)
        id_next=torch.argmax(probs,dim=-1)
        print(id_next)
        next_word=tokenizer.decode(id_next,skip_special_tokens=True)
        response.append(next_word)
        print(response)
        input=torch.cat((input,id_next.reshape(-1,1)),dim=1)
    print("".join(response))





        

