import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


class TextGenerationModel(nn.Module):

    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.hidden_layer=nn.Linear(embedding_dim,hidden_dim)
        self.output_layer=nn.Linear(hidden_dim,vocab_size)

    def forward(self,x):
        # print(x,x.shape)
        x=self.embedding(x)
        # print(x,x.shape)
        x=torch.mean(x,dim=1)
        # print(x,x.shape)
        x=F.relu(self.hidden_layer(x))
        # print(x,x.shape)
        x=F.softmax(self.output_layer(x),dim=1)
        # print(x,x.shape)
        return x
    

if __name__=='__main__':
    model_id='openai-community/gpt2'
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token=tokenizer.eos_token
    vocab_size=tokenizer.vocab_size
    model=TextGenerationModel(vocab_size,128,64)
    sentences=['Life is short, eat dessert first']
    # inputs=tokenizer(sentences,padding='max_length',truncation=True,max_length=10,return_tensors='pt')
    inputs=tokenizer(sentences,truncation=True,return_tensors='pt')
    
    

    # print('inputs is :',inputs)
    input=inputs['input_ids']
    # res=model(input)
    print(f'input is {input} ,{input.shape}')
    response=[]
    for i in range(10):
        res=model(input)
        print('result is:' ,res,res.shape)
        ids=torch.argmax(res,dim=1)
        print('ids is:',ids)
        next_word=tokenizer.decode(ids,skip_special_tokens=True)
        print(f'next word id is {i}: {next_word}')
        response.append(next_word)
        print('Response is ; ',response)
        if i==0:
            print(ids.shape)
        input=torch.cat((input,ids.reshape(1,-1)),dim=1)#.unsqueeze(1)
        print(f'input value is {input},{input.shape}')

    print(''.join(response))




