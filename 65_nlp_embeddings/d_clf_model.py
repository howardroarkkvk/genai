import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

class TextClassificationModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,output_dim):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.hidden_layer=nn.Linear(embedding_dim,hidden_dim)
        self.output_layer=nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        print(x,x.shape)
        x=self.embedding(x)
        print(x,x.shape)
        x=torch.mean(x,dim=1)
        print(x,x.shape)
        x=F.relu(self.hidden_layer(x))
        print(x,x.shape)
        x=F.softmax(self.output_layer(x),dim=1)
        print(x,x.shape)
        return x
    
if __name__=='__main__':
    model_id='openai-community/gpt2'
    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    tokenizer.pad_token=tokenizer.eos_token

    model=TextClassificationModel(tokenizer.vocab_size,128,64,3)
    sentences=[        "Life is short, eat dessert first",
        "I love ice cream",
        "I love chocolate cake",]
    
    inputs=tokenizer(sentences,padding='max_length',truncation=True,max_length=10,return_tensors='pt')
    res=model(inputs['input_ids'])
    print(res)
    print(torch.argmax(res,1))






