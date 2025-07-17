import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

class TextGenerationModel2(nn.Module):
    def __init__(self,vocab_size,context_length,embedding_dim,hidden_dim):
        super().__init__()
        self.tok_embedding=nn.Embedding(vocab_size,embedding_dim)
        self.pos_embedding=nn.Embedding(context_length,embedding_dim)
        self.hidden_layer=nn.Linear(embedding_dim,hidden_dim)
        self.output_layer=nn.Linear(hidden_dim,vocab_size)


    def forward(self,x):
        print(f'input x is :{x},{x.shape}')
        tok_embedding=self.tok_embedding(x)
        print('After token embedding',tok_embedding.shape)
        positions=torch.arange(0,x.shape[1])
        print(positions)
        pos_embedding=self.pos_embedding(positions)
        print('pos embeddings',pos_embedding.shape)
        total_embedding=tok_embedding+pos_embedding
        print('total embedding is :',total_embedding.shape)
        x=F.relu(self.hidden_layer(total_embedding))
        print('After hidden layer',x.shape)
        logits=self.output_layer(x)
        print(logits,logits.shape)
        return logits
    

if __name__=='__main__':
    model_id='openai-community/gpt2'
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    # tokenizer.pad_token=tokenizer.eos_token
    context_length=15
    model=TextGenerationModel2(tokenizer.vocab_size,15,128,64)
    prompt=['Life is short, eat dessert first' ]
    inputs=tokenizer(prompt,return_tensors='pt')
    input=inputs['input_ids']
    print('input shape is ',input.shape)
    response=[]
    for i in range(10):
        print(f'input in for loop befoer is {input.shape} , \n {input}')
        input=input[:,-context_length:]# what it does is it prints the last 15 elements (not starting from last) always....
        print(f'input in for loop is {input.shape} , \n {input}')
        logits=model(input)
        logits_new=logits[:,-1,:] # it takes the last token (i.e. word in a sentence...) from [batch,tokens,embeddings...]
        print('logits is :',logits_new,logits_new.shape)
        probs=torch.softmax(logits_new,dim=1) # applying the softmax to the last dimension so -1....
        print(probs,probs.shape)
        id_next=torch.argmax(probs,dim=1)
        print(id_next)
        next_word=tokenizer.decode(id_next,skip_special_tokens=True)
        response.append(next_word)
        print(response)
        input=torch.cat((input,id_next.reshape(-1,1)),dim=1)
    print("".join(response))





        

