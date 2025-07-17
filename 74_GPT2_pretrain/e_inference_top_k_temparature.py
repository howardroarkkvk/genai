import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoTokenizer
class Inference:
    
    def __init__(self,output_dir=None,check_point_id=None,model=None,tokenizer=None):
        if output_dir is not None:
            check_point_id='final' if check_point_id is  None else str(check_point_id)
            check_point_dir=os.path.join(output_dir,'check_points',check_point_id)
            if not os.path.exists(check_point_dir):
                print(f'no check point directory existis {check_point_dir}')
                return
            model=torch.load(os.path.join(check_point_dir,'model.pkl'),weights_only=False)
            tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
            # tokenizer=torch.load(os.path.join(check_point_dir,'tokenizer.pkl'),weights_only=False)
        elif model is None or tokenizer is None:
            print('model / tokenizer is missing ')
            return
        self.model=model
        self.tokenizer=tokenizer
        self.context_length=model.config['context_length']
        self.device= 'cude' if torch.cuda.is_available() else 'cpu'


    def generate(self,prompt,num_words=10):
        print(' This is in generatte method...')
        self.model.eval() # this sets the model in the evaluation mode
        inputs=self.tokenizer(prompt,return_tensors='pt')
        input=inputs['input_ids'].to(self.device)
        print('input size is ',input.shape)
        self.model.to(self.device)
        response=[]
        # id_next=0
        for i in range(num_words):
            logits=self.model(input)
            logits=logits[:,-1,:] # as logits is in sentence, word and embedding formart and we have to consider the last word hence -1 is used in logits...
            print('logits shape is :',logits.shape)
            probs=torch.softmax(logits,dim=-1) # for the last word, as the last dimension is embedding vector...it regpresent the weighted classification of words giving some value to each of 50257 dimensions....softmax will convert all of them to probabilities
            print('probs shape is :',probs.shape)
            id_next=torch.argmax(probs,dim=-1) # we apply argmax on the last dimension to get the id.....it return a 1d tensor....
            print('id_next is :',id_next.shape)
            input=torch.cat((input,id_next.unsqueeze(0)),dim=1) # if we have to combine input to the id_next, both should be of same dimension 
            print('input is : ',input)
            next_word=self.tokenizer.decode(id_next,skip_special_tokens=True) # for ttokenizer we will directly send the id....
            response.append(next_word)
        response=' '.join(response)
        return response
    
    def generate_creative(self,prompt,num_words=10,top_k=None,temperature=1.0):
        self.model.eval()
        inputs=self.tokenizer(prompt,return_tensors='pt')
        self.model.to(self.device)
        input=inputs['input_ids'].to(self.device)
        print('input shape before applying anything is ',input.shape)
        response=[]
        for i in range(num_words):
            input=input[:,-self.context_length:]
            print('input shape',input.shape)
            with torch.no_grad():
                logits=self.model(input)
                print('logits shape before any changes',logits.shape )
            logits=logits[:,-1,:]
            #/temperature # last word of the logits is taken to predict the next word...
            print('logits after applying temp:',logits.shape)
            if top_k is not None:
                print('with in if loop')
                values,indices=torch.topk(logits,min(top_k,logits.size(-1))) # lets say logits last i.e. embedding layer size is 10 and top_k is 7, 7 is picked here and the top 7 values are displayed...
                print('values:',values,values[:,[-1]])
                print('indices',indices)
                logits[logits < values[:,[-7]]]=-float("inf")
                logits=logits/temperature
                print('logits after updatings with -inf values...',logits)
            probs=torch.softmax(logits,dim=-1)
            # top3_probs=torch.topk(probs,3)
            # print('top3_probs are:',top3_probs)
            print('probs are:',probs)
            id_next=torch.multinomial(probs,num_samples=1)
            print('id next is :',id_next,id_next.shape)
            input=torch.cat((input,id_next),dim=-1)
            print('input aftef concat',input,input.shape)
            next_word=self.tokenizer.decode(id_next.squeeze(0),skip_special_tokens=True)
            print('next word is ',next_word)
            response.append(next_word)

        response=''.join(response)
        return response.replace('\n',' ')

            



if __name__=='__main__':
    output_dir=r'D:\DataFiles\nn\pretraining\bookcorpus2\chunked_ds\gpt2'
    inferencer=Inference(output_dir,check_point_id=None,model=None,tokenizer=None)
    prompt='What is the capital city of india'
    # output=inferencer.generate(prompt)
    output=inferencer.generate_creative(prompt,20,top_k=10000,temperature=4)
    print(output)






