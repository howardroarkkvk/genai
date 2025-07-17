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
    
    def generate_creative(self,prompt,num_words=10,top_p=None,temperature=1.0):
        self.model.eval()
        inputs=self.tokenizer(prompt,return_tensors='pt')
        self.model.to(self.device)
        input=inputs['input_ids'].to(self.device)
        # print('input shape before applying anything is ',input.shape)
        response=[]
        for i in range(num_words):
            input=input[:,-self.context_length:]
            # print('input shape',input.shape)
            with torch.no_grad():
                logits=self.model(input)
                # print('logits shape before any changes',logits.shape )
            logits=logits[:,-1,:] # this converts logits from 3d tensor to 2d tensor....
            logits=logits/temperature
            probs=torch.softmax(logits,dim=-1)
            # print('probs are:',probs)
            probs_values,probs_index=torch.sort(probs,descending=True)
            # print('prob values are:',probs_values)
            # print('prob index are:',probs_index)

            cum_probs=torch.cumsum(probs_values,dim=-1)
            # print('cumulative probs',cum_probs)
            sorted_mask=cum_probs>top_p
            # print('sorted mask is :',sorted_mask)
            sorted_mask[:,1:]=sorted_mask[:,:-1].clone()
            sorted_mask[:,0]=0
            # print("sorted mask after including one after 0.9 prob is ",sorted_mask)

            probs_values_with_masked_fill=probs_values.masked_fill(sorted_mask,value=0.0)
            # print("sorted mask with masked fill is ",probs_values_with_masked_fill)

            id_next=torch.multinomial(probs_values_with_masked_fill,num_samples=1)
            print(id_next)

            input=torch.cat((input,id_next),dim=-1)
            next_word=self.tokenizer.decode(id_next.squeeze(0),skip_special_tokens=True)
            response.append(next_word)

        response=''.join(response)
        return response.replace('\n',' ')

            



if __name__=='__main__':
    output_dir=r'D:\DataFiles\nn\pretraining\bookcorpus2\chunked_ds\gpt2'
    inferencer=Inference(output_dir,check_point_id=None,model=None,tokenizer=None)
    prompt='What is the capital city of india'
    # output=inferencer.generate(prompt)
    output=inferencer.generate_creative(prompt,10,top_p=0.85,temperature=4)
    print(output)






