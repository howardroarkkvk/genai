import os
import torch
from a_data_preparation import *
from c_models import *
from transformers import AutoTokenizer

class Inference:
    def __init__(self,output_dir=None,check_point_id=None,model=None,tokenizer=None):
        if output_dir is not None: # if user doesnt give any check point id it means it has to be picked from final or '-1'
            sub_id='full' if check_point_id is None else str(check_point_id)
            check_point_dir=os.path.join(output_dir,'check_points',str(sub_id))
            if not os.path.exists(check_point_dir):
                print(f'check point directory doesnt exists')
                return
            model=torch.load(os.path.join(check_point_dir,'model.pkl'),weights_only=False)
            tokenizer=torch.load(os.path.join(check_point_dir,'tokenizer.pkl'),weights_only=False)
        elif model is None or tokenizer is None: # here it means that output dir is None hence we check for model and tokenizer values.... user has the option to give a folder path or model/tokenizer details directly
            print('model/tokenizer is missing')
            return
        self.context_length=model.config['context_length']
        self.model=model
        self.tokenizer=tokenizer
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

    
    def generate(self,prompt,num_words=10,temperature=2.0):
        self.model.eval()
        inputs=self.tokenizer(prompt,return_tensors='pt')
        input=inputs['input_ids'].to(self.device)
        print('actual input shape',input, input.shape)
        self.model.to(self.device)
        response=[]
        for i in range(num_words):
            print(f'the value of i is {i}')
            input=input[:,-self.context_length:]  # input is of 2 dimensions 1 sentence and x no. of words.....-self.context: .... displays all the existing elements ...
            print('input value is :',input)
            with torch.no_grad():
                logits=self.model(input) # logits are of 3 d shape  1,7, 50257
            print('logit value is ',logits)
            logits=logits[:,-1,:] # i want the last word embeddings which are of shape(as a result) 1,50257...
            print(logits)
            probs=torch.softmax(logits,dim=-1) # across all columns... i.e a row...
            print('value of probability ',probs)
            id_next=torch.argmax(probs,dim=-1)
            # id_next=torch.multinomial(probs,num_samples=1)
            print(id_next)
            next_word=self.tokenizer.decode(id_next,skip_special_tokens=True)
            print(next_word)
            response.append(next_word)
            input=torch.cat((input,id_next.unsqueeze(0)),dim=-1)
            # if next_word=='<|im_end|>':
            #     break
        response=''.join(response)
        return response
    
    def generate_creative(self,prompt,num_words=10,temperature=None,top_k=None):
        self.model.eval()
        input=self.tokenizer(prompt,return_tensors='pt')
        input=input['input_ids'].to(self.device)
        self.model.to(self.device)
        response=[]
        for i in range(num_words):
            input=input[:,-self.context_length:]
            with torch.no_grad(): # this doesnt initialize any gradients during inference process hence saves memory...
                logits=self.model(input)
            logits=logits[:,-1,:]/temperature
            if top_k is not None:
                values,indices=torch.topk(logits,min(top_k,logits.size(-1)))
                logits[logits<values[:,[-1]]]=-float('inf')
            probs=torch.softmax(logits,dim=-1)
            print('probs is ',probs)
            id_next=torch.multinomial(probs,num_samples=1)
            print('id next is ',id_next)
            input=torch.cat((input,id_next),dim=1)
            print('input after concat is' ,input)
            next_word=self.tokenizer.decode(id_next.squeeze(0),skip_special_tokens=True)
            response.append(next_word)
        response=''.join(response)
        return response
        

    def generate_creative_top_p(self,prompt,num_words=10,temperature=None,top_p=None):
        self.model.eval()
        self.model.eval()
        input=self.tokenizer(prompt,return_tensors='pt')
        input=input['input_ids'].to(self.device)
        self.model.to(self.device)
        response=[]
        for i in range(num_words):
            input=input[:,-self.context_length:]
            with torch.no_grad(): # this doesnt initialize any gradients during inference process hence saves memory...
                logits=self.model(input)
            logits=logits[:,-1,:]/temperature
            probs=torch.softmax(logits,dim=-1)
            values,indices=torch.sort(probs,descending=True)
            cum_sum_values=torch.cumsum(values,dim=-1)
            print(cum_sum_values)
            sorted_mask=cum_sum_values>top_p
            print(sorted_mask)
            probs_with_sort_masked_fill=values.masked_fill(sorted_mask,value=0.0)
            id_next=torch.multinomial(probs_with_sort_masked_fill,num_samples=1)
            print('id next is ',id_next)
            input=torch.cat((input,id_next),dim=1)
            print('input after concat is' ,input)
            next_word=self.tokenizer.decode(id_next.squeeze(0),skip_special_tokens=True)
            response.append(next_word)
        response=''.join(response)
        return response            



    

if __name__=='__main__':
    src_dir=r'D:\DataFiles\nn\fine_tuning\dolly'
    output_dir=os.path.join(src_dir,'gpt2_it')
    tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
    tokenizer.add_special_tokens({'pad_token':tokenizer.eos_token,'bos_token':'<|im_start|>','eos_token':'<|im_end|>'})
    model=GPT2Model.from_pretrained('gpt2',len(tokenizer))
    inference=Inference(None,None,model,tokenizer)
    
    prompt={'system': 'please analyze the input',
            'user':"what is the capital of india",
            'assistant':None
            }
    formatted_prompt=chat_format(prompt)
    # output=inference.generate(formatted_prompt,20)
    # output=inference.generate_creative(formatted_prompt,20,5,10)
    output=inference.generate_creative_top_p(formatted_prompt,20,5,0.9)
    print(output)







        

    
