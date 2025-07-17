import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from b_model_basic import *

def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Inferencer:


    # inference is initialized with model, tokenizer, context length and device,
    def __init__(self,id,model_dir=None,model_file=None,model=None):
        if model is None:
            if not os.path.exists(os.path.join(model_dir)):
                print(f'model directory {model_dir} does not exists')
                return
            self.model=torch.load(os.path.join(model_dir,model_file),weights_only=False)
        self.model=model
        print('selff .model is ',self.model)
        self.context_length=self.model.config['context_length']
        self.tokenizer=AutoTokenizer.from_pretrained(id)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )


    def generate(self,prompt,num_words):
        self.model.eval()
        print("before converting the input to tokens",prompt)
        inputs=self.tokenizer(prompt,return_tensors='pt')
        print('after performing tokenizer1',inputs)
        input=inputs['input_ids']
        print('after performing tokenizer2',input,input.shape)
        self.model.to(self.device)
        response=[]
        for _ in range(num_words):
            print('input size before',input.shape)
            input=input[:,-self.context_length:]
            print('input size after',input,input.shape)
            input.to(self.device)

            with torch.no_grad():
                logits=self.model(input)
            print('logits size bEFORE changing dimension is  :',logits.shape)
            logits=logits[:,-1,:]
            print('logits size after changing dimension is  :',logits.shape)
            probs=torch.softmax(logits,dim=-1)
            print('probs is :',probs, probs.shape)
            predicted_id=torch.argmax(probs,dim=-1,keepdim=True)
            input=torch.cat((input,predicted_id),dim=1)
            print('predicted id is :',predicted_id,predicted_id.shape)
            next_word=self.tokenizer.decode(predicted_id.squeeze(),skip_special_tokens=True)
            print(next_word)
            response.append(next_word)
            
        response=" ".join(response)
        return response.replace('\n',' ')
    
    def generate_creative(self,prompt,num_words,top_k,temperature):
        pass


if __name__=='__main__':
    prommpt=["Hi , How are you?"]
    num_words=2

    model_config={'vocab_size':50257,'context_length':10,'embed_dim':128,'hidden_dim':64,'n_heads':1}
    model=TextGenerationModel4(model_config)
    print(param_count(model))
    inferencer=Inferencer('openai-community/gpt2',model_dir=r'D:\DataFiles\text_generation\shakespear_random\output',model_file='text_generator_1.pkl',model=model)
    output=inferencer.generate(prompt=prommpt,num_words=num_words)
    print(output)
            



        
