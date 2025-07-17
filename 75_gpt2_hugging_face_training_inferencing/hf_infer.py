import os
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline


def infer1(model,tokenizer,prompt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs=tokenizer(prompt,return_tensors='pt').input_ids.to(device)
    outputs=model.generate(inputs,max_new_tokens=100,do_sample=True,top_k=10,top_p=0.95)
    output=tokenizer.batch_decode(outputs,skip_special_tokens=True)
    print(output)


def infer2(model,tokenizer,prompt):
    pipe=pipeline(task="text-generation",
                  model=model,
                  tokenizer=tokenizer,
                  config={'max_new_tokens':100,
                          'do_amnple':True,
                          'top_k':10,
                          'top_p':0.95})
    return pipe(prompt)[0]['generated_text']


if __name__=='__main__':
    src_dir=r'D:\DataFiles\nn\pretraining\bookcorpus2\chunked_ds' 
    check_point_dir=os.path.join(src_dir,'gpt2','check_points','final')
    tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
    model=AutoModelForCausalLM.from_pretrained(check_point_dir,device_map='auto')
    
    infer1(model,tokenizer,'I was telling her that')


