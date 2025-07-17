import os
import torch
from transformers import AutoTokenizer,pipeline,AutoModelForCausalLM

def infer1(model,tokenizer,prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs=tokenizer(prompt,return_tensors='pt')
    print(inputs)
    input=inputs['input_ids'].to(device)
    print(input[0])
    outputs=model.generate(input,max_new_tokens=30,do_sample=True,top_k=10,top_p=0.95)
    print('output before decoding',outputs.squeeze(0))
    output=tokenizer.decode(outputs.squeeze(0),skip_special_tokens=True)
    return output

def infer2(model,tokenizer,prompt):
    pipe=pipeline(task='text-generation',
             model=model,
             tokenizer=tokenizer,
             config={'max_new_tokens':10,
                     'do_sample':True,
                     'top_k':10,
                     'top_p':0.95})
    print('using pipe',pipe(prompt)[0]['generated_text'])
    return pipe(prompt)[0]['generated_text']
    

if __name__=='__main__':
    src_dir=r'D:\DataFiles\nn\fine_tuning\dolly'
    check_point_dir=os.path.join(src_dir,'tokenized_ds_hf')
    model_id='openai-community/gpt2'
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens({'pad_token':tokenizer.eos_token,
                              'bos_token':'<|im_start|>',
                              'eos_token':'<|im_end|>'})
    model=AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,device_map='auto')
    prompt = """{
        "system": "You are an AI assistant to help user queries.",
        "user": "Reverse this array:[10, 20, 30, 40, 50]"
    }"""
    # print(infer1(model, tokenizer, prompt))
    print(infer2(model, tokenizer, prompt))