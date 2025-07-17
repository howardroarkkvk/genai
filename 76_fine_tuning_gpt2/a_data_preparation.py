from datasets import load_dataset
import os
from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
tokenizer.add_special_tokens({'pad_token':tokenizer.eos_token,
                              'bos_token':'<|im_start|>','eos_token':'<|im_end|>'})

def chat_format(message):
    im_start='<|im_start|>'
    im_end='<|im_end|>'
    prompt=''
    if message['system']!=None and len(message['system'])>0:
        prompt+=f'{im_start} system \n {message['system']}  {im_end} \n'
    if message['user']!=None and len(message['user'])>0:
        prompt+=f'{im_start} user \n {message['user']}  {im_end} \n'
    if message['assistant']!=None and len(message['assistant'])>0:
        prompt+=f'{im_start} assistant \n {message['assistant']} \n {im_end}'
    # print(prompt)
    return prompt


def format_dataset(example):
    upgraded_example={'system':example['instruction'],
                      'user':example['context'],
                      'assistant':example['response']}
    expected_example=chat_format(upgraded_example)
    return {'message':expected_example}

def tokenize(example):
    context_length=1024
    tokenized_data=tokenizer(example['message'],padding='max_length',max_length=context_length,truncation=True,return_tensors='pt')
    example['input_ids']=tokenized_data['input_ids']
    example['attention_mask']=tokenized_data['attention_mask']
    return example

    

if __name__=='__main__':
    # extracting dolly dataset
    dir=r'D:\DataFiles\nn\fine_tuning\dolly'
    dataset=load_dataset('databricks/databricks-dolly-15k',trust_remote_code=True,split='train').shuffle(seed=1).select(range(10))
    dataset=dataset.train_test_split(test_size=0.1)
    print('data set load from hugging face.......')
    print(dataset['train'])
    print(dataset['test'][0])
    remove_cols=dataset['train'].column_names
    print('data set load from hugging face.......')
    #formatting dataset so that each message is in format of 'system : instruction, user:context, assistant : response all these appended with '<|im_start|>' ,'<|im_end|>'
    print('columns removed are :',remove_cols)
    print('data set conversion to gpt message format.......')
    templated_dataset=dataset.map(format_dataset,remove_columns=remove_cols,num_proc=os.cpu_count())
    print(templated_dataset['train'][0])
    templated_dataset.save_to_disk(os.path.join(dir,'templated_ds_hf'))
    print('data set conversion end.......')

    # tokenization of dataset
    print('Tokenization of the data set')
    tokenized_dataset=templated_dataset.map(tokenize,remove_columns='message',num_proc=os.cpu_count())
    print(tokenized_dataset['train'][0])
    tokenized_dataset.save_to_disk(os.path.join(dir,'tokenized_ds_hf'))
    print('End of Tokenization of the data set')

    






# print(dataset)
# print('instruction:>> ',dataset[0]['instruction']) # system prompt
# print('context:>>',dataset[0]['context']) # user prompt
# print('response:>>',dataset[0]['response']) # assistant response
# print('category:>>',dataset[0]['category']) # category ......

