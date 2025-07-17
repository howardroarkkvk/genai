from datasets import load_dataset,Dataset
import os
from transformers import AutoTokenizer
from itertools import chain


tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token=tokenizer.eos_token

# def tokenize_function(example):
#     print(f'counter to check how many times it is called....')
#     # print('this is in tokenize function',example['text'])
#     tokens=tokenizer(example['text'])
#     # print('tokens are ',tokens)
#     return tokens

# def concat(examples):
#     examples['input_ids']=list(chain.from_iterable(examples['input_ids']))
#     examples['attention_mask']=list(chain.from_iterable(examples['attention_mask']))
#     return examples

# def chunk(examples):
#     chunk_size=1024
#     input_ids=examples['input_ids']
#     # print('in chunk func input ids',input_ids)
#     attention_mask=examples['attention_mask']
#     # print('in chunk func attenion masks ',attention_mask)
#     input_ids_truncated=[]
#     attention_mask_truncated=[]
#     for i in range(0,len(input_ids),chunk_size):
#         chunk=input_ids[i:i+chunk_size]
#         if len(chunk)==chunk_size:
#             input_ids_truncated.append(chunk)
#             attention_mask_truncated.append(attention_mask[i:i+chunk_size])
#         examples['input_ids']=input_ids_truncated
#         examples['attention_mask']=attention_mask
#     return examples
if __name__=='__main__':
    src_dir=r'D:\DataFiles\nn\pretraining\bookcorpus2'
    print("Data Loading Begin")
    dataset=load_dataset('bookcorpus',trust_remote_code=True,split='train').shuffle(seed=42).select(range(2))
    print(dataset)
    print(">> Data Loading End")


    print(">>> Begin Tokenization :")
    tokenized_data=dataset.map(lambda example:tokenizer(example['text']),batched=True,remove_columns='text',num_proc=8)#,remove_columns='text'
    print(tokenized_data['input_ids'])
    print(tokenized_data['attention_mask'])
    # tokenized_data.save_to_disk(os.path.join(src_dir,'tokenized_ds'))
    print(">>>> Tokenization End")

    

    # make samples to size of context window...
    print(">>>> beigining of  chunkking of tokenized data:")
    concated_tokens_data=tokenized_data.map(lambda example:{'input_ids': list(chain.from_iterable(example['input_ids'])),
                                                            'attention_mask':list(chain.from_iterable(example['attention_mask']))} ,batched=True,num_proc=8)
    print(concated_tokens_data['input_ids'])

    chunked_ds=concated_tokens_data.map(lambda examples:{'input_ids':[examples['input_ids'][i:i+1024] for i in range(0,len(examples['input_ids']),1024)] ,
                                                         'attention_mask':[examples['attention_mask'][i:i+1024] for i in range(0,len(examples['attention_mask']),1024) ]
                                                         }
                                                         ,batch_size=10,batched=True,num_proc=8)
    
    print(chunked_ds['input_ids'])
    # chunked_ds.save_to_disk(os.path.join(src_dir,'chunked_ds'))
    # print(">> End of chunking process...")

