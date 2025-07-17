from datasets import load_dataset,Dataset
import os
from transformers import AutoTokenizer
from itertools import chain


tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token=tokenizer.eos_token

def tokenize_function(example):
    # print('this is in tokenize function',example['text'])
    tokens=tokenizer(example['text'])
    # print('tokens are ',tokens)
    return tokens

def concat(examples):
    # print('with in concat',examples.data)
    examples['input_ids']=[list(chain.from_iterable(examples['input_ids']))]
    examples['attention_mask']=[list(chain.from_iterable(examples['attention_mask']))]
    return examples

def chunk(examples):
    chunk_size=1024
    input_ids=examples['input_ids'][0]
    # print('in chunk func input ids',input_ids)
    attention_mask=examples['attention_mask'][0]
    # print('in chunk func attenion masks ',attention_mask)
    input_ids_truncated=[]
    attention_mask_truncated=[]
    for i in range(0,len(input_ids),chunk_size):
        chunk=input_ids[i:i+chunk_size]
        if len(chunk)==chunk_size:
            input_ids_truncated.append(chunk)
            attention_mask_truncated.append(attention_mask[i:i+chunk_size])
        examples['input_ids']=input_ids_truncated
        examples['attention_mask']=attention_mask_truncated
    return examples
if __name__=='__main__':
    src_dir=r'D:\DataFiles\nn\pretraining\bookcorpus2'
    print("Data Loading Begin")
    dataset=load_dataset('bookcorpus',trust_remote_code=True,split='train').shuffle(seed=42).select(range(10))
    dataset=dataset.train_test_split(test_size=0.1)
    print(">> Data Loading End")


    print(">>> Begin Tokenization :")
    tokenized_data=dataset.map(tokenize_function,remove_columns='text',batched=True,num_proc=8)
    # tokenized_data.save_to_disk(os.path.join(src_dir,'tokenized_ds'))
    print(">>>> Tokenization End")

    # make samples to size of context window...
    print(">>>> beigining of  chunkking of tokenized data:")
    concated_tokens_data=tokenized_data.map(concat,batched=True,batch_size=10000,num_proc=8)
    chunked_ds=concated_tokens_data.map(chunk,batch_size=2,batched=True,num_proc=8)
    print('type of chunked ds',type(chunked_ds))
    print('chunked ds : \n',chunked_ds)
    # chunked_ds.save_to_disk(os.path.join(src_dir,'chunked_ds'))
    print(">> End of chunking process...")

