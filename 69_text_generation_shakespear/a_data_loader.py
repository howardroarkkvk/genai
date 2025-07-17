import torch
from torch.utils.data import  Dataset,DataLoader 
from transformers import AutoTokenizer
import os
from pathlib import Path
from itertools import chain

# dataloader gives the iterable by giving the ablility to sample........#


class CustomDataset(Dataset):

    def __init__(self,dir,context_length):
        super().__init__() # not required here as the parent class Dataset is not initializing anything in it's constructor method...
        model_id='openai-community/gpt2'
        self.tokenizer=AutoTokenizer.from_pretrained(model_id)
        text_from_all_files_as_lists=[]
        for file in Path(dir).glob('**/*.txt'):
            with open(file,'r',encoding='utf-8') as f:
                lines=f.readlines()
            cleaned_lines=[line.strip() for line in lines]
            text_from_all_files_as_lists.append(cleaned_lines)
        # print('Text from all files as lists',text_from_all_files_as_lists)
        token_ids=[]
        for list_of_sentences_from_files in text_from_all_files_as_lists:
            inputs=self.tokenizer(list_of_sentences_from_files)
            # print(inputs['input_ids'])
            flatten_ids=list(chain.from_iterable(inputs['input_ids']))
            # print(f'flatten ids are: {flatten_ids}')
            token_ids.extend(flatten_ids)
        # print('final list of token ids',token_ids)

        self.input_ids=[]
        self.output_ids=[]
        for i in range(0,len(token_ids)-context_length,context_length):
            input=token_ids[i:i+context_length]
            output=token_ids[i+1:i+context_length+1]
            self.input_ids.append(torch.tensor(input))
            self.output_ids.append(torch.tensor(output))

        # print('Input ids are :',self.input_ids)
        # print('output ids are :',self.output_ids)

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index],self.output_ids[index]
    

if __name__=='__main__':
    dir_path=r'D:\DataFiles\text_generation\dummy'
    folder_path='val'
    dir=os.path.join(dir_path,folder_path)
    print(dir)
    dataset=CustomDataset(dir,context_length=4)
    print(dataset)
    print(f'dataset length :{len(dataset)}')
    for item in dataset:
        print(item[0],item[1])

    print('printing complete data set -----------:\n')
    # dataset1=DataLoader(dataset)
    # for i in dataset1:
    #     print(i)
    
    train_dataset=DataLoader(dataset=dataset,batch_size=2,shuffle=True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    for i,j in train_dataset:
        print(f'i is {i} , j is {j}')




            




        

    