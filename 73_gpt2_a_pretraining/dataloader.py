from torch.utils.data import Dataset
from itertools import chain
import torch
from pathlib import Path
from transformers import AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self,tokenizer,input_dir,context_length):
        self.tokenizer=tokenizer
        self.context_length=context_length
        list_of_lines=[] # this holds the lists each lists represent lines from a file..
        for file in Path(input_dir).glob('**/*.txt'):
            with open(file,'r',encoding='utf-8') as f:
                lines_from_file=f.readlines()
            cleaned_lines=[line.strip() for line in lines_from_file]
            list_of_lines.append(cleaned_lines)
        overall_tokens=[]
        for line in list_of_lines:
            encoded=self.tokenizer(line)
            line_ids=encoded['input_ids'] # it's a list of list of token ids
            flatten_ids=[line for lines in line_ids for line in lines]
            overall_tokens.extend(flatten_ids) # it will extend the list , doesnt append to the existing list as list...
        
        self.input_ids=[]
        self.output_ids=[]
        for i in range(0,len(overall_tokens)-context_length,context_length):
            input=overall_tokens[i:i+context_length]
            output=overall_tokens[i+1:i+1+context_length]
            self.input_ids.append(torch.tensor(input))
            self.output_ids.append(torch.tensor(output))





    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,index):
        return self.input_ids[index],self.output_ids[index]


if __name__=='__main__':
    dir=r'D:\DataFiles\text_generation\shakespear_random\train'
    tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
    context_length=5
    dataset=CustomDataset(tokenizer,dir,context_length)
    for i, j in dataset:
        print(i)
        print(j)
        break
