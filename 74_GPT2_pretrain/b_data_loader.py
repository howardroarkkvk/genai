import torch
from  torch.utils.data import Dataset,DataLoader
from datasets import load_from_disk
import os

class CustomDataset(Dataset):
    def __init__(self,dir):
        self.data=load_from_disk(dir)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        tmp=self.data[idx]['input_ids'] # dataset returns list of dictionaries chunked_ds['train'][0]['input_ids'] # first option is to select train /test, as it is list of dictionaries...[0] is for the first dictionary of data ['input_ids'] will return the list of ids from the dict
        return torch.tensor(tmp[:-1]),torch.tensor(tmp[1:])
    
def create_data_loader(dir,batch_size,shuffle):
    print("Begin creating data Loader")
    dataset=CustomDataset(dir)
    print(len(dataset))
    print(dataset[0])
    dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=4)
    print('total Batches :',len(dataloader))
    print("end of creation of data loader")
    return dataloader

if __name__=='__main__':
    dir=r'D:\DataFiles\nn\pretraining\bookcorpus2\chunked_ds\train'
    batch_size=5
    shuffle=True
    dataset=create_data_loader(dir,batch_size,shuffle)
    print(len(dataset))
    for i in dataset:
        input=i[0]
        output=i[1]
        print(f'input is: \n  {input} , {input.shape}')
        print(f'output is: \n  {output}, {output.shape}')
        break
