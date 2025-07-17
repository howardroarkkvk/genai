from datasets import load_from_disk
from torch.utils.data import Dataset,DataLoader
import torch
import os

class CustomDataset(Dataset):
    def __init__(self,dir):
        self.data=load_from_disk(dir)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        ids=self.data[idx]['input_ids'][0]
        mask=self.data[idx]['attention_mask'][0]
        input_ids=ids[:-1]
        output_ids=ids[1:]
        selection_mask=mask[1:]
        for i,e in enumerate(selection_mask):
            if e==0:
                output_ids[i]=-100
        return torch.tensor(input_ids),torch.tensor(output_ids)
    
def create_data_loader(dir,batch_size,shuffle):
    print('>>> Begin data loading process:')
    dataset=CustomDataset(dir)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
    print('Total Batches :',len(dataloader))
    print(">> End of creation of data loader")
    return dataloader

if __name__=='__main__':
    dir='D:/DataFiles/nn/fine_tuning/dolly'
    tokenized_dir=os.path.join(dir,'tokenized_ds/train')
    # dataset=CustomDataset(tokenized_dir)
    # print(len(dataset))
    # for i,j in dataset:
    #     print(i,i.shape)
    #     print(j,j.shape)



    dataset=create_data_loader(tokenized_dir,batch_size=5,shuffle=True)
    for input_ids,output_ids in dataset:
        print('input_ids', input_ids)
        print('output_ids', output_ids)
        break



