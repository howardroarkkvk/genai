
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import os

class CustomDataset(Dataset):

    def __init__(self,file):
        self.df=pd.read_csv(file) # creating a pandas data frame....
        X=np.array(self.df.x,dtype=np.float32).reshape(-1,1) # converting the input features from pandas to numpy array
        self.X=torch.tensor(X) # converting the numpy array to tensor...
        y=np.array(self.df.y,dtype=np.float32).reshape(-1,1) # converting the input features from pandas to numpy array
        self.y=torch.tensor(y)


    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        return self.X[index],self.y[index]
    

def get_loader(dir,file_name,batch_size,shuffle):

    # it can create a train or test dataset....
    dataset=CustomDataset(os.path.join(dir,file_name))

    dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle)

    return dataloader

if __name__=='__main__':
    data_dir=r'D:\DataFiles\llm_regression_exp1'
    batch_size=10
    train_loader=get_loader(dir=data_dir,file_name='train.csv',batch_size=batch_size,shuffle=True)
    for i,j in train_loader:
        print(i.shape,j.shape)
        break