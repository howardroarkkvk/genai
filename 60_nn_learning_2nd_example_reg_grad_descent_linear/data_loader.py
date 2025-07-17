
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
