
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
class CustomDataset(Dataset):

    def __init__(self,file):
        self.df=pd.read_csv(file) # creating a pandas data frame....
        X,y=self.preprocess(self.df)
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)


    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        return self.X[index],self.y[index]
    
        
    def preprocess(self,df):
        features=[]
        for col in df.columns:
            if col!='quality':
                features.append(col)
        scaler=StandardScaler()
        X=scaler.fit_transform(df[features])
        y=np.array(df['quality']).reshape(-1,1)
        return X,y
    

def get_loader(dir,file_name,batch_size,shuffle):

    # it can create a train or test dataset....
    dataset=CustomDataset(os.path.join(dir,file_name))

    dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle)

    return dataloader



if __name__=='__main__':
    src_dir=r'D:\DataFiles\llm_regression_wine_data'
    train_loader=get_loader(src_dir,'train.csv',10,True)
    for item in train_loader:
        print(item)
        break


