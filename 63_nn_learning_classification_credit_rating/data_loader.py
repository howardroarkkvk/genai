
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler,LabelEncoder

class CustomDataset(Dataset):

    def __init__(self,file):
        self.df=pd.read_csv(file) # creating a pandas data frame....
        X,y=self.preprocess(self.df)
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.long)


    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        return self.X[index],self.y[index]
    
        
    def preprocess(self,df):
        print(f'Actual data frame size',df.shape)
        cat_features=['Payment_of_Min_Amount','Credit_Mix','Payment_Behaviour']
        ohe=OneHotEncoder(sparse_output=False)
        cat_data=ohe.fit_transform(df[cat_features])
        print(f'cat features after one hot encoding {cat_data.shape}')

        cont_features=[]
        for col in df.columns:
            if col!='Credit_Score' and col not in cat_features:
                cont_features.append(col)
        # print(cont_features)
        scaler=StandardScaler()
        cont_data=scaler.fit_transform(df[cont_features])
        print(f'cont features shape is : {cont_data.shape}')
        X=np.concat([cat_data,cont_data],axis=1)

        le=LabelEncoder()
        y=le.fit_transform(df['Credit_Score'])
        y
        y=y.reshape(-1,1)
        return X,y






    

def get_loader(dir,file_name,batch_size,shuffle):

    # it can create a train or test dataset....
    dataset=CustomDataset(os.path.join(dir,file_name))

    dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle)

    return dataloader



if __name__=='__main__':
    src_dir=r'D:\DataFiles\llm_classification'
    train_loader=get_loader(src_dir,'test.csv',10,True)
    for inputs,lables in train_loader:
        print('inputs',inputs.shape)
        print(lables)
        print('labels shape',lables.shape)
        x=lables.squeeze()
        print(x)
        print(x.shape)
        break


