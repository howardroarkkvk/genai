
from model import * 
import torch
from itertools import chain
from data_loader import * 
import torch.nn.functional as F
import numpy as np


class Trainer:
    def __init__(self,model):
        self.model=model


    def train(self,epochs,optimizer,loss_fn,train_loader):
        for epoch in range(epochs):
            running_loss=0.0
            for inputs,lables in train_loader:
                outputs=self.model(inputs)
                print('outputs shape is :',outputs.shape)
                print('labels shape is :',lables.shape)
                loss=loss_fn(outputs,lables.squeeze())
                running_loss+=loss.item()

                #backward pass
                optimizer.zero_grad() # resets the gradient values
                loss.backward() # calculates the gradeient values
                optimizer.step() # update weights using gradients calculated in the prev. step...
            print(f"epoch:{epoch+1},loss: {running_loss}")

    def infer(self,test_loader,loss_fn,batch_size):
        match_count=0
        loss=0
        all_predictions=[]
        with torch.no_grad():
            for inputs,lables in test_loader:
                predictions_softmax=self.model(inputs)
                print('predictions softmax in infer',predictions_softmax)
                print(f'labels: {lables.shape}, labels_squeeze: {lables.squeeze().shape}',)
                loss+=loss_fn(predictions_softmax,lables.reshape(-1))#.squeeze()
                indices=torch.argmax(predictions_softmax,dim=1) # here dim=1 represents across columns for a row....,dim=0, across a column i.e for all the rows of that column
                print('indices values are:',indices)
                match_count+=np.sum(indices.numpy()==lables.squeeze().numpy())
                all_predictions.extend(indices.numpy().tolist())
                break
        
        return match_count/(len(test_loader)*batch_size),loss.item()



        








