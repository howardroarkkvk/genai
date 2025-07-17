
from model import * 
import torch
from itertools import chain

class Trainer:
    def __init__(self,model):
        self.model=model


    def train(self,epochs,optimizer,loss_fn,train_loader):
        for epoch in range(epochs):
            running_loss=0.0
            for inputs,labels in train_loader:
                # forward pass 
                outputs=self.model(inputs)
                print('outputs shape',outputs.shape)
                print('labels shape ',labels.shape)
                loss=loss_fn(outputs,labels)
                running_loss+=loss.item()

                #backward pass
                optimizer.zero_grad() # resets the gradient values
                loss.backward() # calculates the gradeient values
                optimizer.step() # update weights using gradients calculated in the prev. step...

            print(f"epoch:{epoch+1},loss: {running_loss}")

    def infer(self,test_loader,loss_fn):
        
        loss=0
        all_predictions=[]
        with torch.no_grad():
            for inputs,labels in test_loader:
                predictions=self.model(inputs)
                loss+=loss_fn(predictions,labels)
                all_predictions.extend(predictions.tolist())
        all_predictions = [list2 for list1 in all_predictions for list2 in list1]

        return loss.item(),all_predictions

        








