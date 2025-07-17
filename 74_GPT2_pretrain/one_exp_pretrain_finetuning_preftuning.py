import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(2,4)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(4,1)

    def forward(self,x):
        x= self.relu(self.fc1(x))
        return self.fc2(x)
def reward_func(pred):
    return -((pred-5.0)**2).mean()
if __name__=='__main__':
    torch.manual_seed(42)
    x_pretrain=torch.randn(100,2)
    print('x pre train is:',x_pretrain,x_pretrain.shape)
    y_pretrain=(x_pretrain[:,0]+x_pretrain[:,1]).unsqueeze(-1) # unsqueez of 1 meaning ....the dimension 1 should be 1, unsqueeze(0), it forms a row 2d vector
    print(f'y_pretrain is {y_pretrain} , its shape is {y_pretrain.shape}')

    model=SimpleModule()
    print(model.state_dict())
    
    optimizer=optim.SGD(model.parameters(),lr=0.01)
    loss_fn=nn.MSELoss()
    # General training on generic data ... i.e. pre training
    for epoch in range(50):
        prediction=model(x_pretrain)
        loss=loss_fn(prediction,y_pretrain)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('For Epoch {epoch} loss is:',loss.item())
    print(f'{model.fc1.weight.data}')

    # training on Specific task ... i.e. Fine Tuning

    x_finetune=torch.randn(20,2)
    print('x_fine tune :',x_finetune,x_finetune.shape)
    y_finetune=(x_finetune[:,0]*2+x_finetune[:,1]*3).unsqueeze(1)
    print('y_fine tune :',y_finetune,y_finetune.shape)

    for epoch in range(20):
        prediction=model(x_finetune)
        loss=loss_fn(prediction,y_finetune)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'For Epoch {epoch} loss is:',loss.item())
    print(f'{model.fc1.weight.data}')


    #  Preference Tuning (Human Feedback Simulated)

  
    
    for epoch in range(10):
        pred = model(x_finetune)
        reward = reward_func(pred)
        loss = -reward  # We want to maximize reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'For Epoch {epoch} loss is:',loss.item())
    print(f'{model.fc1.weight.data}')