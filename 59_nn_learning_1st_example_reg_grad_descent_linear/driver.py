
from trainer import * 
from data_loader import * 
from model import * 


data_dir=r'D:\DataFiles\llm_regression_exp1'
batch_size=10
train_loader=get_loader(dir=data_dir,file_name='train.csv',batch_size=batch_size,shuffle=True)
test_loader=get_loader(dir=data_dir,file_name='test.csv',batch_size=10,shuffle=False)
model=RegressionModel1(1,1)

print('model state dict',model.state_dict())
for i in model.named_parameters():
    print('weight of the nn model',i[1])




optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
loss_fn=nn.MSELoss()
epochs=50

trainer=Trainer(model)
trainer.train(epochs,optimizer,loss_fn,train_loader)


print(trainer.infer(test_loader,loss_fn))