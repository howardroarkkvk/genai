import torch
import torch.nn as nn
import torch.nn.functional as F
from b_data_loader import * 
from d_model import * 
def calc_loss_cross_entropy(model,input_batch,target_batch,loss_fn):
    print('with in loss function.....')
    logits=model(input_batch)
    print('logits',logits,logits.shape)
    print('target batch ',target_batch,target_batch.shape)
    inputs=logits.flatten(0,1)
    outputs=target_batch.flatten()
    print('inputs: ',inputs,inputs.shape)
    print('outputs: ',outputs,outputs.shape)

    loss=loss_fn(inputs,outputs)
    print(loss)

def apply_softmax_then_argmax(model,input_batch,target_batch,loss_fn):
    print('with in loss function.....')
    logits=model(input_batch)
    print('logits',logits,logits.shape)
    print('target batch ',target_batch,target_batch.shape)
    logits=logits.flatten(0,1)
    x=F.softmax(logits,dim=1)
    print(x,x.shape)
    y=torch.argmax(x,dim=-1)
    print(y,y.shape)
    total_loss=0.0
    for i in range(len(y)):
        print(i)
        output=x[i,y[i]]
        loss=-torch.log(output)
        print(output)
        print('loss value is :',loss)
        total_loss+=loss
    print(total_loss)





if __name__=='__main__':
    device= "cuda" if torch.cuda.is_available() else 'cpu'
    dir=r'D:\DataFiles\nn\pretraining\bookcorpus2\chunked_ds\train'
    batch_size=10
    shuffle=True
    dataset=create_data_loader(dir,batch_size,shuffle)
    config = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.25,
    }
    model=GPT2Model(config)
    # print(model.state_dict())
    loss_fn=nn.CrossEntropyLoss()
    for input_batch,target_batch in dataset:
        input_batch=input_batch.to(device)
        target_batch=input_batch.to(device)
        model.to(device)
        model.train()
        # calc_loss_cross_entropy(model,input_batch,target_batch,loss_fn)
        # print(loss)
        apply_softmax_then_argmax(model,input_batch,target_batch,loss_fn)
        break


