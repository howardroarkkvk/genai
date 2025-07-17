import torch
import torch.nn as nn
import torch.nn.functional as F
from b_data_loader import * 
from d_model import * 
import shutil
from transformers import AutoTokenizer




# for a trainer , we intialzie it with a model and a tokenizer....
class Trainer:

    def __init__(self,model,tokenizer):
        self.model=model
        self.tokenizer=tokenizer
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'

    # what is requried for calcualtion of loss input_batch,output_batch,loss_fn
    def calc_loss(self,input_batch,output_batch,loss_fn):
        input_batch=input_batch.to(self.device)
        output_batch=output_batch.to(self.device)
        logits=self.model(input_batch)
        loss=loss_fn(logits.flatten(0,1),output_batch.flatten())
        return loss
    
    # evaluate model for the particualr epoch...i.e. for all the batches of an epoch...
    def evalaute_model(self,dataloader,loss_fn):
        if len(dataloader)==0:
            return float("nan")
        total_loss=0.0
        self.model.eval()
        for input_batch,output_batch in dataloader:
            loss=self.calc_loss(input_batch,output_batch,loss_fn)
            total_loss+=loss.item()
        self.model.train()
        return total_loss/len(dataloader)
    
    def clear_check_points(self,output_dir):
        check_points_dir=os.path.join(output_dir,'check_points')
        if os.path.exists(check_points_dir):
            shutil.rmtree(check_points_dir)

    
    def save_check_point_data(self,output_dir,net_epoch,train_losses,val_losses):
        check_point_folder_name="final" if net_epoch== -1 else str(net_epoch)
        metadata={'last_epoch':net_epoch,
                  'train_losses':train_losses,
                  'val_losses':val_losses}
        check_point_sub_dir=os.path.join(output_dir,'check_points',check_point_folder_name)
        if not os.path.exists(check_point_sub_dir):
            os.makedirs(check_point_sub_dir)

        
        torch.save(self.model,os.path.join(check_point_sub_dir,'model.pkl'))
        torch.save(self.tokenizer,os.path.join(check_point_sub_dir,'tokenizer.pkl'))
        torch.save(metadata,os.path.join(check_point_sub_dir,'metadata.pkl'))
        return metadata

 

    def load_check_point_data(self,output_dir,check_point_id):
        check_point_subdir=os.path.join(output_dir,'check_points',str(check_point_id))
        if not os.path.exists(check_point_subdir):
            print(f"check point sub directory doesnt exist {check_point_subdir}")
            return
        self.model=torch.load(os.path.join(check_point_subdir,'model.pkl'),weights_only=False)
        self.tokenizer=torch.load(os.path.join(check_point_subdir,'tokenizer.pkl'),weights_only=False)
        metadata=torch.load(os.path.join(check_point_subdir,'metdata.pkl'),weights_only=False)
        return metadata
    
    def train(self,optimizer,loss_fn,epochs,train_dataloader,val_dataloader,eval_freq,output_dir,check_point_id=0,check_point_freq=5):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        check_point_dir=os.path.join(output_dir,'check_points')
        if not os.path.exists(check_point_dir):
            os.makedirs(check_point_dir)
        
        print(">>>> Begin Training process:")
        if check_point_id!=0:
            metadata=self.load_check_point_data(output_dir,check_point_id)
            if metadata != None:
                last_epoch=metadata['last_epoch']
                train_losses=metadata['train_losses']
                val_losses=metadata['val_losses']
            else:
                print("for the given check point id , the folder doesnt exist, it will run from the beginning.")
                last_epoch=0
                train_losses,val_losses=[],[]

        else:
            self.clear_check_points(output_dir)
            last_epoch=0
            train_losses,val_losses=[],[]
        self.model.to(self.device)
        self.model.train()
        for epoch in range(epochs):
            print(f'epoch value is: {epoch}')
            for input_batch,outputbatch in train_dataloader:
                
                # forward pass 
                loss=self.calc_loss(input_batch,outputbatch,loss_fn)

                # backward pass
                optimizer.zero_grad() # clearing the gradinents
                loss.backward() # calculating the gradients
                optimizer.step() # updating the weights

            net_epoch=last_epoch+epoch+1
            if net_epoch%eval_freq==0:
                print("in for loop for evalation of model")
                train_loss=self.evalaute_model(train_dataloader,loss_fn)
                val_loss=self.evalaute_model(val_dataloader,loss_fn)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f" Epoch : {net_epoch}\n , Train loss:{train_loss:.3f}\n  val loss:{val_loss:.3f}")

            if net_epoch%check_point_freq==0:
                print("in save check points...")
                self.save_check_point_data(output_dir,net_epoch,train_losses,val_losses)
        self.save_check_point_data(output_dir,-1,train_losses,val_losses)
        print("End Training ")


if __name__=='__main__':
    dir=r'D:\DataFiles\nn\pretraining\bookcorpus2\chunked_ds'
    batch_size=10
    train_dataloader=create_data_loader(os.path.join(dir,'train'),batch_size,shuffle=True)
    val_dataloader=create_data_loader(os.path.join(dir,'test'),int(batch_size/2),shuffle=True)
    tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
    tokenizer.pad_token=tokenizer.eos_token
    config={'vocab_size':tokenizer.vocab_size,
            'context_length':1023,
            'emb_dim':768,
            'n_heads':12,
            'n_layers':12,
            'drop_rate':0.25
            }
    model=GPT2Model(config)
    output_dir=os.path.join(dir,'gpt2')
    epoch_size=3
    optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
    loss_fn=torch.nn.CrossEntropyLoss()
    trainer=Trainer(model,tokenizer)
    eval_freq=1
    trainer.train(optimizer,loss_fn,epoch_size,train_dataloader,val_dataloader,eval_freq,output_dir,0,check_point_freq=2)





        




