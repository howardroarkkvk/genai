import torch
import os
import shutil
from tqdm import tqdm
from transformers import AutoTokenizer
from c_models import * 
from b_data_loader import * 

class Trainer:

    def __init__(self,model,tokenizer):
        self.model=model
        self.tokenizer=tokenizer
        self.device= 'gpu' if torch.cuda.is_available() else 'cpu'

 
    def calcualte_loss(self,input_batch,output_batch,loss_fn):
        input_batch=input_batch.to(self.device)
        output_batch=output_batch.to(self.device)
        logits=self.model(input_batch)
        loss=loss_fn(logits.flatten(0,1),output_batch.flatten())
        return loss
    
    def evaluate_model(self,data_loader,loss_fn):
        if len(data_loader)==0:
            return float("nan") # as this calcualtes the overalll loss for a batch....if dataloader is 0 we are putting the loss as nan
        self.model.eval()
        total_loss=0.0
        with torch.no_grad():
            for input_batch,output_batch in data_loader:
                loss=self.calcualte_loss(input_batch,output_batch,loss_fn)
                total_loss+=loss.item()
        self.model.train()
        return total_loss/len(data_loader)
    
    def load_check_point_data(self,output_dir,check_point_id):
        check_point_subdir=os.path.join(output_dir,'check_points',str(check_point_id))
        if not os.path.exists(check_point_subdir):
            print(f'output dir {check_point_subdir} doesnt exist')
            return
        self.model=torch.load(os.path.join(check_point_subdir,'model.pkl'),weights_only=False)
        self.tokenizer=torch.load(os.path.join(check_point_subdir,'tokenizer.pkl'),weights_only=False)
        metadata=torch.load(os.path.join(check_point_subdir,'metadata.pkl'),weights_only=False)
        return metadata
    
    def clear_check_points(self,output_dir):
        check_point_dir=os.path.join(output_dir,'check_points')
        if os.path.exists(check_point_dir):
            shutil.rmtree(check_point_dir)

    def save_check_points_data(self,output_dir,train_losses,val_losses,last_epoch):
        folder_id ='full' if last_epoch=='-1' else str(last_epoch)
        metadata={'train_losses':train_losses,
                  'val_losses':val_losses,
                  'last_epoch':last_epoch}
        check_point_subdir=os.path.join(output_dir,'check_points',folder_id)
        if not os.path.exists(check_point_subdir):
            os.makedirs(check_point_subdir)
        torch.save(self.model,os.path.join(check_point_subdir,'model.pkl'))
        torch.save(self.tokenizer,os.path.join(check_point_subdir,'tokenizer.pkl'))
        torch.save(metadata,os.path.join(check_point_subdir,'metadata.pkl'))

    
    def train(self,optimizer,loss_fn,epochs,train_dataloader,val_dataloader,eval_freq,output_dir,check_point_id=0,check_point_freq=5):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        check_point_dir=os.path.join(output_dir,'check_points')
        if not os.path.exists(check_point_dir):
            os.makedirs(check_point_dir)
        print(">>>: Beginning of training")
        if check_point_id!=0:
            metadata=self.load_check_point_data(output_dir,check_point_id)
            train_losses=metadata['train_losses']
            val_losses=metadata['val_losses']
            last_epoch=metadata['last_epoch']
        else:
            self.clear_check_points(output_dir)
            train_losses,val_losses=[],[]
            last_epoch=0
            self.model.to(self.device)
            self.model.train()
            for epoch in range(epochs):
                for input_batch,output_batch in tqdm(train_dataloader):
                    
                    #forward pass
                    loss=self.calcualte_loss(input_batch,output_batch,loss_fn)

                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    net_epoch=last_epoch+epoch+1

                    if net_epoch%eval_freq==0:
                        train_loss=self.evaluate_model(train_dataloader,loss_fn)
                        val_loss=self.evaluate_model(val_dataloader,loss_fn)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        print(f"Epoch : {net_epoch}")
                        print(f"Train loss is : {train_loss:.3f} , val loss is :{val_loss:.3f}")
                    if net_epoch%check_point_freq==0:
                        self.save_check_points_data(output_dir,train_losses,val_losses,net_epoch)
            self.save_check_points_data(output_dir,train_losses,val_losses,'-1')
            print("End training")


if __name__=='__main__':
    tokenizer=AutoTokenizer.from_pretrained('openai-community/gpt2')
    tokenizer.add_special_tokens({'pad_token':tokenizer.eos_token,
                                  'bos_token':'<|im_start|>',
                                  'eos_token':'<|im_end|>'})
    
    model=GPT2Model.from_pretrained('gpt2',len(tokenizer))
    epochs=2
    batch_size=8
    optimizer=torch.optim.SGD(model.parameters(),lr=0.1)
    loss_fn=torch.nn.CrossEntropyLoss()
    src_dir=r'D:\DataFiles\nn\fine_tuning\dolly'
    train_dir=os.path.join(src_dir,'tokenized_ds','train')
    val_dir=os.path.join(src_dir,'tokenized_ds','test')
    output_dir=os.path.join(src_dir,'gpt2_it')
    train_dataloader=create_data_loader(train_dir,batch_size,shuffle=True)
    val_dataloader=create_data_loader(val_dir,batch_size,shuffle=True)
    trainer=Trainer(model,tokenizer)
    trainer.train(optimizer,loss_fn,epochs,train_dataloader,val_dataloader,eval_freq=1,output_dir=output_dir,check_point_id=0,check_point_freq=1)

                    







        



        



        
        