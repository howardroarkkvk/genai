import torch
import torch.nn as nn
import torch.nn.functional as F


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TextGenerationModel4(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.tok_embedding=nn.Embedding(config["vocab_size"],config['embed_dim'])
        self.pos_embedding=nn.Embedding(config["context_length"],config['embed_dim'])
        self.hidden_layer=nn.Linear(config['embed_dim'],config['hidden_dim'])
        self.output_layer=nn.Linear(config['hidden_dim'],config['vocab_size'])
        self.config=config
        self.attn=nn.MultiheadAttention(config['embed_dim'],num_heads=config['n_heads'],batch_first=True)

    def forward(self,x):
        print('input shape is :',x.shape)
        tok_embeds=self.tok_embedding(x)

        positions=torch.arange(0,x.shape[1]).unsqueeze(0) # need to check what is the dimension of x while it's inputted
        pos_embeds=self.pos_embedding(positions)

        embeds=tok_embeds+pos_embeds
        mask=torch.triu(torch.ones(x.shape[1],x.shape[1]),diagonal=1).bool()
        x=self.attn(embeds,embeds,embeds,attn_mask=mask)[0]
        x=F.relu(self.hidden_layer(x))
        logits=self.output_layer(x)
        # print('logits:',logits.shape)
        print('logits output is :',logits, logits.shape)
        return logits
    
if __name__=='__main__':
    model_config={'vocab_size':50257,'context_length':10,'embed_dim':128,'hidden_dim':64,'n_heads':1}
    model=TextGenerationModel4(model_config)
    print(model)
    input_tensor=torch.tensor([[  286, 11666,  4430, 31373],[ 1462,   262,   995,   286]])
    expected_tensor=torch.tensor([[11666,  4430, 31373,   703],[  262,   995,   286, 11666]])
    print('expected tensor shape ...',expected_tensor.shape)
    flattend_expected_tensor=expected_tensor.flatten()
    print('flattened expected tensor',flattend_expected_tensor.shape)
    print('input tensor shape is :',input_tensor.shape)
    x=model(input_tensor)
    print('x . shape',x.shape)
    y=x.flatten(0,1)
    print('y . shape',y.shape)
    loss_fn=F.cross_entropy(y,flattend_expected_tensor)
    print(loss_fn.item())
    # print(" This is inside for loop")
    # for i in model.parameters():
    #     print(i,i.shape)
    # print(" This is AFTER for loop")

    # print(" This is inside for loop for named parameter")
    # for name,param in model.named_parameters():
    #     print(f'layername=:{name} ,{param},{param.shape} ')
    # print(" This is AFTER for loop for named parameter")
    # print(param_count(model))

