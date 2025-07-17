import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

def param_count(model):
   return sum(p.numel() for p in  model.parameters() if p.requires_grad) # here from model_parameters we are checking for models with requried_grad is True and using that model parameter, we are trying to get number of elements from that parameters using numel and sum of all that is retured gives the overall model parameters


class MultiheadMaskedSelfAttention(nn.Module):
   
   def __init__(self,config):
      super().__init__()
      assert (config['emb_dim']% config['n_heads']==0),f'embedding dim should be divisible by number of heads'
      self.num_heads=config['n_heads']
      self.emb_dim=config['emb_dim']
      self.attn=nn.Linear(self.emb_dim,3*self.emb_dim)
      self.c_proj=nn.Linear(self.emb_dim,self.emb_dim)
   
   def forward(self,x):
      B,T,C=x.shape   # batch size, i.e no. of sentences, T, no. of tokens, C, channel i.e. embedding size
      qkv=self.attn(x)
      q,k,v=qkv.split(self.emb_dim,dim=-1) # q,k,v are of dimensions 1,3,6 each.....as the split is done against the last dimension of self.attn layer...input i.e passed through self.attention which is converted to 1,3,18
      q=q.view(B,T,self.num_heads,self.emb_dim//self.num_heads).transpose(1,2) # 1,3,6 is changed its view to 1,3,3,2 --> 1 sentence, 3 words, 3 heads eachof 2 embeds...
      # after transpose q becomes B,T,nh,hs --> B,nh,T,hs.....
      k=k.view(B,T,self.num_heads,self.emb_dim//self.num_heads).transpose(1,2)
      v=v.view(B,T,self.num_heads,self.emb_dim//self.num_heads).transpose(1,2)
      out=F.scaled_dot_product_attention(q,k,v,is_causal=True) 
      out=out.transpose(1,2).contiguous().view(B,T,C)
      out=self.c_proj(out)
      return out


class MLP(nn.Module): # multi layered perceptron
   def __init__(self,config):
      super().__init__()
      self.c_fc=nn.Linear(config['emb_dim'],4*config['emb_dim'])
      self.gelu=nn.GELU(approximate='tanh')
      self.c_proj=nn.Linear(4*config['emb_dim'],config['emb_dim'])

   def forward(self,x):
      x=self.c_fc(x)
      x=self.gelu(x)
      x=self.c_proj(x)
      return x


class DecoderBlock(nn.Module): # decoder block
   def __init__(self,config=None):
      super().__init__()
      self.attn=MultiheadMaskedSelfAttention(config)
      self.mlp=MLP(config)
      self.ln_1=nn.LayerNorm(config['emb_dim'])
      self.ln_2=nn.LayerNorm(config['emb_dim'])
      self.drop_shortcut=nn.Dropout(config['drop_rate'])

   def forward(self,x):
      shortcut=x
      x=self.ln_1(x)
      x=self.attn(x)
      x=self.drop_shortcut(x)
      x=x+shortcut

      shortcut=x
      x=self.ln_2(x)
      x=self.mlp(x)
      x=self.drop_shortcut(x)
      x=x+shortcut

      return x

class GPT2Model(nn.Module):
   def __init__(self,config):
      super().__init__()
      self.transformer=nn.ModuleDict(dict(
         wte=nn.Embedding(config['vocab_size'],config['emb_dim']),
         wpe=nn.Embedding(config['context_length'],config['emb_dim']),
         drop=nn.Dropout(config['drop_rate']),
         h=nn.ModuleList([DecoderBlock(config)  for _ in range(config['n_layers'])]),
         ln_f=nn.LayerNorm(config['emb_dim'])
      ))
      self.lm_head=nn.Linear(config['emb_dim'],config['vocab_size'],bias=False)
      self.transformer.wte.weight=self.lm_head.weight  # 
      self.config=config
      self.apply(self._init_weights)
      print(f'model parameter count is :{param_count(self)}')


   def _init_weights(self,module):
      if isinstance(module,nn.Linear):
         torch.nn.init.normal_(module.weight,mean=0.0 ,std=0.2)
         if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
      elif isinstance(module,nn.Embedding):
         torch.nn.init.normal_(module.weight,mean=0.0 ,std=0.2)
      
   def forward(self,x):
      tok_embeds=self.transformer.wte(x)
      positions=torch.arange(0,x.shape[1]).unsqueeze(0)
      pos_embeds=self.transformer.wpe(positions)
      x=pos_embeds+tok_embeds
      x=self.transformer.drop(x)
      for block in self.transformer.h:
         x=block(x)
      x=self.transformer.ln_f(x)
      logits=self.lm_head(x)
      return logits
   
   @classmethod
   def from_pretrained(cls,type,vocab_size):
      assert type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'} # while calling this class method , if the type is not one present in this it will give assert error
      print(f"Begin loading weights of pretrained model: {type}")
      config_args={
         'gpt2':dict(n_layers=12,n_heads=12,emb_dim=768), # 124M params
         'gpt2-medium':dict(n_layers=24,n_heads=16,emb_dim=1024), # 350M params
         'gpt2-large':dict(n_layers=36,n_heads=20,emb_dim=1280), # 774M params
         'gpt2-xl':dict(n_layers=48,n_heads=25,emb_dim=1600), # 1558M params
      }
      config=config_args[type]
      config['vocab_size']=vocab_size
      config['context_length']=1024
      config['drop_rate']=0.0

      model=GPT2Model(config)
      sd=model.state_dict()
      sd_keys=sd.keys()
      sd_keys=[k for k in sd_keys if not k.endswith('.attn.bias')]

      model_hf=GPT2LMHeadModel.from_pretrained(type) # huggiing face model
      sd_hf=model_hf.state_dict()
      sd_keys_hf=sd_hf.keys()
      sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
      sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.bias')]

      transposed=['attn.attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']

      assert len(sd_keys)==len(sd_keys_hf),f'mismathed keys {len(sd_keys)}!={len(sd_keys_hf)}'

      for k in sd_keys_hf:
         if any(k.endswith(w) for w in transposed):
            assert sd_hf[k].shape[::-1]==sd[k].shape
            with torch.no_grad():
               sd[k].copy_(sd_hf[k].T)
         else:
            assert sd_hf[k].shape==sd[k].shape
            with torch.no_grad():
               sd[k].copy_(sd_hf[k])
      print(f"End loading weights of pre trained model :{type}")
      return model


            # Condition	Meaning
            # Omitted start	Defaults to 0 if step is positive, -1 if negative
            # Omitted stop	Goes to the end if step is positive, start of list if step is negative
            # Negative step	Goes in reverse direction
            # stop is exclusive	It is not included in the result





# if __name__=='__main__':

#    config = {
#     "vocab_size": 50257,
#     "context_length": 1024,
#     "emb_dim": 768,
#     "n_heads": 12,
#     "n_layers": 12,
#     "drop_rate": 0.25,
#             }
#    model = GPT2Model(config)
#    print('model state dict keys',model.state_dict().keys())
#    # print('model state dict values ',model.state_dict().values())
#    print('model parameters')
#    for i in model.parameters(recurse=True):
#       print (i.shape)
#    for name, param in model.named_parameters():
#       print(name, param.shape)
#    print(model)   
#    model.from_pretrained('gpt2',50257)   