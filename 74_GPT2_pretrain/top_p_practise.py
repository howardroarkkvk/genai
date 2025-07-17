import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
input=torch.tensor([[2,9,5,4,3],[3,4,5,6,7]])
print(input,input.shape)

embedder=nn.Embedding(10,8)
print(embedder)
output=embedder(input)
# print(output,output.shape)

logits=output
print('logits are:',logits,logits.shape)

temperature = 5.0
logits=logits[:,-1,:] # here we are indexing the logits whose shape changes to 2d from 3d by keeping the last element for each sentence....now no more sentences, it's (1word sentence, embeddings.)..
print('logits without temp change:',logits,logits.shape) # it displays the last word from each sentence...

probs=torch.softmax(logits,dim=-1)
print(probs)


# [[0.2011, 0.2046, 0.1031, 0.2126, 0.0444, 0.0584, 0.0435, 0.1323],
# [0.0156, 0.0660, 0.0597, 0.1251, 0.0577, 0.4026, 0.0195, 0.2539]],



sorted_probs,sorted_indices=torch.sort(probs,descending=True) # sort givess the logit probs in descing order value matrix and indices matrix
print(sorted_probs)
print(sorted_indices)

# [[0.2126, 0.2046, 0.2011, 0.1323, 0.1031, 0.0584, 0.0444, 0.0435],
#  [0.4026, 0.2539, 0.1251, 0.0660, 0.0597, 0.0577, 0.0195, 0.0156]]
# [[3, 1, 0, 7, 2, 5, 4, 6],
# [5, 7, 3, 1, 2, 4, 6, 0]]

cumulative_probs=torch.cumsum(sorted_probs,dim=-1) # on top of sorted_probs, we can appply cumsum which gives cumulative probabilities....
print('cumulative probs is',cumulative_probs)

# [[0.2126, 0.4173, 0.6184, 0.7507, 0.8537, 0.9121, 0.9565, 1.0000],
# [0.4026, 0.6564, 0.7815, 0.8475, 0.9072, 0.9649, 0.9844, 1.0000]],


top_p=0.9
sorted_mask=cumulative_probs>0.9 # we get the masking tensor with values greater than .9 in cumulative probability set to true....
print(sorted_mask)

# [[False, False, False, False, False,  True,  True,  True],
# [False, False, False, False,  True,  True,  True,  True]]


# if we want to include one tensor value which is just exceeded 0.9 probability...
sorted_mask[:,1:]=sorted_mask[:,:-1].clone()
sorted_mask[:,0]=0
print(sorted_mask)
# [[False, False, False, False, False, False,  True,  True],
# [False, False, False, False, False,  True,  True,  True]]


filtered_probs=sorted_probs.masked_fill(sorted_mask,0.0)
print(filtered_probs)

# [[0.2126, 0.2046, 0.2011, 0.1323, 0.1031, 0.0584, 0.0000, 0.0000],
# [0.4026, 0.2539, 0.1251, 0.0660, 0.0597, 0.0000, 0.0000, 0.0000]],

next_token=torch.multinomial(filtered_probs,num_samples=1)
print(next_token)
# [[1],
# [4]]

next_token_id=sorted_probs.gather(dim=-1,index=next_token)
print(next_token_id)