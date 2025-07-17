from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_from_disk
import os
from transformers import  AutoTokenizer
from transformers import GPT2Config,GPT2LMHeadModel,Trainer,TrainingArguments,DataCollatorForLanguageModeling

# for triaining...
# first we need dataset so dataset which is loaded on the local disk, can be loaded in to a variaable...
# for creating batches of input data, we used to use DataLoader, here we use datacollator for language models...
# we have to initilize tokenizer
# we need to fetch the config to be passed for model which is readlily available
# create the model instance using the model form the hugging face...
# set the training arguments
# after initializng trainer object...
# run the trian() method/function.

load_dotenv(override=True)

# authenticate using token for model download
login(token=os.getenv('HF_TOKEN'),add_to_git_credential=True)

src_dir=r'D:\DataFiles\nn\pretraining\bookcorpus2'
dataset=load_from_disk(os.path.join(src_dir,'chunked_ds'))
print(dataset)


# creation of tokenizer
model_id="openai-community/gpt2"
tokenizer=AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'pad_token':tokenizer.eos_token}) # tokenizer.pad_token= tokenier.eos_token...

# create model
config=GPT2Config()
print(config)
model=GPT2LMHeadModel(config)
print(model)


# Training Arguments
output_dir=os.path.join(src_dir,'gpt2_hf')
log_dir=os.path.join(src_dir,'logs')

training_args=TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="epoch",
    save_steps=1,
    eval_strategy="epoch",
    eval_steps=1,
    dataloader_pin_memory=False
)

# initilize trainer
data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=False) # this is for batching the data .....
trainer=Trainer(model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator # it actually prepares batches of data by tokenizing the data
        )

# train the model
trainer.train()
