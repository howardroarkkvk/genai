from datasets import load_from_disk
import os
from transformers import AutoTokenizer
from huggingface_hub import login
from transformers import AutoModelForCausalLM,DataCollatorForLanguageModeling
from trl import SFTConfig,SFTTrainer

login(token=os.getenv("HF_TOKEN"), add_to_git_credential=True)
# loading the dataset from local folder which is the tokenized dataset as the tokenized is the one used for training purpose it has both train and val datasets....
src_dir=r'D:\DataFiles\nn\fine_tuning\dolly'
dataset=load_from_disk(os.path.join(src_dir,'tokenized_ds_hf'))
print(dataset)

# 
model_id='openai-community/gpt2'
tokenizer=AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'pad_token':tokenizer.eos_token,
                              'bos_token':'<|im_start|>',
                              'eos_token':'<|im_end|>'})


model=AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,device_map='auto')
model.resize_token_embeddings(len(tokenizer))
print(model)

output_dir=os.path.join(src_dir,'gpt_it_hf')
log_dir=os.path.join(src_dir,'logs')

training_args=SFTConfig(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy='epoch',
    save_steps=1,
    eval_strategy='epoch',
    eval_steps=1,
    bf16=False
)



data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False) # this prepares the batches of tokens from the input data...

trainer=SFTTrainer(model=model,
           args=training_args,
           train_dataset=dataset['train'],
           eval_dataset=dataset['test'],
           data_collator=data_collator

           )

trainer.train()