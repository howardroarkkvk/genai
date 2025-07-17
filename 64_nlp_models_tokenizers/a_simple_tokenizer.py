import re
import os

def build_vocab(path):
    with open(path,'r',encoding='utf-8') as f:
        raw_text=f.read()
        preprocessed=re.split(r'([,\.:;\?_!"()\']|--|\s)',raw_text)
        preprocessed_list=[item for item in preprocessed if item]
        sorted_preprocessed_list=sorted(set(preprocessed_list))
        vocab_size=len(sorted_preprocessed_list)
        print(vocab_size)
        vocab={token:integer for integer,token in enumerate(sorted_preprocessed_list)}
        # print("in build vocab",vocab)
    return vocab


class SimpleTokenizerV1:
    def __init__(self,src_dir,file):
        vocab=build_vocab(os.path.join(src_dir,file))
        self.str_to_int=vocab
        self.int_to_str={j:i for i,j in vocab.items()}
        print(vocab)


    def encode(self,text):
        preprocessed=re.split(r'([,\.:;\?_!"()\']|--|\s)',text)
        preprocessed_list=[item.strip() for item in preprocessed if item.strip()]
        print(preprocessed_list)
        ids=[self.str_to_int[s] for s in preprocessed_list]

        return ids
    
    def decode(self,ids):
        text=' '.join([self.int_to_str[i] for i in ids])
        text=re.sub(r'\s+([,.?!"()\'])',r'\1',text)
        return text




# text='Hi! my name is vamshi i work for infosys technologies limited. There are (2 ways) to got to | office'
# x=build_vocab(text)
# print(x)


if __name__=='__main__':
    src_dir=r'D:\DataFiles\nn'
    file='sample.txt'
    tokenizer=SimpleTokenizerV1(src_dir=src_dir,file=file)
    text = "the last he painted, you know, Mrs. Gisburn said with pardonable pride."
    ids=tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))