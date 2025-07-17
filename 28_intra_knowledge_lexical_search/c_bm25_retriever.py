from langchain_community.retrievers import BM25Retriever
import os
import pickle


bm25_retriever=BM25Retriever.from_texts(    [
        "Australia won the Cricket World Cup 2023",
        "India and Australia played in the finals",
        "Australia won the sixth time having last won in 2015",
    ])

bm25_dir=r'D:\bm25'

with open(os.path.join(bm25_dir,'bm25.pkl'),'wb') as f:
    pickle.dump(bm25_retriever,f)

with open(os.path.join(bm25_dir,'bm25.pkl'),'rb') as r:
  bm25_retriever=pickle.load(r)  
# bm25_retriever=pickle.load(open(os.path.join(bm25_dir,'bm25.pkl'),'rb'))
results=bm25_retriever.invoke('won')
for result in results:
    print(result.page_content)
