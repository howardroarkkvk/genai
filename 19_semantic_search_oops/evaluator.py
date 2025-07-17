import json
import os
from a_ingestor import DataIngestor
from b_retriever import Retriever
import pandas as pd

main_dir=r'D:\DataFiles\oops'
file_path=r'D:\DataFiles\oops\evaluator'
json_path=os.path.join(file_path,'docs_evaluation_dataset.json')

with open(json_path,'r') as f:
    json_output=json.load(f)
    # print(json_output[0]['correct_chunks'])

main_dir=r'D:\DataFiles\oops'
vector_db_path=r'D:\DataFiles\oops\faiss_oops'

dr=Retriever(main_dir=main_dir,vector_db_dir=vector_db_path)

precisions=[]
recalls=[]
f1s=[]
verdicts=[]

for i,json_oput in enumerate(json_output):
    print('query used:',i,json_oput['question'])
    # print(i,json_oput['correct_chunks'])
    correct_chunks=json_oput['correct_chunks']
    correct_set=set(correct_chunks)
    retrieved_chunks=dr.retriever(json_oput['question'],top_k=2)
    retrieved_set=set(retrieved_chunks)
    true_positives=list(correct_set.intersection(retrieved_set))
    
    precision=len(true_positives)/len(retrieved_chunks) if retrieved_chunks else 0
    recall=len(true_positives)/len(correct_chunks) if correct_chunks else 0
    f1= (2*(precision*recall)/(precision+recall) if precision+recall else 0)
    # print('f1 is :',f1)
    verdict=True if f1>=0.3 else 0
    # print('prescision',precision)
    # print('recall',recall)
    # print('f1',f1)
    # print('verdict',verdict)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    verdicts.append(verdict)
# print('prescisions',precisions)
# print('recalls',recalls)
# print('f1s',f1s)
# print('verdicts',verdicts)

df=pd.DataFrame({'question':[ item['question'] for item in json_output],
                'verdicts':verdicts,
                'retrieval_precision':precisions,
                'retrieval_recalls':recalls,
                'retrieval_f1s':f1s}
                                         )
print(df)

avg_precision=sum(precisions)/len(precisions) if precisions else 0
avg_recall=sum(recalls)/len(recalls) if recalls else 0
avg_f1s=sum(f1s)/len(f1s) if f1s else 0
# df.to_csv(os.path.join(main_dir,'results.csv'),index=False)

with open(os.path.join(main_dir,'summary.json'),'w') as f:
    json.dump({'avg_precision':avg_precision,
               'avg_recall':avg_recall,
               'avg_f1s':avg_f1s
               },f,indent=2)
