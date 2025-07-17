from datasets import load_dataset
from sentence_transformers import SentenceTransformer,util,CrossEncoder
import os

dataset=load_dataset("jamescalam/ai-arxiv-chunked")
chunks=dataset['train']['chunk']
# print(type(chunks))
# print(len(chunks))
# print(dataset['train'][0])

bi_encoder=SentenceTransformer(model_name_or_path=os.getenv('HF_EMBEDDINGS_MODEL'))
# print(bi_encoder.max_seq_length)

embeddings=bi_encoder.encode(chunks[0:100],normalize_embeddings=True,show_progress_bar=True)

query="what is rlhf?"
top_k=10

query_embeddings=bi_encoder.encode(query,normalize_embeddings=True,show_progress_bar=True)

hits=util.semantic_search(query_embeddings,embeddings,top_k=top_k)[0]


top_25_corpus_ids=[hit['corpus_id'] for hit in hits]
print(top_25_corpus_ids)

for i,hit in enumerate(hits[:3]):
    # print('corpus id  is :',hit['corpus_id'])
    sample=dataset['train'][hit['corpus_id']]
    # print(f"Top {i+1} passage with score {hit['score']} from {sample['source']}:")
    # print(sample['chunk'])
    # print('\n')

cross_encoder=CrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
cross_inps=[[query,chunks[hit['corpus_id']]] for hit in hits]
scores=cross_encoder.predict(cross_inps)
print(scores)

for idx in range(len(scores)):
    hits[idx]['cross-score']=scores[idx]
    
hits=sorted(hits,key=lambda x:x['score'],reverse=True)
# print(hits)
for hit in hits:
    print(hit)