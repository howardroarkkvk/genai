from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import time

embeddings_model=SentenceTransformer(model_name_or_path=os.getenv('HF_EMBEDDINGS_MODEL'))
res=embeddings_model.encode("king",normalize_embeddings=True)
# print(res.shape)
print(res.shape)
res=embeddings_model.encode('queen',normalize_embeddings=True)
print(res.shape)

res=embeddings_model.encode('weather',normalize_embeddings=True)
print(res.shape)
sentences=[" The weather is lovely today","its so sunny outside!","he drove to the stadium"]
res=embeddings_model.encode(sentences=sentences,normalize_embeddings=True,show_progress_bar=True)
print(res.shape)