
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os

load_dotenv(override=True)
time.sleep(1)

embedding_model=SentenceTransformer(model_name_or_path=os.getenv('HF_EMBEDDINGS_MODEL'))
words1=['rice','king','missile']
words2=['emperor','queen','war']


embeddings1=embedding_model.encode(words1,normalize_embeddings=True,show_progress_bar=True)
embeddings2=embedding_model.encode(words1,normalize_embeddings=True,show_progress_bar=True)

similarities=embedding_model.similarity(embeddings1,embeddings2)

print(len(similarities))
print(similarities)

for i, word1 in enumerate(words1):
    print(word1)
    for j, word2 in enumerate(words2):
        print(f" - {word2:<10} : {similarities[i][j]:.4f}")
