
import time
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import os

load_dotenv(override=True)
time.sleep(1)


ce_model=CrossEncoder(model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2',max_length=256)


# embedding_model=SentenceTransformer(model_name_or_path=os.getenv('HF_EMBEDDINGS_MODEL'))
words1=['rice','king','missile']
words2=['rice','queen','war','wheat']

for i, word1 in enumerate(words1):
    # print(word1)
    for j,word2 in enumerate(words2):
        # print(word2)
        score=ce_model.predict([word1,word2])
        print(word1,word2,score)





# embeddings1=embedding_model.encode(words1,normalize_embeddings=True,show_progress_bar=True)
# embeddings2=embedding_model.encode(words1,normalize_embeddings=True,show_progress_bar=True)

# similarities=embedding_model.similarity(embeddings1,embeddings2)

# print(len(similarities))
# print(similarities)

# for i, word1 in enumerate(words1):
#     print(word1)
#     for j, word2 in enumerate(words2):
#         print(f" - {word2:<10} : {similarities[i][j]:.4f}")
