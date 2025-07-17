import os
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from plot_embeddings import plot_list_2d

load_dotenv(override=True)
time.sleep(1)

embedding_model=SentenceTransformer(model_name_or_path=os.getenv('HF_EMBEDDINGS_MODEL'))
words = [
    "rice",
    "attraction",
    "love",
    "friendship",
    "twitter",
    "queen",
    "king",
    "minister",
    "google",
    "eating",
    "habbit",
    "food",
    "cook",
    "peace",
    "submarine",
    "war",
    "missile",
]

word_embeddings=embedding_model.encode(words,normalize_embeddings=True,show_progress_bar=True)
similarity_scores=embedding_model.similarity(word_embeddings,word_embeddings)
print(similarity_scores)
plot_list_2d(word_embeddings,words)