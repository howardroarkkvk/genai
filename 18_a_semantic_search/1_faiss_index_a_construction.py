import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy


load_dotenv(override=True)


embeddings_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),model_kwargs={'token':os.getenv('HF_TOKEN')},encode_kwargs={'normalize_embeddings':True})

chunks = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]

# FROM TEXT TAKES 3 INPUTS chunks in the form of list, embedding model object with the type of embedding model and distance strategy. it creates a vector db.

vector_db=FAISS.from_texts(texts=chunks,embedding=embeddings_model,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)



print(vector_db.index)
print(vector_db.docstore)
print(vector_db.index_to_docstore_id)

path1=os.path.expanduser(r"D:\faiss")
vector_db.save_local(folder_path=path1)
