from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv(override=True)

embeddings_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})

vector_db_path=os.path.expanduser(r'D:\faiss')


vector_db=FAISS.load_local(folder_path=vector_db_path,embeddings=embeddings_model,allow_dangerous_deserialization=True)

queries = [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field.",
]


for query in queries:
    base_retriever=vector_db.as_retriever(search_kwargs={'k':5})
    hits=base_retriever.invoke(query)
    print("\n Query: ", query)
    print("\n Top 5 similar chunks")
    print(hits)
    for hit in hits:
        print(hit.page_content)

