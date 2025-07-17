from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv





embeddings_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={"normalize_embeddings":True},model_kwargs={'token':os.getenv('HF_TOKEN')})
vector_db_dir=os.path.expanduser(r'D:\faiss')

# after creating the vector db, we can query the vector db using the load_loacal
vector_db=FAISS.load_local(folder_path=vector_db_dir,embeddings=embeddings_model,allow_dangerous_deserialization=True)


queries= [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field.",
]

# low score represents more similarity as during embedding the algorithm used is distance which is eculidian
for query in queries:
    hits=vector_db.similarity_search_with_score(query,k=5)
    print("\n Query",query)
    print("Top most 5 chunks")
    print(hits)
    for hit in hits:
        print(hit[0].page_content,f"\n Score {hit[1]:.4f}")


