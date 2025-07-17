

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os


embedding_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
vector_db_path=r'D:\faiss2'
vector_db=FAISS.load_local(folder_path=vector_db_path,embeddings=embedding_model,allow_dangerous_deserialization=True)

queries = [
    "How long does international shipping take?",
    "Show the Leather jackets in the store",
    "Contact details of store",
    "What is the return policy?",
]

for query in queries:
    base_retriever=vector_db.as_retriever(search_kwargs={'k':2})
    result=base_retriever.invoke(query)
    print('\n Query : ',query)
    print('---------------------')    
    for hit in result:
        print(hit.page_content,f'Source:{hit.metadata.get('source')}')
        print()