
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os


# we have vector db store now....
path1=r'D:\faiss3'
embeddings_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
vector_db=FAISS.load_local(folder_path=path1,embeddings=embeddings_model,allow_dangerous_deserialization=True)

queries = [
    "What are Recurrent neural networks?",
    "Explain Encoder and Decoder Stacks",
    "What do you mean by Multi-Head Attention?",
]

for query in queries:
    retriever=vector_db.as_retriever(search_kwargs={'k':2})
    documents=retriever.invoke(query)

    print("\n Query used is:",query)
    print('-----------------------------------------------------------------------------------------------------------------------')
    print("Top 2 documents related to the query")
    print('-----------------------------------------------------------------------------------------------------------------------')
    for document in documents:
        print(document.page_content,f'Source:{document.metadata['source']}')