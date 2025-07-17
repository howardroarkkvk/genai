from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from pathlib import Path
from langchain_community.document_loaders import TextLoader


load_dotenv(override=True)

file_path=r'D:\DataFiles\kb1'
text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=10,length_function=len)
HuggingFaceEmbeddings()



documents=[]
for file in Path(file_path).glob('*.txt'):
    loader=TextLoader(file)
    documents.extend(loader.load_and_split(text_splitter=text_splitter))

print(len(documents))
# print(documents)


embedding_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
vector_db=FAISS.from_documents(documents=documents,embedding=embedding_model,distance_strategy=DistanceStrategy.COSINE)
vector_db.save_local(r'D:\DataFiles\faiss_rag_practical')


    




