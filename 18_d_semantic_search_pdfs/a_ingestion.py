

import os
from pathlib import Path
from langchain_community.document_loaders import  PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy


text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20,length_function=len)

path1=r'D:\DataFiles\kb2'
files_path=os.path.abspath(path1)
documents=[]
for file in Path(files_path).glob('*.pdf'):
    loader=PyPDFLoader(file)
    documents.extend(loader.load_and_split(text_splitter=text_splitter))
#print(documents)

embeddings_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})

path_to_store_vecor_db=r'D:\faiss3'
vector_db=FAISS.from_documents(documents=documents,embedding=embeddings_model,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
vector_db.save_local(path_to_store_vecor_db)







