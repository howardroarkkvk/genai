

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import TextLoader

kb_path=r'D:\DataFiles\kb1'
embedding_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20,length_function=len)

documents=[]
for file in Path(kb_path).glob('*.txt'):
    loader=TextLoader(file)
    documents.extend(loader.load_and_split(text_splitter))
print(len(documents))

vector_file_path=r'D:\DataFiles\faiss_kb1'
vector_db=FAISS.from_documents(documents=documents,embedding=embedding_model,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
vector_db.save_local(vector_file_path)