import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy

# we tried to access the text files present in kb1 path
# using Path we are trying to access all the files present in the path
# we iterate over each file, read the conent  using the Textloader class 
# before that we create the RecursiveCharacterTextSplitter object to pass it to load_and_split of TextLoader object
# we store the output of the load_and_split to a list...
# we create the embeddings_model
# using the faiss model and from_documents function,we create the vector db
# we store the vector db as faiss2

load_dotenv(override=True)

dir=r'D:\DataFiles\kb1'
path1=os.path.abspath(dir)
print(path1)
list_of_files=Path(path1).glob('*.txt')
text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20,length_function=len)
documents=[]
for files in list_of_files:
    print(files)
    text_loader_cls=TextLoader(files)
    documents.extend(text_loader_cls.load_and_split(text_splitter=text_splitter))
    
    print(len(documents))
    print(documents)
    # print(documents[0].page_content)
    # print(documents[0].metadata['source'])

embedding_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})

vector_store=FAISS.from_documents(documents=documents,embedding=embedding_model,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
vector_dir=os.path.abspath(r"D:\faiss2")
vector_store.save_local(vector_dir)