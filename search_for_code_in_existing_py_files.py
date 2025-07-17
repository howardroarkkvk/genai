
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path

path = r'D:\genai'
list_of_py_files=[]
for file in Path(path).iterdir():
    if os.path.isfile(file) : 
        if Path(file).match('*.py'):
            list_of_py_files.append(file)
        elif Path(file).match('*.env'): 
            pass
    elif Path(file).is_dir():
        if not ((str(file).split('\\')[-1]=='.git') or  (str(file).split('\\')[-1]=='.gradio') ):
            for py_files in Path(file).glob('*.py'):
                list_of_py_files.append(py_files)

# for path in list_of_py_files:
    # print(path)

text_splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50,length_function=len)
chunks=[]
for file in list_of_py_files:
    with open(file,'r') as f:
        content=f.read()
        chunks.extend(text_splitter.create_documents(texts=[content],metadatas=[{'source':file}]))




embedding_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
if not os.path.exists(os.path.join(path,'FAISS_CODEBASE')):
    os.makedirs(os.path.join(path,'FAISS_CODEBASE'))
else:
    vector_db_dir=os.path.join(path,'FAISS_CODEBASE')
    print(vector_db_dir)
    vector_db=FAISS.from_documents(documents=chunks,embedding=embedding_model,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
    vector_db.save_local(vector_db_dir)

vector_db_dir=os.path.join(path,'FAISS_CODEBASE')

def retrievecontext(query,vector_db_dir):
    documents=[]
    vector_db_loaded=FAISS.load_local(folder_path=vector_db_dir,embeddings=embedding_model,allow_dangerous_deserialization=True)
    base_retriever=vector_db_loaded.as_retriever(search_kwargs={'k':4})
    documents.extend(base_retriever.invoke(query))
    # return ''.join([str({doc.page_content:doc.metadata['source']}) for doc in documents])
    return [doc.metadata['source'] for doc in documents]
    

list1=retrievecontext('multiprocessing',vector_db_dir=vector_db_dir)
print(list1)
for i in list1:
    print(i)




            
    
    

    



