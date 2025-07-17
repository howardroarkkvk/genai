

import os
from pathlib import Path
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy


# text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20,length_function=len)

path1=r'D:\DataFiles\kbn'
files_path=os.path.abspath(path1)
documents=[]
for file in Path(files_path).glob('*.json'):
    print(file)
    loader=JSONLoader(file_path=file,jq_schema='.[]',content_key="[.name,.description]|@tsv",is_content_key_jq_parsable=True)
    # loader=JSONLoader(file_path=file,jq_schema=".",text_content=True)#,content_key="content"
    data=loader.load()
    # print(data)
    documents.extend(loader.load())
    print(documents)
    for document in documents:
            
        print('-----------------------------------------------------------------------------------------------------------------------')
        # print("Top 2 documents related to the query")
        print('-----------------------------------------------------------------------------------------------------------------------')
        print(document)
        # print(document.page_content,f'Source:{document.metadata['source']}')


embeddings_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})

# path_to_store_vecor_db=r'D:\faiss4'
# vector_db=FAISS.from_documents(documents=documents,embedding=embeddings_model,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
# vector_db.save_local(path_to_store_vecor_db)







