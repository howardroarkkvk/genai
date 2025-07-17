

import os
# for using this the mandatory package to be installed is sentence_transformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from agent import *




class DocumentationRAG:
    def __init__(self,src_dir:str,kb_dir:str,vector_db_dir:str,embedding_model:str,top_k:int,llm_name:str,system_prompt:str):
        print("It''s in init()")
        self.src_dir=src_dir
        self.kb_dir=os.path.join(src_dir,kb_dir)
        self.vector_db_dir=os.path.join(src_dir,vector_db_dir)
        self.embedding_model=embedding_model
        self.top_k=top_k
        self.rag_agent=get_agent(llm_name,system_prompt)
        self.embedding_model=HuggingFaceEmbeddings(model_name=self.embedding_model,encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
        self.ingest()

# steps to ingest data
# first check what file to ingest, if it is json, we dont need any textsplitter, other documents would need to define textsplitter
# create embedding model
# load the file to documents out of chunks(creted using textsplitter)
# from the documents create the vector db
# save the vector_db locally

    def metadata_func(self,record:dict,metadata:dict)->dict:
        metadata['source']=record.get('chunk_link')
        metadata['title']=record.get('chunk_heading')
        return metadata

    
    def ingest(self):

        print("Ingestion Begins......")

        
        documents=[]
        for file in Path(self.kb_dir).glob('*.json'):
            loader=JSONLoader(file_path=file,jq_schema='.[]',content_key='.text',is_content_key_jq_parsable=True,metadata_func=self.metadata_func)
            documents.extend(loader.load())

        vector_db=FAISS.from_documents(documents=documents,embedding=self.embedding_model,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
        vector_db.save_local(self.vector_db_dir)
        self.vector_db=vector_db

        print("Ingestion ends......")


# how to retrieve documents
# load vector db
# using as_retriever of vector db and invoke method of retriever, pass the query to invoke methoed, it results in output which is list contains, page_content and metadata
    
    def retrievecontent(self,query:str):
        
        base_retriever=self.vector_db.as_retriever(search_kwargs={'k':self.top_k})
        hits=base_retriever.invoke(query)
        return [hit.page_content for hit in hits]
    
# Execute is to run when we want to get the response from the agent....
    def execute(self,query:str):
        deps=KnowledgeDeps(self.vector_db,query,self.top_k)
        result=self.rag_agent.run_sync(query,deps=deps)
        return result.data



    
        
        





                





