import os
from agent import get_agent,Deps
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy


class TexttoSqlRAG:
    def __init__(self,src_dir:str,kb_dir:str,vector_db_dir:str,embedding_model_name:str,top_k:int,llm_name:str,system_prompt:str):
        self.kb_dir=os.path.join(src_dir,kb_dir)
        self.vector_db_dir=os.path.join(src_dir,vector_db_dir)
        self.embedding_model_name=HuggingFaceEmbeddings(model_name=embedding_model_name,encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
        self.top_k=top_k
        self.agent=get_agent(llm_name,system_prompt)
        self.ingest()

    def metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["table"] = record.get("table")
        metadata["column"] = record.get("column")
        metadata["type"] = record.get("type")
        return metadata

    def ingest(self):
        print("Ingestion process begins here:")
        documents=[]
        for file in Path(self.kb_dir).glob('*.json'):
            loader=JSONLoader(file,jq_schema='.[]',content_key='.text',is_content_key_jq_parsable=True,metadata_func=self.metadata_func)
            documents.extend(loader.load())
        vector_db=FAISS.from_documents(documents,embedding=self.embedding_model_name,distance_strategy=DistanceStrategy.COSINE)
        vector_db.save_local(self.vector_db_dir)
        self.vector_db=vector_db

    def retreiveContext(self,query:str):
        base_retriever=self.vector_db.as_retriever(search_kwargs={'k':self.top_k})
        results=base_retriever.invoke(query)
        return [result.page_content for result in results]
    
    def execute(self,query):
        deps=Deps(self.vector_db,query,self.top_k)
        response=self.agent.run_sync(query,deps=deps)
        return response.data
        



    



