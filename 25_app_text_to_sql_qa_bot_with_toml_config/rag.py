
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from utils import get_chunks_by_field,get_chunks_by_table,execute_query
from langchain_huggingface import HuggingFaceEmbeddings
import os
from config_reader import settings
from agent import get_agent,Deps
import logfire
import time
import json
from pathlib import Path

class TextToSqlRAG:
    def __init__(self):
        self.src_dir=settings.file_paths.src_dir
        self.kb_dir=os.path.join(self.src_dir,settings.file_paths.kb_dir)
        self.vector_db_dir=os.path.join(self.src_dir,settings.file_paths.vector_db_dir)
        self.retriever_top_k=settings.retriever.top_k
        self.agent=get_agent()
        self.embedding_model=HuggingFaceEmbeddings(model_name=settings.embedder.model,encode_kwargs={'normalize_embeddings':settings.embedder.normalize},model_kwargs={'token':settings.embedder.token})
        logfire.configure(token=settings.logfire.token)  
        time.sleep(1)
        # logfire.instrument_pydantic_ai()
        logfire.instrument_openai()


    def chunk_schema(self):
        db_file=settings.file_paths.db_file
        db_file_path=os.path.join(self.src_dir,db_file)
        print(">> Begin Chunking ......\n")
        chunk_strategy=settings.chunker.chunk_strategy
        if chunk_strategy =='table_level':
            chunks=get_chunks_by_table(db_file_path=db_file_path)
        else:
            chunks=get_chunks_by_field(db_file_path=db_file_path)

        if not os.path.exists(self.kb_dir):
            os.makedirs(self.kb_dir)

        json_file_name=db_file.split('.')[0].split('/')[1]+"_schema.json"
        json_file_path=os.path.join(self.kb_dir,json_file_name)

        with open(json_file_path,'w') as f:
            json.dump(chunks,f,indent=2)
        print(">> End of Chunking .....")

    def metadata_func(self,record:dict,metadata:dict)->dict:
        metadata['table']=record.get('table')
        metadata['column']=record.get('column')
        metadata['type']=record.get('type')
        return metadata

    def ingest(self):
        print(">> Ingestion Process Begins : \n")
        documents=[]
        for json in Path(self.kb_dir).glob('*.json'):
            loader=JSONLoader(file_path=json,jq_schema='.[]',content_key='.text',is_content_key_jq_parsable=True,metadata_func=self.metadata_func)
            documents.extend(loader.load())

        vector_db=FAISS.from_documents(documents=documents,embedding=self.embedding_model,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
        vector_db.save_local(self.vector_db_dir)
        self.vector_db=vector_db
        print("End of Ingestion Process >>...")


    def retrievecontext(self,query):
        base_retriever=self.vector_db.as_retriever(search_kwargs={'k':self.retriever_top_k})
        results=base_retriever.invoke(query)
        return [result.page_content for result in results]

    def execute(self,query):
        deps=Deps(vector_db=self.vector_db,query=query,top_k=self.retriever_top_k)
        result=self.agent.run_sync(query,deps=deps)
        return result.data



       
    

















        


