from langchain_huggingface import HuggingFaceEmbeddings
import os
from config_reader import *
import logfire 
import time 
from utils import *
import json
from langchain_community.document_loaders import JSONLoader
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy


class Ingestor:
    def __init__(self):
        self.src_dir=settings.file_paths.src_dir
        self.kb_dir=os.path.join(self.src_dir,settings.file_paths.kb_dir)
        self.vector_db_dir=os.path.join(self.src_dir,settings.file_paths.vector_db_dir)
        self.retriever_top_k=settings.retriever.top_k
        self.embedding_model=HuggingFaceEmbeddings(model_name=settings.embedder.model,encode_kwargs={'normalize_embeddings':settings.embedder.normalize},model_kwargs={'token':settings.embedder.token})
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_openai()

    # chunking for jsons is using the chunk_by_table/chunk_by_field .....it forms the jsons in the kb_dir...using which we form the vector db...
    def chunk_schema(self):
        db_file=settings.file_paths.db_file
        db_file_path=os.path.join(self.src_dir,db_file)

        print('Begin Chunking')
        chunk_strategy=settings.chunker.chunk_strategy
        if chunk_strategy=='table_level':
            chunks=get_chunks_by_table(db_file_path=db_file_path)
        else:
            chunks=get_chunks_by_field(db_file_path=db_file_path)

        # once chunks are created from the database file, create the directory to store it...

        if not os.path.exists(self.kb_dir):
            os.makedirs(self.kb_dir)
            #"db/chinook.sqlite"
        json_file_name=db_file.split('.')[0].split('/')[1]+'_schema.json'
        json_file_path=os.path.join(self.kb_dir,json_file_name)

        # storing the chunked data in kb directory...

        with open(json_file_path,'w') as f:
            json.dump(chunks,f,indent=2)

        print('end of Chunking process')

    def metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["table"] = record.get("table")
        metadata["column"] = record.get("column")
        metadata["type"] = record.get("type")
        return metadata

    def ingest(self):
        print("Ingestion Process Begins >>>>>")
        documents=[]
        for file in Path(self.kb_dir).glob('*.json'):
            loader=JSONLoader(file_path=file,jq_schema='.[]',content_key='.text',is_content_key_jq_parsable=True,metadata_func=self.metadata_func)
            documents.extend(loader.load())
        
        vector_db=FAISS.from_documents(documents=documents,embedding=self.embedding_model,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)

        vector_db.save_local(folder_path=self.vector_db_dir)
        self.vector_db=vector_db
        print('End of Ingestion Process')

    def execute_pipeline(self):
        self.chunk_schema()
        self.ingest()


if __name__=='__main__':
    ingestor=Ingestor()
    ingestor.execute_pipeline()

        



        

