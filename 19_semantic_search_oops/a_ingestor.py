
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
import os
from pathlib import Path
from dotenv import load_dotenv



class DataIngestor:
    def __init__(self,main_dir:str,vector_db_dir:str,kb_dir:str):
        self.embedding_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
        self.main_dir=main_dir
        self.vector_db_dir=vector_db_dir
        self.kb_dir=kb_dir

    def metadata_func(self,record:dict,metadata:dict)->dict:
        metadata["source"]=record.get('chunk_link')
        metadata["title"]=record.get('chunk_heading')
        return metadata
    
    
    def ingestor(self):
        if os.path.exists(self.vector_db_dir):
            print("Vector DB exists")
            return
        print("Ingestion of data begins here....")
        documents=[]
        for file in Path(self.kb_dir).glob('*.json'):
            loader=JSONLoader(file_path=file,jq_schema='.[]',content_key='[.chunk_heading,.summary,.text] | @tsv',is_content_key_jq_parsable=True,metadata_func=self.metadata_func)
            documents.extend(loader.load())

        vector_db=FAISS.from_documents(documents=documents,embedding=self.embedding_model,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
        vector_db.save_local(os.path.join(self.main_dir,self.vector_db_dir))
        print("Ingestion process ends here")



            
if __name__=='__main__':
    load_dotenv(override=True)
    main_dir=r'D:\DataFiles\oops'
    vector_db_dir=os.path.join(main_dir,'faiss_oops')
    kb_dir=os.path.join(main_dir,'kb')
    di=DataIngestor(main_dir,vector_db_dir,kb_dir)
    di.ingestor()






            





















