
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.document_loaders import JSONLoader
from pathlib import Path


class SemanticSearcher:
    def __init__(self,src_dir:str,kb_dir:str,vector_db_dir:str,embeddings_model_name:str,retriever_top_k:int):
        self.src_dir=src_dir
        print(self.src_dir)
        self.kb_dir=os.path.join(src_dir,kb_dir)
        print(self.kb_dir)
        
        self.vector_db_dir=os.path.join(src_dir,vector_db_dir)
        print(self.vector_db_dir)
        self.embeddings_model=HuggingFaceEmbeddings(model_name=embeddings_model_name,encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
        self.retriever_top_k=retriever_top_k
        self.ingest()

    def metadata_func(self,record:dict,metadata:dict)->dict:
        metadata['source']=record.get('chunk_link')
        metadata['title']=record.get('chunk_heading')
        return metadata

    def ingest(self):
        print("Ingestion proces begins")
        documents=[]
        for file in Path(self.kb_dir).glob('*.json'):

            loader=JSONLoader(file_path=file,jq_schema='.[]',content_key='.text',is_content_key_jq_parsable=True,metadata_func=self.metadata_func)
            documents.extend(loader.load())

        vector_db=FAISS.from_documents(documents=documents,embedding=self.embeddings_model,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
        vector_db.save_local(self.vector_db_dir)
        self.vector_db=vector_db
        print("Ingestion process ends here")

    def retriever(self,query:str):
        base_retriever=self.vector_db.as_retriever(search_kwargs={'k':self.retriever_top_k})
        results=base_retriever.invoke(query)
        outputs=[]
        for result in results:
            outputs.append(result.metadata['source'])
        return outputs
    
    def retrievercontent(self,query:str):
        base_retriever=self.vector_db.as_retriever(search_kwargs={'k':self.retriever_top_k})
        outputs=base_retriever.invoke(query)
        results=[]
        for output in outputs:
            results.append(output.page_content)
        return results
    


        
    

   

