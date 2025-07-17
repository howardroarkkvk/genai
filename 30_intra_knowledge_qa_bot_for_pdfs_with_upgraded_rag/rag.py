from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from agent import get_agent,Deps
from config_reader import settings
import os
import logfire
import time
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

class IntraKnowledgeQABot:
    def __init__(self):
        self.src_dir=settings.file_paths.src_dir
        self.kb_dir=os.path.join(self.src_dir,settings.file_paths.kb_dir)
        self.vector_db_dir=os.path.join(self.src_dir,settings.file_paths.vector_db_dir)
        self.bm_25_dir=settings.file_paths.bm_25_dir
        self.agent=get_agent()
        self.embedding_model=HuggingFaceEmbeddings(model_name=settings.embedder.model,encode_kwargs={'normalize_embeddings':settings.embedder.normalize},model_kwargs={'token':settings.embedder.token})
        self.text_splitter=RecursiveCharacterTextSplitter(chunk_size=settings.chunker.chunk_size,chunk_overlap=settings.chunker.chunk_overlap,length_function=len)
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_openai()
        if os.path.exists(self.vector_db_dir) and os.path.isfile(os.path.join(self.bm_25_dir,'bm25.pkl')) :
            self.vector_db=FAISS.load_local(folder_path=self.vector_db_dir,embeddings=self.embedding_model,allow_dangerous_deserialization=True)
            self.bm25_retriever=pickle.load(open(os.path.join(self.bm_25_dir,'bm25.pkl'),'rb'))
        else:
            self.ingest()


    def ingest(self):
        print("Beginning of Ingestion process")
        documents=[]
        print(self.kb_dir)
        for file in Path(self.kb_dir).glob('**/*.pdf'):
            print(file)
            loader=PyPDFLoader(file)
            documents.extend(loader.load_and_split(text_splitter=self.text_splitter))
        # print(documents[0:10])
        print("End of Chunking processes")
        vector_db=FAISS.from_documents(documents=documents,embedding=self.embedding_model,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
        vector_db.save_local(self.vector_db_dir)
        print("Saved vector db to it's dir")
        self.vector_db=vector_db

        if not os.path.exists(self.bm_25_dir):
            os.makedirs(self.bm_25_dir)
        bm25_retriever=BM25Retriever.from_documents(documents=documents)
        with open(os.path.join(self.bm_25_dir,'bm25.pkl'),'wb') as f:
            pickle.dump(bm25_retriever,f)
        self.bm25_retriever=bm25_retriever
        print("ingestion ends here")


    def retrievecontext(self,query:str,retriever_top_k:int=settings.retriever.top_k):
        faiss_retriever=self.vector_db.as_retriever(search_kwargs={'k':retriever_top_k})
        ensemble_retriever=EnsembleRetriever(retrievers=[self.bm25_retriever,faiss_retriever],weights=[0.5,0.5])
        results=ensemble_retriever.invoke(query)
        return [result.page_content for result in results]
    
    

    

    def execute(self,query:str,retriever_top_k:int=settings.retriever.top_k):
        deps=Deps(vector_db=self.vector_db,query=query,top_k=retriever_top_k)
        results=self.agent.run_sync(query,deps=deps)
        return  results.data
    





        



