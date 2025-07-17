from config_reader import * 
import logfire
import time
from agent import get_agent,Deps
from langchain_huggingface import HuggingFaceEmbeddings
import os 
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from pathlib import Path
from langchain_community.document_loaders import JSONLoader
from textwrap import dedent

class TextClassifierRAG:
    def __init__(self):
        self.src_dir=settings.file_paths.src_dir
        self.kb_dir=settings.file_paths.kb_dir
        self.vector_db_dir=settings.file_paths.vector_db_dir
        self.agent=get_agent()
        self.embedding_model=HuggingFaceEmbeddings(model_name=settings.embedder.model,encode_kwargs={"normalize_embeddings": settings.embedder.normalize},model_kwargs={"token": settings.embedder.token})
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_openai()

        if os.path.exists(os.path.join(self.src_dir,self.vector_db_dir)):
            self.vector_db=FAISS.load_local(folder_path=os.path.join(self.src_dir,self.vector_db_dir),embeddings=self.embedding_model,allow_dangerous_deserialization=True)
        else:
            self.ingest()


    def metadata_func(self, record: dict, metadata: dict) -> dict:
        metadata["question"] = record.get("question")
        metadata["answer"] = record.get("answer")
        return metadata

    def ingest(self):
        print('>>>Beginning of the Ingestion Process:')

        documents=[]
        for file in Path(os.path.join(self.src_dir,self.kb_dir)).glob('**/*.json'):
            loader=JSONLoader(file_path=file,jq_schema='.[]',content_key='.question',is_content_key_jq_parsable=True,metadata_func=self.metadata_func)
            documents.extend(loader.load())

        vector_db=FAISS.from_documents(documents=documents,embedding=self.embedding_model,distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE)
        vector_db.save_local(os.path.join(self.src_dir,self.vector_db_dir))
        self.vector_db=vector_db
        print('>>> End of Ingestion Process')

    def retrievecontext(self,query:str,retrieve_top_k:int=settings.retriever.top_k):
        base_retriever=self.vector_db.as_retriever(search_kwargs={'k':retrieve_top_k})
        examples=base_retriever.invoke(query)

        rag_string = "Use the following examples to help you classify the query:"
        rag_string+='\n<examples>\n'
        for example in examples:
            rag_string+=dedent(f"""
                                <example>
                                    <query>
                                        {example.metadata['question']}

                                    </query>
                                    <response>
                                        {example.metadata['question']}
                                    </response>
                               </example>

                               
                               """)
        rag_string+='\n</examples>\n'
        return rag_string
    
    def predict(self,query,retreive_top_k=settings.retriever.top_k):
        deps=Deps(vector_db=self.vector_db,query=query,top_k=retreive_top_k)
        result=self.agent.run_sync(query,deps=deps)
        return result.data

if __name__=='__main__':
    text_classifier=TextClassifierRAG()
    query="I'm confused about a charge on my recent auto insurance bill that's higher than my usual premium payment."
    print(text_classifier.predict(query))












