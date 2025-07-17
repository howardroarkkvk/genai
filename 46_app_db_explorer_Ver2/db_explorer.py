
from sql_executor_agent import *
from sql_generator_agent import * 
from config_reader import * 
from dependencies import * 
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logfire
import time
from pydantic_ai.messages import ModelMessage
class DBExplorer:
    def __init__(self):
        self.db_file_path=os.path.join(settings.file_paths.src_dir,settings.file_paths.db_file)
        self.vector_file_path=os.path.join(settings.file_paths.src_dir,settings.file_paths.vector_db_dir)
        self.embedding_model=HuggingFaceEmbeddings(model_name=settings.embedder.model,encode_kwargs={'normalize_embeddings':settings.embedder.normalize},model_kwargs={'token':settings.embedder.token})
        self.vector_db=FAISS.load_local(folder_path=self.vector_file_path,embeddings=self.embedding_model,allow_dangerous_deserialization=True)
        self.create_agent=create_sql_generator_agent()
        self.execute_agent=create_sql_executor_agent()
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_openai()


    def process(self,user_query:str,retrieve_top_k:int):#,history:list[ModelMessage]
        logfire.info('Generating SQL')

        deps=Deps(vector_db=self.vector_db,retriever_top_k=retrieve_top_k,db_file_path=self.db_file_path,user_query=user_query)

        result=self.create_agent.run_sync("generate the sql query using the database schema "+ user_query,deps=deps)#,message_history=history
        logfire.info(f'thoughts: {result.output.thoughts}')
        logfire.info(f'sql query:{result.output.sql_query}')

        logfire.info(f'Started exeucting the SQL query')
        result=self.execute_agent.run_sync(result.output.sql_query,deps=deps)#message_history=history,
        return result.output


if __name__=='__main__':
    db_explorer=DBExplorer()
    query='How many albums are there?'
    retreive_top_k=3
    result=db_explorer.process(user_query=query,retrieve_top_k=retreive_top_k)
    print(result)


# How many albums are there?
# How many distinct genres are there?
# What is the total number of invoice lines?
