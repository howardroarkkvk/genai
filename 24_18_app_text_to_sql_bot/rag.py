import os
from agent import get_agent,Deps


class TexttoSqlRAG:
    def __init__(self,src_dir:str,db_path:str,llm_name:str,system_prompt:str):
        self.db_path=os.path.join(src_dir,db_path)
        self.rag_agent=get_agent(llm_name,system_prompt)


    def execute(self,query:str)->str:
        deps=Deps(db_path=self.db_path)
        result=self.rag_agent.run_sync(query,deps=deps)
        return result.data

