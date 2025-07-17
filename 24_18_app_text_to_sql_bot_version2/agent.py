
# from utils import get_schema_info
from dataclasses import dataclass
from pydantic_ai import Agent,RunContext
from llm import build_from_model_name
from langchain_community.vectorstores import FAISS

@dataclass
class Deps:
    vector_db:FAISS
    query:str
    top_k:int
    



def get_agent(model_name:str,system_prompt:str):

    model=build_from_model_name(model_name=model_name)
    sql_agent=Agent(model=model,system_prompt=system_prompt,deps_type=Deps)

    @sql_agent.system_prompt
    def add_context_system_prompt(ctx:RunContext[Deps])->str:
        base_retriever=ctx.deps.vector_db.as_retriever(search_kwargs={'k':ctx.deps.top_k})
        results=base_retriever.invoke(ctx.deps.query)
        context = "\n".join([result.page_content for result in results])
        return context



    return sql_agent





