
from langchain_community.vectorstores import FAISS
from pydantic_ai import Agent,RunContext
from dataclasses import dataclass
from llm import build_from_model_name
from prompts import get_prompt
@dataclass
class KnowledgeDeps:
    vector_db:FAISS
    query:str
    top_k:int


def get_agent(model_name:str,system_prompt:str):
    print(build_from_model_name(model_name))
    model_name=build_from_model_name(model_name)
    rag_agent=Agent(model=model_name,system_prompt=system_prompt,deps_type=KnowledgeDeps )


    @rag_agent.system_prompt
    def add_context_system_prompt(ctx:RunContext[KnowledgeDeps])->str:
        base_retriever=ctx.deps.vector_db.as_retriever(search_kwargs={'k':ctx.deps.top_k})
        results=base_retriever.invoke(ctx.deps.query)
        res='\n\n'.join([result.page_content for result in results])
        return '\n<context>\n'+res+'\n</context>'
    
    return rag_agent
    




