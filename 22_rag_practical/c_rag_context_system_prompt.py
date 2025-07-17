import os
import logfire
from pydantic_ai import Agent,RunContext
from pydantic_ai.models.openai import OpenAIModel
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_community.vectorstores import FAISS

load_dotenv(override=True)
logfire.configure(token=os.getenv(''))
logfire.instrument_openai()


@dataclass
class KnowledgeDeps:
    vector_db:FAISS
    query:str
    top_k:int


system_prompt = """
You are a helpful and knowledgeable assistant for the luxury fashion store Harvest & Mill.

Follow these guidelines:
- ALWAYS search the knowledge base that is provided in the context between <context> </context> tags to answer user questions. 
- Generate answer based ONLY on the information retrieved from the knowledge base.
- If information is not found in the knowledge base, politely acknowledge this.
"""

model=OpenAIModel(model_name=os.getenv('OPENAI_CHAT_MODEL'),api_key=os.getenv('OPENAI_API_KEY'))
agent=Agent(model=model,system_prompt=system_prompt)

embedding_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
vector_db=FAISS.load_local(r'D:\DataFiles\faiss_kb1',embeddings=embedding_model,allow_dangerous_deserialization=True)


@agent.system_prompt
def add_context_system_prompt(ctx:RunContext[KnowledgeDeps])->str:
    base_retriever=ctx.deps.vector_db.as_retriever(search_kwargs={'k':ctx.deps.top_k})
    hits=base_retriever.invoke(ctx.deps.query)
    res='\n'.join([hit.page_content for hit in hits])
    print(res)
    return '\n<context>\n'+res+'\n</context>'


queries = [
    # "How long does international shipping take?",
    # "Show the Leather jackets in the store",
    # "Contact details of Harvest & Mill store",
    "What is the return policy?",
]




for query in queries:
    deps=KnowledgeDeps(vector_db=vector_db,query=query,top_k=2)
    resp=agent.run_sync(query,deps=deps)
    print('response of data is :',resp.data)



