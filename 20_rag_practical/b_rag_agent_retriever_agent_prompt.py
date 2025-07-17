from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import logfire
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent,RunContext
from dataclasses import dataclass

load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()

@dataclass
class KnowledgeDeps:
    vector_db:FAISS
    top_k:int
    query:str

model=OpenAIModel(model_name=os.getenv('OPENAI_CHAT_MODEL'),api_key=os.getenv('OPENAI_API_KEY'))
system_prompt="""You are a helpful and knowledgeable assistant for the luxury fashion store H&M.

Follow these guidelines:
- ALWAYS search the knowledge base that is provided in the context between <context> </context> tags to answer user questions.
- Provide accurate product and policy information based ONLY on the information retrieved from the knowledge base. 
- Never make assumptions or provide information not present in the knowledge base.
- If information is not found in the knowledge base, politely acknowledge this.
"""
agent=Agent(model=model,system_prompt=system_prompt,deps_type=KnowledgeDeps)


def build_context_from_results(hits):
    return '\n'.join([hit.page_content for hit in hits])

@agent.system_prompt
def add_context_system_prompt(ctx:RunContext[KnowledgeDeps])->str:
    retriever=ctx.deps.vector_db.as_retriever(search_kwargs={'k':ctx.deps.top_k})
    result=retriever.invoke(ctx.deps.query)
    retrieved_text=build_context_from_results(result)
    return '<context>\n'+retrieved_text+'<\ncontext>'
















vector_db_path=r'D:\DataFiles\faiss_rag_practical'
embedding_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
vector_db=FAISS.load_local(folder_path=vector_db_path,embeddings=embedding_model,allow_dangerous_deserialization=True)
queries= [
    "How long does international shipping take?",
    "Show the Leather jackets in the store",
    "Contact details of store",
    "What is the return policy?",
]

for query in queries:
    deps=KnowledgeDeps(vector_db,top_k=2,query=query)
    result=agent.run_sync(query,deps=deps)
    print(query)
    print(result.data)
















