from dataclasses import dataclass
import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent,RunContext
from pydantic_ai.models.openai import OpenAIModel
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import FAISS


load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()

@dataclass
class FeatureDeps:
    vector_db:FAISS
    query:str
    top_k:int

system_prompt= """
You are a helpful and knowledgeable assistant for the luxury fashion store H&M.

Follow these guidelines:
- ALWAYS search the knowledge base using the search_knowledge_base tool to answer user questions.
- Provide accurate product and policy information based ONLY on the information retrieved from the knowledge base. 
- Never make assumptions or provide information not present in the knowledge base.
- If information is not found in the knowledge base, politely acknowledge this.
"""
model=OpenAIModel(model_name=os.getenv('OPENAI_CHAT_MODEL'),api_key=os.getenv('OPENAI_API_KEY'))
agent=Agent(model=model,system_prompt=system_prompt,deps_type=FeatureDeps)

def build_context_from_results(hits):
    return '\n'.join([hit.page_content for hit in hits])

@agent.tool
def search_knowledge_base(ctx:RunContext[FeatureDeps],query:str)->str:
    retriever=ctx.deps.vector_db.as_retriever(search_kwargs={'k':ctx.deps.top_k})
    results=retriever.invoke(ctx.deps.query)
    res=build_context_from_results(results)
    print("Tool call",query)
    return res







vector_db_file_path=r'D:\DataFiles\faiss_rag_practical'
embedding_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})
vector_db=FAISS.load_local(folder_path=vector_db_file_path,embeddings=embedding_model,allow_dangerous_deserialization=True)

queries = [
    "How long does international shipping take?",
    "Show the Leather jackets in the store",
    "Contact details of store",
    "What is the return policy?",
]  


for query in queries:
    deps=FeatureDeps(vector_db=vector_db,query=query,top_k=2)
    response=agent.run_sync(query,deps=deps)
    print('query:',query)
    print("Agent response is: ",response.data)