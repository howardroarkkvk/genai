
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from dataclasses import dataclass
from pydantic_ai import Agent,RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from dotenv import load_dotenv
import logfire
import os
import time
from pydantic import BaseModel,Field
from typing import Annotated
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(override=True)
logfire.configure(token=os.environ['LOGFIRE_TOKEN'])
logfire.instrument_openai()
time.sleep(1)

@dataclass
class Deps:
    vector_db:FAISS
    top_k:int
    query:str


class Citation(BaseModel):
    path:Annotated[str,"The VERBATIM path name of a SPECIFIC source referred to generate the answer"]

class QuotedAnswer(BaseModel):
    answer:Annotated[str,...,"The answer to the user question, which is based only on the given sources."]
    citation:Annotated[list[Citation],...,"Citations from the given sources that justify the answer."]



system_prompt = """

You are a helpful and knowledgeable assistant for the luxury fashion store Harvest & Mill.

Follow these guidelines:
- ALWAYS search the knowledge base that is provided in the context between <context> </context> tags to answer user questions. 
- Generate answer based ONLY on the information retrieved from the knowledge base.
- If information is not found in the knowledge base, politely acknowledge this.
"""



model=GroqModel(model_name=os.getenv('GROQ_CHAT_MODEL'),provider=GroqProvider(api_key=os.getenv('GROQ_API_KEY')))
agent=Agent(model=model,system_prompt=system_prompt,deps_type=Deps,result_type=QuotedAnswer)
embeddings_model=HuggingFaceEmbeddings(model_name=os.getenv('HF_EMBEDDINGS_MODEL'),encode_kwargs={'normalize_embeddings':True},model_kwargs={'token':os.getenv('HF_TOKEN')})


def format_docs(docs):
    formatted_docs_list=[]
    for i,doc in enumerate(docs):
        formatted_docs_list.append(f'Source ID: {i} \n Source Path:{doc.metadata.get('source')} \n content: {doc.page_content}\n ')
    return '\n'.join(formatted_docs_list)




@agent.system_prompt
def add_system_prompt_context(ctx:RunContext[Deps]):
    base_retreiver=ctx.deps.vector_db.as_retriever(search_kwargs={'k':ctx.deps.top_k})
    documents=base_retreiver.invoke(ctx.deps.query)
    res=format_docs(documents)
    print('context is :',res)
    return '<context>\n' +res + '\n<content>'





vector_db_path=r'D:\faiss2'

vector_db=FAISS.load_local(folder_path=vector_db_path,embeddings=embeddings_model,allow_dangerous_deserialization=True)







# Query sentences:
queries = [
    "How long does international shipping take?",
    "Show the Leather jackets in the store",
    "Contact details of Harvest & Mill store",
    "What is the return policy?",
]

for query in queries:
    deps = Deps(vector_db=vector_db, query=query, top_k=2)
    result = agent.run_sync(query, deps=deps)
    print("\nQuery:", query)
    print("\nAnswer:", result.data)