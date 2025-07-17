from datetime import datetime
from typing import Any, List
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from httpx import Client
import os
from dotenv import load_dotenv
import logfire
# from dataclasses import dataclass
from pydantic_ai.models.groq import GroqModel

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

class SearchResult(BaseModel):
    title:str=Field(...,description='title of search result')
    href:str=Field(...,description='URL of the search')
    body:str=Field(...,description='body of the search result')

class SearchResults(BaseModel):
    final_result:List[SearchResult]=Field(...,description='search results')


model=GroqModel(model_name=os.getenv("GROQ_REASONING_MODEL"), api_key=os.getenv("GROQ_API_KEY"))

system_prompt="""You are a helpful assistant
You must use the tool `get_search` for any query 
"""
#    
agent = Agent(model=model,system_prompt=system_prompt,result_type=SearchResults)

@agent.tool_plain
def get_search(query:str)->dict[str,any]:
    """Run this tool for any search requests to model"""
    print(f"search query is {query}")
    results=DDGS(proxy=None).text(query,max_results=3)
    return results


resp=agent.run_sync("What is the most recent holiday in india")

print(resp.data)