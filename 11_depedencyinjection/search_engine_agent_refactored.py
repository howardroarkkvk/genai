from typing import List
from pydantic import BaseModel,Field
from duckduckgo_search import DDGS
from pydantic_ai import Agent,RunContext
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIModel
from dataclasses import dataclass
import os
import logfire
from dotenv import load_dotenv

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

# model=GroqModel(model_name=os.getenv("GROQ_REASONING_MODEL"), api_key=os.getenv("GROQ_API_KEY"))
# model=GroqModel(model_name=os.getenv("GROQ_REASONING_MODEL"), api_key=os.getenv("GROQ_API_KEY"))
model=OpenAIModel(model_name=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))




@dataclass
class SearchDeps:
    max_results:int

class SearchResult(BaseModel):
    title:str=Field(description='title of the search result')
    href:str=Field(description='URL of the search result')
    body:str=Field(description='body of the search result')

class SearchResults(BaseModel):
    results:List[SearchResult]=Field(description='List of all search results')

system_prompt ="""You are a helpful assistant.You must use the tool 'get_search' for any query"""

agent=Agent(model=model,system_prompt=system_prompt,result_type=SearchResults)#,deps_type=SearchDeps


# here output is defined as dict[str,any] as the output of DDGS().text() results in list of dicts...?
@agent.tool
def get_search(ctx:RunContext[SearchDeps],query:str)-> dict[str,any]:
    """get the search for a keyword query"""
    results=DDGS().text(query,max_results=ctx.deps.max_results)
    # print('results',results)
    # print(type(results))
    return results

deps=SearchDeps(max_results=3)


response = agent.run_sync("Latest news on cricket", deps=deps)
print(response.data)