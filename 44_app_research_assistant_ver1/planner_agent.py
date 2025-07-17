import asyncio
from pydantic_ai import Agent
from pydantic import BaseModel
from typing import Annotated
from llm import *
class WebSearchItem(BaseModel):
    reason:str
    "Your reasoning for why this search is important to the query"

    query:str
    "The search term to be used for web search"


class WebSearchPlan(BaseModel):
    searches:Annotated[list[WebSearchItem],...,'A list of web searches to perform to best answer the query']

    "A list of web searches to perform to best answer the query"

system_prompt="""
You are a helpful research assistant. Given a query, come up with a set of web searches
to perform to best answer the query. Output between 2 and 3 terms to query for.
"""
planner_agent=Agent(name='PlannerAgent',model=build_model(),system_prompt=system_prompt,result_type=WebSearchPlan)


async def main(query:str):
    result=await planner_agent.run(query)
    print(result.data)
    print('query ---------------reason')
    for i in result.data.searches:
        print(f'{i.query} - {i.reason}' )


if __name__=='__main__':
    query='Caribbean vacation spots in April, optimizing for surfing, hiking and water sports'
    print(query)
    asyncio.run(main(query))
