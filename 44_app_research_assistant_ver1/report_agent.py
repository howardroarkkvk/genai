from pydantic_ai import Agent
from llm import *
from pydantic import BaseModel
import asyncio

class ReportData(BaseModel):
    short_summary:str
    """A 2 to 3 sentences summary of the findings"""

    markdown_report:str
    """The final Report"""






system_prompt="""You are a senior researcher tasked with writing a cohesive report for a research query.
You will be provided with the original query, and some initial research done by a research assistant.
You should first come up with an outline for the report that describes the structure and
flow of the report. Then, generate the report and return that as your final output.
The final output should be in markdown format, and it should be lengthy and detailed. 
Aim for 2-5 pages of content, at least 1000 words."""

report_agent=Agent(name='ReportAgent',model=build_model(),system_prompt=system_prompt,result_type=ReportData,retries=3)

async def main(query:str):
    print(query)
    response=await report_agent.run(query)
    print(response.data)


if __name__=='__main__':
    query="About surfing"
    asyncio.run(main(query))



