from planner_agent import *
from report_agent import *
from search_agent import *
import logfire
from config_reader import *
import time
import asyncio

class DeepResearcher:
    def __init__(self):
        logfire.configure(token=settings.logfire.token)
        time.sleep(1)
        logfire.instrument_openai()

    async def plan_searches(self,query:str)->WebSearchPlan:
        logfire.info(f'Planning Searches')
        response=await planner_agent.run(query)
        logfire.info(f'will perform {len(response.data.searches)} searches ...')
        return response.data


    async def search(self,item:WebSearchItem)->str|None:
        query=f"search term :{item.query} \n reason for searching :{item.reason}"
        logfire.info(query)
        try: 
            result=await search_agent.run(query)
            return result.data
        
        except Exception:
            return None
        

    async def perform_searches(self,search_plan:WebSearchPlan)->list[str]:
        logfire.info("Searching web")
        tasks=[asyncio.create_task(self.search(item)) for item in search_plan.searches]
        num_completed=0
        results=[]
        for task in asyncio.as_completed(tasks):
            result=await task
            if result is not None:
                results.append(result)
            num_completed+=1
            logfire.info(f"searching ,,,{num_completed/len(tasks)} completed")
        return results
    
    async def generate_report(self,query,search_results:list[str])->ReportData:
        logfire.info(f'Generating Report...')
        query=f'Original query:{query}\n Summarized Search Results: {search_results}'
        logfire.info(query)
        result=await report_agent.run(query)
        return result.data


    async def do_research(self,query:str)->tuple:
        logfire.info(f'Starting deep research for the query:{query}')
        search_plan=await self.plan_searches(query)
        search_results=await self.perform_searches(search_plan=search_plan)
        logfire.info(f'search_results {search_results}')
        report=await self.generate_report(query=query,search_results=search_results)
        return (report.short_summary,report.markdown_report)
    
async def main(query):
    deep_researcher=DeepResearcher()
    result=await deep_researcher.do_research(query)
    print(result)


if __name__=='__main__':
    query='Caribbean vacation spots in April, optimizing for surfing, hiking and water sports'
    asyncio.run(main(query))


    





