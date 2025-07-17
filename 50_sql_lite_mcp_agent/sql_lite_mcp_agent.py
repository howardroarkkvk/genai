
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import os 
import asyncio
from dotenv import load_dotenv
import logfire
import time



load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
time.sleep(1)
logfire.instrument_pydantic_ai()
logfire.instrument_openai()

class FilteringMCPServer(MCPServerStdio):


    async def list_tools(self):
        """
        list_tools() will list all the existing tools from the MCPserver we are about to invoke. 
        """
        tools=await super().list_tools()
        filter_tools= ["list_tables", "describe_table"]
        return [t for t in tools if t.name in filter_tools]
    

sqllite_mcp_server=FilteringMCPServer(command='docker',
                   args=["run",
                         "-i",
                         "--rm",
                         "-v",
                         "D:/DataFiles/datatext-to-sql-qa-bot/db:/mcp",
                         "mcp/sqlite",
                         "--db-path",
                         "mcp/chinook.sqlite"
                        ])


provider=GroqProvider(api_key=os.getenv('GROQ_API_KEY'))
model=GroqModel(model_name=os.getenv('GROQ_REASONING_MODEL'),provider=provider)
sqllite_agent=Agent(model=model,system_prompt='Use the tools to answer questions based on the database.',mcp_servers=[sqllite_mcp_server])


async def main():
    queries = [
        "list the table names",
        "get the schema of table album",
        "list the columns of table artist",
    ]
    history = []

    async with sqllite_agent.run_mcp_servers():
        for query in queries:
            output=await sqllite_agent.run(query,message_history=history)
            history=output.new_messages()
            print(output.data)


if __name__=='__main__':
    asyncio.run(main())

    

