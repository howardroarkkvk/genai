from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
import logfire


load_dotenv(override=True)
logfire.configure(token=os.environ['LOGFIRE_TOKEN'])
logfire.instrument_openai()
async_openai_client=AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

async def agent(query):
    # print(query)
    completions=await async_openai_client.chat.completions.create(model=os.getenv('OPENAI_CHAT_MODEL'),messages=[{'role':'system','content':"You are a helpful assistant"},                                                                                                            
                                                                                          {'role':'user','content':query}])
    # print(completions.choices[0].message.content)
    return completions.choices[0].message.content

async def calling_agent():
    query=['what is the capital of india','what is the capital of USA','what is the capital of Telangane']
    coroutines=[agent(query[i]) for i in range(3)]
    print(coroutines)
    result=await asyncio.gather(*coroutines)
    print(result)


# query='what is the capital of india'
# asyncio.run(agent(query))
# results=asyncio.run(agent("what is the capital of india"))
# print(results)
asyncio.run(calling_agent())

