# this is synchronous programming
import asyncio
from openai import AsyncOpenAI
import os
import logfire
from dotenv import load_dotenv
import time
import logfire



async_openai_client_instance=AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
logfire.configure(token=os.environ['LOGFIRE_TOKEN'])
logfire.instrument_openai()


async def running_openai_model(query):
    
    completions=await async_openai_client_instance.chat.completions.create(model=os.getenv('OPENAI_CHAT_MODEL'),messages=[{'role':'system','content':'You are a Helpful Assistant'},
                                                                                              {'role':'user','content':query}])
    output=completions.choices[0].message.content

    return output

async def driver():
    print(time.asctime())
    for _ in range(3):
        res=await running_openai_model('What is the capital of india')
        print(res)


st=time.perf_counter()
asyncio.run(driver())
et=time.perf_counter()
print(f' The elapsed time is {et-st} in seconds')





