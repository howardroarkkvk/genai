import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import logfire
import os
import time 
import asyncio

load_dotenv(override=True)
logfire.configure(token=os.environ['LOGFIRE_TOKEN'])
logfire.instrument_openai()

model=GroqModel(model_name=os.getenv('GROQ_CHAT_MODEL'),provider=GroqProvider(api_key=os.getenv('GROQ_API_KEY')))

agent=Agent(model=model,system_prompt='You are a helpful AI assistant')

async def approach1():
    tasks=[agent.run('what is the capital of india') for _ in range(5)]
    responses=await asyncio.gather(*tasks)
    print(responses)

st=time.perf_counter()
asyncio.run(approach1())
et=time.perf_counter()
print(f'Total elapsed time : {et-st:.6f}')