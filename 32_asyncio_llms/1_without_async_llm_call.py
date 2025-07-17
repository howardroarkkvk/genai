import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import logfire
import os
import time 

load_dotenv(override=True)
logfire.configure(token=os.environ['LOGFIRE_TOKEN'])
logfire.instrument_openai()

model=GroqModel(model_name=os.getenv('GROQ_CHAT_MODEL'),provider=GroqProvider(api_key=os.getenv('GROQ_API_KEY')))

agent=Agent(model=model,system_prompt='You are a helpful AI assistant')

def approach1():
    for i in range(5):
        resp=agent.run_sync('what is the capital of india')
        print(resp.data)

    
st=time.perf_counter()
approach1()
et=time.perf_counter()
print(f'Total elapsed time : {et-st:.6f}')