import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
import logfire
import os
import time

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
# logfire.instrument_pydantic_ai()

agent = Agent(
    model=GroqModel(
        model_name=os.getenv("GROQ_CHAT_MODEL"),
        provider=GroqProvider(api_key=os.getenv("GROQ_API_KEY")),
    ),
    system_prompt="You are a helpful AI assistant",
)

async def main(query):
     async with agent.run_stream(query) as response:
          async for text in response.stream_text(delta=False):
                print(text)
        

query='What are your capabilities?'
st=time.perf_counter()
asyncio.run(main(query))
et=time.perf_counter()
print(f'total time elapsed : {et-st}')