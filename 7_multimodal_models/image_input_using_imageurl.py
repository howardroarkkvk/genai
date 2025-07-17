from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai import ImageUrl
from dotenv import load_dotenv
import logfire
import os
from textwrap import dedent

load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()
# logfire.instrument_httpx()

model=OpenAIModel(model_name=os.environ['OPENAI_CHAT_MODEL'],api_key=os.getenv('OPENAI_API_KEY'))
# agent=OpenAIModel(model_name=os.environ['GEMINI_CHAT_MODEL'],api_key=os.getenv('GEMINI_API_KEY'))
system_prompt='You are smart vision agent with smart text extraction skills'

agent=Agent(model=model,system_prompt=dedent(system_prompt))

resp=agent.run_sync(["what's in the image",
                ImageUrl(url='https://iili.io/3Hs4FMg.png')

])

print(resp.data)
