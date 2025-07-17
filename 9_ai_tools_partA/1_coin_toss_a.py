from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire
import os
import random

load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()

model=OpenAIModel(model_name=os.environ['OPENAI_CHAT_MODEL'],api_key=os.getenv('OPENAI_API_KEY'))

system_prompt = """
You are the helpful assistant.
Pick the right tool for the job
"""
# 

# @agent.tool_plain
def coin_toss() -> str:
    """ Toss a coin and return the result"""
    return str(random.randint(0,1))

agent=Agent(model=model,system_prompt=system_prompt,tools=[coin_toss])



user_prompt="tell me the outcome of tossing a coin"
resp=agent.run_sync(user_prompt)
print(resp.data)