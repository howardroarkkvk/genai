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

system_prompt="""
    You are a helpful assistant
    Use 'add' tool to add two integers.
    Use 'mul' tool to multiply two integers.
"""

agent=Agent(model=model,system_prompt=system_prompt)

@agent.tool_plain
def add(a,b) -> int:
    """Adds two numbers
    """
    return a+b

@agent.tool_plain
def mul(a,b)-> int:
    """multiplies two numbers"""
    return a*b

resp=agent.run_sync("10 cross 20")
print(resp.data)

resp=agent.run_sync("10 plus 20")
print(resp.data)