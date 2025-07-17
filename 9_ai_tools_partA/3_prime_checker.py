from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire
import os
import random
from pydantic_ai import Tool


load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()

model=OpenAIModel(model_name=os.environ['OPENAI_CHAT_MODEL'],api_key=os.getenv('OPENAI_API_KEY'))

system_prompt="""
    You are a helpful assistant
    Use 'add' tool to add two integers.
    Use 'is_prime' tool to determine whether an integer is prime or not.
"""

agent=Agent(model=model,system_prompt=system_prompt)


# @Tool
# def add( a:int, b:int) -> int:
#     """Adds two numbers a,b """
#     return a+b

@agent.tool_plain
def add(a,b) -> int:
    """Adds two numbers a , b"""
    return a+b



# @agent.tool_plain
# def is_prime(a)->bool:
#     if a<=1:
#         return False
#     for i in range(2,int(a*0.5)+1):
#         if a%i==0:
#             return False
#     return True

result=agent.run_sync("are there any tools that are captured by the model to use?")
print(result.data)

result=agent.run_sync("what is the sum of 8,6 ")#and is the sum a prime number?
print(result.data)


# result=agent.run_sync("is 13 a prime number and what is the sum of 13,4?")
# print(result.data)

