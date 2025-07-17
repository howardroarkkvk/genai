# pip install whatismyip
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIModel
import logfire
import os
# import whatismyip

load_dotenv(override=True)
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
logfire.instrument_openai()

system_prompt = """
    You are helpful assistant. 
    Pick the right tool for the user request.
"""

model=OpenAIModel(model_name=os.environ['OPENAI_CHAT_MODEL'],api_key=os.getenv('OPENAI_API_KEY'))

agent=Agent(model=model,system_prompt=system_prompt)

response = agent.run_sync("what is the ip address of my machine")
print(response.data)