# same agent, different model, giving model name in run_sync and using message_history in run_sync
from dotenv import load_dotenv
import logfire
import os
from rich.console import Console
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.groq import GroqModel

load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()
model=GroqModel(model_name=os.getenv("GROQ_CHAT_MODEL"), api_key=os.getenv("GROQ_API_KEY"))
model1=GroqModel(model_name=os.getenv("GROQ_REASONING_MODEL"))
agent = Agent(model=model,system_prompt="You are helpful assistant.")


response=agent.run_sync("Tell me a joke")
print(response.data)

response1=agent.run_sync("Explain?",model=model1,message_history=response.all_messages())
print(response1.data)

