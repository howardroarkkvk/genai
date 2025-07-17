from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import logfire
import os
from pydantic import BaseModel

load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()

model=GroqModel(model_name=os.getenv('GROQ_CHAT_MODEL'),api_key=os.environ['GROQ_API_KEY'])
s_prompt="You are a helpful assistant"
agent=Agent(model=model,system_prompt=s_prompt)


resp=agent.run_sync("What is the moat of Awfis workspace")
print(resp.data)