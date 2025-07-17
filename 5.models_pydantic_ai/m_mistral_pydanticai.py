from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from dotenv import load_dotenv
import logfire
import os

load_dotenv(override=True)

logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))

mistral_model=MistralModel(model_name=os.environ['MISTRAL_CHAT_MODEL'],api_key=os.getenv('MISTRAL_API_KEY'))

agent=Agent(model=mistral_model)

response=agent.run_sync('what is the cagr of returns achieved by warren buffet')
print(response.data)