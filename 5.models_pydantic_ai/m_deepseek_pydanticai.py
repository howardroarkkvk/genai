from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import logfire
import os
from dotenv import load_dotenv

load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()


deepseek_model=OpenAIModel(model_name=os.getenv('DEEPSEEK_CHAT_MODEL'),base_url="https://api.deepseek.com",api_key=os.environ['DEEPSEEK_API_KEY'])

agent=Agent(model=deepseek_model)

response=agent.run_sync('What is the meaning of phrase if you want to persuade appeal to interest not to reason')
print(response.data)