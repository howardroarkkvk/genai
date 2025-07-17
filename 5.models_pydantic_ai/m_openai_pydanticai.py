# from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel


load_dotenv(override=True)

logfire.configure(token=os.environ['LOGFIRE_TOKEN'])
logfire.instrument_openai()


openaimodel=OpenAIModel(model_name=os.environ['OPENAI_CHAT_MODEL'],api_key=os.getenv('OPENAI_API_KEY'))

agent=Agent(model=openaimodel)#,system_prompt='You are a helpful assistant'


response=agent.run_sync('what is cloud computing')
print(response.data)

# print(response.new_messages)

# print(response.all_messages)