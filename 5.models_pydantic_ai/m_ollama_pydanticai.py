
# first install ollama for the windows application https://github.com/ollama/ollama?tab=readme-ov-file
# after installing go to windows powersheel promt use the following command ollama run llama3.2
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
import os
import logfire

load_dotenv(override=True)
logfire.configure(token=os.environ['LOGFIRE_TOKEN'])
logfire.instrument_openai()

llama_model=OpenAIModel(model_name=os.getenv('OLLAMA_MODEL'),base_url="http://localhost:11434/v1")
agent=Agent(model=llama_model)

response=agent.run_sync('What is alliteration')
print(response.data)