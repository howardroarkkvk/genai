from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
import logfire
import os
from dotenv import load_dotenv

load_dotenv(override=True)
logfire.configure(token=os.environ['LOGFIRE_TOKEN'])
#logfire.instrument_ # only for open ai there is instrument_openai for gemini there is none

model=GeminiModel(model_name=os.getenv('GEMINI_CHAT_MODEL'),api_key=os.environ['GEMINI_API_KEY'])

agent=Agent(model=model)

# response=agent.run_sync('what is value investing')
# print(response.data)

with logfire.span("Running gemini model span") as span:
    response=agent.run_sync('how will ai disrupt softwware deveoper job')
    print(response.data)
    response=agent.run_sync('are software testers useful once gen ai is implemented in the project')
    print(response.data)