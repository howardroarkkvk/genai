from pydantic_ai import Agent
import os
from dotenv import load_dotenv
import logfire
from pydantic_ai.models.gemini import GeminiModel
from httpx import Client


load_dotenv(override=True)
logfire.configure(token=os.environ['LOGFIRE_TOKEN'])
logfire.instrument_openai()

# open_model=OpenAIModel(model_name=os.getenv('OPENAI_CHAT_MODEL'),api_key=os.environ['OPENAI_API_KEY'])
gemini_model=GeminiModel(model_name=os.getenv('GEMINI_CHAT_MODEL'),api_key=os.environ['GEMINI_API_KEY'])

system_prompt="""     Be concise, reply with one sentence.
    Use the `get_lat_lng` tool to get the latitude and longitude of a location.
"""
agent=Agent(model=gemini_model,system_prompt=system_prompt)

@agent.tool_plain
def get_lat_lng(location:str)->str:
    client=Client()
    r=client.get('https://geocode.maps.co/search',params={'q':location,'api_key':os.getenv('GEO_API_KEY')})
    return r.text

result = agent.run_sync("Find the longitude and lattitude of Hyderabad")
print(result.data)


