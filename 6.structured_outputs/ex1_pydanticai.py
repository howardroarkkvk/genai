from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
import logfire
import os

# this helps in getting the latest ennvironment variables from .env file
load_dotenv(override=True)

# this is to configure the logfire and also setup instrument so that spans for every request is captured
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()


class AddressInfo(BaseModel):
    first_name:str
    last_name:str
    street:str
    house_number:str
    postal_code:str
    city:str
    state:str
    country:str

openai_model=OpenAIModel(model_name=os.environ['OPENAI_CHAT_MODEL'],api_key=os.getenv('OPENAI_API_KEY'))

agent=Agent(model=openai_model,result_type=AddressInfo)

response=agent.run_sync('herlock Holmes lives in the United Kingdom. His residence is in at 221B Baker Street, London, NW1 6XE.')
print(response.data)


