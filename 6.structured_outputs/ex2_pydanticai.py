from typing import List
from datetime import date
from pydantic import BaseModel,Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
import os
import logfire

load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()

class countryInfo(BaseModel):
    continent_name:str
    country_name:str
    capital_name:str
    has_river:bool
    has_sea:bool
    state_names:List[str]
    independence_day:str
    weather:str=Field(description='weather over the year')

model=OpenAIModel(model_name=os.getenv('OPENAI_CHAT_MODEL'),api_key=os.environ['OPENAI_API_KEY'])

agent=Agent(model=model,result_type=countryInfo)

response=agent.run_sync('tell me about India')
print(response.data)