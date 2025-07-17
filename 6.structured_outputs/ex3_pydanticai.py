from pydantic import BaseModel,Field
from typing import List
from datetime import date
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
import logfire
import os

load_dotenv(override=True)
# logfire.configure(token=os.getenv('OPENAI_CHAT_MODEL'))
# logfire.instrument_openai()

class CalendarEvent(BaseModel):
    name:str
    day:List
    participants:List


open_model=OpenAIModel(model_name=os.environ['OPENAI_CHAT_MODEL'],api_key=os.getenv('OPENAI_API_KEY'))

agent=Agent(model=open_model,result_type=CalendarEvent)

resp=agent.run_sync('Alice,vamshi,catchy-pacific and Bob are going to a science fair on Friday,saturday.')
print(resp.data)