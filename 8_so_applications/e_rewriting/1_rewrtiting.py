from pydantic import BaseModel,Field
from typing import List
from textwrap import dedent
import logfire
import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.groq import GroqModel
from datetime import date
from enum import Enum
import json

load_dotenv(override=True)
logfire.configure(token = os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()

class TextStyle(Enum):
    formal="formal"
    informal='informal'
    academic='academic'
    professional='professional'
    business='business'

class NormalizedText(BaseModel):
    style:TextStyle=Field(description='the style of the text')
    text:str

class NormalizedTexts(BaseModel):
    normalized_texts:List[NormalizedText]


def print_summary(text):
    for i in text:
        print(f'{i.style} : {i.text}')
        # print(i.text)



model=GroqModel(model_name=os.getenv('GROQ_CHAT_MODEL'),api_key=os.environ['GROQ_API_KEY'])

system_prompt=(f"Normalize the user-provided text into the following styles: {list(TextStyle)} ")#
# system_prompt=(f"Normalize the user-provided text into the following styles:  "+json.dumps([style.value for style in TextStyle]) )
print(json.dumps([style.value for style in TextStyle]))


agent = Agent(
    model=model,
    result_type=NormalizedTexts,
    system_prompt=dedent(system_prompt)
)

response=agent.run_sync('Large Language Models are a powerful tool for natural language processing')
# print(response.data)

print_summary(response.data.normalized_texts)