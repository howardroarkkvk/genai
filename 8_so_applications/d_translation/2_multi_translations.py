from datetime import date
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from textwrap import dedent
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import logfire
from enum import Enum
import json

load_dotenv(override=True)
logfire.configure(token = os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()

class TargetLanguage(Enum):
    de='denmark'
    fr='france'
    it='italian'
    es='spanish'
    he='hebrew'



class TranslatedString(BaseModel):
    input_language:TargetLanguage=Field(description='the langugage of the original text')
    translation:str=Field(description='the translated text in the new language')

model=GroqModel(model_name=os.getenv('GROQ_CHAT_MODEL'),api_key=os.environ['GROQ_API_KEY'])
# model=GeminiModel(model_name=os.getenv('GEMINI_CHAT_MODEL'),api_key=os.environ['GEMINI_API_KEY'])
# model=OpenAIModel(model_name=os.getenv("OLLAMA_MODEL"), base_url="http://localhost:11434/v1")
system_prompt=("translate the user provided languages in to the following langauages"+ json.dumps([language.value for language in TargetLanguage]))


agent = Agent(
    model=model,
    result_type=TranslatedString,
    system_prompt=dedent(system_prompt)
)

print(system_prompt)

user_prompt = (
    "Large Language Models are a powerful tool for natural language processing."
)

response = agent.run_sync(user_prompt)
print(response.data)