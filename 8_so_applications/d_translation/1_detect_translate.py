from datetime import date
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from textwrap import dedent
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
import logfire


load_dotenv(override=True)
logfire.configure(token = os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()

class TranslatedString(BaseModel):
    input_language:str=Field(description='the langugage of the original text')
    translation:str=Field(description='the translated text in the new language')

system_prompt = """
        Detect the language of the original text and translate it into English.
    """

model=GroqModel(model_name=os.getenv('GROQ_CHAT_MODEL'),api_key=os.environ['GROQ_API_KEY'])
# model=GeminiModel(model_name=os.getenv('GEMINI_CHAT_MODEL'),api_key=os.environ['GEMINI_API_KEY'])
# model=OpenAIModel(model_name=os.getenv("OLLAMA_MODEL"), base_url="http://localhost:11434/v1")
agent = Agent(
    model=model,
    result_type=TranslatedString,
    system_prompt=dedent(system_prompt)
)

user_prompt = "Sprachkenntnisse sind ein wichtiger Bestandteil der Kommunikation."

resp=agent.run_sync(user_prompt)
print(resp.data.input_language)

print(resp.data.translation)