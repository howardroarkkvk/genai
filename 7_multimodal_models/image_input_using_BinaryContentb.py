from pydantic_ai import Agent,BinaryContent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai import ImageUrl
from dotenv import load_dotenv
import logfire
import os
from textwrap import dedent
from pathlib import Path

load_dotenv(override=True)
logfire.configure(token=os.getenv('LOGFIRE_TOKEN'))
logfire.instrument_openai()
# logfire.instrument_httpx()

# model=OpenAIModel(model_name=os.environ['OPENAI_CHAT_MODEL'],api_key=os.getenv('OPENAI_API_KEY'))
# model=GroqModel(model_name=os.environ['GROQ_CHAT_MODEL'],api_key=os.getenv('GROQ_API_KEY'))
model=GeminiModel(model_name=os.environ['GEMINI_CHAT_MODEL'],api_key=os.getenv('GEMINI_API_KEY'))
system_prompt='You are smart vision agent with smart text extraction skills'

agent=Agent(model=model,system_prompt=dedent(system_prompt))

path=r"C:\Users\USER\Downloads\grocery_test.png"
path1=r'D:\All-I-want-to-know-is-where-i-am-going-to-die.png'
path2=r'D:\DataFiles\images\20250324_065416.jpg'
file_path=r'D:\DataFiles\images\20250324_065416.txt'
x=Path(path2).read_bytes()
# print(x)

resp=agent.run_sync(["Extract the complete text fromt he image",
                BinaryContent(x,media_type='image/jpg')

])
#'image/png'
print(resp.data)
with open(file_path,'w') as f:
    f.write(resp.data)
    f.close()

