from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from dotenv import load_dotenv
import os
import logfire
from pathlib import Path

load_dotenv(override=True)
logfire.configure(token=os.environ['LOGFIRE_TOKEN'])
logfire.instrument_openai()

# open_model=OpenAIModel(model_name=os.getenv('OPENAI_CHAT_MODEL'),api_key=os.environ['OPENAI_API_KEY'])
gemini_model=GeminiModel(model_name=os.getenv('GEMINI_CHAT_MODEL'),api_key=os.environ['GEMINI_API_KEY'])

system_prompt=""" You are helpful assistant
Use the tool `list_all_files` to get all the files of a given directory.
"""
agent=Agent(model=gemini_model,system_prompt=system_prompt)

@agent.tool_plain
def list_all_files(dir:str)->list[str]:
    # logfire.info(f"listing files of {dir}")
    files = [str(path) for path in Path(dir).rglob("Screen Recordings/*")]#glob("**/*")
    # Path(dir).rglob('*')
    return files


dir=os.path.abspath(r"C:\Users\USER\Videos")#.expanduser
print(dir)
response=agent.run_sync(f"what are the files in {dir}")

print(response.data)